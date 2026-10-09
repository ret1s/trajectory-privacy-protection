"""Read-only multi-step explanation of a fixed, already frozen TRAIN session.

No protected point, random draw, private key, seed or benchmark is regenerated.
The causal belief replay delegates to the pinned earlier sample extractor; local
POI calculations are checked against the saved frozen L30 utility rows. Native
FCD is used only to verify the evaluator's source-GPS ancestry and draw its path.
"""
from collections import Counter
import argparse
import gzip
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import sys
import xml.etree.ElementTree as ET

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
WORK = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from benchmark.query_purpose import MultiPurposeRoadRanking, QueryPurpose, QuerySpec

OLD_HELPER = ROOT / 'docs/supervisor_meeting/2026-10-09_update/build_sample_walkthrough.py'
OLD_HELPER_SHA = '0635d8612142073ac1b504d7f3e704b8a0573a21a9b1d671d85da30708327c63'
assert hashlib.sha256(OLD_HELPER.read_bytes()).hexdigest() == OLD_HELPER_SHA
spec = importlib.util.spec_from_file_location('frozen_previous_sample_extractor', OLD_HELPER)
old = importlib.util.module_from_spec(spec)
spec.loader.exec_module(old)

TIMES = [0, 20, 60, 120, 180, 240, 300, 600]
CACHE = ROOT / 'artifacts/benchmarks/local_gps_robustness_20261007_v1'


def compact(value):
    return json.dumps(value, ensure_ascii=False, separators=(',', ':'), allow_nan=False).encode()


def source_gps_ancestry(data, family, bundle, rn, slots):
    """Check all public-event GPS points against retained native FCD + dataset."""
    archive = family['evaluator_only']['native_source_archives']['fcd.xml']
    path = ROOT / archive['compressed_path']
    assert old.sha(path) == archive['compressed_sha256']
    native = gzip.decompress(path.read_bytes())
    assert hashlib.sha256(native).hexdigest() == archive['native_sha256']
    sessions = family['evaluator_only']['sessions']
    session_ids = {sessions[slot]['session_id'] for slot in slots}
    fcd = {name: {} for name in session_ids}
    # Full retained FCD is evaluator evidence only; not a mechanism input.
    for time in ET.fromstring(native).findall('timestep'):
        absolute = float(time.attrib['time'])
        for vehicle in time.findall('vehicle'):
            name = vehicle.attrib['id']
            if name in fcd:
                fcd[name][absolute] = [float(vehicle.attrib['y']), float(vehicle.attrib['x'])]
    result = []
    for slot in slots:
        session = sessions[slot]
        trace = data['traces'][session['session_id']]
        trace_by_t = {row['time_s']: row for row in trace}
        raw = bundle['public']['streams']['raw'][slot]['events']
        checks = []
        for event in raw:
            t = event['timestamp_s']
            coordinate = [event['candidates'][0]['lat'], event['candidates'][0]['lon']]
            source = trace_by_t[t]
            assert coordinate == [source['lat'], source['lon']]
            assert coordinate == fcd[session['session_id']][session['depart_s'] + t]
            checks.append({'t_s': t, 'latlon': coordinate,
                           'xy_m': list(map(float, rn.point_xy(*coordinate))),
                           'native_lane_id': source['lane_id'], 'native_speed_m_s': source['speed_m_s']})
        result.append({'slot': slot, 'session_id': session['session_id'],
                       'public_departure_s': session['depart_s'],
                       'public_event_source_GPS': checks,
                       'source_GPS_path_xy': [list(map(float, rn.point_xy(p['lat'], p['lon'])))
                                              for p in trace if p['time_s'] <= 600],
                       'source_GPS_path_times_s': [p['time_s'] for p in trace if p['time_s'] <= 600],
                       'scope': 'Native 1 Hz evaluator path and lane/speed illustrate movement only. Future points, lanes, speed and destination are never inputs to b, REM or Q.'})
    return result, path


def enrich_local(session, bundle, derived, family, rn, reference, full, access, ranking):
    slot = session['slot']
    raw = bundle['public']['streams']['raw'][slot]['events']
    public = bundle['public']['streams'][old.METHOD][slot]['events']
    source_ledger = bundle['evaluator_only']['sessions'][slot]['ledger'][old.METHOD]
    departure = family['evaluator_only']['sessions'][slot]['depart_s']
    destination = raw[-1]['candidates'][0]
    destination_state = rn.nearest(destination['lat'], destination['lon'])[0]
    catalogue, categories = reference.pois, list(reference.categories)
    illustration_category = sorted(categories)[0]
    for row in session['events']:
        index = next(i for i, e in enumerate(public) if e['timestamp_s'] == row['t_s'])
        event = public[index]
        wire = next(r for r in derived['wire'] if r['method'] == 'service_l30'
                    and r['slot'] == slot and r['t'] == row['t_s'])
        utility = next(r for r in derived['utility'] if r['method'] == 'service_l30'
                       and r['slot'] == slot and r['t'] == row['t_s'] and r['cache'] == 'current')
        payloads, replies, pools = [], [], []
        for q, candidate in enumerate(event['candidates']):
            q_state = rn.nearest(candidate['lat'], candidate['lon'])[0]
            indices = [int(p) for p in full[access[q_state], :, :30].ravel() if p >= 0]
            assert indices == wire['reply_poi_ids_by_Q'][q]
            # Keep exact numeric timestamp representation from original wire.
            payloads.append(dict(timestamp_s=departure + wire['t'], lat=candidate['lat'],
                                 lon=candidate['lon'], categories=categories, L=30))
            replies.append({'results': [{k: catalogue[p][k] for k in ('id', 'category', 'lat', 'lon')}
                                        for p in indices]})
            pools.append(indices)
        request_bytes = sum(len(json.dumps(p, separators=(',', ':')).encode()) for p in payloads)
        reply_bytes = sum(len(compact(reply)) for reply in replies)
        assert request_bytes == wire['request_bytes'] and reply_bytes == wire['reply_bytes']
        pool = sorted({p for ids in pools for p in ids})
        received = np.zeros(len(catalogue), dtype=bool)
        received[pool] = True
        assert len(pool) == utility['available_count']
        all_pois = np.ones(len(catalogue), dtype=bool)
        lat, lon = row['evaluator_GPS']['latlon']
        state = rn.nearest(lat, lon)[0]
        outputs = {}
        for purpose in QueryPurpose:
            category_results, recalls = {}, []
            for category in categories:
                query = QuerySpec(purpose, category, k=5,
                    radius_m=1000. if purpose == QueryPurpose.WITHIN_RADIUS else None,
                    destination_state=destination_state if purpose == QueryPurpose.MIN_DETOUR else None)
                scores = ranking.scores(state, query)
                answer = ranking.top(state, received, query)
                target = ranking.top(state, all_pois, query)
                recall = len(set(answer) & set(target)) / len(target) if target else None
                if recall is not None:
                    recalls.append(recall)
                category_results[category] = {
                    'answer_ids': [catalogue[p]['id'] for p in answer],
                    'answer_display_aliases': [f'P{p}' for p in answer],
                    'reference_ids': [catalogue[p]['id'] for p in target],
                    'reference_count': len(target), 'recall': recall,
                    'answer': [dict(index=p, id=catalogue[p]['id'], display_alias=f'P{p}',
                                    lat=catalogue[p]['lat'], lon=catalogue[p]['lon'],
                                    xy_m=list(map(float, rn.point_xy(catalogue[p]['lat'], catalogue[p]['lon']))),
                                    score=float(scores[p]), score_unit='s' if purpose == QueryPurpose.FASTEST else 'm')
                               for p in answer]}
            macro = float(np.mean(recalls)) if recalls else None
            assert macro == utility['purposes'][purpose.value]['recall5']
            assert len(recalls) == utility['purposes'][purpose.value]['reference_category_count']
            outputs[purpose.value] = {'recall_valid_categories_mean': macro,
                'defined_reference_categories': len(recalls), 'categories': category_results}
        previous_display = next((r for r in reversed(session['events']) if r['t_s'] < row['t_s']), None)
        row['event_position'] = index
        row['absolute_public_t_s'] = departure + row['t_s']
        row['component_triggers'] = {
            'protection_GPS_supplier_called': row['protection']['GPS_read'],
            'local_GPS_used_for_ranking': True,
            'new_REM_Z_generated': row['protection']['branch'] == 'fresh',
            'noisy_test_performed': row['protection']['GPS_read'] and row['protection']['Z_before'] is not None,
            'Z_retained': row['protection']['branch'] != 'fresh',
            'belief_motion_prediction': index > 0,
            'belief_protected_emission_update': row['protection']['GPS_read'],
            'road_constrained_K5_queries_planned': True,
            'all_category_L30_requests_sent': True,
            'current_reply_union_used_for_local_purpose': True,
            'private_purpose_or_GPS_in_requests': False,
            'epoch_status_or_static_cache_used_in_this_sample': False}
        row['movement_since_previous_display'] = None if previous_display is None else {
            'from_t_s': previous_display['t_s'],
            'source_GPS_straight_line_m': float(np.linalg.norm(np.asarray(row['gps_xy']) - previous_display['gps_xy'])),
            'Q_straight_line_m_by_track': [float(np.linalg.norm(np.asarray(a) - b))
                                         for a, b in zip(row['Q_xy'], previous_display['Q_xy'])],
            'scope': 'Straight-line displacement between shown markers only; directed feasibility is checked against each immediately prior actual 20 s Q state, not these sparse joins.'}
        row['reads_between_previous_display_and_this_event'] = [] if previous_display is None else [
            r for r in session['all_protection_reads_budget_table'] if previous_display['t_s'] < r['t_s'] <= row['t_s']]
        row['retrieval_and_local_ranking'] = {
            'service': 'Frozen static all-category records; no live availability status supplied.',
            'cache': 'current-only', 'requests': len(payloads), 'payloads': payloads,
            'payload_sha256_shared_by_all_private_purposes': hashlib.sha256(compact(payloads)).hexdigest(),
            'reply_record_count_by_Q': [len(p) for p in pools],
            'reply_catalogue_indices_by_Q': pools,
            'merged_record_count_before_deduplication': sum(map(len, pools)),
            'merged_unique_count': len(pool),
            'duplicate_records_removed': sum(map(len, pools)) - len(pool),
            'merged_unique_catalogue_indices': pool,
            'merged_unique_counts_by_category': dict(Counter(catalogue[p]['category'] for p in pool)),
            'request_bytes': request_bytes, 'reply_bytes': reply_bytes,
            'byte_scope': 'Exact saved compact application JSON, not HTTP or measured bandwidth/latency/energy.',
            'illustration_category': illustration_category,
            'local_GPS_state': int(state), 'local_destination_state': int(destination_state),
            'destination_scope': 'Final synthetic position is evaluator-local known-destination input for detour, not a planner feature or transmitted value; deployment requires a user-known destination.',
            'local_outputs': outputs,
            'defined_reference_scope': 'Empty reference is N/A; reference exists but no retrieved match gives zero. Scores are this fixed TRAIN illustration, never a benchmark mean.'}
    session['public_track_trajectory'] = [
        {'t_s': event['timestamp_s'], 'event_id': event['event_id'],
         'Q_xy': [list(map(float, rn.point_xy(q['lat'], q['lon']))) for q in event['candidates']],
         'Q_state_ids': list(source_ledger['states'][i]),
         'Z_xy': list(map(float, rn.point_xy(*source_ledger['anchors'][i]))),
         'protection_GPS_read': source_ledger['ledger'][i]['private_read'],
         'branch': source_ledger['ledger'][i]['branch'],
         'spent_units': source_ledger['ledger'][i]['spent_units']}
        for i, event in enumerate(public)]
    session['trajectory_scope'] = 'All actual 20 s public Q endpoints, no regenerated points. Straight joins are display guides; no recovered physical dummy route is claimed.'


def component_changes():
    return [
        {'component': 'REM và phép thử tái sử dụng Z có nhiễu', 'status': 'giữ',
         'old': 'Đã có REM, ngưỡng 200 m và phép thử có nhiễu.',
         'current': 'Cùng cơ chế; u giảm từ 0,01 xuống 0,00125 /m trong cấu hình Epoch8.'},
        {'component': 'Nhịp đọc GPS bảo vệ và ngân sách', 'status': 'cập nhật',
         'old': 'Đọc bảo vệ cách ít nhất 60 s; cận 0,23 /m cho mỗi phiên.',
         'current': 'Giữ nhịp ≥60 s, nhưng dự trữ ngân sách trước khi đọc; tám phiên chung cận 0,23 /m, mỗi phiên tối đa 0,02875 /m. Không hoàn lại theo nhánh riêng tư.'},
        {'component': 'Ước lượng b từ lịch sử đã bảo vệ', 'status': 'giữ',
         'old': 'Đã có dự đoán chuyển động và cập nhật từ Z.',
         'current': 'Có đọc GPS bảo vệ thì cập nhật emission, kể cả noisy-reuse; không đọc thì chỉ dự đoán. Không dùng GPS thật để tìm Q.'},
        {'component': 'Chọn năm Q khả thi trên đường', 'status': 'giữ + cập nhật tài nguyên',
         'old': 'Đã có K5, ràng buộc đường có hướng, độ phủ POI, tiến độ và slack 0,03; bản walkthrough dùng đồ thị nhỏ dựng lại.',
         'current': 'Giữ thuật toán legacy L10; dùng đúng đồ thị SUMO đã chuyển đổi gồm đoạn nội bộ nút giao. Q là đầu ra bộ chọn, không phải năm mẫu REM độc lập.'},
        {'component': 'Truy hồi cố định và phản hồi máy chủ', 'status': 'cập nhật L',
         'old': 'Đã truy vấn tất cả loại POI với L10/loại/Q.',
         'current': 'L30/loại/Q cố định; đánh giá mới so sánh L20→L30 trên chính cùng Q. Chữ ký planner vẫn L10.'},
        {'component': 'Hợp nhất và xếp hạng POI tại thiết bị', 'status': 'mở rộng',
         'old': 'Đã hợp nhất, bỏ trùng và lấy top5 gần nhất cho từng loại.',
         'current': 'Bốn mục đích local: gần nhất, nhanh nhất, trong bán kính và ít vòng đường đến đích; GPS, mục đích, bán kính và đích không vào payload.'},
        {'component': 'Cache và trạng thái POI', 'status': 'mở rộng kiểm chứng',
         'old': 'Walkthrough đã có availability và cache trạng thái theo epoch60.',
         'current': 'Có API/static-versioned cache tùy chọn và kiểm chứng received-only/current-status; mẫu này chỉ current-only/static, không dùng stale status.'},
        {'component': 'Bảo vệ điểm đầu/cuối', 'status': 'đổi chính sách; cấu hình riêng',
         'old': 'Ví dụ riêng bỏ đầu60 s và giữ chậm60 s rồi bỏ phần cuối.',
         'current': 'Mẫu Epoch8/L30 không bỏ đầu, không trì hoãn, không giữ đuôi. Endpoint20 là thí nghiệm tăng nhiễu đồng đều riêng; không ghép ngân sách/số liệu của nó vào mẫu này.'},
        {'component': 'Phân biệt GPS bảo vệ và GPS local', 'status': 'làm rõ giả định',
         'old': 'Luồng chọn POI local và luồng bảo vệ đã khác vai trò.',
         'current': 'Nhịp ≥60 s chỉ tính lần đọc của cơ chế bảo vệ; utility mẫu giả định GPS local đúng tại mỗi event20 s. Không phải hai cảm biến GPS và không phải chứng minh tiết kiệm GNSS.'}
    ]


def build(output=WORK / 'multistep_sample.json', public_work=old.PUBLIC_WORK):
    output = Path(output).resolve()
    if not output.is_relative_to(WORK.resolve()):
        raise ValueError('New output must remain inside focus_v2')
    if output.exists():
        raise FileExistsError('Write once; use a new filename for a recheck')
    public_work = Path(public_work).resolve()
    if public_work.is_relative_to(ROOT):
        raise ValueError('Public-only reconstruction caches must stay outside repository')
    base_protocol, depth_protocol = old.pinned_protocol(old.BASE), old.pinned_protocol(old.DEPTH)
    assert depth_protocol['selected_depth'] == 30
    assert depth_protocol['base_q_protocol_sha256'] == old.sha(old.BASE / 'protocol.json')
    source_path = sorted((old.BASE / 'families').glob('*.json.gz'))[0]
    derived_path = old.DEPTH / 'families' / source_path.name
    bundle, derived = old.read(source_path), old.read(derived_path)
    assert bundle['evaluator_only']['family_id'] == 'freshqp-001'
    assert bundle['evaluator_only']['split'] == 'train' and bundle['evaluator_only']['draw'] == 1
    assert derived['source_bundle_sha256'] == old.sha(source_path)
    assert derived['frozen_controls']['Q_not_regenerated'] and derived['frozen_controls']['private_reads_not_performed']
    dataset_path = ROOT / base_protocol['dataset_path']
    assert old.sha(dataset_path) == base_protocol['dataset_sha256']
    data = old.read(dataset_path)
    family = next(f for f in data['families'] if f['family_id'] == 'freshqp-001')
    print('Replaying saved protected history on public map; no sampler/private key access', flush=True)
    rn, reference, legacy, beliefs, metadata = old.native_resources(data, public_work)
    saved = old.read(old.BASE / 'resources.json')
    assert rn.catalogue_sha256 == saved['catalogue']['sha256']
    assert reference.sha256 == saved['reference_sha256'] and legacy.sha256 == saved['reply_sha256']
    assert beliefs[.00125].base.sha256 == saved['belief_sha256']['0.00125']
    model, travel = beliefs[.00125], old.SparseTravel(rn)
    main = old.session_rows(bundle, 0, TIMES, rn, model, travel)
    reuse = old.first_reuse(bundle)
    inset = None if reuse is None else old.session_rows(bundle, reuse[0], [0, reuse[1]], rn, model, travel)
    print('Protected history and per-event budget checked; recovering frozen replies/local rankings', flush=True)
    cache_protocol = old.read(CACHE / 'protocol.json')
    cache_path = CACHE / 'public_reply60.npz'
    assert old.sha(cache_path) == cache_protocol['reply_cache_sha256']
    with np.load(cache_path, allow_pickle=False) as arrays:
        metadata_json = str(arrays['metadata'])
        service_metadata = json.loads(metadata_json)
        signatures, access = arrays['signatures'], arrays['access']
        digest = hashlib.sha256(metadata_json.encode())
        digest.update(signatures.astype('<i4').tobytes())
        digest.update(access.astype('<i4').tobytes())
        assert digest.hexdigest() == old.read(old.DEPTH / 'resources.json')['full_reply60_sha256']
        assert service_metadata['catalogue_sha256'] == rn.catalogue_sha256
        assert service_metadata['pois'] == list(reference.pois) and service_metadata['categories'] == list(reference.categories)
        ranking = MultiPurposeRoadRanking(reference, cache_limit=64)
        for session in [main] + ([] if inset is None else [inset]):
            enrich_local(session, bundle, derived, family, rn, reference, signatures, access, ranking)
    ancestry, fcd_path = source_gps_ancestry(data, family, bundle, rn, [0] + ([] if reuse is None else [reuse[0]]))
    print('Native FCD ancestry checked; preparing shared map crop', flush=True)
    names = {Path(__file__), OLD_HELPER, source_path, derived_path, dataset_path, fcd_path,
             ROOT / data['network']['compressed_path'], CACHE / 'protocol.json', cache_path,
             old.BASE / 'protocol.json', old.BASE / 'validation.json', old.BASE / 'resources.json',
             old.DEPTH / 'protocol.json', old.DEPTH / 'depth_freeze.json', old.DEPTH / 'validation.json', old.DEPTH / 'resources.json',
             ROOT / 'docs/supervisor_meeting/2026-10-09_update/sample_walkthrough.json',
             ROOT / 'docs/supervisor_meeting/2026-09-26_brief/model_architecture.tex',
             ROOT / 'docs/supervisor_meeting/2026-09-26_brief/walkthrough/build_walkthrough.py',
             ROOT / 'docs/supervisor_meeting/2026-09-26_brief/walkthrough/walkthrough.json',
             ROOT / 'benchmark/query_purpose.py'}
    names.update(ROOT / p for p in base_protocol['source_sha256'])
    original = old.read(ROOT / 'docs/supervisor_meeting/2026-10-09_update/sample_walkthrough.json')
    for previous in original['main_sample']['events']:
        now = next(row for row in main['events'] if row['t_s'] == previous['t_s'])
        for field in ('protection', 'belief', 'Q_sent', 'planner', 'gps_xy', 'Z_xy', 'Q_xy'):
            assert now[field] == previous[field], ('Earlier sample changed', previous['t_s'], field)
    value = {
        'schema': 'frozen-GeoI-REM-multistep-walkthrough-v2', 'prepared_date': '2026-10-09',
        'selection': {'family_id': 'freshqp-001', 'split': 'train', 'draw': 1, 'main_slot': 0,
                      'main_times_s': TIMES,
                      'rule': 'First lexicographic already frozen TRAIN family/draw/session; public fixed illustrative times before looking at utility. First actual noisy-reuse by slot/event in same family is a separate-session inset. No benchmark score is used to choose examples.',
                      'first_noisy_reuse': None if reuse is None else {'slot': reuse[0], 't_s': reuse[1]}},
        'configuration': original['configuration'], 'scope': original['scope'],
        'metadata': original['metadata'],
        'main_sample': main, 'test_reuse_inset': inset, 'native_GPS_ancestry': ancestry,
        'map_main': old.geometry(rn, main['events']),
        'map_reuse_inset': None if inset is None else old.geometry(rn, inset['events'], include_segments=False),
        'current_vs_2026_09_26': component_changes(),
        'presentation_notes_vi': [
            'Đi hết chuỗi: GPS bảo vệ → phép thử/Z → b → 5Q → 5 truy vấn cố định → hợp nhất → top5 local.',
            't20: không đọc GPS bảo vệ, không chi thêm, Z giữ nguyên; b vẫn dự đoán và Q vẫn có thể di chuyển trên đường.',
            't60–t600: vị trí đã đi xa; các phép thử thực tế của phiên0 đều tạo Z mới, mỗi lần chi2 đơn vị.',
            't300→t600 có các lần đọc360/420/480/540 ở giữa: phải cộng đủ, không nhầm chi phí của hai hình liên tiếp.',
            'Ô phụ phiên5/t60 cho thấy giữ Z sau phép thử có nhiễu: đã đọc GPS, chi1 đơn vị, b vẫn cập nhật. Không trộn nó vào timeline phiên0.',
            'b chỉ là ước lượng dùng để ưu tiên POI; chấm top12 là phần nhỏ của phân bố, không phải vùng chứa người dùng với xác suất cao.',
            'Top5 POI dùng GPS/mục đích local; đây không phải5Q. Máy chủ không nhận GPS thật, đích, bán kính hay mục đích.',
            'Đây là ví dụ TRAIN mô phỏng và dịch vụ static/current-only; không thay thế bảng benchmark hoặc chứng minh mọi kịch bản.'],
        'source_pins_sha256': {old.relative(path): old.sha(path) for path in sorted(names)},
        'validation': {'pinned_source_and_independent_receipts': 'pass',
                       'all_31_main_events_processed_causally_for_belief': True,
                       'saved_POI_proxy_matches_reconstructed_belief': 'pass',
                       'directed_Q_reachability_at_shown_events': 'pass',
                       'all_source_GPS_public_events_equal_native_FCD_and_dataset': 'pass',
                       'saved_L30_reply_prefixes_and_JSON_bytes': 'pass',
                       'four_local_purpose_scores_equal_saved_current_scores': 'pass',
                       'earlier_main_sample_fields_preserved': 'pass',
                       'sampler_run': False, 'private_keys_or_resolved_seeds_read': False,
                       'old_sources_scores_or_proofs_changed': False}}
    value['scope']['multistep_extension'] = 'Read-only display extension on existing TRAIN tapes. All actual intervening events are used in causal belief/budget replay; sparse displayed times do not skip cost. Native ancestry is evaluator-only.'
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open('xb') as handle:
        handle.write(compact(value) + b'\n')
    print('Written', old.relative(output), 'SHA256', old.sha(output), flush=True)
    for session in [main] + ([] if inset is None else [inset]):
        print('slot', session['slot'], [(r['t_s'], r['protection']['branch'], r['protection']['spent_after_units'],
                                        r['retrieval_and_local_ranking']['merged_unique_count']) for r in session['events']], flush=True)
    return value


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=WORK / 'multistep_sample.json')
    parser.add_argument('--public-work', type=Path, default=old.PUBLIC_WORK)
    args = parser.parse_args()
    build(args.output, args.public_work)
