"""Expanded endpoint evidence, read directly from verified frozen experiments."""
import hashlib
import json
from pathlib import Path


def add_endpoint_content(ns):
    p, key, sec, sub, page, table = (ns[k] for k in ('p', 'key', 'sec', 'sub', 'page', 'table'))
    root, out = ns['ROOT'], ns['OUT']
    base = root / 'artifacts/benchmarks/endpoint_calendar_expanded_v1'
    def load(name): return json.loads((base / name).read_text())
    def sha(path): return hashlib.sha256(path.read_bytes()).hexdigest()
    def pct(v): return '—' if v is None else f'{100*v:.1f}%'.replace('.', ',')
    def num(v, digits=0): return '—' if v is None else f'{v:.{digits}f}'.replace('.', ',')
    read, check, meta = load('readout.json'), load('verification.json'), load('presentation.json')
    diagnosis_dir = root / 'artifacts/benchmarks/s10b_diagnostic_v1'
    diagnosis = json.loads((diagnosis_dir / 'readout.json').read_text())
    diagnosis_check = json.loads((diagnosis_dir / 'verification.json').read_text())
    assert diagnosis_check['status'] == 'passed' and diagnosis_check['readout_sha256'] == sha(diagnosis_dir / 'readout.json')
    assert check['status'] == 'passed' and check['readout_sha256'] == sha(base / 'readout.json')
    for path, digest in meta['sources'].items(): assert sha(root / path) == digest, path
    scope=ns['SCOPE'];read=scope['endpoint'];inventory=scope['inventory']['expanded']
    scope_base=root/'artifacts/benchmarks/active_scope_ac_v2'
    scope_check=json.loads((scope_base/'verification.json').read_text())
    assert scope_check['status']=='passed' and scope_check['readout_sha256']==sha(scope_base/'readout.json')
    for path,digest in scope['sources'].items():assert sha(root/path)==digest,path
    a = {r['method']: r for r in read['aggregates']}
    s = {(r['method'], r['case_id']): r for r in read['summaries']}
    labels = {'raw': 'Vị trí thật', 'dls': 'DLS', 'rdg': 'RDG', 'transprotect_markov': 'TransProtect*',
              'semantic_poi': 'Semantic*', 'fake_queries': 'Fake-query*', 'anotherme_offline': 'AnotherMe†',
              'calendar30': 'Đề xuất 30 + lịch', 'calendar67': 'Đề xuất 67 + lịch'}
    evidence = json.loads((out / 'method_evidence.json').read_text())
    files = [base / n for n in ('protocol.json', 'selection.json', 'readout.json', 'verification.json',
                               'calendar_flow_checks.json', 'calendar_ablation.json', 'presentation.json')]
    files += [root / n for n in ('benchmark/scheduled_category_client.py', 'evaluation/road_endpoint_attack.py',
                                'experiments/endpoint_calendar_study.py', 'experiments/extend_endpoint_cohort.py')]
    files.append(Path(__file__))
    files += [diagnosis_dir / n for n in ('protocol.json', 'selection.json', 'readout.json', 'verification.json', 'implementation_amendment.json')]
    files += [root / n for n in ('evaluation/prefix_destination_attack.py', 'experiments/diagnose_s10b.py',
                                'experiments/verify_s10b.py', 'docs/research/s10b_diagnostic.md')]
    files += [scope_base/n for n in ('readout.json','verification.json')]
    files += [root/n for n in ('evaluation/report_scope.py','experiments/reaggregate_active_scope.py','experiments/verify_active_scope.py')]
    files += [root/n for n in ('experiments/summarize_active_scope.py','docs/research/active_scope_results.md')]
    evidence['sources'].update({str(f.relative_to(root)): sha(f) for f in files})
    evidence['endpoint_expansion'] = {**meta, 'pilot_pooled': False, 'city_generator_unchanged': True,
                                      'protects_fixed_subscription_payload_not_whole_network': True}
    evidence['active_scope']={'cases':scope['active_all_cases'],'priority_cases':scope['active_priority_cases'],
        'decision':scope['scope_change'],'inventory':scope['inventory'],'aggregation':scope['aggregation'],
        'cost_policy':scope['cost_policy'],'original_results_preserved':True,'not_new_confirmation':True}
    (out / 'method_evidence.json').write_text(json.dumps(evidence, ensure_ascii=False, indent=2) + '\n')

    page(); sec('S9 và S10.A/C trên 32 nhóm tuyến')
    p('**Cohort dựng sau khi khóa kế hoạch và attacker:** đủ 32 seed, '+str(meta['completed_trips'])+
      ' chuyến. Phạm vi hiện tại giữ '+str(inventory['active_records'])+'/'+str(meta['all_records'])+
      ' record gốc; utility năm scenario dùng '+str(inventory['priority_records'])+
      ' record từ '+str(inventory['priority_source_sessions'])+' chuyến, privacy S9/S10 dùng '+str(inventory['endpoint_records'])+
      '. Không gộp pilot bảy nhóm. Trùng nguyên tuyến với tập phát triển/pilot: '+
      str(check['exact_route_overlaps_with_historical_sessions'])+'/'+str(meta['exact_route_overlaps_with_pilot'])+
      '; vẫn cùng thành phố và bộ sinh SUMO. Thu hẹp S10 sau chẩn đoán; đây là tổng hợp lại, không phải đợt test mới.')
    p('Giữ bank cũ; thêm suy luận xuôi/ngược theo **mạng đường có hướng**, vận tốc quan sát và nhiều horizon; ca C kết hợp chuyến liên kết. Fit/chọn trên 10/10 nhóm khác; giữ nguyên lựa chọn cho 32 nhóm mới. Đối thủ không nhận đoạn bị giấu, endpoint thật hoặc thời gian cắt thật.')
    p('Mỗi ô privacy là **Hit100 ↓ / MAE (m) ↑**: S9 gộp A/B/C, S10 gộp A/C. Recall gộp 14 ca rồi đều qua năm scenario ở p=0,8. Chi phí chỉ trên chuyến được giữ, gồm toàn clock gốc và cả giờ cover traffic; chưa có HTTP/TLS. Đây không phải so ở cùng byte.')
    table(['Phương pháp', 'S9: Hit / MAE', 'S10.A/C: Hit / MAE', 'Recall@5 ↑', 'Byte ↓'], [
        [label, pct(a[m]['S9']['hit100'])+' / '+num(a[m]['S9']['mae_m']),
         pct(a[m]['S10']['hit100'])+' / '+num(a[m]['S10']['mae_m']), pct(a[m]['recall_0.8']),
         num(a[m]['request_bytes_per_service_event']+a[m]['response_bytes_per_service_event'])]
        for m, label in labels.items()], [4.1, 3.5, 3.5, 2.8, 3.0])
    p('* Giữ giới hạn adapter ở mục 9. † AnotherMe offline: privacy chỉ trên tập con thành công; utility giữ yêu cầu lỗi bằng 0. Stress p=0,95: đề xuất 30 đạt '+pct(a['calendar30']['recall_0.95'])+', '+str(a['calendar30']['gates_0.95'])+'/14 ca ≥90%; bản 67 đạt '+pct(a['calendar67']['recall_0.95'])+'.')
    sub('Control theo ca: không phải mọi Hit=0 đều có sức phân biệt')
    table(['Ca / nhóm', 'Raw: Hit100 / Hit200', '67 + lịch: Hit100 / Hit200', 'MAE raw → đề xuất (m)'], [
        [c+' / '+str(s['raw', c]['valid_families']), pct(s['raw', c]['hit100'])+' / '+pct(s['raw', c]['hit200']),
         pct(s['calendar67', c]['hit100'])+' / '+pct(s['calendar67', c]['hit200']),
         num(s['raw', c]['mae_m'])+' → '+num(s['calendar67', c]['mae_m'])]
        for c in ('S10.A', 'S10.C')], [2.5, 4.2, 4.7, 5.5])
    separated = {sc: sum(r['ci95'][1] < 0 for r in read['comparisons'] if r['target'] == 'calendar67' and r['scenario'] == sc and r['metric'] == 'hit100') for sc in ('S9', 'S10')}
    ac = [a[m]['S10']['hit100'] for m in ('dls', 'rdg', 'transprotect_markov', 'semantic_poi', 'fake_queries')]
    p('**Lợi thế trong phạm vi hiện tại:** Hit100 S10.A/C của năm adapter là '+num(100*min(ac), 2)+'–'+num(100*max(ac), 2)+'%, lịch công khai 0%. Raw control có hit ở cả A/C nên các ca có khả năng phân biệt. Hit=0 không đồng nghĩa an toàn trước mọi đối thủ; đây là kết quả trong bank đã thử.')
    p('CI 95% của ΔHit100 **được tính lại đúng phạm vi** nằm dưới 0 ở '+str(separated['S9'])+'/5 đối chứng S9 và '+str(separated['S10'])+'/5 đối chứng S10.A/C. Bootstrap 3.000 lần theo nhóm, chưa hiệu chỉnh nhiều so sánh; không dùng CI A/B/C cũ. Số chi tiết và điều kiện: active_scope_results.md.')
    ratio = scope['ablation']['calendar_to_active_epoch_byte_ratio']['calendar67']
    p('**Giá của việc che giờ hoạt động:** so cùng kế hoạch chỉ refresh khi có hoạt động, lịch công khai dùng '+num(ratio, 2)+
      ' lần byte trung bình theo nhóm; giữ '+str(scope['ablation']['equal_utility_event_checks'])+
      ' lượt có utility bằng nhau. Lịch che quan hệ giữa giờ gửi và giờ đi lại trong khoảng đăng ký trước; không tạo thêm độ phủ POI.')
    key('Recall 100% của tập thật + dummy là hệ quả chứa truy vấn tại vị trí thật; tăng mẫu không tự làm chỉ số này giảm. Đọc cùng privacy và chi phí. Kết luận giữ điều kiện vùng/lịch công khai, không suy thành bảo vệ IP, account hay mọi thành phố.')
