"""Read-only local POI walkthrough on one deterministically chosen frozen tape.

This is an illustrative TRAIN event, not selection of an accuracy winner.
No location protection, RNG, budget ledger, or benchmark output is regenerated.
"""
import argparse
from collections import Counter
import gzip
import hashlib
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from benchmark.query_purpose import MultiPurposeRoadRanking, QueryPurpose, QuerySpec
from data.lane_states import build_lane_states

BASE = 'artifacts/benchmarks/qplanner_depth_base_q_generalization_20261006_v1/'
DEPTH = 'artifacts/benchmarks/qplanner_response_depth_generalization_20261006_v1/'
CACHE = 'artifacts/benchmarks/local_gps_robustness_20261007_v1/'
FAMILY = 'freshqp-001'
DRAW, SLOT, CLOCK = 1, 0, 60
SOURCES = {}


def read(path):
    raw = (ROOT / path).read_bytes()
    SOURCES[path] = hashlib.sha256(raw).hexdigest()
    return json.loads(gzip.decompress(raw) if path.endswith('.gz') else raw)


def pin(path):
    raw = (ROOT / path).read_bytes()
    SOURCES[path] = hashlib.sha256(raw).hexdigest()
    return raw


def digest(value):
    return hashlib.sha256(json.dumps(value, separators=(',', ':'),
                         ensure_ascii=False, allow_nan=False).encode()).hexdigest()


def extract():
    SOURCES.clear()
    file_name = f'{FAMILY}--draw{DRAW}.json.gz'
    path = 'families/' + file_name
    protocol = read(DEPTH + 'protocol.json')
    depth = read(DEPTH + path)
    base = read(BASE + path)
    readout = read(DEPTH + 'readout.json')
    resources = read(DEPTH + 'resources.json')
    certificate = read(DEPTH + 'validation.json')
    assert certificate['status'] == 'pass'
    assert certificate['readout_sha256'] == SOURCES[DEPTH + 'readout.json']
    assert readout['family_files_sha256'][file_name] == SOURCES[DEPTH + path]
    assert depth['source_bundle_sha256'] == SOURCES[BASE + path]
    data = read(protocol['dataset_path'])
    assert SOURCES[protocol['dataset_path']] == protocol['dataset_sha256']
    family = next(f for f in data['families'] if f['family_id'] == FAMILY)
    assert family['split'] == base['evaluator_only']['split'] == 'train'
    departure = family['evaluator_only']['sessions'][SLOT]['depart_s']
    network = data['network']
    raw_network = pin(network['compressed_path'])
    assert SOURCES[network['compressed_path']] == network['compressed_sha256']
    xml = gzip.decompress(raw_network)
    assert hashlib.sha256(xml).hexdigest() == network['native_sha256']
    sensor_protocol = read(CACHE + 'protocol.json')
    pin(CACHE + 'public_reply60.npz')
    assert SOURCES[CACHE + 'public_reply60.npz'] == sensor_protocol['reply_cache_sha256']
    with np.load(ROOT / CACHE / 'public_reply60.npz', allow_pickle=False) as arrays:
        metadata_json = str(arrays['metadata'])
        metadata = json.loads(metadata_json)
        signatures, access = arrays['signatures'], arrays['access']
        context_digest = hashlib.sha256(metadata_json.encode())
        context_digest.update(signatures.astype('<i4').tobytes())
        context_digest.update(access.astype('<i4').tobytes())
        assert context_digest.hexdigest() == resources['full_reply60_sha256']
        with tempfile.TemporaryDirectory(prefix='utility-walkthrough-public-map-') as work:
            public_xml = Path(work) / 'native.net.xml'
            public_xml.write_bytes(xml)
            road = build_lane_states(public_xml, spacing_m=40.)
        assert road.catalogue_sha256 == metadata['catalogue_sha256']
        catalogue = metadata['pois']
        context = SimpleNamespace(rn=road, pois=catalogue, categories=metadata['categories'])
        ranking = MultiPurposeRoadRanking(context, cache_limit=16)
        raw_events = base['public']['streams']['raw'][SLOT]['events']
        q_events = base['public']['streams']['legacy_l10'][SLOT]['events']
        i = next(i for i, event in enumerate(q_events) if event['timestamp_s'] == CLOCK)
        event = q_events[i]
        gps = raw_events[i]['candidates'][0]
        destination = raw_events[-1]['candidates'][0]
        state = road.nearest(gps['lat'], gps['lon'])[0]
        destination_state = road.nearest(destination['lat'], destination['lon'])[0]
        ledger = base['evaluator_only']['sessions'][SLOT]['ledger']['legacy_l10']
        anchor = ledger['anchors'][i]
        wire = next(r for r in depth['wire'] if r['method'] == 'service_l30'
                    and r['slot'] == SLOT and r['t'] == CLOCK)
        saved_score = next(r for r in depth['utility'] if r['method'] == 'service_l30'
                          and r['slot'] == SLOT and r['t'] == CLOCK and r['cache'] == 'current')
        requests, responses = [], []
        for q, candidate in enumerate(event['candidates']):
            query_state = road.nearest(candidate['lat'], candidate['lon'])[0]
            by_category = signatures[access[query_state], :, :30]
            indices = [int(p) for p in by_category.ravel() if p >= 0]
            assert indices == wire['reply_poi_ids_by_Q'][q]
            payload = dict(timestamp_s=departure + wire['t'], lat=candidate['lat'],
                           lon=candidate['lon'], categories=metadata['categories'], L=30)
            requests.append(dict(Q=f'Q{q+1}', lane_state=ledger['states'][i][q],
                                 provider_access_state=int(access[query_state]), payload=payload))
            responses.append(dict(Q=f'Q{q+1}', records=[catalogue[j] for j in indices],
                ordered_indices=indices, counts_by_category=dict(Counter(catalogue[j]['category'] for j in indices)),
                ordered_ids_by_category={category: [catalogue[int(j)]['id'] for j in ids if j >= 0]
                                        for category, ids in zip(metadata['categories'], by_category)}))
        payloads = [r['payload'] for r in requests]
        request_bytes = sum(len(json.dumps(p, separators=(',', ':')).encode()) for p in payloads)
        reply_bytes = sum(len(json.dumps({'results': [{k: p[k] for k in ('id', 'category', 'lat', 'lon')}
                                                     for p in reply['records']]}, separators=(',', ':'),
                                       ensure_ascii=False).encode()) for reply in responses)
        assert request_bytes == wire['request_bytes'] and reply_bytes == wire['reply_bytes']
        pool = sorted({j for reply in responses for j in reply['ordered_indices']})
        available = np.zeros(len(catalogue), dtype=bool)
        available[pool] = True
        all_pois = np.ones(len(catalogue), dtype=bool)
        results = {}
        for purpose in QueryPurpose:
            rows, defined_recalls = {}, []
            for category in metadata['categories']:
                query = QuerySpec(purpose, category, k=5,
                    radius_m=1000. if purpose == QueryPurpose.WITHIN_RADIUS else None,
                    destination_state=destination_state if purpose == QueryPurpose.MIN_DETOUR else None)
                scores = ranking.scores(state, query)
                answer = ranking.top(state, available, query)
                reference = ranking.top(state, all_pois, query)
                recall = len(set(answer) & set(reference)) / len(reference) if reference else None
                if recall is not None:
                    defined_recalls.append(recall)
                rows[category] = dict(ordered_ids=[catalogue[j]['id'] for j in answer],
                    ordered_indices=answer, answer=[dict(catalogue[j], catalogue_index=j,
                        display_alias=f'P{j}', score=float(scores[j]),
                        score_unit='s' if purpose == QueryPurpose.FASTEST else 'm') for j in answer],
                    reference_ids=[catalogue[j]['id'] for j in reference],
                    reference_size=len(reference), recall_sample=recall)
            macro = float(np.mean(defined_recalls)) if defined_recalls else None
            assert macro == saved_score['purposes'][purpose.value]['recall5']
            results[purpose.value] = rows
    for source in ['benchmark/query_purpose.py', 'data/lane_states.py', 'core/road_network.py',
                   'evaluation/lane_travel.py', 'benchmark/public_poi_context.py',
                   str(Path(__file__).resolve().relative_to(ROOT))]:
        pin(source)
    for source, sha in SOURCES.items():
        assert hashlib.sha256((ROOT / source).read_bytes()).hexdigest() == sha
    return dict(schema='frozen-L30-local-utility-walkthrough-v1',
        sample=dict(family_id=FAMILY, split='train', draw=DRAW, slot=SLOT, t_s=CLOCK, event_id=event['event_id']),
        selection_rule='First lexicographic frozen family, draw1, slot0, fixed60s event; first lexicographic public category cafe. No score/Recall-based selection.',
        current_configuration=dict(K=5, L=30, local_k=5, radius_m=1000, planner_signature_L=10),
        inputs=dict(GPS=dict(lat=gps['lat'], lon=gps['lon'], lane_state=int(state)),
                    protected_Z=dict(lat=anchor[0], lon=anchor[1]),
                    destination_oracle=dict(lat=destination['lat'], lon=destination['lon'], lane_state=int(destination_state),
                        scope='Actual final synthetic destination supplied ONLY to evaluator-local detour ranking; a deployment would require a user-provided known destination. Not read by the Q planner.')),
        requests=requests, responses=responses,
        merged=dict(raw_record_count=sum(len(r['ordered_indices']) for r in responses), unique_count=len(pool),
            removed_duplicate_count=sum(len(r['ordered_indices']) for r in responses)-len(pool),
            unique_indices=pool, unique_ids=[catalogue[j]['id'] for j in pool],
            counts_by_category=dict(Counter(catalogue[j]['category'] for j in pool)),
            records=[dict(catalogue[j], catalogue_index=j, display_alias=f'P{j}') for j in pool]),
        local_results=results, illustration_category='cafe',
        cost=dict(requests=wire['requests'], request_bytes=request_bytes, reply_bytes=reply_bytes,
                  scope='Saved application compact-JSON bytes, not network/latency/energy measurements'),
        identical_payload_sha256_for_all_local_purposes={p.value:digest(payloads) for p in QueryPurpose},
        frozen_sample_utility_row=saved_score,
        invariants=dict(exact_five_Q_service_responses_recovered=True, exact_saved_cost_recovered=True,
            exact_saved_current_purpose_scores_recovered=True, source_files_unchanged=True,
            Q_protection_not_regenerated=True, private_keys_or_sampler_not_used=True),
        scope='A single synthetic TRAIN example for algorithm explanation, not evidence of average Recall, privacy superiority, real GPS, live availability, or an independent benchmark.',
        notes=['Server top-L is per CATEGORY per Q, not30 POIs total and not guaranteed to contain30 records.',
               'Local top-k is per chosen purpose/category; four purposes need not produce different answers.',
               'Nearest/radius/detour scores are directed-road metres to public POI access states; fastest uses static free-flow seconds, not measured traffic.',
               'P-index aliases are presentation labels for exact OSM IDs, not invented place names.',
               'This event uses exact synthetic GPS locally; the60-second local sensor sensitivity is a separate diagnostic.'],
        source_sha256=dict(sorted(SOURCES.items())))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, default=Path(__file__).with_name('utility_sample.json'))
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError('Keep the existing sample; use a new output path')
    result = extract()
    with args.output.open('x') as stream:
        json.dump(result, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write('\n')
    print('Saved deterministic TRAIN utility sample:', result['sample'], 'merged', result['merged']['unique_count'])
