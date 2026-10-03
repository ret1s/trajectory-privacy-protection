"""Read frozen Geo-I experiments; derive the presentation's 14-case scope.

No model is executed. Bootstrap resamples paired route families, averaging
cases within each scenario and then the five scenarios equally.
"""
from pathlib import Path
import hashlib
import json
import numpy as np

OUT = Path(__file__).resolve().parent
ROOT = OUT.parents[2]
SCENARIOS = ('S1', 'S2', 'S3', 'S9', 'S10')
CASES = [s+'.'+c for s in SCENARIOS for c in ('AC' if s == 'S10' else 'ABC')]
LABELS = {('response_paced', 'epoch_cache'): 'GeoI-Paced',
          ('response_paced_slack03', 'epoch_cache'): 'GeoI-Slack',
          ('response_paced_slack03', 'fresh'): 'GeoI-Slack, bỏ cache',
          ('raw_current', 'fresh'): 'GPS thật'}
LEGACY_LABELS = {'unprotected': 'GPS thật', 'dls_graph_adaptation': 'DLS*',
                 'transprotect_adaptation': 'TransProtect*',
                 'semantic_correlation_local_adaptation': 'Semantic*',
                 'geo_i_anchored_dummy': 'Geo-I + dummy',
                 'geo_i_anchored_dummy_road': 'Geo-I + dummy đường',
                 'br_private': 'BR tái dùng (Geo-I, v2)'}


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def macro(values):
    return float(np.mean([np.mean([values[c] for c in CASES if c.startswith(s+'.')])
                          for s in SCENARIOS]))


def build_evidence():
    base = ROOT/'artifacts/benchmarks/research_loop'
    sources = {}
    def read(path):
        sources[str(path.relative_to(ROOT))] = sha(path)
        return json.loads(path.read_text())
    live = read(base/'iteration28_live_service.json')
    check = read(base/'iteration28_verification.json')
    assert check['status'] == 'passed' and check['sha256'] == sha(base/'iteration28_live_service.json')
    attacks = read(base/'iteration18_expanded_attacks.json')
    screen = read(base/'iteration18_expanded_screening.json')
    dataset = read(ROOT/live['protocol']['dataset'])
    assert sha(ROOT/live['protocol']['dataset']) == live['provenance']['dataset_sha256'] == attacks['dataset_sha256']
    assert sha(base/'iteration18_expanded_screening.json') == live['provenance']['screen_sha256'] == attacks['screen_sha256']
    for receipt in (live['provenance'], screen['provenance']):
        for name, digest in receipt['source_sha256'].items():
            assert sha(ROOT/name) == digest, name
            sources[name] = digest
    records = [r for r in dataset['records'] if r['case_id'] in CASES]
    sessions = {s for r in records for s in r['session_ids']}
    assert len(records) == 165 and len(sessions) == 94
    summaries = {(r['method'], r['mode'], r['case_id']): r
                 for r in live['summaries'] if r['probability'] == .8}
    privacy = {(r['method'], r['case_id']): r for r in attacks['summaries']}
    costs = {k: {'events': 0, 'request_json_bytes': 0, 'response_body_bytes': 0,
                 'coordinate_queries': 0} for k in LABELS}
    for item in live['shards']:
        assert sha(ROOT/item['file']) == item['sha256']
        sources[item['file']] = item['sha256']
        if '/p80-' not in item['file']:
            continue
        shard = json.loads((ROOT/item['file']).read_text())
        for row in shard['sessions']:
            k = row['method'], row['mode']
            if k not in costs or row['session_id'] not in sessions:
                continue
            costs[k]['events'] += row['events']
            for field in ('request_json_bytes', 'response_body_bytes', 'coordinate_queries'):
                costs[k][field] += row['communication'][field]
    service = []
    for k, label in LABELS.items():
        scores = {c: summaries[k+(c,)]['recall'] for c in CASES}
        cost = costs[k]
        service.append({'method': k[0], 'mode': k[1], 'label': label,
                        'recall': macro(scores), 'gates': sum(v >= .9 for v in scores.values()),
                        'request_bytes': cost['request_json_bytes']/cost['events'],
                        'response_bytes': cost['response_body_bytes']/cost['events'],
                        'queries': cost['coordinate_queries']/cost['events'],
                        'event_denominator': cost['events'], 'case_recall': scores})
    case_rows = []
    for c in CASES:
        pr = privacy['response_paced_slack03', c]
        sr = summaries['response_paced_slack03', 'epoch_cache', c]
        case_rows.append({'source_case': c, 'case': 'S10.B' if c == 'S10.C' else c,
                          'families': pr['family_count'], 'records': pr['record_count'],
                          'hit100': pr['metrics']['hit100'], 'mae_m': pr['metrics']['mae_m'],
                          'hit500': pr['metrics']['hit500'], 'recall': sr['recall'],
                          'raw_hit100': privacy['raw', c]['metrics']['hit100']})
    scenario_rows = []
    for scenario in SCENARIOS:
        rows = [r for r in case_rows if r['source_case'].startswith(scenario+'.')]
        scenario_rows.append({'scenario': scenario, 'conditions': len(rows),
                              **{k: float(np.mean([r[k] for r in rows])) for k in
                                 ('raw_hit100', 'hit100', 'mae_m', 'hit500', 'recall')},
                              'min_recall': min(r['recall'] for r in rows)})
    for row in service:
        row['scenario_recall'] = {s: float(np.mean([v for c, v in row['case_recall'].items()
                                                    if c.startswith(s+'.')])) for s in SCENARIOS}
        row['scenario_gates'] = sum(v >= .9 for v in row['scenario_recall'].values())
    families = sorted({f for c in CASES for f in summaries['response_paced_slack03', 'epoch_cache', c]['family_recall']})
    assert len(families) == 12
    indices = np.random.default_rng(20261003).integers(0, len(families), (10000, len(families)))
    target = ('response_paced_slack03', 'epoch_cache')
    contrasts = []
    for control in [('response_paced_slack03', 'fresh'), ('response_paced', 'epoch_cache')]:
        diffs = {c: np.array([summaries[target+(c,)]['family_recall'].get(f, np.nan)
                              - summaries[control+(c,)]['family_recall'].get(f, np.nan)
                              for f in families]) for c in CASES}
        boot = np.mean([np.mean([np.nanmean(diffs[c][indices], axis=1)
                                for c in CASES if c.startswith(s+'.')], axis=0)
                        for s in SCENARIOS], axis=0)
        contrasts.append({'control': LABELS[control], 'delta': macro({c: np.nanmean(v) for c, v in diffs.items()}),
                          'ci95': np.quantile(boot, [.025, .975]).tolist(),
                          'draws': 10000, 'seed': 20261003, 'multiplicity_adjusted': False})
    paper_path = ROOT/'artifacts/benchmarks/paper_benchmark/results.json'
    paper = read(paper_path)
    # Reviewed immutable v2 artifact, not the current adapter implementations.
    assert sha(paper_path) == '2df043bfb73c3b2f630046381daf5b4bc9f6941fb5be925b0967d6de98f27538'
    for name in ('evaluation/research_protocol.py', 'evaluation/scenario_metrics.py',
                 'experiments/run_paper_benchmark.py'):
        assert sha(ROOT/name) == paper['source_sha256'][name], name
        sources[name] = paper['source_sha256'][name]
    for name, digest in attacks['source_sha256'].items():
        assert sha(ROOT/name) == digest, name
        sources[name] = digest
    legacy = []
    for method, label in LEGACY_LABELS.items():
        scores = {r['scenario']: r for r in paper['summary'] if r['k'] == 5 and r['method'] == method}
        assert set(scores) == set(SCENARIOS) and all(r['completed'] == 12 for r in scores.values())
        legacy.append({'method': method, 'label': label,
                       'scenarios': {s: {'hit100': scores[s]['location_hit_100m'],
                                          'mae_m': scores[s]['location_mae_m'],
                                          'recall': scores[s]['poi_recall_at_k']} for s in SCENARIOS}})
    extra = ['docs/reviews/verification_paper_benchmark_v2.md',
             'docs/research/contribution_positioning_20260924.md',
             'core/boundary_release.py', 'benchmark/engines/budgeted.py',
             'benchmark/engines/contextual_lane.py', 'evaluation/live_poi.py']
    for name in extra:
        sources[name] = sha(ROOT/name)
    sources[str(Path(__file__).relative_to(ROOT))] = sha(Path(__file__))
    result = {'status': 'source_verified_scope_derived', 'updated_on': '2026-10-03',
              'new_model_runs': False, 'sources': sources, 'source_cases': CASES,
              'aggregation': 'record -> family -> case; equal cases within scenario, equal five scenarios; availability worlds and RNG replicates averaged within family',
              'cost_policy': 'all original traffic of the 94 included full sessions, pooled event-weighted compact request + response JSON; no HTTP/TLS',
              'development_groups': 12, 'source_sessions': 94, 'records': 165,
              'configuration': {'K': 5, 'L': 10, 'reference_k': 5, 'epsilon_test_per_m': .01,
                                'epsilon_release_per_m': .01, 'threshold_m': 200,
                                'session_bound_per_m': .23, 'private_read_interval_s': 60,
                                'slack': .03, 'availability_epoch_s': 60},
              'service_rows': service, 'case_rows': case_rows, 'scenario_rows': scenario_rows,
              'presentation_grain': 'scenario; equally weighted original conditions within each scenario',
              'contrasts': contrasts,
              'legacy_rows': legacy, 'legacy_protocol': {'k': 5, 'seeds': paper['seeds'],
                                                        'test_trips': 12, 'scenario_windows': 60,
                                                        'formal_budget_per_m': .24},
              'full_boundary_service_benchmark': False,
              'six_original_papers_superiority_established': False,
              'independent_confirmation': False}
    (OUT/'method_evidence.json').write_text(json.dumps(result, ensure_ascii=False, indent=2)+'\n')
    return result


if __name__ == '__main__':
    d = build_evidence()
    print(json.dumps({k: d[k] for k in ('source_sessions', 'records', 'service_rows', 'contrasts')}, ensure_ascii=False))
