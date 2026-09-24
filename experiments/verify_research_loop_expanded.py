"""Verify the separate expanded-development source and completed screening."""
import gzip
import json
import numpy as np
from experiments.research_loop_resources import ROOT, sha
from experiments.verify_research_loop_extended import ledger_check
from experiments.research_loop_cases import public_view, mean_optional, passes_recall_gate

BASE = ROOT/'artifacts/benchmarks/research_loop'
DATA = ROOT/'artifacts/datasets/research_loop_expanded_v1/dataset.json'


def check():
    if not DATA.exists():
        return {'status': 'expanded_source_not_complete'}
    data = json.loads(DATA.read_text()); records = {r['record_id']: r for r in data['records']}
    assert len(data['families']) == 12 and len(data['traces']) == 264 and len(records) == 415
    assert len({f['family_id'] for f in data['families']}) == 12
    for family in data['families']:
        assert family['split'] == 'expanded_development' and len(family['sessions']) == 22
        assert all(float(family['actual_vehicles'][s['session_id']]['arrival']) >= 0 for s in family['sessions'])
    for record in records.values():
        for sid, indices in zip(record['session_ids'], record['observed_indices']):
            assert all(0 <= i < len(data['traces'][sid]) for i in indices)
    result = {'dataset_sha256': sha(DATA), 'families': 12, 'completed_sessions': 264,
              'records': 415, 'role': 'expanded_development_never_final_confirmation'}
    source_path = BASE/'iteration18_expanded_screening.json'
    if not source_path.exists():
        return {**result, 'screening_status': 'incomplete',
                'completed_shards_present': len(list((BASE/'expanded_screening').glob('*.json.gz')))}
    screen = json.loads(source_path.read_text()); json.dumps(screen, allow_nan=False)
    assert screen['provenance']['dataset_sha256'] == sha(DATA)
    assert len(screen['shards']) == 12
    prior_selection = {}
    for name, digest in screen['provenance']['attack_selection_source_sha256'].items():
        path = BASE/name; assert sha(path) == digest
        for row in json.loads(path.read_text())['summaries']:
            prior_selection[row['method']+'/'+row['case_id']] = row['selected_attack']
    assert all(prior_selection[key] == value for key, value in screen['selection'].items())
    all_rows = []; total_executions = 0
    for item in screen['shards']:
        path = BASE/'expanded_screening'/item['file']; assert sha(path) == item['sha256']
        shard = json.loads(gzip.decompress(path.read_bytes()))
        assert shard['provenance'] == screen['provenance']
        lookup = {}; eligibility = {}; anchors = {}
        for ex in shard['executions']:
            key = ex['session_id'], ex['rep'], ex['method']; assert key not in lookup
            lookup[key] = ex; ledger_check(ex, False)
            assert list(map(int, ex['events'])) == ex['clock_indices']
            if 'paced' in ex['method']:
                paired = ex['session_id'], ex['rep']
                assert paired not in anchors or anchors[paired] == ex['evaluator_anchor_sha256']
                anchors[paired] = ex['evaluator_anchor_sha256']
            if ex['method'] == 'raw':
                assert all(u['10'] is None or abs(u['10']-1.) < 1e-12 for u in ex['utility_by_index'].values())
        for row in shard['rows']:
            record = records[row['record_id']]
            assert row['selected_attack'] == screen['selection'][row['method']+'/'+row['case_id']]
            flags = row['eligible_events'], row['empty_reference_events']; key = row['record_id'], row['rep']
            assert key not in eligibility or eligibility[key] == flags; eligibility[key] = flags
            for slot, (sid, indices) in enumerate(zip(record['session_ids'], record['observed_indices'])):
                ex = lookup[sid, row['rep'], row['method']]
                assert public_view({int(i): e for i, e in ex['events'].items()}, indices) == row['public_views'][slot]
            assert all(a in row['errors'] for a in row['selected_attack'].values())
        total_executions += len(shard['executions']); all_rows.extend(shard['rows'])
    for summary in screen['summaries']:
        rows = [r for r in all_rows if r['method'] == summary['method'] and r['case_id'] == summary['case_id']]
        value = mean_optional([r['recall']['10'] for r in rows])
        assert value == summary['recall']['10']
        assert passes_recall_gate(value) == summary['pass_90pct_case_recall_L10']
        assert len({r['family_id'] for r in rows}) == summary['family_count']
        for family, value in summary['family_L10'].items():
            assert value == mean_optional([r['recall']['10'] for r in rows if r['family_id'] == family])
    result.update(screening_status='completed_checked', screening_sha256=sha(source_path),
        full_session_executions=total_executions, case_rows=len(all_rows),
        matched_paced_anchor_streams=True, attack_selection_frozen_outside_new_groups=True)
    attack_path = BASE/'iteration18_expanded_attacks.json'
    if attack_path.exists():
        from experiments.research_loop_expanded_attack import SPECS, summarize
        attack = json.loads(attack_path.read_text()); json.dumps(attack, allow_nan=False)
        assert attack['dataset_sha256'] == sha(DATA)
        assert attack['screen_sha256'] == sha(source_path)
        assert attack['code_sha256'] == sha(ROOT/'experiments/research_loop_expanded_attack.py')
        for p, digest in attack['source_sha256'].items():
            assert sha(ROOT/p) == digest
        frozen = {}
        for item in attack['selection_sources']:
            path = BASE/item['file']; assert sha(path) == item['sha256']
            frozen[item['method']] = json.loads(path.read_text())['selection']
        raw_path = BASE/'iteration14_sequence_attacks.json'
        assert sha(raw_path) == attack['raw_current_selection_sha256']
        raw_geometry = json.loads(raw_path.read_text())['selection']
        for item in attack['training']:
            path = BASE/item['file']; assert sha(path) == item['sha256']
            with np.load(path, allow_pickle=False) as arrays:
                assert len(set(arrays['families'])) == 64
                assert not set(arrays['families']) & {f['family_id'] for f in data['families']}
        original = {(r['method'], r['record_id'], r['rep']): r for r in all_rows}
        seen = set()
        for row in attack['rows']:
            key = row['method'], row['record_id'], row['rep']
            assert key not in seen; seen.add(key)
            assert row['recall'] == original[key]['recall']
            assert row['split'] == 'expanded_development'
            if row['method'] == 'raw':
                expected = (raw_geometry['raw/'+row['task']] if row['task'].startswith('current')
                            else frozen['raw'][row['task']])
            else:
                expected = frozen[row['method']][SPECS[row['method']][1]+'/'+row['task']]
            assert row['selected_attack'] == expected
            assert all(a in row['errors'] for a in expected.values())
            assert all(np.isfinite(v).all() for v in row['errors'].values())
        assert seen == set(original)
        assert attack['summaries'] == summarize(attack['rows'])
        result['learned_attacks'] = {'sha256': sha(attack_path), 'case_rows': len(seen),
            'training_and_selection_frozen_outside_expanded_groups': True,
            'raw_endpoint_learners_included': True}
    paired_path = BASE/'iteration18_paired_comparisons.json'
    if paired_path.exists():
        paired = json.loads(paired_path.read_text())
        assert paired['source_sha256'] == sha(attack_path)
        assert paired['code_sha256'] == sha(ROOT/'experiments/research_loop_expanded_readout.py')
        assert len(paired['rows']) == 45
        for row in paired['rows']:
            for metric in row['metrics'].values():
                values = np.array(list(metric['family_deltas'].values()))
                assert len(values) == row['families']
                assert metric['mean_delta'] == float(values.mean())
                draws = np.random.default_rng(paired['seed']).integers(0, len(values), (paired['draws'], len(values)))
                assert metric['family_bootstrap_percentile_95'] == np.quantile(values[draws].mean(axis=1), [.025, .975]).tolist()
        result['paired_comparisons'] = {'sha256': sha(paired_path), 'comparisons': 45,
                                        'bootstrap_resamples_checked': True}
    rare_path = BASE/'iteration19_expanded_rare_diagnostic.json'
    if rare_path.exists():
        rare = json.loads(rare_path.read_text()); json.dumps(rare, allow_nan=False)
        assert rare['source_sha256'] == sha(source_path) and rare['dataset_sha256'] == sha(DATA)
        assert rare['code_sha256'] == sha(ROOT/'experiments/research_loop_reserve_rare.py')
        for p, digest in rare['implementation_sha256'].items():
            assert sha(ROOT/p) == digest
        controls = {}
        for item in screen['shards']:
            shard = json.loads(gzip.decompress((BASE/'expanded_screening'/item['file']).read_bytes()))
            controls.update({(ex['session_id'], ex['rep'], ex['method']): ex for ex in shard['executions']})
        lookup = {}; new = replayed = 0
        for ex in rare['executions']:
            key = ex['session_id'], ex['rep'], ex['method']; assert key not in lookup
            lookup[key] = ex; ledger_check(ex, False)
            if key in controls:
                assert ex == controls[key]; replayed += 1
            else:
                new += 1
                assert ex['clock_indices'] == controls[ex['session_id'], ex['rep'], 'raw']['clock_indices']
        assert new == rare['new_executions'] == 48
        assert replayed == rare['replayed_control_executions'] == 48
        assert {r['record_id'] for r in rare['rows']} == {r['record_id'] for r in records.values() if r['case_id'] == 'S1.C'}
        assert len(rare['rows']) == 96
        for row in rare['rows']:
            record = records[row['record_id']]; ex = lookup[row['session_id'], row['rep'], row['method']]
            assert row['recall_L10'] == mean_optional([ex['utility_by_index'][str(i)]['10'] for i in record['observed_indices'][0]])
        for summary in rare['summaries']:
            rows = [r for r in rare['rows'] if r['method'] == summary['method']]
            assert len(rows) == 24
            assert summary['recall_L10'] == float(np.mean([r['recall_L10'] for r in rows]))
        result['reserve_expanded_rare'] = {'sha256': sha(rare_path), 'new_executions': new,
            'replayed_controls': replayed, 'all_12_source_families': True}
    return result


if __name__ == '__main__':
    print(json.dumps(check(), indent=2))
