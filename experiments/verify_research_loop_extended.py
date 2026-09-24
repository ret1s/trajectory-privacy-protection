"""Evidence checks for phase/pacing experiments and auxiliary attacker data."""
import gzip
import json
import numpy as np
from experiments.research_loop_resources import ROOT, sha

OUT = ROOT/'artifacts/benchmarks/research_loop'


def ledger_check(row, quarter):
    ledger = row.get('evaluator_ledger')
    if not ledger:
        return
    assert sum(v['cost_units'] for v in ledger) == ledger[-1]['spent_units']
    unit = .0025 if quarter else .01
    assert abs(unit*ledger[-1]['spent_units']-row['budget_bound']) < 1e-12
    assert row['budget_bound'] <= .23+1e-12
    spent = 0
    for v in ledger:
        assert v['cost_units'] >= 0
        spent += v['cost_units']
        assert spent == v['spent_units']
        if not v['private_read']:
            assert v['cost_units'] == 0


def check():
    from experiments.research_loop_cases import public_view, mean_optional
    result = {}
    old = json.loads((OUT/'iteration10_cases_checked.json').read_text())
    old_lookup = {(r['session_id'], r['rep'], r['method']): r for r in old['executions']}
    data = json.loads((ROOT/'artifacts/datasets/research_loop_development_v1/dataset.json').read_text())
    records = {r['record_id']: r for r in data['records']}
    for name in ('iteration12_origin_guard.json', 'iteration13_paced_cases.json'):
        p = OUT/name
        x = json.loads(p.read_text()); json.dumps(x, allow_nan=False)
        executions = x.get('executions', x['rows'])
        lookup = {}
        for row in executions:
            ledger_check(row, row['quarter_units'])
            if name.startswith('iteration13'):
                key = row['session_id'], row['rep'], row['method']
                assert key not in lookup
                lookup[key] = row
                if key in old_lookup:
                    assert row['events'] == old_lookup[key]['events']
                    assert row['utility_by_index'] == old_lookup[key]['utility_by_index']
                for depth in ('5', '10'):
                    expected = mean_optional([u[depth] for u in row['utility_by_index'].values()])
                    assert expected == row['whole_session_recall'][depth]
        if name.startswith('iteration13'):
            eligibility = {}
            for row in x['rows']:
                record = records[row['record_id']]
                for slot, (sid, indices) in enumerate(zip(record['session_ids'], record['observed_indices'])):
                    ex = lookup[sid, row['rep'], row['method']]
                    pview = public_view({int(i): e for i, e in ex['events'].items()}, indices)
                    assert pview == row['public_views'][slot]
                flags = row['eligible_events'], row['empty_reference_events']
                key = row['record_id'], row['rep']
                assert key not in eligibility or eligibility[key] == flags
                eligibility[key] = flags
        result[name] = {'sha256': sha(p), 'full_session_executions': len(executions)}
    manifest_path = OUT/'sequential_shadows/manifest.json'
    manifest = json.loads(manifest_path.read_text())
    auxiliary_path = ROOT/'artifacts/datasets/research_loop_shadow_v1/dataset.json'
    auxiliary = json.loads(auxiliary_path.read_text())
    assert manifest['provenance']['dataset_sha256'] == sha(auxiliary_path)
    assert len(auxiliary['families']) == 80 and len(auxiliary['traces']) == 160
    assert len({tuple(f['departure_cell']) for f in auxiliary['families']}) == 80
    for f in auxiliary['families']:
        assert all(float(f['actual_vehicles'][s['session_id']]['arrival']) >= 0 for s in f['sessions'])
    splits = {}
    runs = 0
    for item in manifest['shards']:
        p = manifest_path.parent/item['file']; assert sha(p) == item['sha256']
        shard = json.loads(gzip.decompress(p.read_bytes()))
        assert shard['provenance'] == manifest['provenance']
        splits.setdefault(shard['split'], set()).add(shard['family_id'])
        assert len(shard['runs']) == 8
        for row in shard['runs']:
            ledger_check(row, row['method'] in ('origin_first', 'paced_origin'))
            assert len(row['public']['events']) == len(row['evaluation_truth']['current_xy'])
            assert set(row['public']) == {'events'}
            for event in row['public']['events']:
                assert set(event) == {'event_id', 'timestamp_s', 'candidates'}
                assert len(event['candidates']) == 5
                assert all(set(c) == {'candidate_id', 'lat', 'lon'} for c in event['candidates'])
            runs += 1
    assert len(splits['auxiliary_train']) == 64 and len(splits['auxiliary_selection']) == 16
    assert not splits['auxiliary_train'] & splits['auxiliary_selection']
    result['auxiliary_sequence_shadows'] = {'manifest_sha256': sha(manifest_path),
        'completed_sumo_sessions': 160, 'full_session_executions': runs,
        'train_families': 64, 'selection_families': 16}
    attacks = OUT/'iteration14_sequence_attacks.json'
    if attacks.exists():
        x = json.loads(attacks.read_text()); json.dumps(x, allow_nan=False)
        assert x['shadow_manifest_sha256'] == sha(manifest_path)
        assert x['core_source_sha256'] == sha(OUT/'iteration13_paced_cases.json')
        for item in x['training']:
            p = manifest_path.parent/item['file']; assert sha(p) == item['sha256']
            with np.load(p, allow_pickle=False) as arrays:
                assert set(arrays['families']) == splits['auxiliary_train']
                assert len(arrays['x']) == item['rows']
                assert all(np.isfinite(arrays[k]).all() for k in ('x', 'y', 'centers'))
        for row in x['auxiliary_selection_rows']:
            assert row['family_id'] in splits['auxiliary_selection']
        for row in x['core_rows']:
            assert row['family_id'] not in splits['auxiliary_selection'] | splits['auxiliary_train']
            assert row['selected_attack'] == x['selection'][row['method']+'/'+row['task']]
            assert all(a in row['errors'] for a in row['selected_attack'].values())
        result['sequential_attacks'] = {'sha256': sha(attacks), 'training_sets': len(x['training']),
            'core_case_rows': len(x['core_rows']), 'independent_defender_confirmation': False}
    raw_audit = OUT/'iteration14_raw_sequence_audit.json'
    if raw_audit.exists():
        x = json.loads(raw_audit.read_text()); json.dumps(x, allow_nan=False)
        assert x['shadow_manifest_sha256'] == sha(manifest_path)
        for item in x['training']:
            p = manifest_path.parent/item['file']; assert sha(p) == item['sha256']
            with np.load(p, allow_pickle=False) as arrays:
                assert set(arrays['families']) == splits['auxiliary_train']
        for row in x['auxiliary_selection_rows']:
            assert row['family_id'] in splits['auxiliary_selection']
        for row in x['core_rows']:
            assert row['selected_attack'] == x['selection'][row['task']]
        result['raw_sequence_control'] = {'sha256': sha(raw_audit), 'core_rows': len(x['core_rows'])}
    response = OUT/'iteration15_response_cases.json'
    if response.exists():
        x = json.loads(response.read_text()); json.dumps(x, allow_nan=False)
        source = json.loads((OUT/'iteration13_paced_cases.json').read_text())
        controls = {(r['session_id'], r['rep'], r['method']): r for r in source['executions']}
        assert x['control_source_sha256'] == sha(OUT/'iteration13_paced_cases.json')
        new = replayed = 0; lookup = {}
        for row in x['executions']:
            key = row['session_id'], row['rep'], row['method']; lookup[key] = row
            ledger_check(row, row['quarter_units'])
            if key in controls:
                assert row == controls[key]
                replayed += 1
            else:
                new += 1
        assert new == x['new_executions'] == 198
        assert replayed == x['replayed_control_executions'] == 264
        for row in x['rows']:
            record = records[row['record_id']]
            for slot, (sid, indices) in enumerate(zip(record['session_ids'], record['observed_indices'])):
                ex = lookup[sid, row['rep'], row['method']]
                assert public_view({int(i): e for i, e in ex['events'].items()}, indices) == row['public_views'][slot]
        result['response_aware_cases'] = {'sha256': sha(response), 'new_executions': new,
            'replayed_controls': replayed, 'case_rows': len(x['rows']), 'exact_allowed_views_checked': True}
    response_manifest = OUT/'response_shadows/manifest.json'
    if response_manifest.exists():
        rm = json.loads(response_manifest.read_text()); count = 0
        assert rm['provenance']['dataset_sha256'] == sha(auxiliary_path)
        for item in rm['shards']:
            p = response_manifest.parent/item['file']; assert sha(p) == item['sha256']
            shard = json.loads(gzip.decompress(p.read_bytes()))
            assert shard['provenance'] == rm['provenance'] and len(shard['runs']) == 4
            assert shard['family_id'] in splits[shard['split']]
            for row in shard['runs']:
                ledger_check(row, False)
                assert len(row['public']['events']) == len(row['evaluation_truth']['current_xy'])
                assert set(row['public']) == {'events'}
                count += 1
        assert count == 320
        result['response_shadows'] = {'manifest_sha256': sha(response_manifest), 'new_executions': count}
    attacks = OUT/'iteration16_response_attacks.json'
    if attacks.exists():
        x = json.loads(attacks.read_text()); json.dumps(x, allow_nan=False)
        assert x['shadow_manifest_sha256'] == sha(response_manifest)
        assert x['core_source_sha256'] == sha(OUT/'iteration15_response_cases.json')
        for item in x['training']:
            p = response_manifest.parent/item['file']; assert sha(p) == item['sha256']
            with np.load(p, allow_pickle=False) as a:
                assert set(a['families']) == splits['auxiliary_train']
        for row in x['auxiliary_selection_rows']:
            assert row['family_id'] in splits['auxiliary_selection']
        for row in x['core_rows']+x['whole_session_probe']:
            assert row['selected_attack'] == x['selection'][row['method']+'/'+row['task']]
            assert all(a in row['errors'] for a in row['selected_attack'].values())
        result['response_matched_attacks'] = {'sha256': sha(attacks), 'case_rows': len(x['core_rows']),
                                             'whole_endpoint_rows': len(x['whole_session_probe'])}
    first_attack = OUT/'iteration16_first_query_attack.json'
    if first_attack.exists():
        x = json.loads(first_attack.read_text()); json.dumps(x, allow_nan=False)
        train_path = OUT/'iteration16_first_query_training.json'
        train = json.loads(train_path.read_text())
        original = json.loads((OUT/'iteration11_shadow_training.json').read_text())
        assert x['training_sha256'] == sha(train_path)
        assert train['state_ids'] == original['state_ids'] and train['y'] == original['y']
        assert len(train['x']) == len(train['y']) == len(train['public']) == 2000
        assert len({r['family_id'] for r in x['auxiliary_selection_rows']}) == 16
        for row in x['auxiliary_selection_rows']:
            assert row['family_id'] in splits['auxiliary_selection']
        result['response_first_query_attack'] = {'sha256': sha(first_attack), 'public_training_positions': 2000,
                                                'core_rows': len(x['core_rows'])}
    diagnostic = OUT/'iteration16_s1_oracle_diagnosis.json'
    if diagnostic.exists():
        x = json.loads(diagnostic.read_text()); json.dumps(x, allow_nan=False)
        assert x['source_sha256'] == sha(OUT/'iteration15_response_cases.json')
        assert len(x['rows']) == 24
        for row in x['rows']:
            oracle = row['reachable_true_reference_oracle']
            assert oracle['feasible_value'] <= oracle['upper_bound']+1e-9
            assert row['actual_recall_L10'] <= oracle['upper_bound']+1e-9
            assert row['budget_spent'] <= .23+1e-12
            assert row['prefix_matches_saved_outputs'] and oracle['evaluator_only_uses_reference_weights']
        result['S1_C_oracle_diagnosis'] = {'sha256': sha(diagnostic), 'rows': 24,
            'all_oracles_solved_to_optimality': all(r['reachable_true_reference_oracle']['optimal'] for r in x['rows'])}
    slack = OUT/'iteration17_paced_slack_cases.json'
    if slack.exists():
        x = json.loads(slack.read_text()); json.dumps(x, allow_nan=False)
        source = json.loads((OUT/'iteration15_response_cases.json').read_text())
        controls = {(r['session_id'], r['rep'], r['method']): r for r in source['executions']}
        assert x['control_source_sha256'] == sha(OUT/'iteration15_response_cases.json')
        lookup = {}; new = replayed = 0
        for row in x['executions']:
            ledger_check(row, False)
            key = row['session_id'], row['rep'], row['method']; lookup[key] = row
            if key in controls:
                assert row == controls[key]; replayed += 1
            else:
                new += 1
        assert new == x['new_executions'] == 66 and replayed == x['replayed_control_executions'] == 198
        for row in x['rows']:
            record = records[row['record_id']]
            for slot, (sid, indices) in enumerate(zip(record['session_ids'], record['observed_indices'])):
                ex = lookup[sid, row['rep'], row['method']]
                assert public_view({int(i): e for i, e in ex['events'].items()}, indices) == row['public_views'][slot]
        result['paced_slack'] = {'sha256': sha(slack), 'new_executions': new, 'replayed_controls': replayed,
                                'case_rows': len(x['rows'])}
    slack_manifest = OUT/'paced_slack_shadows/manifest.json'
    if slack_manifest.exists():
        sm = json.loads(slack_manifest.read_text()); count = 0
        assert sm['provenance']['dataset_sha256'] == sha(auxiliary_path)
        for item in sm['shards']:
            path = slack_manifest.parent/item['file']; assert sha(path) == item['sha256']
            shard = json.loads(gzip.decompress(path.read_bytes()))
            assert shard['provenance'] == sm['provenance'] and len(shard['runs']) == 2
            assert shard['family_id'] in splits[shard['split']]
            response = json.loads(gzip.decompress((response_manifest.parent/item['file']).read_bytes()))
            firsts = {r['session_id_evaluator_only']: r['public']['events'][0]
                      for r in response['runs'] if r['method'] == 'response_paced'}
            for run in shard['runs']:
                ledger_check(run, False)
                assert run['public']['events'][0] == firsts[run['session_id_evaluator_only']]
                count += 1
        assert count == 160
        result['slack_shadows'] = {'manifest_sha256': sha(slack_manifest), 'new_executions': count,
                                  'first_query_equals_response_paced': True}
    slack_attacks = OUT/'iteration18_slack_attacks.json'
    if slack_attacks.exists():
        x = json.loads(slack_attacks.read_text()); json.dumps(x, allow_nan=False)
        assert x['shadow_manifest_sha256'] == sha(slack_manifest)
        assert x['core_source_sha256'] == sha(slack)
        for item in x['training']:
            path = slack_manifest.parent/item['file']; assert sha(path) == item['sha256']
            with np.load(path, allow_pickle=False) as a:
                assert set(a['families']) == splits['auxiliary_train']
        for row in x['auxiliary_selection_rows']:
            assert row['family_id'] in splits['auxiliary_selection']
        for row in x['core_rows']+x['whole_session_probe']:
            assert row['selected_attack'] == x['selection'][row['method']+'/'+row['task']]
            assert all(a in row['errors'] for a in row['selected_attack'].values())
        result['slack_matched_attacks'] = {'sha256': sha(slack_attacks), 'case_rows': len(x['core_rows'])}
    reserve = OUT/'iteration19_reserve_cases.json'
    if reserve.exists():
        from experiments.research_loop_reserve_cases import summaries
        x = json.loads(reserve.read_text()); json.dumps(x, allow_nan=False)
        assert x['control_source_sha256'] == sha(slack)
        assert x['dataset_sha256'] == sha(ROOT/'artifacts/datasets/research_loop_development_v1/dataset.json')
        assert x['code_sha256'] == sha(ROOT/'experiments/research_loop_reserve_cases.py')
        for p, digest in x['source_sha256'].items():
            assert sha(ROOT/p) == digest
        controls = {(r['session_id'], r['rep'], r['method']): r
                    for r in json.loads(slack.read_text())['executions']}
        lookup = {}; new = replayed = 0
        for ex in x['executions']:
            ledger_check(ex, False)
            key = ex['session_id'], ex['rep'], ex['method']; assert key not in lookup
            lookup[key] = ex
            if key in controls:
                assert ex == controls[key]; replayed += 1
            else:
                new += 1
        assert new == x['new_executions'] == 132
        assert replayed == x['replayed_control_executions'] == 198
        for row in x['rows']:
            record = records[row['record_id']]
            for slot, (sid, indices) in enumerate(zip(record['session_ids'], record['observed_indices'])):
                ex = lookup[sid, row['rep'], row['method']]
                assert public_view({int(i): e for i, e in ex['events'].items()}, indices) == row['public_views'][slot]
        assert x['summaries'] == summaries(x['rows'])
        result['reserve_pacing'] = {'sha256': sha(reserve), 'new_executions': new,
            'replayed_controls': replayed, 'case_rows': len(x['rows']), 'exact_allowed_views_checked': True}
    return result


if __name__ == '__main__':
    print(json.dumps(check(), indent=2))
