"""Audit auxiliary-only mobility fitting and paired defender integration."""
import json
import numpy as np
from scipy.sparse import coo_matrix, load_npz
from benchmark.empirical_mobility import smoothed_generator
from experiments.research_loop_resources import ROOT, load, sha
from experiments.verify_research_loop_extended import ledger_check
from experiments.research_loop_cases import public_view

BASE = ROOT/'artifacts/benchmarks/research_loop'


def check():
    path = BASE/'iteration20_mobility_fit.json'
    if not path.exists():
        return {'status': 'not_fitted'}
    fit = json.loads(path.read_text()); json.dumps(fit, allow_nan=False)
    assert fit['code_sha256'] == sha(ROOT/'experiments/research_loop_mobility_fit.py')
    assert fit['model_source_sha256'] == sha(ROOT/'benchmark/empirical_mobility.py')
    data_path = ROOT/'artifacts/datasets/research_loop_shadow_v1/dataset.json'
    assert fit['dataset_sha256'] == sha(data_path)
    data = json.loads(data_path.read_text())
    groups = {split: sorted(f['family_id'] for f in data['families'] if f['split'] == split)
              for split in ('auxiliary_train', 'auxiliary_selection')}
    assert fit['fit_families'] == groups['auxiliary_train'] and len(fit['fit_families']) == 64
    assert fit['selection_families'] == groups['auxiliary_selection'] and len(fit['selection_families']) == 16
    training = BASE/fit['training_file']; assert sha(training) == fit['training_sha256']
    rn, _, _, model, metadata = load()
    assert fit['resource_sha256'] == metadata['resource_sha256']
    with np.load(training, allow_pickle=False) as a:
        assert a['fit_families'].tolist() == fit['fit_families']
        assert np.array_equal(a['state_ids'], model.state_ids)
        counts = coo_matrix((a['counts'], (a['row'], a['col'])), shape=(len(model.xy), len(model.xy))).tocsr()
        exposures = a['exposures'].copy()
    # Independently recover exposure and transitions from only the declared families.
    check_exposure = np.zeros(len(model.xy)); transitions = {}
    for family in data['families']:
        if family['split'] != 'auxiliary_train':
            continue
        for session in family['sessions']:
            trace = data['traces'][session['session_id']]
            ids = model.tree.query(np.array([rn.point_xy(p['lat'], p['lon']) for p in trace]))[1]
            np.add.at(check_exposure, ids[:-1], np.diff([p['time_s'] for p in trace]))
            for a, b in zip(ids[:-1], ids[1:]):
                if a != b:
                    transitions[a, b] = transitions.get((a, b), 0)+1
    assert np.array_equal(exposures, check_exposure)
    assert counts.nnz == len(transitions)
    assert all(counts[a, b] == value for (a, b), value in transitions.items())
    for item in fit['generator_files']:
        p = BASE/item['file']; assert sha(p) == item['sha256']
        q = load_npz(p); expected = smoothed_generator(counts, exposures, model.transition(20.), item['pseudo_exposure_s'])
        assert (q != expected).nnz == 0
    for row in fit['selection_rows']:
        assert row['family_id'] in groups['auxiliary_selection']
    for score in fit['scores']:
        values = {f: float(np.mean([-np.log(max(r['probability'][score['name']], 1e-300))
                  for r in fit['selection_rows'] if r['family_id'] == f])) for f in fit['selection_families']}
        assert score['family_NLL_nats'] == values
        assert score['mean_family_NLL_nats'] == float(np.mean(list(values.values())))
    selected = min([r for r in fit['scores'] if r['name'] != 'original_discrete_P20'],
                   key=lambda r: (r['mean_family_NLL_nats'], r['name']))
    assert selected['name'] == fit['selected_generator']['name']
    result = {'status': 'fit_checked', 'fit_sha256': sha(path), 'training_reconstructed': True,
              'fit_families': 64, 'selection_families': 16, 'defender_labels_used': False}
    cases_path = BASE/'iteration20_mobility_cases.json'
    if not cases_path.exists():
        return result
    from experiments.research_loop_mobility_cases import summaries
    cases = json.loads(cases_path.read_text()); json.dumps(cases, allow_nan=False)
    assert cases['mobility_fit_sha256'] == sha(path)
    assert cases['code_sha256'] == sha(ROOT/'experiments/research_loop_mobility_cases.py')
    for p, digest in cases['source_sha256'].items():
        assert sha(ROOT/p) == digest
    core_path = ROOT/'artifacts/datasets/research_loop_development_v1/dataset.json'
    assert cases['dataset_sha256'] == sha(core_path)
    core = json.loads(core_path.read_text()); records = {r['record_id']: r for r in core['records']}
    control_path = BASE/'iteration17_paced_slack_cases.json'
    assert cases['control_source_sha256'] == sha(control_path)
    controls = {(r['session_id'], r['rep'], r['method']): r for r in json.loads(control_path.read_text())['executions']}
    lookup = {}; new = replayed = 0; anchors = {}
    for ex in cases['executions']:
        key = ex['session_id'], ex['rep'], ex['method']; assert key not in lookup
        lookup[key] = ex; ledger_check(ex, False)
        if key in controls:
            assert ex == controls[key]; replayed += 1
        else:
            new += 1; pair = key[:2]
            assert ex['evaluator_ledger'] == controls[*pair, 'response_paced']['evaluator_ledger']
            assert pair not in anchors or anchors[pair] == ex['evaluator_anchor_sha256']
            anchors[pair] = ex['evaluator_anchor_sha256']
    assert new == cases['new_executions'] == 132
    assert replayed == cases['replayed_control_executions'] == 198
    for row in cases['rows']:
        record = records[row['record_id']]
        for slot, (sid, indices) in enumerate(zip(record['session_ids'], record['observed_indices'])):
            ex = lookup[sid, row['rep'], row['method']]
            assert public_view({int(i): e for i, e in ex['events'].items()}, indices) == row['public_views'][slot]
    assert cases['summaries'] == summaries(cases['rows'])
    result.update(status='fit_and_cases_checked', case_sha256=sha(cases_path),
                  new_executions=new, replayed_controls=replayed, case_rows=len(cases['rows']),
                  matching_ledgers_and_paired_anchor_streams=True)
    pilot_path = BASE/'iteration21_lookahead_pilot.json'
    if pilot_path.exists():
        pilot = json.loads(pilot_path.read_text()); json.dumps(pilot, allow_nan=False)
        assert pilot['source_sha256'] == sha(cases_path)
        assert pilot['dataset_sha256'] == sha(core_path)
        assert pilot['mobility_fit_sha256'] == sha(path)
        assert pilot['mobility_model_sha256'] == cases['mobility_model_sha256']
        assert pilot['code_sha256'] == sha(ROOT/'experiments/research_loop_lookahead_pilot.py')
        for p, digest in pilot['implementation_sha256'].items():
            assert sha(ROOT/p) == digest
        lookup_pilot = {}; nnew = nreplayed = 0
        for ex in pilot['executions']:
            key = ex['session_id'], ex['rep'], ex['method']; assert key not in lookup_pilot
            lookup_pilot[key] = ex; ledger_check(ex, False)
            if key in lookup:
                assert ex == lookup[key]; nreplayed += 1
            else:
                nnew += 1
                control = lookup[ex['session_id'], ex['rep'], 'response_empirical_paced']
                assert ex['clock_indices'] == control['clock_indices']
                assert ex['evaluator_anchor_sha256'] == control['evaluator_anchor_sha256']
                assert ex['evaluator_ledger'] == control['evaluator_ledger']
                for v in ex['lookahead_diagnostics'].values():
                    if v:
                        assert v['current_objective_after_lookahead'] >= v['current_objective_before_lookahead']-v['current_objective_slack']-1e-12
                        assert v['lookahead_s'] == 120.
        assert nnew == pilot['new_executions'] == 24 and nreplayed == pilot['replayed_controls'] == 16
        assert len(pilot['rows']) == 40
        assert {r['record_id'] for r in pilot['rows']} == {r['record_id'] for r in records.values() if r['case_id'] == 'S1.C'}
        for row in pilot['rows']:
            ex = lookup_pilot[row['session_id'], row['rep'], row['method']]
            idx = records[row['record_id']]['observed_indices'][0][0]
            assert row['recall_L10'] == ex['utility_by_index'][str(idx)]['10']
        for summary in pilot['summaries']:
            rows = [r for r in pilot['rows'] if r['method'] == summary['method'] and r['split'] == summary['split']]
            assert len(rows) == 4 and len({r['family_id'] for r in rows}) == 2
            assert summary['recall_L10'] == float(np.mean([r['recall_L10'] for r in rows]))
        result['lookahead_pilot'] = {'sha256': sha(pilot_path), 'new_executions': nnew,
            'replayed_controls': nreplayed, 'paired_anchor_and_ledger_match': True,
            'current_protected_objective_floor_checked': True, 'scope': 'S1.C pilot only'}
    return result


if __name__ == '__main__':
    print(json.dumps(check(), indent=2))
