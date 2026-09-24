"""Recompute planar first-query features, auxiliary selection and all errors."""
import json
import numpy as np
from evaluation.service_shadow import fit, features
from experiments.research_loop_response_first import prediction_bank
from experiments.research_loop_sequence_attack import select
from experiments.research_loop_resources import ROOT, load, sha
from experiments.research_loop_planar_first import BASE, TRAINING, TREES, OUT, CORE, AUX, DATA


def check():
    if not OUT.exists():
        return None
    result = json.loads(OUT.read_text()); training = json.loads(TRAINING.read_text())
    for name, digest in result['provenance']['source_sha256'].items():
        assert sha(ROOT/name) == digest
    assert result['provenance'] == training['provenance']
    assert result['training_sha256'] == sha(TRAINING) and result['trees_sha256'] == sha(TREES)
    rn, _, _, _, metadata = load()
    assert result['provenance']['resources'] == metadata
    ids = json.loads((BASE/'iteration11_shadow_training.json').read_text())['state_ids']
    assert ids == training['state_ids'] and len(ids) == 2000
    assert training['y'] == rn.xy[ids].tolist()
    assert training['x'] == [features(view, rn)[0].tolist() for view in training['public']]
    model = fit(training['x'], training['y'], {'training_sha256': sha(TRAINING)})
    with np.load(TREES, allow_pickle=False) as archive:
        trees = {key: archive[key].copy() for key in archive.files}
    source = json.loads(CORE.read_text()); data = json.loads(DATA.read_text()); aux = json.loads(AUX.read_text())
    expected_selection = {session['session_id']: family['family_id'] for family in aux['families']
                          if family['split'] == 'auxiliary_selection' for session in family['sessions']}
    selection = result['auxiliary_selection_rows']
    assert len(selection) == len(expected_selection) == 32
    assert len({r['family_id'] for r in selection}) == 16
    assert {(r['session_id_evaluator_only'], r['family_id']) for r in selection} == set(expected_selection.items())
    for row in selection:
        p = aux['traces'][row['session_id_evaluator_only']][0]
        assert row['truth_xy'] == list(rn.point_xy(p['lat'], p['lon']))
    predictions = prediction_bank(model, trees, np.array([features(r['public'], rn)[0] for r in selection]))
    for i, row in enumerate(selection):
        assert row['errors'] == {a: [float(np.linalg.norm(p[i]-row['truth_xy']))] for a, p in predictions.items()}
    chosen = select(selection)
    assert chosen == result['summary']['selection']
    lookup = {(ex['session_id'], ex['rep'], ex['method']): ex for ex in source['executions']}
    rows = result['core_rows']; assert len(rows) == 66
    assert len({(r['session_id_evaluator_only'], r['rep']) for r in rows}) == len(rows)
    assert {r['family_id'] for r in rows}.isdisjoint(r['family_id'] for r in selection)
    predictions = prediction_bank(model, trees, np.array([features(r['public'], rn)[0] for r in rows]))
    for i, row in enumerate(rows):
        sid, rep = row['session_id_evaluator_only'], row['rep']
        ex = lookup[sid, rep, 'planar_paced']; other = lookup[sid, rep, 'planar_paced_slack03']
        assert row['split'] == ex['split'] and row['family_id'] == ex['family_id']
        event = next(iter(ex['events'].values()))
        assert row['public'] == {'events': [event]}
        assert event == next(iter(other['events'].values()))
        p = data['traces'][sid][0]; truth = np.array(rn.point_xy(p['lat'], p['lon']))
        assert row['errors'] == {a: [float(np.linalg.norm(v[i]-truth))] for a, v in predictions.items()}
        raw = next(iter(lookup[sid, rep, 'raw']['events'].values()))['candidates'][0]
        assert row['raw_first_query_error_m'] == np.linalg.norm(rn.point_xy(raw['lat'], raw['lon'])-truth) == 0.
    val = [r for r in rows if r['split'] == 'development_validation']; summary = result['summary']
    assert summary['selected_mae_m'] == float(np.mean([r['errors'][chosen['mae']][0] for r in val]))
    assert summary['selected_hits'] == {str(rad): float(np.mean([r['errors'][chosen[f'hit{rad}']][0] <= rad for r in val])) for rad in (50,100,200,500)}
    assert summary['validation_families'] == len({r['family_id'] for r in val}) == 2
    assert summary['validation_source_sessions'] == len({r['session_id_evaluator_only'] for r in val})
    assert summary['validation_record_RNG_pairs'] == len(val)
    assert result['new_first_query_only_executions'] == 2032
    return {'file': OUT.name, 'sha256': sha(OUT), 'training_locations': 2000, 'auxiliary_selection_families': 16,
            'core_rows': 66, 'features_selection_and_errors_recomputed': True,
            'scope': 'first-query probe only; not masked S9 cases or independent confirmation'}


if __name__ == '__main__':
    print(json.dumps(check(), indent=2))
