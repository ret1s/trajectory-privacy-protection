"""Check auxiliary-only fitting/selection and replay every expanded density decision."""
import gzip
import json
import numpy as np
from experiments.research_loop_site_density import BASE, DATA, SCREEN, OLD, OUT, METHODS, BANDWIDTHS, density_bank
from experiments.research_loop_resources import ROOT, load, sha
from experiments.research_loop_sequence_attack import select, errors
from experiments.research_loop_raw_sequence import repeated_public
from experiments.research_loop_cases import target_xy
from evaluation.site_density import SiteDensity, GridDecisions


def check():
    if not OUT.exists():
        return None
    result = json.loads(OUT.read_text()); json.dumps(result, allow_nan=False)
    for name, digest in result['source_sha256'].items():
        assert sha(ROOT/name) == digest
    rn, _, _, belief, resources = load(); decisions = GridDecisions(belief.xy)
    assert result['grid_points'] == len(belief.xy) == 2073
    assert result['resources_sha256'] == resources['resource_sha256']
    data = json.loads(DATA.read_text()); records = {r['record_id']: r for r in data['records']}
    old = json.loads(OLD.read_text()); old_rows = {(r['method'], r['record_id'], r['rep']): r for r in old['rows']}
    screen = json.loads(SCREEN.read_text()); public = {}
    for item in screen['shards']:
        path = BASE/'expanded_screening'/item['file']; assert sha(path) == item['sha256']
        shard = json.loads(gzip.decompress(path.read_bytes()))
        public.update({(r['method'], r['record_id'], r['rep']): r for r in shard['rows'] if r['method'] in METHODS and r['case_id'] in ('S9.C', 'S10.C')})
    assert len(result['rows']) == len(public) == 144
    for method, (directory, _) in METHODS.items():
        manifest = json.loads((BASE/directory/'manifest.json').read_text())
        fit = {i['family_id'] for i in manifest['shards'] if i['split'] == 'auxiliary_train'}
        selection = {i['family_id'] for i in manifest['shards'] if i['split'] == 'auxiliary_selection'}
        assert len(fit) == 64 and len(selection) == 16 and not fit & selection
        assert not (fit | selection) & {f['family_id'] for f in data['families']}
        for task in ('S9', 'S10'):
            training = BASE/directory/f'training_{method}_{task}.npz'
            with np.load(training, allow_pickle=False) as a:
                assert set(a['families']) == fit
                models = {h: SiteDensity(a['x'], a['y'], belief.xy, bandwidth_m=h) for h in BANDWIDTHS}
            auxiliary = [r for r in result['auxiliary_selection_rows'] if r['method'] == method and r['task'] == task+'_joint']
            assert len(auxiliary) == 16 and {r['family_id'] for r in auxiliary} == selection
            chosen = result['selection'][method+'/'+task]
            assert chosen['combined'] == select(auxiliary)
            assert chosen['old_only'] == select([{**r, 'errors': {a: v for a, v in r['errors'].items() if a.startswith('old_')}} for r in auxiliary])
            for row in result['rows']:
                if row['method'] != method or row['case_id'] != task+'.C':
                    continue
                key = method, row['record_id'], row['rep']; original = public[key]; record = records[row['record_id']]
                views = [repeated_public(p) for p in original['public_views']] if method == 'raw' else original['public_views']
                targets = np.concatenate([target_xy(record, slot, data['traces'][sid], rn) for slot, sid in enumerate(record['session_ids'])])
                predicted = density_bank(models, views, rn, task, decisions)
                expected = {'old_'+a: e for a, e in old_rows[key]['errors'].items()}
                expected.update(errors(predicted, targets))
                assert expected == row['errors'] and row['selected'] == chosen
                assert row['site_separation_m'] == float(np.linalg.norm(targets[0]-targets[1]))
    for summary in result['summaries']:
        subset = [r for r in result['rows'] if r['method'] == summary['method'] and r['case_id'] == summary['case_id']]
        assert len(subset) == 24 and len({r['family_id'] for r in subset}) == 12
        for bank, values in summary['metrics'].items():
            chosen = subset[0]['selected'][bank]
            for family, recorded in values['family_metrics'].items():
                group = [r for r in subset if r['family_id'] == family]
                actual = {'mae_m': float(np.mean([np.mean(r['errors'][chosen['mae']]) for r in group])),
                    **{f'hit{radius}': float(np.mean([np.mean(np.array(r['errors'][chosen[f'hit{radius}']]) <= radius) for r in group])) for radius in (50, 100, 200, 500)}}
                assert actual == recorded
            assert values['mean'] == {key: float(np.mean([v[key] for v in values['family_metrics'].values()])) for key in values['mean']}
    components_path = BASE/'iteration24_site_density_components.json'
    if components_path.exists():
        components = json.loads(components_path.read_text())
        assert components['source_sha256'] == sha(OUT)
        assert components['code_sha256'] == sha(ROOT/'experiments/research_loop_site_density_readout.py')
        for row in components['rows']:
            auxiliary = [r for r in result['auxiliary_selection_rows'] if r['method'] == row['method'] and r['task'] == row['case_id'].split('.')[0]+'_joint']
            assert row['selected_attack'] == select([{**r, 'errors': {a: v for a, v in r['errors'].items() if '_'+row['component']+'_' in a}} for r in auxiliary])
            chosen = row['selected_attack']
            for family, values in row['family_metrics'].items():
                group = [r for r in result['rows'] if r['method'] == row['method'] and r['case_id'] == row['case_id'] and r['family_id'] == family]
                expected = {'mae_m': float(np.mean([np.mean(r['errors'][chosen['mae']]) for r in group])),
                    **{f'hit{radius}': float(np.mean([np.mean(np.array(r['errors'][chosen[f'hit{radius}']]) <= radius) for r in group])) for radius in (50, 100, 200, 500)}}
                assert values == expected
            assert row['mean'] == {key: float(np.mean([v[key] for v in row['family_metrics'].values()])) for key in row['mean']}
    return {'file': OUT.name, 'sha256': sha(OUT), 'all_expanded_predictions_replayed': 144,
            'fit_families': 64, 'selection_families': 16, 'expanded_families': 12,
            'scope': 'prior-corrected same-site empirical density score; not directed-road or calibrated posterior certification'}


if __name__ == '__main__':
    print(json.dumps(check(), indent=2))
