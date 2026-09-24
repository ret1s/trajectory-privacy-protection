"""Auxiliary-selected repeated-site density attacks on exposed expanded C cases."""
from collections import defaultdict
import gzip
import json
from pathlib import Path
import numpy as np
from evaluation.site_density import SiteDensity, GridDecisions, combine_density
from evaluation.sequence_shadow import endpoint_features
from experiments.research_loop_sequence_attack import view, select, errors
from experiments.research_loop_raw_sequence import repeated_public
from experiments.research_loop_cases import target_xy
from experiments.research_loop_resources import ROOT, load, sha

BASE = ROOT/'artifacts/benchmarks/research_loop'
DATA = ROOT/'artifacts/datasets/research_loop_expanded_v1/dataset.json'
SCREEN = BASE/'iteration18_expanded_screening.json'
OLD = BASE/'iteration18_expanded_attacks.json'
OUT = BASE/'iteration24_site_density.json'
METHODS = {
    'raw': ('sequential_shadows', 'iteration14_raw_sequence_audit.json'),
    'response_paced': ('response_shadows', 'iteration16_response_attacks.json'),
    'response_paced_slack03': ('paced_slack_shadows', 'iteration18_slack_attacks.json'),
}
BANDWIDTHS = (100, 250, 500)
RADII = (50, 100, 200, 500)


def density_bank(models, public_views, rn, task, decisions):
    features = np.concatenate([endpoint_features(v, rn, task)[0] for v in public_views])
    predictions = {}
    for bandwidth, model in models.items():
        probabilities = model.probabilities(features)
        distributions = {'pool': combine_density(probabilities, model.prior, 'pool'),
                         'product': combine_density(probabilities, model.prior), 'prior': model.prior}
        for mode, distribution in distributions.items():
            predictions.update({f'kde_h{bandwidth}_{mode}_{policy}':
                np.repeat(value[None, :], len(public_views), axis=0)
                for policy, value in decisions.predict(distribution).items()})
    return predictions


def main():
    if OUT.exists():
        raise FileExistsError('Preserve completed evidence')
    rn, _, _, belief, resources = load(); decisions = GridDecisions(belief.xy)
    data = json.loads(DATA.read_text()); records = {r['record_id']: r for r in data['records']}
    old_expanded = json.loads(OLD.read_text())
    old_lookup = {(r['method'], r['record_id'], r['rep']): r for r in old_expanded['rows']
                  if r['method'] in METHODS and r['case_id'] in ('S9.C', 'S10.C')}
    screen = json.loads(SCREEN.read_text()); public_rows = []
    for item in screen['shards']:
        path = BASE/'expanded_screening'/item['file']; assert sha(path) == item['sha256']
        shard = json.loads(gzip.decompress(path.read_bytes()))
        public_rows.extend(r for r in shard['rows'] if r['method'] in METHODS and r['case_id'] in ('S9.C', 'S10.C'))
    provenance = {str(p.relative_to(ROOT)): sha(p) for p in (DATA, SCREEN, OLD, Path(__file__),
        ROOT/'evaluation/site_density.py', ROOT/'evaluation/sequence_shadow.py',
        ROOT/'experiments/research_loop_sequence_attack.py', ROOT/'experiments/research_loop_raw_sequence.py')}
    holdout, rows, choices, training = [], [], {}, []
    for method, (directory, old_file) in METHODS.items():
        shadow = BASE/directory; manifest_path = shadow/'manifest.json'
        manifest = json.loads(manifest_path.read_text()); provenance[str(manifest_path.relative_to(ROOT))] = sha(manifest_path)
        old_path = BASE/old_file; old = json.loads(old_path.read_text()); provenance[str(old_path.relative_to(ROOT))] = sha(old_path)
        old_holdout = {(r['family_id'], r['task']): r for r in old['auxiliary_selection_rows']
                       if r.get('method', method) == method and r['task'] in ('S9_joint', 'S10_joint')}
        models = {}
        for task in ('S9', 'S10'):
            training_path = shadow/f'training_{method}_{task}.npz'
            provenance[str(training_path.relative_to(ROOT))] = sha(training_path)
            with np.load(training_path, allow_pickle=False) as a:
                assert len(set(a['families'])) == 64
                allowed = {i['family_id'] for i in manifest['shards'] if i['split'] == 'auxiliary_train'}
                assert set(a['families']) == allowed
                models[task] = {h: SiteDensity(a['x'], a['y'], belief.xy, bandwidth_m=h) for h in BANDWIDTHS}
                training.append({'method': method, 'task': task, 'path': str(training_path.relative_to(ROOT)),
                    'families': sorted(allowed), 'rows': len(a['x']), 'neighbors': 64,
                    'bandwidths_m': list(BANDWIDTHS), 'prior_mix': .05})
        for item in manifest['shards']:
            if item['split'] != 'auxiliary_selection':
                continue
            path = shadow/item['file']; assert sha(path) == item['sha256']
            shard = json.loads(gzip.decompress(path.read_bytes()))
            selected_method = 'filter_progress' if method == 'raw' else method
            runs = [r for r in shard['runs'] if r['method'] == selected_method]; assert len(runs) == 2
            for task in ('S9', 'S10'):
                views, targets = [], []
                for run in runs:
                    public = run['public']
                    if method == 'raw':
                        events = []
                        for event, xy in zip(public['events'], run['evaluation_truth']['current_xy']):
                            lat, lon = rn.proj.to_latlon(*xy)
                            events.append({'timestamp_s': event['timestamp_s'], 'candidates': [{'lat': float(lat), 'lon': float(lon)}]*5})
                        public = {'events': events}
                    times = np.array([e['timestamp_s'] for e in public['events']])
                    indices = np.flatnonzero(times >= 60) if task == 'S9' else np.flatnonzero(times <= run['evaluation_truth']['last_time_s']-60)
                    views.append(view(public, indices))
                    targets.append(run['evaluation_truth']['origin_xy' if task == 'S9' else 'endpoint_xy'])
                target = np.asarray(targets)
                predictions = density_bank(models[task], views, rn, task, decisions)
                previous = old_holdout[item['family_id'], task+'_joint']
                old_errors = {'old_'+a: e for a, e in previous['errors'].items()}
                holdout.append({'method': method, 'family_id': item['family_id'], 'task': task+'_joint',
                    'site_separation_m': float(np.linalg.norm(target[0]-target[1])),
                    'errors': {**old_errors, **errors(predictions, target)}})
        for task in ('S9', 'S10'):
            group = [r for r in holdout if r['method'] == method and r['task'] == task+'_joint']
            assert len(group) == 16
            combined = select(group)
            old_group = [{**r, 'errors': {a: v for a, v in r['errors'].items() if a.startswith('old_')}} for r in group]
            old_selected = select(old_group)
            key = task+'_joint' if method == 'raw' else method+'/'+task+'_joint'
            assert old_selected == {metric: 'old_'+a for metric, a in old['selection'][key].items()}
            choices[method+'/'+task] = {'combined': combined, 'old_only': old_selected}
        print('Selected', method, 'on 16 auxiliary families', flush=True)
        for row in public_rows:
            if row['method'] != method:
                continue
            record = records[row['record_id']]; task = record['scenario']
            views = [repeated_public(p) for p in row['public_views']] if method == 'raw' else row['public_views']
            target = np.concatenate([target_xy(record, slot, data['traces'][sid], rn) for slot, sid in enumerate(record['session_ids'])])
            prediction = density_bank(models[task], views, rn, task, decisions)
            previous = old_lookup[method, row['record_id'], row['rep']]
            old_errors = {'old_'+a: e for a, e in previous['errors'].items()}
            error = {**old_errors, **errors(prediction, target)}
            expected_names = set(next(r['errors'] for r in holdout if r['method'] == method and r['task'] == task+'_joint'))
            assert set(error) == expected_names
            rows.append({'method': method, 'case_id': record['case_id'], 'family_id': record['family_id'],
                'record_id': record['record_id'], 'rep': row['rep'], 'site_separation_m': float(np.linalg.norm(target[0]-target[1])),
                'errors': error, 'selected': choices[method+'/'+task]})
        print('Scored', method, 'expanded S9.C/S10.C', flush=True)
    summaries = []
    for method in METHODS:
        for case in ('S9.C', 'S10.C'):
            subset = [r for r in rows if r['method'] == method and r['case_id'] == case]
            families = sorted({r['family_id'] for r in subset}); assert len(families) == 12
            metrics = {}
            for bank in ('old_only', 'combined'):
                chosen = subset[0]['selected'][bank]
                fmetrics = {}
                for f in families:
                    group = [r for r in subset if r['family_id'] == f]
                    fmetrics[f] = {'mae_m': float(np.mean([np.mean(r['errors'][chosen['mae']]) for r in group])),
                        **{f'hit{radius}': float(np.mean([np.mean(np.array(r['errors'][chosen[f'hit{radius}']]) <= radius) for r in group])) for radius in RADII}}
                metrics[bank] = {'selected_attack': chosen, 'family_metrics': fmetrics,
                    'mean': {key: float(np.mean([v[key] for v in fmetrics.values()])) for key in fmetrics[families[0]]}}
            summaries.append({'method': method, 'case_id': case, 'families': 12, 'records': 12,
                'RNG_repetitions': 2, 'metrics': metrics})
    OUT.write_text(json.dumps({'scope': 'same-site KDE fusion, auxiliary-selected; all expanded C records; exposed development, not confirmation',
        'conditional_independence_is_approximate': True, 'grid_points': len(belief.xy), 'resources_sha256': resources['resource_sha256'],
        'source_sha256': provenance, 'training': training, 'selection': choices,
        'auxiliary_selection_rows': holdout, 'rows': rows, 'summaries': summaries}, indent=2, allow_nan=False)+'\n')
    for s in summaries:
        print(s['method'], s['case_id'], {k: v['mean'] for k, v in s['metrics'].items()}, flush=True)


if __name__ == '__main__':
    main()
