"""Fit on auxiliary families, select on auxiliary holdout, score exposed core cases."""
from collections import defaultdict
import gzip
import json
from pathlib import Path
import numpy as np
import sklearn
from evaluation.sequence_shadow import current_features, endpoint_features, EmpiricalShadow, road_variants, TREE_PARAMS
from experiments.research_loop_resources import ROOT, load, sha
from experiments.research_loop_cases import geometric_predictions, target_xy

SHADOW = ROOT/'artifacts/benchmarks/research_loop/sequential_shadows'
OUT = ROOT/'artifacts/benchmarks/research_loop/iteration14_sequence_attacks.json'
CORE = ROOT/'artifacts/benchmarks/research_loop/iteration13_paced_cases.json'
DATA = ROOT/'artifacts/datasets/research_loop_development_v1/dataset.json'
ALIASES = {'filter_progress': 'filter_progress', 'filter_paced': 'paced', 'filter_paced_origin': 'paced_origin'}
RADII = (50, 100, 200, 500)


def view(public, indices):
    events = [public['events'][int(i)] for i in indices]
    if not events:
        raise ValueError('Empty window')
    epoch = events[0]['timestamp_s']
    return {'events': [{'timestamp_s': e['timestamp_s']-epoch, 'candidates': e['candidates']}
                       for e in events]}


def windows(run):
    public, truth = run['public'], run['evaluation_truth']
    n = len(public['events']); times = np.array([e['timestamp_s'] for e in public['events']])
    result = [('current', 'whole', list(range(n))), ('current', 'late', list(range(n//2, n)))]
    result.extend(('current', f'snapshot{i}', [i]) for i in range(n))
    for scenario in ('S9', 'S10'):
        result.append((scenario, 'whole', list(range(n))))
    result.extend([('S9', 'head60', np.flatnonzero(times >= 60).tolist()),
                   ('S9', 'late', list(range(n//2, n))),
                   ('S10', 'tail60', np.flatnonzero(times <= truth['last_time_s']-60).tolist()),
                   ('S10', 'half', list(range(max(1, n//2)))),
                   ('S10', 'first12', list(range(min(12, n))))])
    return [r for r in result if r[2]]


def examples(run, rn):
    for task, label, indices in windows(run):
        public = view(run['public'], indices)
        if task == 'current':
            x, centers = current_features(public, rn)
            target = np.asarray(run['evaluation_truth']['current_xy'])[indices]
        else:
            x, centers = endpoint_features(public, rn, task)
            target = np.array([run['evaluation_truth']['origin_xy' if task == 'S9' else 'endpoint_xy']])
        yield task, label, public, x, centers, target


def learned_bank(model, public, rn, task):
    x, centers = current_features(public, rn) if task == 'current' else endpoint_features(public, rn, task)
    learned = {'shadow_'+a: p for a, p in model.predict(x, centers).items()}
    result = geometric_predictions(public, 'S3' if task == 'current' else task, rn)
    result.update(road_variants(learned, rn))
    return result


def errors(predictions, target):
    return {a: np.linalg.norm(p-target, axis=1).tolist() for a, p in predictions.items()}


def select(rows):
    names = sorted(rows[0]['errors'])
    families = sorted({r['family_id'] for r in rows})
    def score(name, radius=None):
        vals = []
        for family in families:
            group = [np.asarray(r['errors'][name]) for r in rows if r['family_id'] == family]
            vals.append(np.mean([v.mean() if radius is None else np.mean(v <= radius) for v in group]))
        return float(np.mean(vals))
    chosen = {'mae': min(names, key=lambda a: (score(a), a))}
    chosen.update({f'hit{r}': min(names, key=lambda a: (-score(a, r), score(a), a)) for r in RADII})
    return chosen


def selection_key(case, task):
    if case.startswith('S1.'):
        return 'current_snapshot'
    if case in ('S9.C', 'S10.C'):
        return task+'_joint'
    if case in ('S9.A', 'S10.A'):
        return task+'_mask60'
    if case == 'S10.B':
        return 'S10_prefix'
    return task


def add_mask_selections(method, rows, choices):
    for task, labels, suffix in (('S9', {'head60'}, 'mask60'),
                                ('S10', {'tail60'}, 'mask60'),
                                ('S10', {'half', 'first12'}, 'prefix'),
                                ('S9', {'whole'}, 'full'), ('S10', {'whole'}, 'full')):
        choices[method+'/'+task+'_'+suffix] = select([r for r in rows if r['task'] == task and r['window'] in labels])
    choices[method+'/current_snapshot'] = select([
        r for r in rows if r['task'] == 'current_snapshot'])


def raw_controls(auxiliary, core, data, by_id, rn):
    holdout, scored, choices, pairs = [], [], {}, defaultdict(list)
    for source in auxiliary:
        if source['method'] != 'filter_progress' or source['split'] != 'auxiliary_selection':
            continue
        truth = source['evaluation_truth']
        public = {'events': [{'timestamp_s': e['timestamp_s'], 'candidates': [
            {'candidate_id': 'raw', 'lat': float(rn.proj.to_latlon(*xy)[0]), 'lon': float(rn.proj.to_latlon(*xy)[1])}]}
            for e, xy in zip(source['public']['events'], truth['current_xy'])]}
        run = {**source, 'public': public}
        for task, label, indices in windows(run):
            p = view(public, indices)
            target = (np.asarray(truth['current_xy'])[indices] if task == 'current' else
                      np.array([truth['origin_xy' if task == 'S9' else 'endpoint_xy']]))
            pred = geometric_predictions(p, 'S3' if task == 'current' else task, rn)
            holdout.append({'method': 'raw', 'family_id': run['family_id'],
                            'task': 'current_snapshot' if label.startswith('snapshot') else task,
                            'window': label, 'errors': errors(pred, target)})
            if (task, label) in (('S9', 'head60'), ('S10', 'tail60')):
                pairs[run['family_id'], task].append((pred, target))
    for (family, task), pair in pairs.items():
        target = np.concatenate([p[1] for p in pair])
        pred = {a: np.concatenate([p[0][a] for p in pair]) for a in pair[0][0]}
        pred.update({a+'_joint_mean': np.repeat(p.mean(axis=0, keepdims=True), 2, axis=0) for a, p in list(pred.items())})
        holdout.append({'method': 'raw', 'family_id': family, 'task': task+'_joint',
                        'window': 'same_site_two_trips', 'errors': errors(pred, target)})
    for task in sorted({r['task'] for r in holdout}):
        choices['raw/'+task] = select([r for r in holdout if r['task'] == task])
    add_mask_selections('raw', holdout, choices)
    for row in core['rows']:
        if row['method'] != 'raw':
            continue
        record = by_id[row['record_id']]
        task = record['scenario'] if record['scenario'] in ('S9', 'S10') else 'current'
        predictions, targets = [], []
        for slot, (p, sid) in enumerate(zip(row['public_views'], record['session_ids'])):
            pred = geometric_predictions(p, 'S3' if task == 'current' else task, rn)
            if record['scenario'] == 'S2':
                pred = {a: x.mean(axis=0, keepdims=True) for a, x in pred.items()}
            predictions.append(pred); targets.append(target_xy(record, slot, data['traces'][sid], rn))
        pred = {a: np.concatenate([p[a] for p in predictions]) for a in predictions[0]}
        key = selection_key(record['case_id'], task)
        if key.endswith('_joint'):
            pred.update({a+'_joint_mean': np.repeat(p.mean(axis=0, keepdims=True), 2, axis=0) for a, p in list(pred.items())})
        scored.append({'method': 'raw', 'task': key, 'case_id': record['case_id'], 'record_id': row['record_id'],
                       'rep': row['rep'], 'family_id': row['family_id'], 'split': row['split'],
                       'errors': errors(pred, np.concatenate(targets)), 'selected_attack': choices['raw/'+key], 'recall': row['recall']})
    return holdout, scored, choices


def whole_probe(source, source_method, method, data, rn, models, choices):
    rows = []
    for row in source['rows']:
        if row['method'] != source_method:
            continue
        trace = data['traces'][row['session_id']]
        for scenario in ('S9', 'S10'):
            public = row['public']
            pred = (geometric_predictions(public, scenario, rn) if method == 'raw' else
                    learned_bank(models[scenario], public, rn, scenario))
            p = trace[0 if scenario == 'S9' else -1]
            target = np.array([rn.point_xy(p['lat'], p['lon'])])
            rows.append({'method': method, 'scenario': scenario, 'split': row['split'],
                         'session_id_evaluator_only': row['session_id'], 'family_id': row['family_id'],
                         'errors': errors(pred, target), 'selected_attack': choices[method+'/'+scenario+'_full']})
    return rows


def main():
    if OUT.exists():
        raise FileExistsError('Preserve completed evidence')
    manifest_path = SHADOW/'manifest.json'
    manifest = json.loads(manifest_path.read_text())
    rn, service, context, belief, metadata = load()
    assert metadata['resource_sha256'] == manifest['provenance']['resources_sha256']
    auxiliary = []
    for item in manifest['shards']:
        path = SHADOW/item['file']; assert sha(path) == item['sha256']
        shard = json.loads(gzip.decompress(path.read_bytes()))
        auxiliary.extend({**run, 'family_id': shard['family_id'], 'split': shard['split']} for run in shard['runs'])
    assert len({r['family_id'] for r in auxiliary if r['split'] == 'auxiliary_train'}) == 64
    assert len({r['family_id'] for r in auxiliary if r['split'] == 'auxiliary_selection'}) == 16
    core = json.loads(CORE.read_text()); data = json.loads(DATA.read_text())
    by_id = {r['record_id']: r for r in data['records']}
    holdout_rows, core_rows, choices = raw_controls(auxiliary, core, data, by_id, rn)
    whole_source_path = CORE.with_name('iteration12_origin_guard.json')
    whole_source = json.loads(whole_source_path.read_text())
    whole_rows = whole_probe(whole_source, 'raw', 'raw', data, rn, None, choices)
    training_manifest = []
    for method in manifest['methods']:
        training = defaultdict(list)
        for run in auxiliary:
            if run['method'] != method or run['split'] != 'auxiliary_train':
                continue
            for task, label, public, x, centers, y in examples(run, rn):
                training[task].append((x, y, centers, run['family_id']))
        models = {}
        for task, chunks in training.items():
            x, y, centers = (np.concatenate([r[i] for r in chunks]) for i in range(3))
            families = np.concatenate([np.repeat(r[3], len(r[0])) for r in chunks])
            path = SHADOW/f'training_{method}_{task}.npz'
            if not path.exists():
                np.savez_compressed(path, x=x, y=y, centers=centers, families=families)
            else:
                with np.load(path, allow_pickle=False) as cached:
                    assert all(np.array_equal(cached[key], value) for key, value in
                               (('x', x), ('y', y), ('centers', centers), ('families', families)))
            models[task] = EmpiricalShadow(x, y, centers)
            training_manifest.append({'method': method, 'task': task, 'file': path.name,
                                      'sha256': sha(path), 'rows': len(x), 'families': 64})
            print('Fitted', method, task, len(x), flush=True)
        pair_views = defaultdict(list)
        method_holdout = []
        for run in auxiliary:
            if run['method'] != method or run['split'] != 'auxiliary_selection':
                continue
            for task, label, public, x, centers, y in examples(run, rn):
                predictions = learned_bank(models[task], public, rn, task)
                row = {'method': method, 'family_id': run['family_id'],
                       'task': 'current_snapshot' if label.startswith('snapshot') else task, 'window': label,
                       'session_id_evaluator_only': run['session_id_evaluator_only'], 'errors': errors(predictions, y)}
                method_holdout.append(row)
                if (task, label) in (('S9', 'head60'), ('S10', 'tail60')):
                    pair_views[run['family_id'], task].append((predictions, y))
        for (family, task), pair in pair_views.items():
            assert len(pair) == 2
            y = np.concatenate([p[1] for p in pair])
            predictions = {a: np.concatenate([p[0][a] for p in pair]) for a in pair[0][0]}
            predictions.update({a+'_joint_mean': np.repeat(p.mean(axis=0, keepdims=True), 2, axis=0)
                                for a, p in list(predictions.items())})
            method_holdout.append({'method': method, 'family_id': family, 'task': task+'_joint',
                                   'window': 'same_site_two_trips', 'errors': errors(predictions, y)})
        for task in sorted({r['task'] for r in method_holdout}):
            choices[method+'/'+task] = select([r for r in method_holdout if r['task'] == task])
        add_mask_selections(method, method_holdout, choices)
        holdout_rows.extend(method_holdout)
        if method in ('filter_progress', 'origin_first'):
            whole_rows.extend(whole_probe(whole_source, 'filter_progress' if method == 'filter_progress' else 'filter_origin_first',
                                           method, data, rn, models, choices))
        for row in core['rows']:
            if ALIASES.get(row['method']) != method:
                continue
            record = by_id[row['record_id']]
            task = record['scenario'] if record['scenario'] in ('S9', 'S10') else 'current'
            predictions, truths = [], []
            for slot, (public, sid) in enumerate(zip(row['public_views'], record['session_ids'])):
                pred = learned_bank(models[task], public, rn, task)
                if record['scenario'] == 'S2':
                    pred = {a: p.mean(axis=0, keepdims=True) for a, p in pred.items()}
                predictions.append(pred)
                truths.append(target_xy(record, slot, data['traces'][sid], rn))
            target = np.concatenate(truths)
            pred = {a: np.concatenate([p[a] for p in predictions]) for a in predictions[0]}
            task = selection_key(record['case_id'], task)
            if task.endswith('_joint'):
                pred.update({a+'_joint_mean': np.repeat(p.mean(axis=0, keepdims=True), 2, axis=0)
                             for a, p in list(pred.items())})
            core_rows.append({'method': method, 'task': task, 'case_id': record['case_id'],
                              'record_id': row['record_id'], 'rep': row['rep'],
                              'family_id': row['family_id'], 'split': row['split'],
                              'errors': errors(pred, target), 'selected_attack': choices[method+'/'+task],
                              'recall': row['recall']})
        print('Scored auxiliary/core', method, flush=True)
    summaries = []
    for method in sorted({r['method'] for r in core_rows}):
        for case in sorted({r['case_id'] for r in core_rows}):
            rows = [r for r in core_rows if r['method'] == method and r['case_id'] == case
                    and r['split'] == 'development_validation']
            selected = rows[0]['selected_attack']
            summaries.append({'method': method, 'case_id': case, 'selected_attack': selected,
                              'attacker_fit_families': 0 if method == 'raw' else 64, 'attacker_selection_families': 16,
                              'core_validation_families': len({r['family_id'] for r in rows}),
                              'mae_m': float(np.mean([np.mean(r['errors'][selected['mae']]) for r in rows])),
                              'hits': {str(radius): float(np.mean([np.mean(np.array(r['errors'][selected[f'hit{radius}']]) <= radius) for r in rows])) for radius in RADII},
                              'recall_L10': float(np.mean([r['recall']['10'] for r in rows]))})
    result = {'schema': 'auxiliary-selected-sequential-attacks-v1',
              'scope': 'new attacker training/selection; exposed defender development, not final confirmation; approximate learned and road-snap attacks',
              'shadow_manifest_sha256': sha(manifest_path), 'core_source_sha256': sha(CORE),
              'whole_session_probe_source_sha256': sha(whole_source_path),
              'core_dataset_sha256': sha(DATA), 'code_sha256': sha(Path(__file__)),
              'source_sha256': {p: sha(ROOT/p) for p in ('evaluation/sequence_shadow.py',
                  'evaluation/loss_aware_shadow.py', 'experiments/research_loop_cases.py', 'experiments/contribution_stress.py')},
              'versions': {'numpy': np.__version__, 'sklearn': sklearn.__version__},
              'tree_params': TREE_PARAMS, 'training': training_manifest, 'selection': choices,
              'auxiliary_selection_rows': holdout_rows, 'core_rows': core_rows, 'summaries': summaries,
              'whole_session_probe': whole_rows}
    OUT.write_text(json.dumps(result, indent=2, allow_nan=False)+'\n')
    print(json.dumps(summaries, indent=2), flush=True)


if __name__ == '__main__':
    main()
