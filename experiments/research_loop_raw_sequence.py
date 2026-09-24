"""Strengthen the endpoint positive control using matched raw shadow learners."""
from collections import defaultdict
import gzip
import json
from pathlib import Path
import numpy as np
from experiments.research_loop_resources import ROOT, load, sha
from experiments.research_loop_sequence_attack import (SHADOW, DATA, CORE, examples,
    learned_bank, select, selection_key, errors)
from experiments.research_loop_cases import target_xy
from evaluation.sequence_shadow import EmpiricalShadow

OUT = ROOT/'artifacts/benchmarks/research_loop/iteration14_raw_sequence_audit.json'


def repeated_public(public):
    return {'events': [{'timestamp_s': e['timestamp_s'], 'candidates': [e['candidates'][0]]*5}
                       for e in public['events']]}


def main():
    if OUT.exists():
        raise FileExistsError('Preserve completed evidence')
    rn, *_ = load()
    manifest = json.loads((SHADOW/'manifest.json').read_text())
    runs = []
    for item in manifest['shards']:
        path = SHADOW/item['file']; assert sha(path) == item['sha256']
        shard = json.loads(gzip.decompress(path.read_bytes()))
        for source in shard['runs']:
            if source['method'] != 'filter_progress':
                continue
            events = []
            for event, xy in zip(source['public']['events'], source['evaluation_truth']['current_xy']):
                lat, lon = rn.proj.to_latlon(*xy)
                events.append({'timestamp_s': event['timestamp_s'], 'candidates': [
                    {'lat': float(lat), 'lon': float(lon)}]*5})
            runs.append({**source, 'public': {'events': events}, 'family_id': shard['family_id'], 'split': shard['split']})
    training = defaultdict(list)
    for run in runs:
        if run['split'] != 'auxiliary_train':
            continue
        for task, label, public, x, centers, y in examples(run, rn):
            if task in ('S9', 'S10'):
                training[task].append((x, y, centers, run['family_id']))
    models, training_manifest = {}, []
    for task, chunks in training.items():
        x, y, centers = (np.concatenate([r[i] for r in chunks]) for i in range(3))
        families = np.concatenate([np.repeat(r[3], len(r[0])) for r in chunks])
        path = SHADOW/f'training_raw_{task}.npz'
        if path.exists():
            raise FileExistsError(path)
        np.savez_compressed(path, x=x, y=y, centers=centers, families=families)
        models[task] = EmpiricalShadow(x, y, centers)
        training_manifest.append({'task': task, 'file': path.name, 'sha256': sha(path), 'rows': len(x)})
    holdout, pairs = [], defaultdict(list)
    for run in runs:
        if run['split'] != 'auxiliary_selection':
            continue
        for task, label, public, x, centers, y in examples(run, rn):
            if task not in models:
                continue
            pred = learned_bank(models[task], public, rn, task)
            holdout.append({'family_id': run['family_id'], 'task': task, 'window': label, 'errors': errors(pred, y)})
            if (task, label) in (('S9', 'head60'), ('S10', 'tail60')):
                pairs[run['family_id'], task].append((pred, y))
    for (family, task), pair in pairs.items():
        assert len(pair) == 2
        y = np.concatenate([p[1] for p in pair])
        pred = {a: np.concatenate([p[0][a] for p in pair]) for a in pair[0][0]}
        pred.update({a+'_joint_mean': np.repeat(p.mean(axis=0, keepdims=True), 2, axis=0) for a, p in list(pred.items())})
        holdout.append({'family_id': family, 'task': task+'_joint', 'window': 'same_site_two_trips', 'errors': errors(pred, y)})
    choices = {task: select([r for r in holdout if r['task'] == task]) for task in ('S9', 'S10', 'S9_joint', 'S10_joint')}
    for task, labels, key in (('S9', {'head60'}, 'S9_mask60'), ('S10', {'tail60'}, 'S10_mask60'),
                              ('S10', {'half', 'first12'}, 'S10_prefix')):
        choices[key] = select([r for r in holdout if r['task'] == task and r['window'] in labels])
    data, core = json.loads(DATA.read_text()), json.loads(CORE.read_text())
    records = {r['record_id']: r for r in data['records']}; rows = []
    for row in core['rows']:
        if row['method'] != 'raw' or row['scenario'] not in models:
            continue
        record = records[row['record_id']]; task = row['scenario']
        predictions, targets = [], []
        for slot, (public, sid) in enumerate(zip(row['public_views'], record['session_ids'])):
            predictions.append(learned_bank(models[task], repeated_public(public), rn, task))
            targets.append(target_xy(record, slot, data['traces'][sid], rn))
        pred = {a: np.concatenate([p[a] for p in predictions]) for a in predictions[0]}
        key = selection_key(record['case_id'], task)
        if key.endswith('_joint'):
            pred.update({a+'_joint_mean': np.repeat(p.mean(axis=0, keepdims=True), 2, axis=0) for a, p in list(pred.items())})
        rows.append({'case_id': record['case_id'], 'record_id': row['record_id'], 'rep': row['rep'],
            'family_id': row['family_id'], 'split': row['split'], 'errors': errors(pred, np.concatenate(targets)),
            'task': key, 'selected_attack': choices[key]})
    summaries = []
    for case in sorted({r['case_id'] for r in rows}):
        val = [r for r in rows if r['case_id'] == case and r['split'] == 'development_validation']; chosen = val[0]['selected_attack']
        summaries.append({'case_id': case, 'mae_m': float(np.mean([np.mean(r['errors'][chosen['mae']]) for r in val])),
            'hits': {str(radius): float(np.mean([np.mean(np.array(r['errors'][chosen[f'hit{radius}']]) <= radius) for r in val])) for radius in (50, 100, 200, 500)},
            'selected_attack': chosen})
    OUT.write_text(json.dumps({'scope': 'raw positive-control attack strengthening; not defender confirmation',
        'shadow_manifest_sha256': sha(SHADOW/'manifest.json'), 'core_source_sha256': sha(CORE),
        'code_sha256': sha(Path(__file__)), 'source_sha256': {p: sha(ROOT/p) for p in (
            'experiments/research_loop_sequence_attack.py', 'evaluation/sequence_shadow.py')},
        'training': training_manifest, 'selection': choices, 'auxiliary_selection_rows': holdout,
        'core_rows': rows, 'summaries': summaries}, indent=2, allow_nan=False)+'\n')
    print(json.dumps(summaries, indent=2))


if __name__ == '__main__':
    main()
