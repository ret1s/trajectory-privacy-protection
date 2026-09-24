"""Refit fixed sequence attackers to response-aware public transcripts."""
from collections import defaultdict
import gzip
import json
from pathlib import Path
import numpy as np
from evaluation.sequence_shadow import EmpiricalShadow, TREE_PARAMS
from experiments.research_loop_sequence_attack import (examples, learned_bank, errors, select,
    add_mask_selections, selection_key)
from experiments.research_loop_cases import target_xy
from experiments.research_loop_resources import ROOT, load, sha

SHADOW = ROOT/'artifacts/benchmarks/research_loop/response_shadows'
CORE = ROOT/'artifacts/benchmarks/research_loop/iteration15_response_cases.json'
DATA = ROOT/'artifacts/datasets/research_loop_development_v1/dataset.json'
OUT = ROOT/'artifacts/benchmarks/research_loop/iteration16_response_attacks.json'


def main():
    if OUT.exists():
        raise FileExistsError('Preserve completed evidence')
    rn, *_ = load()
    manifest = json.loads((SHADOW/'manifest.json').read_text()); auxiliary = []
    for item in manifest['shards']:
        path = SHADOW/item['file']; assert sha(path) == item['sha256']
        shard = json.loads(gzip.decompress(path.read_bytes()))
        auxiliary.extend({**r, 'family_id': shard['family_id'], 'split': shard['split']} for r in shard['runs'])
    source = json.loads(CORE.read_text()); data = json.loads(DATA.read_text())
    records = {r['record_id']: r for r in data['records']}
    holdout, core_rows, whole_rows, choices, training_manifest = [], [], [], {}, []
    for method in manifest['methods']:
        training = defaultdict(list)
        for run in auxiliary:
            if run['method'] == method and run['split'] == 'auxiliary_train':
                for task, label, public, x, center, y in examples(run, rn):
                    training[task].append((x, y, center, run['family_id']))
        models = {}
        for task, chunks in training.items():
            x, y, centers = (np.concatenate([r[i] for r in chunks]) for i in range(3))
            families = np.concatenate([np.repeat(r[3], len(r[0])) for r in chunks])
            path = SHADOW/f'training_{method}_{task}.npz'
            if path.exists():
                with np.load(path, allow_pickle=False) as a:
                    assert all(np.array_equal(a[k], v) for k, v in [('x', x), ('y', y), ('centers', centers), ('families', families)])
            else:
                np.savez_compressed(path, x=x, y=y, centers=centers, families=families)
            models[task] = EmpiricalShadow(x, y, centers)
            training_manifest.append({'method': method, 'task': task, 'file': path.name, 'sha256': sha(path),
                                      'rows': len(x), 'families': len(set(families))})
            print('Fitted', method, task, len(x), flush=True)
        rows, pairs = [], defaultdict(list)
        for run in auxiliary:
            if run['method'] != method or run['split'] != 'auxiliary_selection':
                continue
            for task, label, public, x, center, y in examples(run, rn):
                pred = learned_bank(models[task], public, rn, task)
                rows.append({'method': method, 'family_id': run['family_id'],
                    'task': 'current_snapshot' if label.startswith('snapshot') else task,
                    'window': label, 'errors': errors(pred, y)})
                if (task, label) in [('S9', 'head60'), ('S10', 'tail60')]:
                    pairs[run['family_id'], task].append((pred, y))
        for (family, task), pair in pairs.items():
            assert len(pair) == 2
            y = np.concatenate([p[1] for p in pair])
            pred = {a: np.concatenate([p[0][a] for p in pair]) for a in pair[0][0]}
            pred.update({a+'_joint_mean': np.repeat(p.mean(axis=0, keepdims=True), 2, axis=0) for a, p in list(pred.items())})
            rows.append({'method': method, 'family_id': family, 'task': task+'_joint',
                         'window': 'same_site_two_trips', 'errors': errors(pred, y)})
        for task in sorted({r['task'] for r in rows}):
            choices[method+'/'+task] = select([r for r in rows if r['task'] == task])
        add_mask_selections(method, rows, choices); holdout.extend(rows)
        for row in source['rows']:
            if row['method'] != method:
                continue
            record = records[row['record_id']]
            task = record['scenario'] if record['scenario'] in ('S9', 'S10') else 'current'
            predictions, targets = [], []
            for slot, (public, sid) in enumerate(zip(row['public_views'], record['session_ids'])):
                pred = learned_bank(models[task], public, rn, task)
                if record['scenario'] == 'S2':
                    pred = {a: p.mean(axis=0, keepdims=True) for a, p in pred.items()}
                predictions.append(pred); targets.append(target_xy(record, slot, data['traces'][sid], rn))
            pred = {a: np.concatenate([p[a] for p in predictions]) for a in predictions[0]}
            key = selection_key(record['case_id'], task)
            if key.endswith('_joint'):
                pred.update({a+'_joint_mean': np.repeat(p.mean(axis=0, keepdims=True), 2, axis=0) for a, p in list(pred.items())})
            core_rows.append({'method': method, 'task': key, 'case_id': record['case_id'], 'record_id': row['record_id'],
                'rep': row['rep'], 'family_id': row['family_id'], 'split': row['split'],
                'errors': errors(pred, np.concatenate(targets)), 'selected_attack': choices[method+'/'+key], 'recall': row['recall']})
        for ex in source['executions']:
            if ex['method'] != method:
                continue
            public = {'events': list(ex['events'].values())}
            trace = data['traces'][ex['session_id']]
            for task in ('S9', 'S10'):
                p = trace[0 if task == 'S9' else -1]
                target = np.array([rn.point_xy(p['lat'], p['lon'])])
                whole_rows.append({'method': method, 'task': task+'_full', 'session_id_evaluator_only': ex['session_id'],
                    'rep': ex['rep'], 'family_id': ex['family_id'], 'split': ex['split'],
                    'errors': errors(learned_bank(models[task], public, rn, task), target),
                    'selected_attack': choices[method+'/'+task+'_full']})
        print('Scored', method, flush=True)
    summaries = []
    for method in manifest['methods']:
        for case in sorted({r['case_id'] for r in core_rows}):
            rows = [r for r in core_rows if r['method'] == method and r['case_id'] == case and r['split'] == 'development_validation']
            chosen = rows[0]['selected_attack']
            summaries.append({'method': method, 'case_id': case, 'selected_attack': chosen,
                'validation_families': len({r['family_id'] for r in rows}),
                'mae_m': float(np.mean([np.mean(r['errors'][chosen['mae']]) for r in rows])),
                'hits': {str(rad): float(np.mean([np.mean(np.array(r['errors'][chosen[f'hit{rad}']]) <= rad) for r in rows])) for rad in (50, 100, 200, 500)},
                'recall_L10': float(np.mean([r['recall']['10'] for r in rows]))})
    OUT.write_text(json.dumps({'schema': 'response-aware-matched-sequence-attacks-v1',
        'scope': 'auxiliary fitted/selected attackers; exposed defender development; full-session probe separate from case windows',
        'shadow_manifest_sha256': sha(SHADOW/'manifest.json'), 'core_source_sha256': sha(CORE), 'dataset_sha256': sha(DATA),
        'raw_control_source_sha256': sha(CORE.with_name('iteration14_raw_sequence_audit.json')),
        'code_sha256': sha(Path(__file__)), 'source_sha256': {p: sha(ROOT/p) for p in (
            'evaluation/sequence_shadow.py', 'experiments/research_loop_sequence_attack.py', 'experiments/research_loop_cases.py')},
        'tree_params': TREE_PARAMS, 'training': training_manifest, 'selection': choices,
        'auxiliary_selection_rows': holdout, 'core_rows': core_rows, 'whole_session_probe': whole_rows,
        'summaries': summaries}, indent=2, allow_nan=False)+'\n')
    print(json.dumps(summaries, indent=2), flush=True)


if __name__ == '__main__':
    main()
