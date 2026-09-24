"""Score the frozen shortlist with auxiliary-only fitted/selected attackers.

No parameter, training row or selection rule is fitted on expanded development.
Raw endpoint learners are an explicit positive control. All bank errors remain
available; a strongest-on-evaluation readout is not an independently selected attack.
"""
from collections import defaultdict
import gzip
import json
from pathlib import Path
import numpy as np
from evaluation.sequence_shadow import EmpiricalShadow, TREE_PARAMS
from experiments.research_loop_sequence_attack import learned_bank, errors, selection_key
from experiments.research_loop_raw_sequence import repeated_public
from experiments.research_loop_cases import geometric_predictions, target_xy
from experiments.research_loop_resources import ROOT, load, sha

BASE = ROOT/'artifacts/benchmarks/research_loop'
DATA = ROOT/'artifacts/datasets/research_loop_expanded_v1/dataset.json'
SCREEN = BASE/'iteration18_expanded_screening.json'
OUT = BASE/'iteration18_expanded_attacks.json'
SPECS = {
    'raw': ('sequential_shadows', 'raw', 'iteration14_raw_sequence_audit.json'),
    'filter_paced': ('sequential_shadows', 'paced', 'iteration14_sequence_attacks.json'),
    'response_progress': ('response_shadows', 'response_progress', 'iteration16_response_attacks.json'),
    'response_paced': ('response_shadows', 'response_paced', 'iteration16_response_attacks.json'),
    'response_paced_slack03': ('paced_slack_shadows', 'response_paced_slack03', 'iteration18_slack_attacks.json'),
}
RADII = (50, 100, 200, 500)


def row_metrics(row):
    choice, err = row['selected_attack'], row['errors']
    return {'mae_m': float(np.mean(err[choice['mae']])),
            **{f'hit{r}': float(np.mean(np.array(err[choice[f'hit{r}']]) <= r)) for r in RADII},
            'recall_L10': row['recall']['10']}


def summarize(rows):
    result = []
    for method in SPECS:
        for case in sorted({r['case_id'] for r in rows}):
            subset = [r for r in rows if r['method'] == method and r['case_id'] == case]
            by_family = defaultdict(list)
            for row in subset:
                by_family[row['family_id']].append(row_metrics(row))
            means = {f: {k: float(np.mean([v[k] for v in vals])) for k in vals[0]}
                     for f, vals in sorted(by_family.items())}
            names = subset[0]['errors']
            bank = {a: {'mae_m': float(np.mean([np.mean(r['errors'][a]) for r in subset])),
                       **{f'hit{rad}': float(np.mean([np.mean(np.array(r['errors'][a]) <= rad)
                                                     for r in subset])) for rad in RADII}}
                    for a in names}
            result.append({'method': method, 'case_id': case, 'family_count': len(means),
                'record_count': len({r['record_id'] for r in subset}), 'RNG_repetitions': 2,
                'selected_attack': subset[0]['selected_attack'],
                'metrics': {k: float(np.mean([row_metrics(r)[k] for r in subset])) for k in next(iter(means.values()))},
                'family_metrics': means, 'all_bank_descriptive_only': bank})
    return result


def main():
    if OUT.exists():
        raise FileExistsError('Preserve completed evidence')
    screen = json.loads(SCREEN.read_text()); data = json.loads(DATA.read_text())
    assert screen['provenance']['dataset_sha256'] == sha(DATA)
    records = {r['record_id']: r for r in data['records']}
    rn, *_ = load()
    source_rows = []
    for item in screen['shards']:
        path = BASE/'expanded_screening'/item['file']; assert sha(path) == item['sha256']
        source_rows.extend(json.loads(gzip.decompress(path.read_bytes()))['rows'])
    rows, training_manifest, selection_manifest = [], [], []
    raw_geometry = json.loads((BASE/'iteration14_sequence_attacks.json').read_text())['selection']
    for method, (folder, alias, name) in SPECS.items():
        selection_path = BASE/name; evidence = json.loads(selection_path.read_text())
        models = {}
        for task in (('S9', 'S10') if method == 'raw' else ('current', 'S9', 'S10')):
            item = next(i for i in evidence['training'] if i['task'] == task and
                        (method == 'raw' or i['method'] == alias))
            path = BASE/folder/item['file']; assert sha(path) == item['sha256']
            with np.load(path, allow_pickle=False) as arrays:
                assert len(set(arrays['families'])) == 64
                assert not set(arrays['families']) & {f['family_id'] for f in data['families']}
                models[task] = EmpiricalShadow(arrays['x'], arrays['y'], arrays['centers'])
            training_manifest.append({'method': method, 'task': task, 'file': str(path.relative_to(BASE)),
                                      'sha256': sha(path), 'selection_evidence': name})
            print('Fitted auxiliary-only', method, task, flush=True)
        selection_manifest.append({'method': method, 'file': name, 'sha256': sha(selection_path)})
        for row in source_rows:
            if row['method'] != method:
                continue
            record = records[row['record_id']]
            task = record['scenario'] if record['scenario'] in ('S9', 'S10') else 'current'
            predictions, targets = [], []
            for slot, (public, sid) in enumerate(zip(row['public_views'], record['session_ids'])):
                if method == 'raw' and task == 'current':
                    pred = geometric_predictions(public, 'S3', rn)
                else:
                    pred = learned_bank(models[task], repeated_public(public) if method == 'raw' else public, rn, task)
                if record['scenario'] == 'S2':
                    pred = {a: p.mean(axis=0, keepdims=True) for a, p in pred.items()}
                predictions.append(pred); targets.append(target_xy(record, slot, data['traces'][sid], rn))
            pred = {a: np.concatenate([p[a] for p in predictions]) for a in predictions[0]}
            key = selection_key(record['case_id'], task)
            if key.endswith('_joint'):
                pred.update({a+'_joint_mean': np.repeat(p.mean(axis=0, keepdims=True), 2, axis=0)
                             for a, p in list(pred.items())})
            if method == 'raw':
                choice = raw_geometry['raw/'+key] if task == 'current' else evidence['selection'][key]
            else:
                choice = evidence['selection'][alias+'/'+key]
            assert all(a in pred for a in choice.values())
            rows.append({'method': method, 'case_id': record['case_id'], 'task': key,
                'record_id': row['record_id'], 'rep': row['rep'], 'family_id': row['family_id'],
                'split': 'expanded_development', 'errors': errors(pred, np.concatenate(targets)),
                'selected_attack': choice, 'recall': row['recall']})
        print('Scored', method, flush=True)
    result = {'schema': 'frozen-auxiliary-attacks-expanded-development-v1',
        'scope': 'expanded development, NOT final confirmation; empirical learned and road-snap bank, not an exact road-path posterior',
        'dataset_sha256': sha(DATA), 'screen_sha256': sha(SCREEN), 'code_sha256': sha(Path(__file__)),
        'source_sha256': {p: sha(ROOT/p) for p in ('evaluation/sequence_shadow.py',
            'experiments/research_loop_sequence_attack.py', 'experiments/research_loop_raw_sequence.py',
            'experiments/research_loop_cases.py')}, 'tree_params': TREE_PARAMS,
        'raw_current_selection_sha256': sha(BASE/'iteration14_sequence_attacks.json'),
        'training': training_manifest, 'selection_sources': selection_manifest,
        'rows': rows, 'summaries': summarize(rows)}
    OUT.write_text(json.dumps(result, indent=2, allow_nan=False)+'\n')
    print('Saved', OUT, len(rows), 'case rows', flush=True)


if __name__ == '__main__':
    main()
