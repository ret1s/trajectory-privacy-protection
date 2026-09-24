"""Matched first-query control: same core sessions/RNG and auxiliary selection."""
import gzip
import json
from pathlib import Path
import numpy as np
from evaluation.service_shadow import fit, features
from evaluation.expanded_shadow import fit_trees
from experiments.research_loop_response_first import prediction_bank
from experiments.research_loop_sequence_attack import select
from experiments.research_loop_resources import ROOT, load, sha

BASE = ROOT/'artifacts/benchmarks/research_loop'
OUT = BASE/'iteration16_first_query_paired.json'


def main():
    if OUT.exists():
        raise FileExistsError('Preserve completed evidence')
    rn, *_ = load()
    train_path = BASE/'iteration11_shadow_training.json'; train = json.loads(train_path.read_text())
    model = fit(train['x'], train['y'], {'training_sha256': sha(train_path)}); trees = fit_trees(model)
    manifest_path = BASE/'sequential_shadows/manifest.json'; manifest = json.loads(manifest_path.read_text())
    auxiliary = []
    for item in manifest['shards']:
        if item['split'] != 'auxiliary_selection':
            continue
        path = manifest_path.parent/item['file']; assert sha(path) == item['sha256']
        shard = json.loads(gzip.decompress(path.read_bytes()))
        auxiliary.extend({**r, 'family_id': shard['family_id']} for r in shard['runs'] if r['method'] == 'filter_progress')
    x = np.array([features({'events': r['public']['events'][:1]}, rn)[0] for r in auxiliary])
    pred = prediction_bank(model, trees, x)
    holdout = [{'family_id': r['family_id'], 'errors': {a: [float(np.linalg.norm(p[i]-r['evaluation_truth']['origin_xy']))]
                for a, p in pred.items()}} for i, r in enumerate(auxiliary)]
    chosen = select(holdout)
    core_path = BASE/'iteration15_response_cases.json'; source = json.loads(core_path.read_text())
    data = json.loads((ROOT/'artifacts/datasets/research_loop_development_v1/dataset.json').read_text())
    executions = [r for r in source['executions'] if r['method'] == 'filter_paced']
    x = np.array([features({'events': [next(iter(r['events'].values()))]}, rn)[0] for r in executions])
    pred = prediction_bank(model, trees, x); rows = []
    for i, ex in enumerate(executions):
        p = data['traces'][ex['session_id']][0]; truth = rn.point_xy(p['lat'], p['lon'])
        rows.append({'method': 'filter_paced', 'family_id': ex['family_id'], 'split': ex['split'],
            'session_id_evaluator_only': ex['session_id'], 'rep': ex['rep'], 'selected_attack': chosen,
            'errors': {a: [float(np.linalg.norm(v[i]-truth))] for a, v in pred.items()}})
    response_path = BASE/'iteration16_first_query_attack.json'; response = json.loads(response_path.read_text())
    response_rows = [r for r in response['core_rows'] if r['method'] == 'response_paced']
    key = lambda r: (r['session_id_evaluator_only'], r['rep'])
    assert sorted(map(key, rows)) == sorted(map(key, response_rows))
    rows.extend(response_rows); summaries = []
    for method in ('filter_paced', 'response_paced'):
        val = [r for r in rows if r['method'] == method and r['split'] == 'development_validation']
        choices = val[0]['selected_attack']
        summaries.append({'method': method, 'selection': choices,
            'selected_mae_m': float(np.mean([r['errors'][choices['mae']][0] for r in val])),
            'selected_hits': {str(rad): float(np.mean([r['errors'][choices[f'hit{rad}']][0] <= rad for r in val])) for rad in (50, 100, 200, 500)},
            'fixed_knn15_mae_m': float(np.mean([r['errors']['expanded_knn_15'][0] for r in val])),
            'fixed_knn15_hit100': float(np.mean([r['errors']['expanded_knn_15'][0] <= 100 for r in val])),
            'validation_sessions': len({r['session_id_evaluator_only'] for r in val}), 'RNG_repetitions': 2,
            'validation_families': len({r['family_id'] for r in val})})
    OUT.write_text(json.dumps({'scope': 'paired first-step control only; same sessions/RNG, 16 auxiliary selection families, exposed core development',
        'control_training_sha256': sha(train_path), 'core_source_sha256': sha(core_path),
        'control_shadow_manifest_sha256': sha(manifest_path), 'response_attack_sha256': sha(response_path),
        'code_sha256': sha(Path(__file__)), 'predictor_source_sha256': sha(ROOT/'experiments/research_loop_response_first.py'),
        'control_auxiliary_selection_rows': holdout, 'core_rows': rows, 'summaries': summaries}, indent=2, allow_nan=False)+'\n')
    print(json.dumps(summaries, indent=2), flush=True)


if __name__ == '__main__':
    main()
