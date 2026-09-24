"""Separate continuous-time reformulation from learning the mobility rates.

This diagnostic is added after auxiliary model selection, without changing its
chosen tau or inspecting defender case scores. The original fit is immutable.
"""
import json
from pathlib import Path
import numpy as np
from scipy.sparse import eye
from scipy.sparse.linalg import expm_multiply
from experiments.research_loop_resources import ROOT, load, sha

BASE = ROOT/'artifacts/benchmarks/research_loop'
SOURCE = BASE/'iteration20_mobility_fit.json'
OUT = BASE/'iteration20_mobility_null.json'


def main():
    if OUT.exists():
        raise FileExistsError('Preserve completed evidence')
    _, _, _, model, _ = load(); source = json.loads(SOURCE.read_text())
    q = (model.transition(20.)-eye(len(model.xy), format='csr'))/20.
    rows = source['selection_rows']; starts = np.array([r['start'] for r in rows]); ends = np.array([r['end'] for r in rows])
    ids = np.unique(starts); lookup = {s: i for i, s in enumerate(ids)}
    probs = np.zeros((len(model.xy), len(ids)))
    for begin in range(0, len(ids), 64):
        batch = ids[begin:begin+64]; initial = np.zeros((len(model.xy), len(batch)))
        initial[batch, np.arange(len(batch))] = 1.
        probs[:, begin:begin+len(batch)] = expm_multiply(q.T*20., initial, traceA=float(q.diagonal().sum()*20.))
    values = np.maximum(0., probs[ends, [lookup[s] for s in starts]])
    loss = -np.log(np.maximum(values, 1e-300)); families = np.array([r['family_id'] for r in rows])
    scores = {f: float(loss[families == f].mean()) for f in sorted(set(families))}
    result = {'scope': 'auxiliary-selection null control, added after tau selection; no defender case labels; no parameter change',
        'source_sha256': sha(SOURCE), 'code_sha256': sha(Path(__file__)),
        'name': 'public_diffusion_generator_no_training', 'mean_family_NLL_nats': float(np.mean(list(scores.values()))),
        'family_NLL_nats': scores, 'selection_probabilities': values.tolist(),
        'zero_probability_count': int(np.sum(values <= 0))}
    OUT.write_text(json.dumps(result, indent=2, allow_nan=False)+'\n')
    print('No-training CTMC NLL:', result['mean_family_NLL_nats'], flush=True)


if __name__ == '__main__':
    main()
