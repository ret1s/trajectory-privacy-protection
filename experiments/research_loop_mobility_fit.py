"""Fit/choose a mobility generator outside all defender case labels."""
from collections import defaultdict
import json
from pathlib import Path
import numpy as np
from scipy.sparse import coo_matrix, save_npz
from scipy.sparse.linalg import expm_multiply
from benchmark.empirical_mobility import smoothed_generator
from experiments.research_loop_resources import ROOT, load, sha

BASE = ROOT/'artifacts/benchmarks/research_loop'
DATA = ROOT/'artifacts/datasets/research_loop_shadow_v1/dataset.json'
OUT = BASE/'iteration20_mobility_fit.json'
PARAMETERS = (20., 100., 500.)


def main():
    if OUT.exists():
        raise FileExistsError('Preserve completed evidence')
    rn, _, _, belief, metadata = load(); data = json.loads(DATA.read_text())
    n = len(belief.xy); counts = defaultdict(float); exposure = np.zeros(n)
    validation, fit_families, selection_families = [], [], []
    for family in data['families']:
        training = family['split'] == 'auxiliary_train'
        (fit_families if training else selection_families).append(family['family_id'])
        for session in family['sessions']:
            trace = data['traces'][session['session_id']]
            xy = np.array([rn.point_xy(p['lat'], p['lon']) for p in trace])
            states = belief.tree.query(xy)[1]
            times = np.array([p['time_s'] for p in trace]); dt = np.diff(times)
            assert np.allclose(dt, 1.)
            if training:
                np.add.at(exposure, states[:-1], dt)
                for a, b in zip(states[:-1], states[1:]):
                    if a != b:
                        counts[int(a), int(b)] += 1.
            else:
                for i in range(0, len(states)-20, 20):
                    assert times[i+20]-times[i] == 20.
                    validation.append({'family_id': family['family_id'], 'session_id': session['session_id'],
                                       'source_index': i, 'start': int(states[i]), 'end': int(states[i+20])})
    assert len(fit_families) == 64 and len(selection_families) == 16
    assert not set(fit_families) & set(selection_families)
    keys = sorted(counts); row, col = np.array(keys).T
    c = coo_matrix(([counts[k] for k in keys], (row, col)), shape=(n, n)).tocsr()
    training_path = BASE/'iteration20_mobility_training.npz'
    assert not training_path.exists()
    np.savez_compressed(training_path, row=row, col=col, counts=np.array([counts[k] for k in keys]),
                        exposures=exposure, state_ids=belief.state_ids,
                        fit_families=np.array(sorted(fit_families)), dataset_sha256=sha(DATA))
    p20 = belief.transition(20.)
    starts = np.array([r['start'] for r in validation]); ends = np.array([r['end'] for r in validation])
    unique = np.unique(starts); lookup = {int(s): i for i, s in enumerate(unique)}
    variants = {'original_discrete_P20': np.asarray(p20[starts, ends]).ravel()}
    generators = []
    for pseudo in PARAMETERS:
        q = smoothed_generator(c, exposure, p20, pseudo)
        path = BASE/f'iteration20_generator_tau{int(pseudo)}.npz'
        assert not path.exists(); save_npz(path, q)
        probabilities = np.zeros((n, len(unique)))
        for begin in range(0, len(unique), 64):
            ids = unique[begin:begin+64]
            initial = np.zeros((n, len(ids))); initial[ids, np.arange(len(ids))] = 1.
            probabilities[:, begin:begin+len(ids)] = expm_multiply(q.T*20., initial, traceA=float(q.diagonal().sum()*20.))
        assert probabilities.min() >= -1e-10
        assert np.allclose(probabilities.sum(axis=0), 1., atol=1e-10)
        variants[f'tau{int(pseudo)}'] = np.maximum(0., probabilities[ends, [lookup[int(i)] for i in starts]])
        generators.append({'name': f'tau{int(pseudo)}', 'pseudo_exposure_s': pseudo,
                           'file': path.name, 'sha256': sha(path), 'nonzero_rates_including_diagonal': q.nnz})
        print('Scored', pseudo, flush=True)
    scores = []
    families = np.array([r['family_id'] for r in validation])
    for name, probability in variants.items():
        losses = -np.log(np.maximum(probability, 1e-300))
        family_scores = {f: float(losses[families == f].mean()) for f in sorted(selection_families)}
        scores.append({'name': name, 'mean_family_NLL_nats': float(np.mean(list(family_scores.values()))),
                       'family_NLL_nats': family_scores, 'zero_probability_count': int(np.sum(probability <= 0))})
        for row, prob in zip(validation, probability):
            row.setdefault('probability', {})[name] = float(prob)
    selected = min([r for r in scores if r['name'] != 'original_discrete_P20'],
                   key=lambda r: (r['mean_family_NLL_nats'], r['name']))
    chosen = next(g for g in generators if g['name'] == selected['name'])
    improved = selected['mean_family_NLL_nats'] < next(r['mean_family_NLL_nats'] for r in scores if r['name'] == 'original_discrete_P20')
    OUT.write_text(json.dumps({'scope': 'public synthetic auxiliary mobility fitting/selection; no defender core or expanded labels',
        'dataset_sha256': sha(DATA), 'resource_sha256': metadata['resource_sha256'],
        'base_belief_sha256': belief.sha256, 'code_sha256': sha(Path(__file__)),
        'model_source_sha256': sha(ROOT/'benchmark/empirical_mobility.py'),
        'training_file': training_path.name, 'training_sha256': sha(training_path),
        'fit_families': sorted(fit_families), 'selection_families': sorted(selection_families),
        'latent_states': n, 'states_with_training_exposure': int(np.sum(exposure > 0)),
        'total_exposure_s': float(exposure.sum()), 'observed_cell_changes': float(c.sum()),
        'scores': scores, 'generator_files': generators, 'selected_generator': chosen,
        'improves_auxiliary_NLL': improved, 'selection_rows': validation}, indent=2, allow_nan=False)+'\n')
    print(json.dumps({'scores': scores, 'selected': chosen, 'improves': improved}, indent=2), flush=True)


if __name__ == '__main__':
    main()
