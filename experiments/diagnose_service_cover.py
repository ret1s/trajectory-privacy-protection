"""Fixed-history interventions on the failed, now-development confirmation."""
import json
from pathlib import Path

import numpy as np

from benchmark.anchor_belief import AnchorBelief, PublicAnchorModel
from benchmark.engines.service_cover import ServiceCoverLaneDummy, greedy_cover
from experiments.run_service_cover import prepare, read, write, OUTPUT as PREVIOUS
from experiments.run_contextual_lane import sha

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / 'artifacts/benchmarks/service_recovery'


def uniform_cell_prior(rn, spacing_m):
    _, inverse, counts = np.unique(np.floor(rn.xy / spacing_m).astype(np.int64),
                                   axis=0, return_inverse=True, return_counts=True)
    prior = 1. / counts[inverse]
    return prior / prior.sum()


def recall(context, states, references):
    responses = [set() for _ in context.categories]
    for state in states:
        for c, ids in enumerate(context.query_indices(state)):
            responses[c].update(ids[ids >= 0].tolist())
    values = []
    for c, ids in enumerate(references):
        ref = set(ids[ids >= 0].tolist())
        values.append(len(ref & responses[c]) / len(ref) if ref else None)
    return values


def run():
    target = OUTPUT / 'diagnostic.json'
    if target.exists(): raise FileExistsError(target)
    records, rn, service, context, prior, learned, provenance = prepare('confirmation')
    print(f'Context: {len(rn)} lane states / {len(learned.xy)} latent cells', flush=True)
    flat = PublicAnchorModel(rn, context, uniform_cell_prior(rn, learned.spacing_m),
                             cache_path=ROOT/'cache/service_recovery/uniform.npz')
    assert np.allclose(flat.prior, 1. / len(flat.prior))
    frozen = read(PREVIOUS / 'confirmation.json')
    lookup = {r['record_id']: r for r in records}
    model = ServiceCoverLaneDummy(rn, belief_model=learned, k=5)
    viable = model.viable_ids
    public_center = model.public_center
    rows = []
    for original in frozen['rows']:
        if (original['method'], original['k'], original['replicate']) != ('service_cover', 5, 1): continue
        record = lookup[original['record_id']]
        b, u = AnchorBelief(learned), AnchorBelief(flat)
        for t, x in enumerate(record['points']):
            anchor = original['evaluator_anchors'][t]
            w, v = b.update(anchor, x['timestamp_s']), u.update(anchor, x['timestamp_s'])
            truth_state, _ = rn.nearest(x['lat'], x['lon'])
            references = context.signatures[truth_state]
            truth_xy = rn.xy[truth_state]
            prev = original['evaluator_states'][t-1] if t else None
            groups = [viable] * 5 if prev is None else [
                np.array(sorted(i for i in model.travel.reachable(s, x['timestamp_s'] - record['points'][t-1]['timestamp_s'])
                                if model.viable[i])) for s in prev]
            def ties(j, ids):
                movement = np.zeros(len(ids)) if prev is None else np.linalg.norm(rn.xy[ids] - rn.xy[prev[j]], axis=1)
                return movement, np.linalg.norm(rn.xy[ids] - public_center, axis=1)
            weights = {'learned': np.asarray(w @ learned.poi_weights).ravel(),
                       'uniform': np.asarray(v @ flat.poi_weights).ravel(),
                       'anchor': context.reference_weights(anchor),
                       'truth_oracle': context.reference_weights((x['lat'], x['lon']))}
            outputs = {}
            for name, poi_weights in weights.items():
                for reachable in (True, False):
                    selected, gains = greedy_cover(groups if reachable else [viable] * 5,
                        lambda ids, chosen: context.marginal_gain(ids, poi_weights, chosen), ties)
                    values = recall(context, selected, references)
                    outputs[f'{name}/{"reachable" if reachable else "free"}'] = {
                        'states': selected, 'objective': float(sum(gains)),
                        'category_recall': values}
            assert outputs['learned/reachable']['states'] == original['evaluator_states'][t]
            rows.append({'record_id': record['record_id'], 'family_id': record['family_id'],
                'case_id': record['case_id'], 'event': t, 'timestamp_s': x['timestamp_s'],
                'learned_mean_error_m': float(np.linalg.norm(w @ learned.xy - truth_xy)),
                'uniform_mean_error_m': float(np.linalg.norm(v @ flat.xy - truth_xy)),
                'anchor_error_m': float(np.linalg.norm(rn.point_xy(*anchor) - truth_xy)),
                'outputs': outputs})
        print(f'Diagnosed {record["record_id"]}', flush=True)
    summary = {}
    for name in rows[0]['outputs']:
        per_record = [np.mean([v for r in rows if r['record_id'] == rid
                               for v in r['outputs'][name]['category_recall'] if v is not None]) for rid in lookup]
        summary[name] = float(np.mean(per_record))
    for key in ('learned_mean_error_m', 'uniform_mean_error_m', 'anchor_error_m'):
        summary[key] = float(np.mean([np.mean([r[key] for r in rows if r['record_id'] == rid]) for rid in lookup]))
    write(target, {'schema': 'service-recovery-fixed-history-v1',
        'scope': 'development_intervention_not_new_confirmation',
        'source_sha256': {**provenance['source_sha256'], **{p: sha(ROOT/p) for p in (
            'experiments/diagnose_service_cover.py', 'thesis/notes/service_recovery_protocol.md')}},
        'previous_confirmation_sha256': sha(PREVIOUS/'confirmation.json'),
        'dataset_release': provenance['dataset_release'],
        'dataset_content_sha256': provenance['dataset_content_sha256'],
        'uniform_model_sha256': flat.sha256, 'categories': context.categories,
        'rows': rows, 'summary': summary})
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == '__main__': run()
