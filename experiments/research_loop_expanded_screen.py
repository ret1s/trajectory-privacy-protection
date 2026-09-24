"""Frozen shortlist screening on twelve additional development families.

Two CPU workers, immutable per-family shards. Runtime is descriptive only.
Finite geometric decoders are frozen from OLD train families, never selected here.
"""
from concurrent.futures import ProcessPoolExecutor, as_completed
from collections import defaultdict
from functools import partial
import gzip
import hashlib
import json
from pathlib import Path
import numpy as np
from benchmark.public_poi_context import PublicPoiContext
from benchmark.response_aware_belief import ResponseAwareAnchorModel
from benchmark.engines.paced_guard import PacedProgressLaneDummy
from benchmark.engines.paced_slack import PacedSlackProgressLaneDummy
from benchmark.engines.matched_filter import MatchedFilteredProgressCoverLaneDummy
from evaluation.lane_travel import LanePoiService
from experiments.research_loop_resources import ROOT, CACHE, load, sha
from experiments.research_loop_cases import (public_view, geometric_predictions, target_xy,
    recall_pair, mean_optional, passes_recall_gate)
from experiments.rng_util import rng_from_key

BASE = ROOT/'artifacts/benchmarks/research_loop'
DATA = ROOT/'artifacts/datasets/research_loop_expanded_v1/dataset.json'
SHARDS = BASE/'expanded_screening'
OUT = BASE/'iteration18_expanded_screening.json'
METHODS = ('raw', 'filter_paced', 'response_progress', 'response_paced', 'response_paced_slack03')
SOURCES = ('benchmark/engines/paced_slack.py', 'benchmark/engines/slack_progress.py',
    'benchmark/engines/paced_guard.py', 'benchmark/engines/matched_filter.py',
    'benchmark/engines/filtered_cover.py', 'benchmark/engines/progress_cover.py',
    'benchmark/engines/quotient_cover.py', 'benchmark/engines/fair_cover.py',
    'benchmark/engines/service_cover.py', 'benchmark/anchor_belief.py',
    'benchmark/response_aware_belief.py', 'core/mechanisms.py',
    'experiments/research_loop_cases.py', 'evaluation/lane_travel.py')
WORKER = None


def initialize(provenance, selections):
    global WORKER
    rn, _, reference, base, metadata = load()
    reply = PublicPoiContext(LanePoiService(rn, list(reference.pois), k=10), CACHE/'poi10.npz')
    response = ResponseAwareAnchorModel(base, reply)
    factories = {'filter_paced': (PacedProgressLaneDummy, base),
        'response_progress': (MatchedFilteredProgressCoverLaneDummy, response),
        'response_paced': (PacedProgressLaneDummy, response),
        'response_paced_slack03': (partial(PacedSlackProgressLaneDummy, utility_slack=.03), response)}
    models = {m: cls(rn, belief_model=b, k=5, budget=.24, horizon=12, rng=np.random.default_rng(0))
              for m, (cls, b) in factories.items()}
    WORKER = (rn, reference, reply, models, json.loads(DATA.read_text()), provenance, selections)


def family_run(family_id):
    rn, reference, reply, models, data, provenance, selections = WORKER
    path = SHARDS/(family_id+'.json.gz')
    if path.exists():
        old = json.loads(gzip.decompress(path.read_bytes()))
        assert old['provenance'] == provenance and old['family_id'] == family_id
        return {'family_id': family_id, 'file': path.name, 'sha256': sha(path),
                'executions': len(old['executions']), 'case_rows': len(old['rows'])}
    records = [r for r in data['records'] if r['family_id'] == family_id and r['scenario'] in {'S1', 'S2', 'S3', 'S9', 'S10'}]
    clocks = defaultdict(set)
    for record in records:
        for sid, indices in zip(record['session_ids'], record['observed_indices']):
            clocks[sid].update(indices)
    for sid in clocks:
        clocks[sid].update(range(0, len(data['traces'][sid]), 20))
    executions, lookup = [], {}
    for sid, clock in sorted(clocks.items()):
        trace = data['traces'][sid]
        for rep in range(2):
            seeds = rng_from_key(sid+f'/{rep}', schema='persistent-exact-case-clock-v1').integers(0, 2**63, 2, dtype=np.int64)
            paced_anchor_hash = None
            for method in METHODS:
                model = models.get(method)
                if model:
                    model.anchor_rng, model.dummy_rng = (np.random.default_rng(int(s)) for s in seeds)
                    model.reset()
                events, utilities, category, items = {}, {}, {}, {}
                for i in sorted(clock):
                    p = trace[i]; t = p['time_s']-trace[0]['time_s']
                    coords = model.protect_step(p['lat'], p['lon'], t) if model else [(p['lat'], p['lon'])]
                    events[i] = {'timestamp_s': t, 'candidates': [
                        {'candidate_id': f'candidate_{j:04d}', 'lat': lat, 'lon': lon} for j, (lat, lon) in enumerate(coords)]}
                    states = [rn.nearest(lat, lon)[0] for lat, lon in coords]
                    truth, _ = rn.nearest(p['lat'], p['lon'])
                    utilities[i] = recall_pair(reference, reply, states, truth)
                    category[i] = {}
                    for ci, name in enumerate(reference.categories):
                        refs = set(reference.signatures[truth, ci])-{-1}
                        got = set(reply.signatures[states, ci].ravel())-{-1}
                        category[i][name] = len(refs & got)/len(refs) if refs else None
                    items[i] = int(np.sum(reply.signatures[states] >= 0))
                anchor_hash = hashlib.sha256(json.dumps(model.evaluator_anchors).encode()).hexdigest() if model else None
                if 'paced' in method:
                    assert paced_anchor_hash is None or paced_anchor_hash == anchor_hash
                    paced_anchor_hash = anchor_hash
                bound = model.spent_bound if model else None
                assert bound is None or bound <= .23+1e-12
                ex = {'method': method, 'family_id': family_id, 'split': 'expanded_development',
                    'session_id': sid, 'rep': rep, 'clock_indices': sorted(clock), 'events': events,
                    'utility_by_index': utilities, 'category_recall_L10_by_index': category,
                    'reply_items_L10_by_index': items, 'budget_bound': bound,
                    'evaluator_anchor_sha256': anchor_hash, 'evaluator_ledger': model.evaluator_ledger if model else None,
                    'step_ms': list(model.step_ms) if model else [],
                    'whole_session_recall': {L: mean_optional([u[L] for u in utilities.values()]) for L in ('5', '10')}}
                executions.append(ex); lookup[sid, rep, method] = ex
    rows = []
    for record in records:
        for rep in range(2):
            for method in METHODS:
                predictions, targets, utility, public, categories = [], [], [], [], []
                for slot, (sid, indices) in enumerate(zip(record['session_ids'], record['observed_indices'])):
                    ex = lookup[sid, rep, method]; p = public_view(ex['events'], indices); public.append(p)
                    predictions.append(geometric_predictions(p, record['scenario'], rn))
                    targets.append(target_xy(record, slot, data['traces'][sid], rn))
                    utility.extend(ex['utility_by_index'][i] for i in indices)
                    categories.extend(ex['category_recall_L10_by_index'][i] for i in indices)
                pred = {a: np.concatenate([p[a] for p in predictions]) for a in predictions[0]}
                if record['case_id'] in ('S9.C', 'S10.C'):
                    pred.update({a+'_joint_mean': np.repeat(p.mean(axis=0, keepdims=True), 2, axis=0) for a, p in list(pred.items())})
                truth = np.concatenate(targets); selection = selections[method+'/'+record['case_id']]
                assert all(a in pred for a in selection.values())
                rows.append({'method': method, 'family_id': family_id, 'split': 'expanded_development',
                    'record_id': record['record_id'], 'case_id': record['case_id'], 'rep': rep, 'public_views': public,
                    'errors': {a: np.linalg.norm(p-truth, axis=1).tolist() for a, p in pred.items()},
                    'selected_attack': selection, 'composition_bound': .23*len(set(record['session_ids'])) if method != 'raw' else None,
                    'eligible_events': sum(u['5'] is not None for u in utility),
                    'empty_reference_events': sum(u['5'] is None for u in utility),
                    'recall': {L: mean_optional([u[L] for u in utility]) for L in ('5', '10')},
                    'category_recall_L10': {c: mean_optional([v[c] for v in categories]) for c in reference.categories}})
    result = {'schema': 'frozen-shortlist-expanded-development-v1', 'provenance': provenance,
              'family_id': family_id, 'executions': executions, 'rows': rows}
    pending = path.with_suffix('.pending')
    pending.write_bytes(gzip.compress(json.dumps(result, separators=(',', ':'), allow_nan=False).encode(), mtime=0))
    pending.replace(path)
    return {'family_id': family_id, 'file': path.name, 'sha256': sha(path), 'executions': len(executions), 'case_rows': len(rows)}


def main():
    if OUT.exists():
        raise FileExistsError('Preserve completed evidence')
    SHARDS.mkdir(exist_ok=True)
    selections = {}
    source_paths = [BASE/'iteration15_response_cases.json', BASE/'iteration17_paced_slack_cases.json']
    for path in source_paths:
        for row in json.loads(path.read_text())['summaries']:
            if row['method'] in METHODS:
                selections[row['method']+'/'+row['case_id']] = row['selected_attack']
    data = json.loads(DATA.read_text())
    assert len(data['families']) == 12
    assert all(f['split'] == 'expanded_development' for f in data['families'])
    assert all(float(f['actual_vehicles'][s['session_id']]['arrival']) >= 0 for f in data['families'] for s in f['sessions'])
    provenance = {'dataset_sha256': sha(DATA), 'code_sha256': sha(Path(__file__)),
        'source_sha256': {p: sha(ROOT/p) for p in SOURCES}, 'method_shortlist': METHODS,
        'attack_selection_source_sha256': {p.name: sha(p) for p in source_paths},
        'resources_sha256': json.loads((CACHE/'resources.json').read_text())['resource_sha256'],
        'K': 5, 'reference_k': 5, 'reply_L': 10, 'privacy_cap': .23, 'workers': 2}
    # JSON normalisation makes resume comparisons independent of tuple/list types.
    provenance = json.loads(json.dumps(provenance))
    manifests = []
    with ProcessPoolExecutor(max_workers=2, initializer=initialize, initargs=(provenance, selections)) as pool:
        pending = {pool.submit(family_run, f['family_id']): f['family_id'] for f in data['families']}
        for future in as_completed(pending):
            item = future.result(); manifests.append(item)
            print('Completed', item['family_id'], len(manifests), '/12', flush=True)
    rows = []
    for item in sorted(manifests, key=lambda x: x['family_id']):
        shard = json.loads(gzip.decompress((SHARDS/item['file']).read_bytes())); rows.extend(shard['rows'])
    summaries = []
    for method in METHODS:
        for case in sorted({r['case_id'] for r in rows}):
            selected = [r for r in rows if r['method'] == method and r['case_id'] == case]
            selection = selections[method+'/'+case]
            recall = {L: mean_optional([r['recall'][L] for r in selected]) for L in ('5', '10')}
            summaries.append({'method': method, 'case_id': case, 'family_count': len({r['family_id'] for r in selected}),
                'record_count': len({r['record_id'] for r in selected}), 'RNG_repetitions': 2, 'recall': recall,
                'pass_90pct_case_recall_L10': passes_recall_gate(recall['10']), 'selected_attack': selection,
                'mae_m': float(np.mean([np.mean(r['errors'][selection['mae']]) for r in selected])),
                'hits': {str(rad): float(np.mean([np.mean(np.array(r['errors'][selection[f'hit{rad}']]) <= rad) for r in selected])) for rad in (50, 100, 200, 500)},
                'family_L10': {f: mean_optional([r['recall']['10'] for r in selected if r['family_id'] == f]) for f in sorted({r['family_id'] for r in selected})},
                'category_L10': {c: mean_optional([r['category_recall_L10'][c] for r in selected]) for c in selected[0]['category_recall_L10']}})
    OUT.write_text(json.dumps({'scope': 'expanded development screening, frozen shortlist; fixed old-train finite decoders; NOT final confirmation or sufficient attack coverage',
        'provenance': provenance, 'shards': sorted(manifests, key=lambda x: x['family_id']),
        'selection': selections, 'summaries': summaries}, indent=2, allow_nan=False)+'\n')
    print(json.dumps(summaries, indent=2), flush=True)


if __name__ == '__main__':
    main()
