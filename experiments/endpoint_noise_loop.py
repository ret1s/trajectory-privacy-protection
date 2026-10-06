"""Bounded endpoint-noise development, sealed selection, then family holdout.

This is a new full-city reconstructed-map development experiment. It never
overwrites old benchmarks. All planned seeds/configurations, unsuccessful
utility thresholds and generation failures are retained. Public session close
is available to attackers, as it is in the actual no-delay transport.
"""
import argparse
from collections import defaultdict
import gzip
import hashlib
import json
from pathlib import Path
import pickle

import numpy as np

from benchmark.anchor_belief import PublicAnchorModel
from benchmark.engines.endpoint_noise import EndpointNoiseProgressLaneDummy
from benchmark.engines.paced_slack import PacedSlackProgressLaneDummy
from benchmark.paper_comparators import PublicHistory
from benchmark.response_aware_belief import ResponseAwareAnchorModel
from core.boundary_release import BoundaryPolicy, BoundaryProtectedStream
from evaluation.endpoint_noise_attacks import EndpointShadowBank, endpoint_features, family_mean, select_attackers
from evaluation.live_poi import AvailabilityWorld, LivePointService, EpochResponseCache, score_returned
from experiments.public_research_resources import ROOT, load_public_research_resources, sha

DATA = ROOT/'artifacts/datasets/research_loop_expanded_v1/dataset.json'
SCALES = (1., .5, .25, .125)
METHODS = ('raw', 'plain', 'delay60', 'noise50', 'noise25', 'noise125')
SCALE = dict(plain=1., delay60=1., noise50=.5, noise25=.25, noise125=.125)


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(value, indent=2, allow_nan=False, ensure_ascii=False)+'\n'
    if path.suffix == '.gz':
        path.write_bytes(gzip.compress(text.encode(), mtime=0))
    else:
        path.write_text(text)


def read(path):
    path = Path(path)
    return json.loads(gzip.decompress(path.read_bytes()) if path.suffix == '.gz' else path.read_text())


def seal(out):
    path = out/'protocol.json'
    if path.exists():
        raise FileExistsError('Protocol already sealed; use verify or a fresh output directory')
    files = ['experiments/endpoint_noise_loop.py', 'experiments/public_research_resources.py',
        'benchmark/engines/endpoint_noise.py', 'evaluation/endpoint_noise_attacks.py',
        'benchmark/engines/paced_slack.py', 'benchmark/engines/paced_guard.py',
        'benchmark/engines/matched_filter.py', 'benchmark/engines/filtered_cover.py',
        'benchmark/engines/progress_cover.py', 'benchmark/engines/slack_progress.py',
        'benchmark/anchor_belief.py', 'core/mechanisms.py', 'core/boundary_release.py',
        'evaluation/live_comparison_attacks.py', 'evaluation/live_comparison_endpoint_attacks.py',
        'evaluation/live_poi.py', 'benchmark/paper_comparators.py']
    write(path, {'schema': 'endpoint-noise-loop-v1', 'date': '2026-10-05',
        'source_sha256': {f: sha(ROOT/f) for f in files}, 'dataset_sha256': sha(DATA),
        'fit_families': [f'family-{i}' for i in range(701, 709)],
        'selection_families': ['family-709', 'family-710'],
        'test_families': ['family-711', 'family-712'], 'sessions_per_family': 2,
        'session_rule': 'lexically first two existing IDs per planned family; no score-based exclusions',
        'query_interval_s': 60., 'read_interval_s': 60., 'include_true_last_input': True,
        'rep_seeds': [2026100501, 2026100502], 'methods': METHODS,
        'scale_grid': SCALES, 'budget_per_m': .24, 'H': 12, 'theta_m': 200.,
        'K': 5, 'L': 10, 'slack': .03, 'availability_p': .8,
        'world_seed': 2026100541, 'map_spacing_m': 40., 'latent_spacing_m': 200.,
        'scope': 'Small same-generator development on reconstructed full-city map. '
                 'Heldout families only within this run; source data were previously used in historical research. '
                 'Not independent confirmation or rerun of original endpoint-calendar benchmark.',
        'attacker_bank': 'per-method aggregate and whole-sequence kNN/ExtraTrees; '
                         'centroid/median/history/motion Viterbi; OLS and observable-boundary OLS; fixed training priors',
        'visible_timing': 'session start 0, close, request coordinates and actual publication timestamps',
        'defense_selection': 'minimum mean S9/S10 selected Hit100 subject to all-input Recall>=0.90; '
                             'ties maximize Recall then choose largest epsilon scale; '
                             'if none feasible, maximize Recall then minimize Hit100 and mark failed target',
        'test_rule': 'freeze attacker/defense selection before test; do not refit after test',
        'aggregation': 'session/rep within family, then equal-family mean; S9 and S10 separately',
        'excluded_targets': 'No S9.B partial-prefix or linked-session S9.C/S10.C claim; complete first/last endpoints only'})
    (out/'protocol.sha256').write_text(sha(path)+'\n')


def protocol(out):
    p = read(out/'protocol.json')
    assert sha(out/'protocol.json') == (out/'protocol.sha256').read_text().strip()
    assert sha(DATA) == p['dataset_sha256']
    for file, digest in p['source_sha256'].items():
        assert sha(ROOT/file) == digest, file
    return p


def sessions(data, families):
    for family in data['families']:
        if family['family_id'] not in families:
            continue
        ids = sorted(s['session_id'] for s in family['sessions'] if s['session_id'] in data['traces'])[:2]
        assert len(ids) == 2, family['family_id']
        for sid in ids:
            trace = data['traces'][sid]
            start = trace[0]['time_s']
            duration = trace[-1]['time_s']-start
            # Times are chosen by public clock, not by distance/accuracy outcome.
            selected = [0]
            next_time = 60.
            for i, point in enumerate(trace[1:], 1):
                if point['time_s']-start >= next_time:
                    selected.append(i)
                    next_time = point['time_s']-start+60.
            if selected[-1] != len(trace)-1:
                selected.append(len(trace)-1)
            yield {'family_id': family['family_id'], 'session_id': sid, 'close_s': duration,
                   'points': [dict(trace[i], t=trace[i]['time_s']-start) for i in selected]}


def make_beliefs(rn, reference, reply, base, cache):
    result = {1.: base}
    # Preserve the same public latent grid/prior across epsilon settings.
    _, inv, counts = np.unique(np.floor(rn.xy/120.).astype(np.int64), axis=0,
                               return_inverse=True, return_counts=True)
    prior = 1./counts[inv]
    prior /= prior.sum()
    for scale in SCALES[1:]:
        b = PublicAnchorModel(rn, reference, prior, spacing_m=200.,
            epsilon_release=.01*scale, epsilon_test=.01*scale,
            cache_path=cache/f'endpoint-belief-{rn.catalogue_sha256[:12]}-{scale:g}.npz')
        result[scale] = ResponseAwareAnchorModel(b, reply)
    return result


def generate(session, method, seed, rn, beliefs, ranking, world):
    if method != 'raw':
        scale = SCALE[method]
        options = dict(belief_model=beliefs[scale], k=5, budget=.24, horizon=12,
            theta_m=200., read_interval_s=60., utility_slack=.03, rng=np.random.default_rng(seed))
        engine = (PacedSlackProgressLaneDummy(rn, **options) if method == 'delay60'
                  else EndpointNoiseProgressLaneDummy(rn, privacy_scale=scale, **options))
        stream = BoundaryProtectedStream(engine, BoundaryPolicy(60., 60.) if method == 'delay60'
                                         else BoundaryPolicy(0., 0.), session_start_s=0.)
    server, cache = LivePointService(ranking, world, response_l=10), EpochResponseCache(ranking.n)
    events, scores, delays, mapping = [], [], [], []
    source_times = []
    for point in session['points']:
        t = point['t']
        if method == 'raw':
            emitted = [{'timestamp_s': t, 'coordinates': [[point['lat'], point['lon']]]}]
            delays.append(0.)
        else:
            before = stream.generated
            emitted = []
            released = stream.ingest(t, point['lat'], point['lon'])
            if stream.generated > before:
                source_times.append(t)
            for event in released:
                emitted.append({'timestamp_s': event.timestamp_s,
                    'coordinates': [[c.lat, c.lon] for c in event.candidates]})
                delays.append(t-source_times.pop(0))
        epoch, replies = world.epoch(t), []
        for event in emitted:
            events.append(event)
            replies.extend(server.query(rn.nearest(*p)[0], epoch) for p in event['coordinates'])
        _, known = cache.receive(epoch, replies)
        state, distance = rn.nearest(point['lat'], point['lon'])
        mapping.append(float(distance))
        available = world.at_epoch(epoch)
        scores.append(score_returned(ranking.top(state, available, 5),
                                    ranking.top(state, known, 5), available)['recall'])
    if method != 'raw':
        stream.close(session['close_s'])
        accounting = stream.evaluator_summary()
        spent = engine.spent_bound
        assert spent <= .23*SCALE[method]+1e-12
        for left, right, a, b in zip(engine.evaluator_states, engine.evaluator_states[1:],
                                   [p['t'] for p in session['points']][-len(engine.evaluator_states):],
                                   [p['t'] for p in session['points']][-len(engine.evaluator_states):][1:]):
            assert all(v in engine.travel.reachable(u, b-a) for u, v in zip(left, right))
    else:
        accounting = dict(input_events=len(session['points']), released_events=len(events),
                          head_suppressed=0, tail_cancelled=0)
        spent = None
    if method != 'delay60':
        assert [e['timestamp_s'] for e in events] == [p['t'] for p in session['points']]
    eligible = [value for value in scores if value is not None]
    return {k: session[k] for k in ('family_id', 'session_id', 'close_s')} | {
        'method': method, 'seed': seed, 'events': events, 'accounting': accounting,
        'budget_spent_per_m': spent, 'recall': float(np.mean(eligible)) if eligible else None,
        'eligible_service_times': len(eligible), 'empty_service_times': len(scores)-len(eligible),
        'publication_mean_delay_s': float(np.mean(delays)) if delays else None,
        'publication_max_delay_s': max(delays) if delays else None,
        'mapping_mean_m': float(np.mean(mapping)), 'mapping_max_m': max(mapping),
        'target_xy_evaluator_only': {scenario: list(rn.point_xy(point['lat'], point['lon']))
                                   for scenario, point in [('S9', session['points'][0]), ('S10', session['points'][-1])]}}


def bank_rows(generated, scenario, bank, rn, history):
    rows = []
    for item in generated:
        if not item['events']:
            rows.append(dict(item, scenario=scenario, status='empty_transcript', errors={}))
            continue
        truth = np.asarray(item['target_xy_evaluator_only'][scenario])
        predictions = bank.predictions(item['events'], scenario, rn, history,
                                       observable_close_s=item['close_s'])
        errors = {name: float(np.linalg.norm(value[0]-truth)) for name, value in predictions.items()}
        rows.append({k: item[k] for k in ('family_id', 'session_id', 'method', 'seed', 'recall', 'close_s')}
                    | {'scenario': scenario, 'status': 'ok', 'errors': errors})
    return rows


def score_summary(rows, selections):
    summaries = {}
    for scenario in ('S9', 'S10'):
        selected = selections[scenario]
        values = [dict(r, mae_m=r['errors'][selected['mae']],
                       **{'hit'+str(radius): float(r['errors'][selected['hit'+str(radius)]] <= radius)
                          for radius in (50, 100, 200, 500)}) for r in rows if r['scenario']==scenario and r['status']=='ok']
        summaries[scenario] = {key: family_mean(values, key) for key in ('mae_m', 'hit50', 'hit100', 'hit200', 'hit500')}
        summaries[scenario]['selected_attackers'] = selected
        summaries[scenario]['families'] = len({r['family_id'] for r in values})
        summaries[scenario]['observations'] = len(values)
    return summaries


def setup(p, cache):
    rn, service, reference, reply, base, ranking, metadata = load_public_research_resources(cache)
    beliefs = make_beliefs(rn, reference, reply, base, cache)
    data = read(DATA)
    fit = list(sessions(data, p['fit_families']))
    history = PublicHistory(rn, [[rn.nearest(point['lat'], point['lon'])[0] for point in s['points']] for s in fit])
    world = AvailabilityWorld(ranking.n, p['world_seed'], probability=.8, epoch_seconds=60)
    return rn, beliefs, ranking, metadata, data, fit, history, world


def fit_select(out, cache):
    p = protocol(out)
    if (out/'selection.json').exists():
        raise FileExistsError('Selection already sealed')
    rn, beliefs, ranking, metadata, data, fit, history, world = setup(p, cache)
    development = list(sessions(data, p['selection_families']))
    fitted, selections, summaries, all_rows, executions, failures = {}, {}, {}, [], [], []
    for method in METHODS:
        groups = {}
        for split, inputs in [('fit', fit), ('selection', development)]:
            groups[split] = []
            for session in inputs:
                for seed in p['rep_seeds']:
                    try:
                        item = generate(session, method, seed, rn, beliefs, ranking, world)
                    except Exception as error:
                        failures.append({'split': split, 'method': method, 'family_id': session['family_id'],
                            'session_id': session['session_id'], 'seed': seed,
                            'error': type(error).__name__+': '+str(error)})
                        write(out/'generation_failures.json', failures)
                        raise RuntimeError('Generation failure retained; no replacement') from error
                    groups[split].append(item)
            write(out/f'{split}-{method}.json.gz', groups[split])
        chosen, scored = {}, []
        for scenario in ('S9', 'S10'):
            x, seq, truth = [], [], []
            for item in groups['fit']:
                if not item['events']:
                    continue
                aggregate, sequence, _ = endpoint_features(item['events'], scenario, rn, history,
                                                           observable_close_s=item['close_s'])
                x.append(aggregate[0]); seq.append(sequence[0]); truth.append(item['target_xy_evaluator_only'][scenario])
            fitted[method, scenario] = EndpointShadowBank(x, seq, truth)
            rows = bank_rows(groups['selection'], scenario, fitted[method, scenario], rn, history)
            scored.extend(rows)
            chosen[scenario] = select_attackers([r for r in rows if r['status']=='ok'])
        selections[method] = chosen
        summary = score_summary(scored, chosen)
        utility = family_mean([r for r in groups['selection'] if r['recall'] is not None], 'recall')
        summaries[method] = dict(summary, recall=utility,
            target_recall_met=utility is not None and utility >= .90,
            mean_selected_hit100=float(np.mean([summary[s]['hit100'] for s in ('S9', 'S10')])),
            mapping_max_m=max(r['mapping_max_m'] for r in groups['selection']))
        all_rows.extend(scored)
        executions.extend(groups['selection'])
        print('Development', method, summaries[method], flush=True)
    candidates = [method for method in SCALE if method != 'delay60']
    feasible = [m for m in candidates if summaries[m]['target_recall_met']]
    if feasible:
        selected = min(feasible, key=lambda m: (summaries[m]['mean_selected_hit100'], -summaries[m]['recall'], -SCALE[m], m))
    else:
        selected = min(candidates, key=lambda m: (-summaries[m]['recall'], summaries[m]['mean_selected_hit100'], -SCALE[m], m))
    with (out/'attackers.pkl').open('wb') as f:
        pickle.dump(dict(bank=fitted, history=history), f)
    write(out/'development_attack_rows.json.gz', all_rows)
    write(out/'development_summary.json', summaries)
    write(out/'generation_failures.json', failures)
    write(out/'resources.json', metadata)
    write(out/'selection.json', {'protocol_sha256': sha(out/'protocol.json'),
        'selected_defense': selected, 'target_recall_met': bool(feasible),
        'all_development_configs_retained': True, 'test_metrics_not_used': True,
        'selected_attackers': selections, 'attackers_sha256': sha(out/'attackers.pkl'),
        'development_summary_sha256': sha(out/'development_summary.json')})
    print('Selection sealed:', selected, flush=True)


def evaluate(out, cache):
    p = protocol(out)
    if (out/'heldout.json').exists():
        raise FileExistsError('Retain the first heldout result; no post-test re-selection')
    selection = read(out/'selection.json')
    assert sha(out/'attackers.pkl') == selection['attackers_sha256']
    assert sha(out/'development_summary.json') == selection['development_summary_sha256']
    rn, beliefs, ranking, metadata, data, _, _, world = setup(p, cache)
    with (out/'attackers.pkl').open('rb') as f:
        saved = pickle.load(f)  # trusted locally generated, checksum-verified
    methods = list(dict.fromkeys(('raw', 'plain', 'delay60', selection['selected_defense'])))
    summaries, all_rows = {}, []
    for method in methods:
        executions = [generate(session, method, seed, rn, beliefs, ranking, world)
                      for session in sessions(data, p['test_families']) for seed in p['rep_seeds']]
        rows = [row for scenario in ('S9', 'S10')
                for row in bank_rows(executions, scenario, saved['bank'][method, scenario], rn, saved['history'])]
        chosen = selection['selected_attackers'][method]
        summary = score_summary(rows, chosen)
        summaries[method] = dict(summary,
            recall=family_mean([r for r in executions if r['recall'] is not None], 'recall'),
            input_events=sum(r['accounting']['input_events'] for r in executions),
            released_events=sum(r['accounting']['released_events'] for r in executions),
            mean_publication_delay_s=family_mean(executions, 'publication_mean_delay_s'),
            max_mapping_error_m=max(r['mapping_max_m'] for r in executions),
            mean_mapping_error_m=family_mean(executions, 'mapping_mean_m'),
            max_budget_spent_per_m=None if method=='raw' else max(r['budget_spent_per_m'] for r in executions))
        all_rows.extend(rows)
        write(out/f'test-{method}.json.gz', executions)
        print('Heldout', method, summaries[method], flush=True)
    write(out/'heldout_attack_rows.json.gz', all_rows)
    write(out/'heldout.json', {'schema': 'endpoint-noise-heldout-v1',
        'protocol_sha256': sha(out/'protocol.json'), 'selection_sha256': sha(out/'selection.json'),
        'selected_defense': selection['selected_defense'], 'rows': summaries,
        'scope': p['scope'], 'raw_positive_control': {
            scenario: summaries['raw'][scenario]['hit100'] == 1. for scenario in ('S9', 'S10')}})
    # Check all old input hashes after the complete run.
    protocol(out)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage', choices=['prepare', 'select', 'test', 'all', 'verify'])
    parser.add_argument('--out', type=Path, default=Path('/private/tmp/endpoint-noise-20261005'))
    parser.add_argument('--cache', type=Path, default=Path('/private/tmp/trajectory-research-20261005-public-map'))
    args = parser.parse_args()
    if args.stage in ('prepare', 'all'):
        seal(args.out)
    if args.stage in ('select', 'all'):
        fit_select(args.out, args.cache)
    if args.stage in ('test', 'all'):
        evaluate(args.out, args.cache)
    if args.stage == 'verify':
        protocol(args.out)
        selection, heldout = read(args.out/'selection.json'), read(args.out/'heldout.json')
        assert heldout['selection_sha256'] == sha(args.out/'selection.json')
        assert heldout['protocol_sha256'] == sha(args.out/'protocol.json')
        assert heldout['selected_defense'] == selection['selected_defense']
        print('Protocol, immutable inputs and frozen selection verified')


if __name__ == '__main__':
    main()
