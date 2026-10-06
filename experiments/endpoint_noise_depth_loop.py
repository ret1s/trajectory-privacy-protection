"""Round two: stronger accounted noise versus public reply depth, without delay.

Round-one failures remain intact. Scale/depth and attacker decisions use only
701--710. Fresh within-study heldout families1201--1204 are opened after the
selection file is written and never used for re-selection. Both source cohorts
are existing simulated data, so this is not independent future confirmation.
"""
import argparse
import gzip
import json
from pathlib import Path
import pickle

import numpy as np

from benchmark.anchor_belief import PublicAnchorModel
from benchmark.engines.endpoint_noise import EndpointNoiseProgressLaneDummy
from benchmark.engines.endpoint_phase_noise import EndpointPhaseNoiseProgressLaneDummy
from benchmark.engines.paced_slack import PacedSlackProgressLaneDummy
from benchmark.paper_comparators import PublicHistory
from benchmark.public_poi_context import PublicPoiContext
from benchmark.response_aware_belief import ResponseAwareAnchorModel
from core.boundary_release import BoundaryPolicy, BoundaryProtectedStream
from evaluation.endpoint_noise_attacks import EndpointShadowBank, endpoint_features, family_mean, select_attackers
from evaluation.lane_travel import LanePoiService
from evaluation.live_poi import AvailabilityWorld, LivePointService, EpochResponseCache, score_returned
from experiments import endpoint_noise_loop as common
from experiments.public_research_resources import ROOT, POIS, sha, load_public_research_resources

FRESH = ROOT/'artifacts/datasets/endpoint_holdout_expanded_v1/dataset.json.gz'
CONFIGS = [{'id': f'scale{int(scale*100):03d}_L{depth}', 'scale': scale, 'L': depth,
            'phase': False, 'delay': False} for scale in (1., .5, .25) for depth in (10, 20, 40)]
CONFIGS += [dict(id='originquarter_L10', scale=1., L=10, phase=True, delay=False),
            dict(id='delay60_L10', scale=1., L=10, phase=False, delay=True)]


def seal(out, round_one_protocol):
    if (out/'protocol.json').exists():
        raise FileExistsError('Never overwrite sealed round-two protocol')
    original = common.read(round_one_protocol)
    sources = {file: sha(ROOT/file) for file in original['source_sha256']}
    sources.update({str(p.relative_to(ROOT)): sha(p) for p in
        (Path(__file__), ROOT/'benchmark/engines/endpoint_phase_noise.py',
         ROOT/'benchmark/engines/origin_guard.py', ROOT/'benchmark/response_aware_belief.py',
         ROOT/'benchmark/public_poi_context.py')})
    # Do not open the fresh dataset, including its summary, during selection.
    # Byte hash only pins the source without inspecting labels/results.
    common.write(out/'protocol.json', dict(original,
        schema='endpoint-noise-depth-v2', source_sha256=sources,
        fresh_dataset_sha256=sha(FRESH), test_families=[f'family-{i}' for i in range(1201, 1205)],
        configurations=CONFIGS,
        selection_rule='Recall >= 90%; minimize mean selected S9/S10 Hit100, '
            'then maximize mean MAE, then lower JSON bytes; retain all failures; '
            'matched same-L baseline comparison; do not infer dominance from tied Hit100',
        byte_contract='Compact JSON requests + POI ID responses, no HTTP/TLS; per input tick; '
                      'deeper replies are public and position-independent',
        scope='Second bounded development loop on full reconstructed public map, '
              'existing SUMO families; new within-study heldout 1201–1204 after '
              'round-one test 711–712 was already examined; no reuse of touched test for selection'))
    (out/'protocol.sha256').write_text(sha(out/'protocol.json')+'\n')


def protocol(out):
    p = common.read(out/'protocol.json')
    assert sha(out/'protocol.json') == (out/'protocol.sha256').read_text().strip()
    assert sha(common.DATA) == p['dataset_sha256']
    assert sha(FRESH) == p['fresh_dataset_sha256']
    for file, digest in p['source_sha256'].items():
        assert sha(ROOT/file) == digest, file
    return p


def resources(cache):
    rn, service, reference, reply, base, ranking, metadata = load_public_research_resources(cache)
    pois = [{k: v for k, v in p.items() if k not in ('vertex', 'access_offset_m')}
            for p in json.loads(POIS.read_text())['pois_used']]
    contexts = {10: reply}
    for depth in (20, 40):
        contexts[depth] = PublicPoiContext(LanePoiService(rn, pois, k=depth),
            cache/f'endpoint-depth-{rn.catalogue_sha256[:12]}-L{depth}.npz')
    _, inv, counts = np.unique(np.floor(rn.xy/120.).astype(np.int64), axis=0,
                               return_inverse=True, return_counts=True)
    prior = 1./counts[inv]
    prior /= prior.sum()
    bases = {1.: base.base}
    for scale in (.5, .25):
        bases[scale] = PublicAnchorModel(rn, reference, prior, spacing_m=200.,
            epsilon_release=.01*scale, epsilon_test=.01*scale,
            cache_path=cache/f'endpoint-belief-{rn.catalogue_sha256[:12]}-{scale:g}.npz')
    beliefs = {(scale, depth): ResponseAwareAnchorModel(b, ctx)
               for scale, b in bases.items() for depth, ctx in contexts.items()}
    return rn, beliefs, ranking, metadata


def generate(session, config, seed, rn, beliefs, ranking, world):
    raw = config['id'] == 'raw'
    if not raw:
        options = dict(belief_model=beliefs[config['scale'], config['L']], k=5, budget=.24,
            horizon=12, theta_m=200., read_interval_s=60., utility_slack=.03,
            rng=np.random.default_rng(seed))
        if config['phase']:
            engine = EndpointPhaseNoiseProgressLaneDummy(rn, guard_seconds=60.,
                early_belief_model=beliefs[.25, config['L']], **options)
        elif config['delay']:
            engine = PacedSlackProgressLaneDummy(rn, **options)
        else:
            engine = EndpointNoiseProgressLaneDummy(rn, privacy_scale=config['scale'], **options)
        stream = BoundaryProtectedStream(engine, BoundaryPolicy(60., 60.) if config['delay']
                                         else BoundaryPolicy(0., 0.), session_start_s=0.)
    server = LivePointService(ranking, world, response_l=config['L'])
    cache = EpochResponseCache(ranking.n)
    events, scores, mapping, delays, pending = [], [], [], [], []
    request_bytes = response_bytes = 0
    for point in session['points']:
        t = point['t']
        if raw:
            emitted = [{'timestamp_s': t, 'coordinates': [[point['lat'], point['lon']]]}]
            delays.append(0.)
        else:
            before = stream.generated
            released = stream.ingest(t, point['lat'], point['lon'])
            if stream.generated > before:
                pending.append(t)
            emitted = []
            for event in released:
                emitted.append({'timestamp_s': event.timestamp_s,
                                'coordinates': [[c.lat, c.lon] for c in event.candidates]})
                delays.append(t-pending.pop(0))
        epoch, replies = world.epoch(t), []
        for event in emitted:
            events.append(event)
            reply = [server.query(rn.nearest(*p)[0], epoch) for p in event['coordinates']]
            replies.extend(reply)
            request_bytes += len(json.dumps(dict(event, L=config['L']), separators=(',', ':')).encode())
            response_bytes += len(json.dumps([[ [ranking.pois[i]['id'] for i in category]
                for category in point_reply] for point_reply in reply], separators=(',', ':')).encode())
        _, known = cache.receive(epoch, replies)
        state, distance = rn.nearest(point['lat'], point['lon'])
        mapping.append(float(distance))
        available = world.at_epoch(epoch)
        scores.append(score_returned(ranking.top(state, available, 5),
                                    ranking.top(state, known, 5), available)['recall'])
    if not raw:
        stream.close(session['close_s'])
        accounting = stream.evaluator_summary()
        bound = .23 if config['phase'] else .23*config['scale']
        assert engine.spent_bound <= bound+1e-12
        spent = engine.spent_bound
    else:
        spent = None
        accounting = dict(input_events=len(session['points']), released_events=len(events),
                          head_suppressed=0, tail_cancelled=0)
    if raw or not config['delay']:
        assert [e['timestamp_s'] for e in events] == [p['t'] for p in session['points']]
    valid = [v for v in scores if v is not None]
    return {k: session[k] for k in ('family_id', 'session_id', 'close_s')} | {
        'method': config['id'], 'config': config, 'seed': seed, 'events': events,
        'accounting': accounting, 'budget_spent_per_m': spent,
        'recall': float(np.mean(valid)) if valid else None,
        'eligible_service_times': len(valid), 'empty_service_times': len(scores)-len(valid),
        'request_bytes': request_bytes, 'response_bytes': response_bytes,
        'bytes_per_input': (request_bytes+response_bytes)/len(session['points']),
        'publication_mean_delay_s': float(np.mean(delays)) if delays else None,
        'mapping_mean_m': float(np.mean(mapping)), 'mapping_max_m': max(mapping),
        'target_xy_evaluator_only': {s: list(rn.point_xy(p['lat'], p['lon']))
                                   for s, p in [('S9', session['points'][0]), ('S10', session['points'][-1])]}}


def summarize(executions, rows, attackers):
    result = common.score_summary(rows, attackers)
    result.update(recall=family_mean([r for r in executions if r['recall'] is not None], 'recall'),
        bytes_per_input=family_mean(executions, 'bytes_per_input'),
        mean_publication_delay_s=family_mean(executions, 'publication_mean_delay_s'),
        input_events=sum(r['accounting']['input_events'] for r in executions),
        released_events=sum(r['accounting']['released_events'] for r in executions),
        mapping_max_m=max(r['mapping_max_m'] for r in executions))
    result['mean_selected_hit100'] = float(np.mean([result[s]['hit100'] for s in ('S9', 'S10')]))
    result['mean_selected_mae_m'] = float(np.mean([result[s]['mae_m'] for s in ('S9', 'S10')]))
    return result


def fit_select(out, cache):
    p = protocol(out)
    if (out/'selection.json').exists():
        raise FileExistsError('Round-two selection already sealed')
    rn, beliefs, ranking, metadata = resources(cache)
    data = common.read(common.DATA)
    fit = list(common.sessions(data, p['fit_families']))
    dev = list(common.sessions(data, p['selection_families']))
    history = PublicHistory(rn, [[rn.nearest(point['lat'], point['lon'])[0] for point in s['points']] for s in fit])
    world = AvailabilityWorld(ranking.n, p['world_seed'], probability=.8, epoch_seconds=60)
    banks, selections, summaries, all_rows = {}, {}, {}, []
    configs = [dict(id='raw', L=10)] + p['configurations']
    for config in configs:
        method = config['id']
        generated = {}
        for split, inputs in [('fit', fit), ('selection', dev)]:
            generated[split] = []
            for session in inputs:
                for seed in p['rep_seeds']:
                    try:
                        generated[split].append(generate(session, config, seed, rn, beliefs, ranking, world))
                    except Exception as error:
                        common.write(out/'generation_failure.json', dict(split=split, config=config,
                            family_id=session['family_id'], session_id=session['session_id'], seed=seed,
                            error=type(error).__name__+': '+str(error), replacement=None))
                        raise
            common.write(out/f'{split}-{method}.json.gz', generated[split])
        chosen, rows = {}, []
        for scenario in ('S9', 'S10'):
            aggregates, sequences, targets = [], [], []
            for item in generated['fit']:
                a, sequence, _ = endpoint_features(item['events'], scenario, rn, history,
                                                   observable_close_s=item['close_s'])
                aggregates.append(a[0]); sequences.append(sequence[0])
                targets.append(item['target_xy_evaluator_only'][scenario])
            bank = EndpointShadowBank(aggregates, sequences, targets)
            banks[method, scenario] = bank
            scored = common.bank_rows(generated['selection'], scenario, bank, rn, history)
            chosen[scenario] = select_attackers(scored)
            rows.extend(scored)
        summaries[method] = summarize(generated['selection'], rows, chosen)
        selections[method] = chosen
        all_rows.extend(rows)
        print('Development', method, 'Recall', summaries[method]['recall'],
              'Hit100', summaries[method]['mean_selected_hit100'],
              'MAE', summaries[method]['mean_selected_mae_m'],
              'bytes', summaries[method]['bytes_per_input'], flush=True)
    candidates = [c['id'] for c in p['configurations'] if not c['delay']]
    feasible = [m for m in candidates if summaries[m]['recall'] is not None and summaries[m]['recall'] >= .90]
    selected = min(feasible, key=lambda m: (summaries[m]['mean_selected_hit100'],
        -summaries[m]['mean_selected_mae_m'], summaries[m]['bytes_per_input'], m)) if feasible else None
    with (out/'attackers.pkl').open('wb') as f:
        pickle.dump(dict(banks=banks, history=history), f)
    common.write(out/'development_summary.json', summaries)
    common.write(out/'development_attack_rows.json.gz', all_rows)
    common.write(out/'resources.json', metadata)
    common.write(out/'selection.json', dict(protocol_sha256=sha(out/'protocol.json'),
        selected_defense=selected, selected_attackers=selections,
        target_recall_met=bool(feasible), all_development_configs_retained=True,
        fresh_test_not_opened_by_fit_select=True,
        attackers_sha256=sha(out/'attackers.pkl'), development_summary_sha256=sha(out/'development_summary.json')))
    print('Round-two selection sealed', selected, flush=True)


def test(out, cache):
    p = protocol(out)
    if (out/'heldout.json').exists():
        raise FileExistsError('Retain first fresh heldout result')
    selection = common.read(out/'selection.json')
    assert sha(out/'attackers.pkl') == selection['attackers_sha256']
    assert sha(out/'development_summary.json') == selection['development_summary_sha256']
    rn, beliefs, ranking, _ = resources(cache)
    data = common.read(FRESH)  # opened only after selection is frozen above
    with (out/'attackers.pkl').open('rb') as f:
        saved = pickle.load(f)
    by_id = {c['id']: c for c in p['configurations']}
    selected = selection['selected_defense']
    methods = ['raw', 'scale100_L10', 'delay60_L10']
    if selected:
        depth = by_id[selected]['L']
        methods += [f'scale100_L{depth}', selected]
    summaries, all_rows = {}, []
    world = AvailabilityWorld(ranking.n, p['world_seed'], probability=.8, epoch_seconds=60)
    for method in dict.fromkeys(methods):
        config = dict(id='raw', L=10) if method=='raw' else by_id[method]
        executions = [generate(session, config, seed, rn, beliefs, ranking, world)
                      for session in common.sessions(data, p['test_families']) for seed in p['rep_seeds']]
        rows = [row for scenario in ('S9', 'S10') for row in common.bank_rows(executions,
            scenario, saved['banks'][method, scenario], rn, saved['history'])]
        summaries[method] = summarize(executions, rows, selection['selected_attackers'][method])
        all_rows.extend(rows)
        common.write(out/f'test-{method}.json.gz', executions)
        print('Fresh heldout', method, summaries[method], flush=True)
    common.write(out/'heldout_attack_rows.json.gz', all_rows)
    common.write(out/'heldout.json', dict(schema='endpoint-noise-depth-heldout-v2',
        protocol_sha256=sha(out/'protocol.json'), selection_sha256=sha(out/'selection.json'),
        selected_defense=selected, rows=summaries, scope=p['scope'],
        no_post_test_reselection=True,
        raw_positive_control={s: summaries['raw'][s]['hit100']==1. for s in ('S9', 'S10')}))
    protocol(out)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage', choices=['prepare', 'select', 'test', 'all', 'verify'])
    parser.add_argument('--out', type=Path, default=Path('/private/tmp/endpoint-noise-depth-20261005-review'))
    parser.add_argument('--cache', type=Path, default=Path('/private/tmp/trajectory-research-20261005-public-map'))
    parser.add_argument('--round-one-protocol', type=Path,
        default=ROOT/'artifacts/benchmarks/endpoint_noise_20261005/round1/protocol.json')
    args = parser.parse_args()
    if args.stage in ('prepare', 'all'):
        seal(args.out, args.round_one_protocol)
    if args.stage in ('select', 'all'):
        fit_select(args.out, args.cache)
    if args.stage in ('test', 'all'):
        test(args.out, args.cache)
    if args.stage == 'verify':
        protocol(args.out)
        heldout = common.read(args.out/'heldout.json')
        assert heldout['selection_sha256'] == sha(args.out/'selection.json')
        assert heldout['protocol_sha256'] == sha(args.out/'protocol.json')
        print('Round-two protocol and frozen selection verified')


if __name__ == '__main__':
    main()
