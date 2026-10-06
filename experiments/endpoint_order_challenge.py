"""Post-inspection endpoint challenge: stable Q order versus private shuffle.

Reuse all frozen endpoint-generalization streams without changing their GPS,
Q multiset, clock, replies, utility or byte count. This is a DEVELOPMENT
diagnostic on previously inspected test families, not a new holdout claim.
"""
import argparse
from collections import defaultdict
import gzip
import hashlib
import hmac
from pathlib import Path
import pickle
import secrets

import numpy as np

from benchmark.paper_comparators import PublicHistory
from evaluation.endpoint_noise_attacks import family_mean, select_attackers
from evaluation.ordered_endpoint_attacks import OrderedEndpointBank, ordered_endpoint_features
from experiments import endpoint_noise_loop as common
from experiments.endpoint_generalization import CONFIGS
from experiments.endpoint_noise_readout import paired_interval
from experiments.public_research_resources import ROOT, load_public_research_resources, sha

SOURCE = ROOT/'artifacts/benchmarks/endpoint_generalization_20261005'
METHODS = [c['id'] for c in CONFIGS]
VIEWS = ('ordered', 'shuffled')
GROUPS = ('full', 'invariant', 'observed_slots', 'geometry_associated')
RNG_SCHEMA = 'endpoint-order-private-event-v1'


def permutation(master_hex, session_id, rep, event_index, size):
    """Private independent event/session substream, shared only matched methods."""
    master = bytes.fromhex(master_hex)
    if len(master) != 32 or not session_id or any(int(v) != v or v < 0 for v in (rep, event_index, size)):
        raise ValueError('32-byte master, session ID and nonnegative integers required')
    token = f'{RNG_SCHEMA}\0{session_id}\0{int(rep)}\0{int(event_index)}'.encode()
    seed = int.from_bytes(hmac.new(master, token, hashlib.sha256).digest()[:16], 'little')
    return np.random.default_rng(seed).permutation(size)


def publication_events(item, view, master_hex):
    """The attacker receives only public coordinates and the same public clock."""
    if view not in VIEWS:
        raise ValueError('Unknown publication view')
    result = []
    for j, event in enumerate(item['events']):
        coordinates = [list(p) for p in event['coordinates']]
        order = np.arange(len(coordinates)) if view == 'ordered' else permutation(
            master_hex, item['session_id'], item['seed'], j, len(coordinates))
        result.append(dict(timestamp_s=event['timestamp_s'], coordinates=[coordinates[int(i)] for i in order]))
    return result


def validate_view(original, published):
    assert len(original) == len(published)
    for a, b in zip(original, published):
        assert a['timestamp_s'] == b['timestamp_s']
        assert sorted(map(tuple, a['coordinates'])) == sorted(map(tuple, b['coordinates']))
        assert set(b) == {'timestamp_s', 'coordinates'}


def bank_group(rows, group):
    def allowed(name):
        observed = name.startswith('observed_slots_')
        geometry = name.startswith(('geometry_nearest_', 'geometry_velocity_'))
        return (group == 'full' or group == 'invariant' and not observed and not geometry
                or group == 'observed_slots' and not geometry
                or group == 'geometry_associated' and not observed)
    if group not in GROUPS:
        raise ValueError('Unknown attack group')
    return [dict(row, errors={name: value for name, value in row['errors'].items() if allowed(name)}) for row in rows]


def prepare(out, source):
    if (out/'protocol.json').exists():
        raise FileExistsError('Never overwrite a sealed ordered endpoint challenge')
    original = common.read(source/'protocol.json')
    out.mkdir(parents=True, exist_ok=True)
    files = [f'{split}-{method}.json.gz' for split in ('fit', 'selection', 'test') for method in METHODS]
    files += ['protocol.json', 'selection.json', 'heldout.json', 'resources.json']
    code = ['evaluation/ordered_endpoint_attacks.py', 'experiments/endpoint_order_challenge.py',
        'evaluation/endpoint_noise_attacks.py', 'evaluation/live_comparison_attacks.py',
        'evaluation/live_comparison_endpoint_attacks.py', 'benchmark/paper_comparators.py',
        'experiments/public_research_resources.py', 'experiments/endpoint_noise_readout.py',
        'experiments/endpoint_noise_loop.py']
    common.write(out/'evaluator_shuffle_randomness.json.gz', dict(schema=RNG_SCHEMA,
        master_hex=secrets.token_bytes(32).hex(), role='evaluator-only synthetic reproducibility key'))
    protocol = dict(schema='endpoint-ordered-view-challenge-v1', date='2026-10-05',
        source_reference=str(source.relative_to(ROOT)) if source.is_relative_to(ROOT) else str(source),
        input_sha256={file: sha(source/file) for file in files}, source_sha256={file: sha(ROOT/file) for file in code},
        private_shuffle_sha256=sha(out/'evaluator_shuffle_randomness.json.gz'),
        fit_families=original['fit_families'], selection_families=original['selection_families'],
        inspected_test_families=original['test_families'], methods=METHODS, views=list(VIEWS),
        scenarios=['S9', 'S10'], groups=list(GROUPS),
        status='POST-INSPECTION DEVELOPMENT DIAGNOSTIC; test families were already inspected',
        defense='Frozen raw/plainL10/plainL20/Endpoint20 Q sets; no new defense generation or tuning',
        intervention='Private independent permutation of each event coordinate list, retaining every duplicate; '
            'no candidate IDs published; original engine tracks unaffected',
        attacker='Per-slot endpoint and OLS 2/3/6 at observed boundary and horizons 0/30/60/120; '
            'ordered per-track ExtraTrees64/kNN1/kNN5 features; initial spatial sorting + consecutive Hungarian '
            'nearest and velocity matching with equivalent learned features; existing invariant controls',
        selection='Fit only 701–708; loss-specific decoder selection only 709–710, separately by method/view/group; '
            'freeze before this runner opens existing test labels; no defense selection or test-based bank changes',
        randomization=RNG_SCHEMA+' HMAC of evaluator-private master/session/rep/event; independent across unrelated sessions',
        inference='Target each trip\'s original GPS endpoint; public session close allowed; no IDs, internal Z, '
            'random seed, GPS or truth enter attack features',
        uncertainty='Paired bootstrap over all 28 already-inspected families; exploratory, no multiplicity correction',
        limits='Same synthetic SUMO generator and reconstructed public map; finite attacker bank; this removes '
            'explicit list-position labels only, not geometric trackability or account/IP/identity linkability; '
            'cannot replace earlier coordinate-set primary results with an independent confirmation claim')
    common.write(out/'protocol.json', protocol)
    (out/'protocol.sha256').write_text(sha(out/'protocol.json')+'\n')
    print('Sealed post-inspection challenge, all 28 previously inspected families', flush=True)


def protocol(out, source):
    p = common.read(out/'protocol.json')
    assert sha(out/'protocol.json') == (out/'protocol.sha256').read_text().strip()
    assert p['methods'] == METHODS and p['views'] == list(VIEWS)
    for file, digest in p['input_sha256'].items():
        assert sha(source/file) == digest, file
    for file, digest in p['source_sha256'].items():
        assert sha(ROOT/file) == digest, file
    assert sha(out/'evaluator_shuffle_randomness.json.gz') == p['private_shuffle_sha256']
    return p


def resources(cache, p):
    rn, _, _, _, _, _, metadata = load_public_research_resources(cache)
    data = common.read(common.DATA)
    fit = list(common.sessions(data, p['fit_families']))
    history = PublicHistory(rn, [[rn.nearest(point['lat'], point['lon'])[0] for point in session['points']] for session in fit])
    return rn, history, metadata


def attack_rows(items, view, master, scenario, bank, rn, history):
    rows = []
    for item in items:
        events = publication_events(item, view, master)
        validate_view(item['events'], events)
        predictions = bank.predictions(events, scenario, rn, history, observable_close_s=item['close_s'])
        truth = np.asarray(item['target_xy_evaluator_only'][scenario])
        rows.append({k: item[k] for k in ('family_id', 'session_id', 'seed', 'recall', 'close_s')}
            | dict(method=item['method']+'/'+view, defense=item['method'], view=view, scenario=scenario, status='ok',
                   errors={name: float(np.linalg.norm(value[0]-truth)) for name, value in predictions.items()}))
    return rows


def select(out, source, cache):
    p = protocol(out, source)
    if (out/'selection.json').exists():
        raise FileExistsError('Ordered challenge attackers already frozen')
    master = common.read(out/'evaluator_shuffle_randomness.json.gz')['master_hex']
    rn, history, metadata = resources(cache, p)
    banks, selected, scores, summaries = {}, {}, [], {}
    for method in METHODS:
        fit, development = (common.read(source/f'{split}-{method}.json.gz') for split in ('fit', 'selection'))
        assert {r['family_id'] for r in fit} == set(p['fit_families'])
        assert {r['family_id'] for r in development} == set(p['selection_families'])
        for view in VIEWS:
            name = method+'/'+view
            selected[name] = {group: {} for group in GROUPS}
            view_rows = []
            for scenario in p['scenarios']:
                features, targets = defaultdict(list), []
                for item in fit:
                    events = publication_events(item, view, master)
                    validate_view(item['events'], events)
                    x, _ = ordered_endpoint_features(events, scenario, rn, history, observable_close_s=item['close_s'])
                    for channel, value in x.items():
                        features[channel].append(value[0])
                    targets.append(item['target_xy_evaluator_only'][scenario])
                bank = OrderedEndpointBank(features, targets)
                banks[name, scenario] = bank
                rows = attack_rows(development, view, master, scenario, bank, rn, history)
                for group in GROUPS:
                    selected[name][group][scenario] = select_attackers(bank_group(rows, group))
                view_rows.extend(rows)
            scores.extend(view_rows)
            summaries[name] = {group: common.score_summary(view_rows, selected[name][group]) for group in GROUPS}
            print('Selected development-only decoders', name, 'S9/S10 full MAE',
                  summaries[name]['full']['S9']['mae_m'], summaries[name]['full']['S10']['mae_m'], flush=True)
    with gzip.open(out/'attackers.pkl.gz', 'wb') as file:
        pickle.dump(dict(banks=banks, history=history), file)
    common.write(out/'selection_rows.json.gz', scores)
    common.write(out/'development_summary.json', summaries)
    common.write(out/'resources.json', metadata)
    common.write(out/'selection.json', dict(protocol_sha256=sha(out/'protocol.json'),
        selected_attackers=selected, attackers_sha256=sha(out/'attackers.pkl.gz'),
        development_summary_sha256=sha(out/'development_summary.json'),
        no_test_labels_opened_by_select=True, defense_changed=False, status=p['status']))
    print('Ordered challenge decoders frozen; existing test remains a development diagnostic', flush=True)


def score(out, source, cache):
    p = protocol(out, source)
    if (out/'diagnostic.json').exists():
        raise FileExistsError('Retain first complete diagnostic, never overwrite scores')
    selection = common.read(out/'selection.json')
    assert selection['protocol_sha256'] == sha(out/'protocol.json')
    assert selection['attackers_sha256'] == sha(out/'attackers.pkl.gz')
    master = common.read(out/'evaluator_shuffle_randomness.json.gz')['master_hex']
    rn, _, _ = resources(cache, p)
    with gzip.open(out/'attackers.pkl.gz', 'rb') as file:
        saved = pickle.load(file)  # trusted local model, SHA checked above
    rows, results, unchanged = [], {}, {}
    for method in METHODS:
        items = common.read(source/f'test-{method}.json.gz')
        assert {r['family_id'] for r in items} == set(p['inspected_test_families']) and len(items) == 112
        unchanged[method] = dict(recall=family_mean(items, 'recall'), bytes_per_input=family_mean(items, 'bytes_per_input'),
            events=sum(len(r['events']) for r in items), coordinate_multisets_and_public_clock_unchanged=True)
        for view in VIEWS:
            name = method+'/'+view
            current = [row for scenario in p['scenarios'] for row in attack_rows(items, view, master,
                scenario, saved['banks'][name, scenario], rn, saved['history'])]
            results[name] = {group: common.score_summary(current, selection['selected_attackers'][name][group]) for group in GROUPS}
            rows.extend(current)
            print('Already-inspected diagnostic', name, 'S9/S10 full MAE',
                  results[name]['full']['S9']['mae_m'], results[name]['full']['S10']['mae_m'], flush=True)
    uncertainty = {}
    for group in GROUPS:
        selector = {'selected_attackers': {name: chosen[group] for name, chosen in selection['selected_attackers'].items()}}
        for scenario in p['scenarios']:
            for key in ('mae_m', 'hit100', 'hit500'):
                for method in METHODS:
                    uncertainty[f'{group}/{scenario}/{method}/shuffled-ordered/{key}'] = paired_interval(
                        rows, selector, method+'/shuffled', method+'/ordered', scenario, key)
                for view in VIEWS:
                    uncertainty[f'{group}/{scenario}/{view}/Endpoint20-plainL20/{key}'] = paired_interval(
                        rows, selector, 'scale025_L20/'+view, 'scale100_L20/'+view, scenario, key)
    common.write(out/'diagnostic_rows.json.gz', rows)
    common.write(out/'diagnostic.json', dict(schema='endpoint-order-diagnostic-v1',
        protocol_sha256=sha(out/'protocol.json'), selection_sha256=sha(out/'selection.json'),
        status=p['status'], results=results, unchanged_service=unchanged,
        paired_family_uncertainty=uncertainty, complete_all_28_inspected_families=True,
        defense_changed=False, primary_earlier_coordinate_set_claim_unchanged=True, limits=p['limits']))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage', choices=('prepare', 'select', 'score', 'all'))
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--source', type=Path, default=SOURCE)
    parser.add_argument('--cache', type=Path, default=Path('/private/tmp/trajectory-research-20261005-public-map'))
    args = parser.parse_args()
    if args.stage in ('prepare', 'all'):
        prepare(args.out, args.source)
    if args.stage in ('select', 'all'):
        select(args.out, args.source, args.cache)
    if args.stage in ('score', 'all'):
        score(args.out, args.source, args.cache)
