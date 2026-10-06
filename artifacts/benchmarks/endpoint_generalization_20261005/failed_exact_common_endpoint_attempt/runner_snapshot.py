"""Locked GeoI-Endpoint20 generalization with independent session RNG streams.

Only attacker fitting/decoder selection uses existing train/selection families.
All 28 unused within-study families are predeclared before labels are opened.
The defense stays scale=.25, L=20; no post-test tuning or optional stopping.
Linked-two-trip inference is a separately reported secondary stress.
"""
import argparse
from collections import defaultdict
import gzip
import hashlib
import hmac
import json
import os
from pathlib import Path
import pickle
import secrets

import numpy as np

from benchmark.paper_comparators import PublicHistory
from evaluation.endpoint_noise_attacks import EndpointShadowBank, endpoint_features, family_mean, select_attackers
from experiments import endpoint_noise_loop as common
from experiments import endpoint_noise_depth_loop as depth
from experiments.public_research_resources import ROOT, sha

CONFIGS = [dict(id='raw', L=10),
    dict(id='scale100_L10', scale=1., L=10, phase=False, delay=False),
    dict(id='scale100_L20', scale=1., L=20, phase=False, delay=False),
    dict(id='scale025_L20', scale=.25, L=20, phase=False, delay=False)]
LOCKED = 'scale025_L20'
RNG_SCHEMA = 'endpoint-generalization-private-session-v1'
FAMILY_KEYS = ('family_id', 'seed')


def session_seed(master_hex, session_id, rep):
    """Independent 128-bit substream per session/rep; method intentionally absent.

    The evaluator keeps this key and resolved seed outside public events and
    attacker features. Pairing across methods is only for paired comparisons.
    A production client needs its own private RNG, not this experiment key.
    """
    master = bytes.fromhex(master_hex)
    if len(master) != 32 or not isinstance(session_id, str) or not session_id or int(rep) != rep or rep < 0:
        raise ValueError('32-byte master, nonempty session ID and nonnegative integer rep required')
    token = f'{RNG_SCHEMA}\0{session_id}\0{int(rep)}'.encode()
    return int.from_bytes(hmac.new(master, token, hashlib.sha256).digest()[:16], 'little')


def prepare(out):
    if (out/'protocol.json').exists():
        raise FileExistsError('Never replace a sealed endpoint generalization protocol')
    previous = common.read(ROOT/'artifacts/benchmarks/endpoint_noise_20261005/round2/protocol.json')
    sources = {file: sha(ROOT/file) for file in previous['source_sha256']}
    sources[str(Path(__file__).relative_to(ROOT))] = sha(Path(__file__))
    sources['experiments/rng_util.py'] = sha(ROOT/'experiments/rng_util.py')
    out.mkdir(parents=True, exist_ok=True)
    # Evaluator-only seed material is separate from the public transcript.
    common.write(out/'evaluator_randomness.json.gz', dict(schema=RNG_SCHEMA,
        master_hex=secrets.token_bytes(32).hex(), role='evaluator-only, never passed to attacker'))
    os.chmod(out/'evaluator_randomness.json.gz', 0o600)
    p = dict(previous, schema='endpoint-locked-generalization-v3', date='2026-10-05',
        source_sha256=sources, configurations=CONFIGS, fixed_defense=LOCKED,
        previously_examined_test_families=[f'family-{i}' for i in range(1201, 1205)],
        test_families=[f'family-{i}' for i in range(1205, 1233)],
        mechanism_reps=[0, 1], selection_rule='Defense remains locked; select only attackers on 709–710',
        private_randomness_sha256=sha(out/'evaluator_randomness.json.gz'),
        randomization='HMAC-SHA256 private evaluator master + session ID + rep; '
            'distinct sessions get distinct substreams; matched methods share the same session/rep key',
        primary='S10 paired MAE and Hit100/Hit500 against scale100_L20; S9 secondary; '
            'utility average, lower-tail trips and request/response bytes',
        linked_secondary='Attacker may link the two predeclared repeated-base trips; '
            'pool public features and average per-trip estimates; train/select this bank separately. '
            'Endpoints must agree by construction, without score-based record filtering.',
        evaluation_rule='All 28 planned groups, first two sessions, both reps, all four methods; '
            'no optional stopping, replacement families, defense tuning or post-test attacker selection',
        guarantee_scope='Per-session ideal-kernel privacy bound; linked sessions compose; '
            'published synthetic experiment RNG material is evaluator-only, not production secrets',
        scope='Locked-config generalization on 28 previously unused within-study endpoint families, '
            'existing SUMO data and reconstructed map; not new independent real-world confirmation')
    common.write(out/'protocol.json', p)
    (out/'protocol.sha256').write_text(sha(out/'protocol.json')+'\n')
    print('Sealed all 28 families, locked defense and session RNG scheme', flush=True)


def protocol(out):
    p = common.read(out/'protocol.json')
    assert sha(out/'protocol.json') == (out/'protocol.sha256').read_text().strip()
    assert p['fixed_defense'] == LOCKED and p['configurations'] == CONFIGS
    assert p['test_families'] == [f'family-{i}' for i in range(1205, 1233)]
    assert sha(common.DATA) == p['dataset_sha256'] and sha(depth.FRESH) == p['fresh_dataset_sha256']
    assert sha(out/'evaluator_randomness.json.gz') == p['private_randomness_sha256']
    for file, digest in p['source_sha256'].items():
        assert sha(ROOT/file) == digest, file
    return p


def private_master(out):
    return common.read(out/'evaluator_randomness.json.gz')['master_hex']


def generate(session, config, rep, master, rn, beliefs, ranking, world):
    resolved = session_seed(master, session['session_id'], rep)
    row = depth.generate(session, config, resolved, rn, beliefs, ranking, world)
    # Rep labels allow pairing but are not a feature; actual randomization is
    # stored separately and never included in the events sent to the attacker.
    row['rng_seed_evaluator_only'] = row.pop('seed')
    row['seed'] = int(rep)
    row['rep'] = int(rep)
    return row


def linked_groups(executions, scenario, rn, history):
    groups = defaultdict(list)
    for item in executions:
        groups[item['family_id'], item['seed']].append(item)
    result = []
    for (family, rep), items in sorted(groups.items()):
        assert len(items) == 2
        features, sequences, targets = [], [], []
        for item in items:
            aggregate, sequence, _ = endpoint_features(item['events'], scenario, rn, history,
                                                       observable_close_s=item['close_s'])
            features.append(aggregate[0]); sequences.append(sequence[0])
            targets.append(np.asarray(item['target_xy_evaluator_only'][scenario]))
        # This checks the predeclared repeated-route pair, not privacy outcomes.
        if not np.allclose(targets[0], targets[1], atol=.2, rtol=0):
            raise ValueError('Predeclared linked pair has different endpoints; retain failure, do not replace')
        f, s = np.asarray(features), np.asarray(sequences)
        result.append(dict(family_id=family, seed=rep, items=items,
            aggregate=np.r_[f.mean(axis=0), f.std(axis=0)],
            sequence=np.r_[s.mean(axis=0), s.std(axis=0)], target=targets[0]))
    return result


def linked_rows(executions, scenario, individual_bank, linked_bank, rn, history):
    rows = []
    for group in linked_groups(executions, scenario, rn, history):
        banks = [individual_bank.predictions(item['events'], scenario, rn, history,
                    observable_close_s=item['close_s']) for item in group['items']]
        predictions = {'linked_mean_'+name: np.mean([bank[name][0] for bank in banks], axis=0)
                       for name in banks[0]}
        predictions.update({'linked_shadow_'+name: value[0]
                            for name, value in linked_bank.aggregate.predict(group['aggregate'][None, :]).items()})
        predictions.update({'linked_sequence_'+name: value[0]
                            for name, value in linked_bank.sequence.predict(group['sequence'][None, :]).items()})
        rows.append(dict(family_id=group['family_id'], session_id='linked_pair', seed=group['seed'],
            method=group['items'][0]['method'], scenario=scenario, status='ok',
            errors={name: float(np.linalg.norm(prediction-group['target']))
                    for name, prediction in predictions.items()}))
    return rows


def train_select(out, cache):
    p = protocol(out)
    if (out/'selection.json').exists():
        raise FileExistsError('Attackers already frozen')
    rn, beliefs, ranking, metadata = depth.resources(cache)
    data = common.read(common.DATA)
    fit, dev = (list(common.sessions(data, p[key])) for key in ('fit_families', 'selection_families'))
    history = PublicHistory(rn, [[rn.nearest(point['lat'], point['lon'])[0] for point in s['points']] for s in fit])
    world = depth.AvailabilityWorld(ranking.n, p['world_seed'], probability=.8, epoch_seconds=60)
    master = private_master(out)
    banks, linked, chosen, linked_chosen, summaries, all_rows, all_linked = {}, {}, {}, {}, {}, [], []
    for config in p['configurations']:
        method = config['id']
        by_split = {}
        for split, inputs in [('fit', fit), ('selection', dev)]:
            by_split[split] = []
            for session in inputs:
                for rep in p['mechanism_reps']:
                    try:
                        by_split[split].append(generate(session, config, rep, master, rn, beliefs, ranking, world))
                    except Exception as error:
                        common.write(out/'failure.json', dict(split=split, method=method,
                            session_id=session['session_id'], rep=rep, error=str(error), replacement=None))
                        raise
            common.write(out/f'{split}-{method}.json.gz', by_split[split])
        selected, selected_linked, rows, linked_scores = {}, {}, [], []
        for scenario in ('S9', 'S10'):
            x, sequence, targets = [], [], []
            for item in by_split['fit']:
                aggregate, seq, _ = endpoint_features(item['events'], scenario, rn, history,
                                                       observable_close_s=item['close_s'])
                x.append(aggregate[0]); sequence.append(seq[0]); targets.append(item['target_xy_evaluator_only'][scenario])
            bank = EndpointShadowBank(x, sequence, targets)
            banks[method, scenario] = bank
            scores = common.bank_rows(by_split['selection'], scenario, bank, rn, history)
            selected[scenario] = select_attackers(scores)
            rows.extend(scores)
            pairs = linked_groups(by_split['fit'], scenario, rn, history)
            linked_bank = EndpointShadowBank([g['aggregate'] for g in pairs],
                                            [g['sequence'] for g in pairs], [g['target'] for g in pairs])
            linked[method, scenario] = linked_bank
            pair_scores = linked_rows(by_split['selection'], scenario, bank, linked_bank, rn, history)
            selected_linked[scenario] = select_attackers(pair_scores)
            linked_scores.extend(pair_scores)
        chosen[method], linked_chosen[method] = selected, selected_linked
        summaries[method] = depth.summarize(by_split['selection'], rows, selected)
        summaries[method]['linked_secondary'] = common.score_summary(linked_scores, selected_linked)
        all_rows.extend(rows); all_linked.extend(linked_scores)
        print('Selection attacker calibration', method, 'Recall', summaries[method]['recall'],
            'S10 MAE', summaries[method]['S10']['mae_m'], flush=True)
    with (out/'attackers.pkl').open('wb') as file:
        pickle.dump(dict(banks=banks, linked=linked, history=history), file)
    common.write(out/'selection_rows.json.gz', all_rows)
    common.write(out/'selection_linked_rows.json.gz', all_linked)
    common.write(out/'development_summary.json', summaries)
    common.write(out/'resources.json', metadata)
    common.write(out/'selection.json', dict(protocol_sha256=sha(out/'protocol.json'),
        selected_defense=LOCKED, defense_not_reselected=True, selected_attackers=chosen,
        linked_attackers=linked_chosen, attackers_sha256=sha(out/'attackers.pkl'),
        selection_summary_sha256=sha(out/'development_summary.json'),
        test_labels_not_opened_by_train_select=True,
        development_utility_gate_met=summaries[LOCKED]['recall'] >= .90))
    print('Attacker selection sealed; defense stays', LOCKED, flush=True)


def fresh_dataset(out):
    selection_path = out/'selection.json'
    if not selection_path.exists():
        raise RuntimeError('Freeze attacker selection before opening unused test labels')
    selection = common.read(selection_path)
    assert selection['protocol_sha256'] == sha(out/'protocol.json')
    assert selection['defense_not_reselected'] and selection['selected_defense'] == LOCKED
    assert selection['attackers_sha256'] == sha(out/'attackers.pkl')
    return common.read(depth.FRESH)


def test(out, cache):
    p = protocol(out)
    if (out/'heldout.json').exists():
        raise FileExistsError('Retain first full 28-group result, never reselect')
    selection = common.read(out/'selection.json')
    data = fresh_dataset(out)
    rn, beliefs, ranking, metadata = depth.resources(cache)
    with (out/'attackers.pkl').open('rb') as file:
        saved = pickle.load(file)  # verified trusted local model above
    sessions = list(common.sessions(data, p['test_families']))
    assert {s['family_id'] for s in sessions} == set(p['test_families']) and len(sessions) == 56
    master = private_master(out)
    seeds = {session_seed(master, s['session_id'], rep) for s in sessions for rep in p['mechanism_reps']}
    assert len(seeds) == len(sessions)*len(p['mechanism_reps'])
    world = depth.AvailabilityWorld(ranking.n, p['world_seed'], probability=.8, epoch_seconds=60)
    summaries, all_rows, all_linked = {}, [], []
    for config in p['configurations']:
        method, executions = config['id'], []
        for session in sessions:
            for rep in p['mechanism_reps']:
                try:
                    executions.append(generate(session, config, rep, master, rn, beliefs, ranking, world))
                except Exception as error:
                    common.write(out/'failure.json', dict(split='test', method=method,
                        session_id=session['session_id'], rep=rep, error=str(error), replacement=None))
                    raise
        rows = [row for scenario in ('S9', 'S10') for row in common.bank_rows(executions,
            scenario, saved['banks'][method, scenario], rn, saved['history'])]
        linked_scores = [row for scenario in ('S9', 'S10') for row in linked_rows(executions,
            scenario, saved['banks'][method, scenario], saved['linked'][method, scenario], rn, saved['history'])]
        summary = depth.summarize(executions, rows, selection['selected_attackers'][method])
        summary['linked_secondary'] = common.score_summary(linked_scores, selection['linked_attackers'][method])
        values = np.asarray([e['recall'] for e in executions])
        summary['trip_run_recall'] = dict(min=float(values.min()), p10=float(np.quantile(values, .1)),
            median=float(np.median(values)), below90_fraction=float(np.mean(values < .9)))
        summary['max_budget_spent_per_m'] = None if method=='raw' else max(e['budget_spent_per_m'] for e in executions)
        summaries[method] = summary
        all_rows.extend(rows); all_linked.extend(linked_scores)
        common.write(out/f'test-{method}.json.gz', executions)
        print('Heldout all 28 groups', method, 'Recall', summary['recall'],
            'S9', summary['S9']['mae_m'], summary['S9']['hit100'],
            'S10', summary['S10']['mae_m'], summary['S10']['hit100'],
            'linked S10', summary['linked_secondary']['S10']['mae_m'], flush=True)
    common.write(out/'heldout_attack_rows.json.gz', all_rows)
    common.write(out/'heldout_linked_rows.json.gz', all_linked)
    common.write(out/'heldout.json', dict(schema='endpoint-generalization-heldout-v3',
        protocol_sha256=sha(out/'protocol.json'), selection_sha256=sha(out/'selection.json'),
        selected_defense=LOCKED, rows=summaries, scope=p['scope'],
        defense_not_reselected=True, complete_predeclared_test=True,
        unique_session_rep_streams=len(seeds), independent_session_streams=True,
        raw_positive_control={s: summaries['raw'][s]['hit100']==1. for s in ('S9', 'S10')}))
    protocol(out)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage', choices=['prepare', 'select', 'test', 'all'])
    parser.add_argument('--out', type=Path, default=Path('/private/tmp/endpoint-generalization-20261005-review'))
    parser.add_argument('--cache', type=Path, default=Path('/private/tmp/trajectory-research-20261005-public-map'))
    args = parser.parse_args()
    if args.stage in ('prepare', 'all'):
        prepare(args.out)
    if args.stage in ('select', 'all'):
        train_select(args.out, args.cache)
    if args.stage in ('test', 'all'):
        test(args.out, args.cache)


if __name__ == '__main__':
    main()
