"""Cross-fit selector refinement on frozen Geo-I endpoint streams.

The private-shuffled coordinate view and defense stay frozen. One family-risk
rule is sealed before this runner reads any previously inspected test score.
The 28 old test families remain development diagnostics, not a fresh holdout.
"""
import argparse
from collections import defaultdict
import gzip
from pathlib import Path
import pickle

import numpy as np

from benchmark.paper_comparators import PublicHistory
from evaluation.endpoint_noise_attacks import select_attackers
from evaluation.ordered_endpoint_attacks import OrderedEndpointBank, ordered_endpoint_features
from evaluation.robust_endpoint_selection import robust_select
from experiments import endpoint_noise_loop as common
from experiments import endpoint_order_challenge as ordered
from experiments.public_research_resources import ROOT, load_public_research_resources, sha

SOURCE = ordered.SOURCE
ORDER_SOURCE = ROOT/'artifacts/benchmarks/endpoint_order_20261005'
SELECTORS = ('robust_crossfit', 'old_two_family_invariant', 'old_two_family_expanded')


def prepare(out, source, order_source):
    if (out/'protocol.json').exists():
        raise FileExistsError('Never overwrite a sealed robust endpoint selector')
    prior = common.read(source/'protocol.json')
    code = list(common.read(order_source/'protocol.json')['source_sha256'])
    code += ['evaluation/robust_endpoint_selection.py', 'experiments/endpoint_robust_selection_20261006.py',
             'experiments/verify_endpoint_robust_selection_20261006.py',
             'experiments/endpoint_robust_readout_20261006.py', 'tests/test_robust_endpoint_selection.py']
    inputs = [f'{split}-{method}.json.gz' for split in ('fit', 'selection', 'test') for method in ordered.METHODS]
    inputs += ['protocol.json', 'resources.json', 'heldout.json']
    out.mkdir(parents=True, exist_ok=True)
    p = dict(schema='endpoint-robust-selector-20261006-v1', date='2026-10-06',
        status='POST-INSPECTION DEVELOPMENT; all 28 diagnostic test families were previously inspected',
        source_reference=str(source.relative_to(ROOT)), order_reference=str(order_source.relative_to(ROOT)),
        input_sha256={file: sha(source/file) for file in inputs},
        order_input_sha256={file: sha(order_source/file) for file in
            ('protocol.json', 'evaluator_shuffle_randomness.json.gz', 'diagnostic.json')},
        source_sha256={file: sha(ROOT/file) for file in sorted(set(code))}, dataset_sha256=sha(common.DATA),
        methods=ordered.METHODS, view='shuffled', scenarios=['S9', 'S10'], selectors=list(SELECTORS),
        fit_families=prior['fit_families'], selection_families=prior['selection_families'],
        validation_families=prior['fit_families']+prior['selection_families'],
        inspected_test_families=prior['test_families'],
        folds=[dict(validation_family=f, training_families=[other for other in prior['fit_families'] if other != f])
               for f in prior['fit_families']],
        rule='One fixed rule: minimize mean family MAE + 1 sample SE; maximize mean family Hit - 1 sample SE; '
            'SE = sample SD(ddof=1)/sqrt(10); compare objectives rounded to 12 decimal places, '
            'then lexical candidate name only; this numerical tie policy is fixed before scores',
        validation='Leave-one-family-out 701–708: learners AND mobility history trained on the other seven families; '
            '709–710 uses all eight fit families; all ten validation families receive equal weight, not all repetitions',
        final_fit='After decoder selection, use the bank fit on all 701–708 only; no 709–710 labels fit final models',
        features='Frozen invariant + observed-slot + Hungarian nearest/velocity union; ExtraTrees64/kNN1/kNN5 '
            'unchanged. Learner feature cache uses uniform no-GPS history; fold geometry uses fold-only trained history.',
        controls='Original invariant and expanded two-family selectors recomputed on exactly 709–710; '
            'controls do not become fallback candidates selected from test',
        defense='No regeneration, tuning or replacement of Q/GPS/time/POI replies/bytes/seed/session/family',
        evaluation='Freeze all decoders/models before opening test streams or historical test scores in this runner; '
            'score all 28 existing families × two trips × two reps; no optional stopping or test-based fallback',
        limits='Overlapping cross-fit training sets make SE a heuristic selection penalty, not an independent '
            'confidence interval. Validation training sizes differ (7 vs 8 families). Same synthetic SUMO and '
            'reconstructed public map; finite attacker bank; no fresh confirmation or universal superiority claim.')
    common.write(out/'protocol.json', p)
    (out/'protocol.sha256').write_text(sha(out/'protocol.json')+'\n')
    print('Sealed one robust 10-family rule before this runner reads inspected test scores', flush=True)


def protocol(out, source, order_source):
    p = common.read(out/'protocol.json')
    assert sha(out/'protocol.json') == (out/'protocol.sha256').read_text().strip()
    for file, digest in p['input_sha256'].items():
        assert sha(source/file) == digest, file
    for file, digest in p['order_input_sha256'].items():
        assert sha(order_source/file) == digest, file
    for file, digest in p['source_sha256'].items():
        assert sha(ROOT/file) == digest, file
    assert sha(common.DATA) == p['dataset_sha256']
    assert p['view'] == 'shuffled' and p['selectors'] == list(SELECTORS)
    return p


def resources(cache, p):
    rn, _, _, _, _, _, metadata = load_public_research_resources(cache)
    data = common.read(common.DATA)
    sessions = list(common.sessions(data, p['fit_families']))
    tracks = {f: [[rn.nearest(point['lat'], point['lon'])[0] for point in item['points']]
                  for item in sessions if item['family_id']==f] for f in p['fit_families']}
    make_history = lambda families: PublicHistory(rn, [track for f in families for track in tracks[f]])
    return rn, make_history, metadata


def features_cache(items, scenario, rn):
    # All learned features in this pinned bank depend on public coordinates
    # and clock, not the mobility history. Avoid even an implicit held-out
    # support dependency by caching them with an empty uniform history.
    uniform = PublicHistory(rn, [])
    return {(item['session_id'], item['seed']): ordered_endpoint_features(
        item['events'], scenario, rn, uniform, observable_close_s=item['close_s'])[0] for item in items}


def fit_cached(items, scenario, cache):
    features = defaultdict(list)
    targets = []
    for item in items:
        for channel, value in cache[item['session_id'], item['seed']].items():
            features[channel].append(value[0])
        targets.append(item['target_xy_evaluator_only'][scenario])
    return OrderedEndpointBank(features, targets)


def prepared(items, master):
    result = []
    for item in items:
        events = ordered.publication_events(item, 'shuffled', master)
        ordered.validate_view(item['events'], events)
        result.append(dict(item, events=events))
    return result


def rows_for(items, scenario, bank, rn, history, *, fold, training_families):
    rows = common.bank_rows(items, scenario, bank, rn, history)
    for row in rows:
        row.update(fold=fold, training_families=list(training_families))
    return rows


def select(out, source, order_source, cache):
    p = protocol(out, source, order_source)
    if (out/'selection.json').exists():
        raise FileExistsError('Robust endpoint decoders already frozen')
    rn, make_history, metadata = resources(cache, p)
    master = common.read(order_source/'evaluator_shuffle_randomness.json.gz')['master_hex']
    full_history = make_history(p['fit_families'])
    fold_histories = {fold['validation_family']: make_history(fold['training_families']) for fold in p['folds']}
    final_banks, fold_banks, chosen, statistics, all_rows, summaries = {}, {}, {}, {}, [], {}
    for method in ordered.METHODS:
        fit, development = [prepared(common.read(source/f'{split}-{method}.json.gz'), master) for split in ('fit', 'selection')]
        assert len(fit) == 32 and len(development) == 8
        assert {r['family_id'] for r in fit} == set(p['fit_families'])
        assert {r['family_id'] for r in development} == set(p['selection_families'])
        chosen[method] = {selector: {} for selector in SELECTORS}
        method_rows, old_rows = [], []
        for scenario in p['scenarios']:
            cached = features_cache(fit+development, scenario, rn)
            final_bank = fit_cached(fit, scenario, cached)
            final_banks[method, scenario] = final_bank
            cv_rows = []
            for fold in p['folds']:
                family, training = fold['validation_family'], fold['training_families']
                inputs = [item for item in fit if item['family_id'] in training]
                held = [item for item in fit if item['family_id']==family]
                assert len(inputs) == 28 and len(held) == 4 and family not in training
                bank = fit_cached(inputs, scenario, cached)
                fold_banks[method, scenario, family] = bank
                cv_rows.extend(rows_for(held, scenario, bank, rn, fold_histories[family],
                                        fold=family, training_families=training))
            dev_rows = rows_for(development, scenario, final_bank, rn, full_history,
                                fold='original_selection', training_families=p['fit_families'])
            validation = cv_rows+dev_rows
            selected, stats = robust_select(validation, p['validation_families'])
            chosen[method]['robust_crossfit'][scenario] = selected
            statistics[method+'/'+scenario] = stats
            chosen[method]['old_two_family_expanded'][scenario] = select_attackers(dev_rows)
            chosen[method]['old_two_family_invariant'][scenario] = select_attackers(ordered.bank_group(dev_rows, 'invariant'))
            method_rows.extend(validation)
            old_rows.extend(dev_rows)
        summaries[method] = dict(robust_validation=common.score_summary(method_rows, chosen[method]['robust_crossfit']),
            old_invariant_selection=common.score_summary(old_rows, chosen[method]['old_two_family_invariant']),
            old_expanded_selection=common.score_summary(old_rows, chosen[method]['old_two_family_expanded']))
        all_rows.extend(method_rows)
        print('Robust rule selected', method, chosen[method]['robust_crossfit'], flush=True)
    with gzip.open(out/'models.pkl.gz', 'wb') as file:
        pickle.dump(dict(final_banks=final_banks, fold_banks=fold_banks,
                         full_history=full_history, fold_histories=fold_histories), file)
    common.write(out/'fold_predictions.json.gz', all_rows)
    common.write(out/'candidate_statistics.json.gz', statistics)
    common.write(out/'validation_summary.json', summaries)
    common.write(out/'resources.json', metadata)
    common.write(out/'fold_manifest.json', dict(folds=p['folds'],
        geometry_history_excludes_validation=True, learner_cache_history='uniform no GPS history',
        fold_model_training_rows=28, final_model_training_rows=32, final_fit_families=p['fit_families']))
    common.write(out/'selection.json', dict(protocol_sha256=sha(out/'protocol.json'),
        selected_attackers=chosen, models_sha256=sha(out/'models.pkl.gz'),
        fold_predictions_sha256=sha(out/'fold_predictions.json.gz'),
        candidate_statistics_sha256=sha(out/'candidate_statistics.json.gz'),
        defense_changed=False, no_test_labels_or_historical_test_scores_opened_by_select=True,
        status=p['status']))
    print('Robust decoders and both original controls frozen', flush=True)


def frozen_selection(out):
    if not (out/'selection.json').exists():
        raise RuntimeError('Freeze selectors before opening inspected test streams')
    selection = common.read(out/'selection.json')
    assert selection['protocol_sha256'] == sha(out/'protocol.json')
    for file, key in (('models.pkl.gz', 'models_sha256'), ('fold_predictions.json.gz', 'fold_predictions_sha256'),
                      ('candidate_statistics.json.gz', 'candidate_statistics_sha256')):
        assert sha(out/file) == selection[key], file
    return selection


def score(out, source, order_source, cache):
    p = protocol(out, source, order_source)
    if (out/'diagnostic.json').exists():
        raise FileExistsError('Retain first complete diagnostic')
    selection = frozen_selection(out)
    rn, _, _ = resources(cache, p)
    master = common.read(order_source/'evaluator_shuffle_randomness.json.gz')['master_hex']
    with gzip.open(out/'models.pkl.gz', 'rb') as file:
        saved = pickle.load(file)  # trusted locally generated and SHA-checked model
    results, all_rows, service = {}, [], {}
    for method in ordered.METHODS:
        original = common.read(source/f'test-{method}.json.gz')
        assert len(original) == 112 and {r['family_id'] for r in original} == set(p['inspected_test_families'])
        items = prepared(original, master)
        rows = [row for scenario in p['scenarios'] for row in rows_for(items, scenario,
            saved['final_banks'][method, scenario], rn, saved['full_history'],
            fold='inspected_test', training_families=p['fit_families'])]
        results[method] = {selector: common.score_summary(rows, selection['selected_attackers'][method][selector])
                           for selector in SELECTORS}
        service[method] = dict(recall=common.family_mean(original, 'recall'),
            bytes_per_input=common.family_mean(original, 'bytes_per_input'), events=sum(len(i['events']) for i in original),
            Q_GPS_clock_replies_unchanged=True)
        all_rows.extend(rows)
        print('Inspected development score', method, 'robust S9/S10 MAE',
              results[method]['robust_crossfit']['S9']['mae_m'], results[method]['robust_crossfit']['S10']['mae_m'], flush=True)
    common.write(out/'diagnostic_rows.json.gz', all_rows)
    common.write(out/'diagnostic.json', dict(schema='endpoint-robust-selector-diagnostic-v1', status=p['status'],
        protocol_sha256=sha(out/'protocol.json'), selection_sha256=sha(out/'selection.json'),
        rows=results, unchanged_service=service, all_28_inspected_families_retained=True,
        defense_changed=False, no_test_based_attacker_fallback=True, limits=p['limits']))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage', choices=('prepare', 'select', 'score', 'all'))
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--source', type=Path, default=SOURCE)
    parser.add_argument('--order-source', type=Path, default=ORDER_SOURCE)
    parser.add_argument('--cache', type=Path, default=Path('/private/tmp/trajectory-research-20261005-public-map'))
    args = parser.parse_args()
    if args.stage in ('prepare', 'all'):
        prepare(args.out, args.source, args.order_source)
    if args.stage in ('select', 'all'):
        select(args.out, args.source, args.order_source, args.cache)
    if args.stage in ('score', 'all'):
        score(args.out, args.source, args.order_source, args.cache)
