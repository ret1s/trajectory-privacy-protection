"""Independent rule arithmetic, fold exclusion and actual-model verification."""
import argparse
import gzip
import math
from pathlib import Path
import pickle
import statistics

import numpy as np

from evaluation.ordered_endpoint_attacks import ordered_endpoint_features
from experiments import endpoint_noise_loop as common
from experiments import endpoint_robust_selection_20261006 as run
from experiments.public_research_resources import sha


def independent_rule(rows, families):
    """Use Python statistics, not the NumPy selector implementation."""
    names = sorted(rows[0]['errors'])
    metrics = {}
    for name in names:
        family_errors = [[r['errors'][name] for r in rows if r['family_id']==family] for family in families]
        values = {'mae': [statistics.mean(errors) for errors in family_errors]}
        values.update({'hit'+str(radius): [statistics.mean(float(error <= radius) for error in errors)
                                           for errors in family_errors] for radius in (50, 100, 200, 500)})
        metrics[name] = {key: dict(mean=statistics.mean(v), se=statistics.stdev(v)/math.sqrt(len(v)))
                         for key, v in values.items()}
        for key, item in metrics[name].items():
            item['objective'] = item['mean']+(item['se'] if key=='mae' else -item['se'])
    selected = {'mae': min(names, key=lambda n: (round(metrics[n]['mae']['objective'], 12), n))}
    for key in ('hit50', 'hit100', 'hit200', 'hit500'):
        selected[key] = min(names, key=lambda n: (-round(metrics[n][key]['objective'], 12), n))
    return selected, metrics


def verify_training(bank, items, scenario, cache):
    y = np.asarray([item['target_xy_evaluator_only'][scenario] for item in items])
    for channel, learner in bank.learners.items():
        x = np.asarray([cache[item['session_id'], item['seed']][channel][0] for item in items])
        mean, scale = x.mean(axis=0), x.std(axis=0)
        scale[scale < 1e-9] = 1.
        assert np.array_equal(learner.y, y)
        assert np.allclose(learner.mean, mean, rtol=0., atol=1e-12)
        assert np.allclose(learner.scale, scale, rtol=0., atol=1e-12)
        assert np.allclose(learner.x, (x-mean)/scale, rtol=0., atol=1e-12)


def validate(out, source, order_source, cache):
    p = run.protocol(out, source, order_source)
    selection = run.frozen_selection(out)
    diagnostic = common.read(out/'diagnostic.json')
    assert diagnostic['status'].startswith('POST-INSPECTION')
    assert diagnostic['selection_sha256'] == sha(out/'selection.json')
    assert diagnostic['all_28_inspected_families_retained'] and not diagnostic['defense_changed']
    with gzip.open(out/'models.pkl.gz', 'rb') as file:
        saved = pickle.load(file)  # trusted local model, compressed SHA checked by frozen_selection
    rn = saved['full_history'].rn
    _, make_history, _ = run.resources(cache, p)
    master = common.read(order_source/'evaluator_shuffle_randomness.json.gz')['master_hex']
    fold_recorded = common.read(out/'fold_predictions.json.gz')
    test_recorded = common.read(out/'diagnostic_rows.json.gz')
    stats = common.read(out/'candidate_statistics.json.gz')
    counters = dict(predictions=0, arithmetic=0, training_models=0, events=0, metrics=0)
    for method in p['methods']:
        fit, dev, test = [run.prepared(common.read(source/f'{split}-{method}.json.gz'), master)
                          for split in ('fit', 'selection', 'test')]
        assert len(fit)==32 and len(dev)==8 and len(test)==112
        assert {item['family_id'] for item in test} == set(p['inspected_test_families'])
        for original, shuffled in zip(common.read(source/f'test-{method}.json.gz'), test):
            assert original['target_xy_evaluator_only'] == shuffled['target_xy_evaluator_only']
            assert sorted(tuple(c) for c in original['events'][0]['coordinates']) == sorted(tuple(c) for c in shuffled['events'][0]['coordinates'])
            counters['events'] += len(original['events'])
        for scenario in p['scenarios']:
            feature_cache = run.features_cache(fit+dev, scenario, rn)
            actual_validation = []
            for fold in p['folds']:
                family, training = fold['validation_family'], fold['training_families']
                assert set(training) == set(p['fit_families'])-{family}
                inputs = [item for item in fit if item['family_id'] in training]
                held = [item for item in fit if item['family_id']==family]
                assert len(inputs)==28 and len(held)==4
                history = saved['fold_histories'][family]
                expected_history = make_history(training)
                assert np.array_equal(history.q, expected_history.q) and history.counts == expected_history.counts
                assert history.training_sequences == 14
                bank = saved['fold_banks'][method, scenario, family]
                verify_training(bank, inputs, scenario, feature_cache)
                counters['training_models'] += len(bank.learners)
                for item in held:
                    # Certify that the cached learner features contain no
                    # held-out support through the history argument.
                    direct, _ = ordered_endpoint_features(item['events'], scenario, rn, history,
                                                         observable_close_s=item['close_s'])
                    assert all(np.array_equal(direct[channel], feature_cache[item['session_id'], item['seed']][channel]) for channel in direct)
                actual_validation.extend(run.rows_for(held, scenario, bank, rn, history,
                    fold=family, training_families=training))
            final_bank = saved['final_banks'][method, scenario]
            verify_training(final_bank, fit, scenario, feature_cache)
            counters['training_models'] += len(final_bank.learners)
            expected_history = make_history(p['fit_families'])
            assert np.array_equal(saved['full_history'].q, expected_history.q)
            actual_validation.extend(run.rows_for(dev, scenario, final_bank, rn, saved['full_history'],
                fold='original_selection', training_families=p['fit_families']))
            chosen, calculated = independent_rule(actual_validation, p['validation_families'])
            assert chosen == selection['selected_attackers'][method]['robust_crossfit'][scenario]
            for name, values in calculated.items():
                for metric, numbers in values.items():
                    for key, value in numbers.items():
                        assert np.isclose(value, stats[method+'/'+scenario][name][metric][key], rtol=0., atol=1e-9)
                        counters['arithmetic'] += 1
            actual_test = run.rows_for(test, scenario, final_bank, rn, saved['full_history'],
                fold='inspected_test', training_families=p['fit_families'])
            for actual, recorded in ((actual_validation, fold_recorded), (actual_test, test_recorded)):
                expected = {(row['session_id'], row['seed']): row for row in recorded if row['method']==method and row['scenario']==scenario}
                assert len(expected) == len(actual)
                for row in actual:
                    target = expected[row['session_id'], row['seed']]
                    assert row['fold'] == target['fold'] and row['training_families'] == target['training_families']
                    assert set(row['errors']) == set(target['errors'])
                    for name, value in row['errors'].items():
                        assert np.isclose(value, target['errors'][name], rtol=0., atol=1e-9)
                        counters['predictions'] += 1
            for selector in run.SELECTORS:
                chosen = selection['selected_attackers'][method][selector][scenario]
                for metric in ('mae_m', 'hit50', 'hit100', 'hit200', 'hit500'):
                    decoder = chosen['mae' if metric=='mae_m' else metric]
                    family_values = [statistics.mean(row['errors'][decoder] if metric=='mae_m'
                        else float(row['errors'][decoder] <= int(metric[3:])) for row in actual_test if row['family_id']==family)
                        for family in p['inspected_test_families']]
                    assert np.isclose(statistics.mean(family_values), diagnostic['rows'][method][selector][scenario][metric], rtol=0., atol=1e-9)
                    counters['metrics'] += 1
    result = dict(schema='endpoint-robust-verification-v1', status=p['status'],
        protocol_sources_inputs_models_sha256_verified=True, all_eight_folds_exclude_validation_from_learners_and_history=True,
        actual_predictions_recomputed=counters['predictions'], independent_rule_numbers_verified=counters['arithmetic'],
        fitted_learner_training_arrays_verified=counters['training_models'], event_multisets_and_clock_verified=counters['events'],
        reported_metrics_verified=counters['metrics'], all_28_inspected_groups_retained=True,
        defense_changed=False, no_test_based_fallback=True)
    common.write(out/'verification.json', result)
    print(result, flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('out', type=Path)
    parser.add_argument('--source', type=Path, default=run.SOURCE)
    parser.add_argument('--order-source', type=Path, default=run.ORDER_SOURCE)
    parser.add_argument('--cache', type=Path, default=Path('/private/tmp/trajectory-research-20261005-public-map'))
    args = parser.parse_args()
    validate(args.out, args.source, args.order_source, args.cache)
