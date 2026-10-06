from collections import Counter, defaultdict

from experiments.verify_endpoint_robust_selection_20261006_review import independent_rule, positive_counts


def test_history_review_ignores_zero_cache_keys_but_detects_changed_training_counts():
    clean = defaultdict(Counter, {1: Counter({2: 3})})
    cached = defaultdict(Counter, {1: Counter({2: 3, 4: 0}), 5: Counter({6: 0})})
    corrupt = defaultdict(Counter, {1: Counter({2: 4})})
    assert clean != cached
    assert positive_counts(clean) == positive_counts(cached)
    assert positive_counts(clean) != positive_counts(corrupt)


def test_independent_arithmetic_uses_the_declared_variance_penalty():
    rows = [dict(family_id='a', errors={'unstable': 0., 'stable': 100.}),
            dict(family_id='b', errors={'unstable': 180., 'stable': 100.})]
    selected, stats = independent_rule(rows, ['a', 'b'])
    assert selected['mae'] == 'stable'
    assert stats['unstable']['mae'] == {'mean': 90., 'se': 90., 'objective': 180.}
