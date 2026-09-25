from itertools import product
import numpy as np
import pytest

from evaluation.category_cover import greedy_category_cover, reply_mask


def test_budget_counts_category_requests_and_redistributes_unused_slots():
    profiles = [[{'state': 0, 'ids': [0]}],
                [{'state': 1, 'ids': [1, 2]}, {'state': 2, 'ids': [3, 4]}]]
    result = greedy_category_cover(profiles, [[0], [1, 2, 3, 4]], budget=3)
    assert result['full_catalogue_cover'] and result['category_queries'] == 3
    assert [q['category_index'] for q in result['queries']] == [0, 1, 1]
    assert result['private_reads'] == 0
    limited = greedy_category_cover(profiles, [[0], [1, 2, 3, 4]], budget=2)
    assert limited['uncovered_ids'] == [[], [3, 4]]


def test_impossible_cover_is_reported_not_silently_certified():
    r = greedy_category_cover([[{'state': 0, 'ids': [0]}]], [[0, 1]])
    assert r['uncovered_ids'] == [[1]] and not r['full_catalogue_cover']
    for budget in [0, -1, 1.5, True]:
        with pytest.raises(ValueError):
            greedy_category_cover([[{'state': 0, 'ids': [0]}]], [[0]], budget)


def test_static_cover_certificate_survives_every_availability_mask():
    # If a POI is in static top-L, removing unavailable predecessors cannot
    # push it out of live top-L while that POI itself remains available.
    orders = [[0, 1, 2, 3], [2, 3, 0, 1]]
    plan = greedy_category_cover([[{'state': i, 'ids': row[:2]}
                                   for i, row in enumerate(orders)]], [list(range(4))])
    assert plan['full_catalogue_cover']
    for bits in product([False, True], repeat=4):
        replies = [[i for i in orders[q['state']] if bits[i]][:2] for q in plan['queries']]
        known = reply_mask(replies, 4)
        assert np.array_equal(known, bits)
