import numpy as np
import pytest
from evaluation.live_poi import AvailabilityWorld, RankedRoadPois, EpochResponseCache, score_returned
from evaluation.lane_travel import LanePoiService
from tests.test_belief_lane import fixture


def test_full_ranking_matches_direct_shortest_path_with_missing_and_tied_pois(tmp_path):
    rn, context, _ = fixture()
    service = LanePoiService(rn, list(context.pois))
    ranking = RankedRoadPois(service, tmp_path/'rank.npy')
    reloaded = RankedRoadPois(service, tmp_path/'rank.npy')
    assert np.array_equal(ranking.rank, reloaded.rank)
    for state in range(len(rn)):
        point = rn.latlon(state)
        actual = rn.nearest(*point)[0]
        distances = service.distances(point)
        for mask in [np.ones(ranking.n, dtype=bool), np.zeros(ranking.n, dtype=bool),
                     np.arange(ranking.n) % 2 == 0]:
            got = ranking.top(actual, mask, 5)
            for category, ids in zip(ranking.categories, got):
                expected = sorted((i for i, p in enumerate(ranking.pois)
                                   if mask[i] and p['category'] == category and p['id'] in distances),
                                  key=lambda i: (distances[ranking.pois[i]['id']], ranking.pois[i]['id']))[:5]
                assert ids == expected


def test_availability_epoch_order_and_paired_probability_worlds():
    low = AvailabilityWorld(100, 123, .5)
    high = AvailabilityWorld(100, 123, .95)
    assert low.epoch(59.99) == 0 and low.epoch(60) == 1
    first = low.at_epoch(3).copy()
    assert np.all(~first | high.at_epoch(3))
    low.at_epoch(0)
    assert np.array_equal(first, AvailabilityWorld(100, 123, .5).at_epoch(3))
    assert not np.array_equal(first, low.at_epoch(4))
    with pytest.raises(ValueError):
        low.at_epoch(0)[0] = True


def test_cache_prefix_only_expiry_and_unavailable_stale_results():
    cache = EpochResponseCache(5)
    a, b = cache.receive(0, [[[0, 1], []]])
    assert np.array_equal(a, b)
    a, b = cache.receive(0, [[[2], []]])
    assert np.flatnonzero(a).tolist() == [2]
    assert np.flatnonzero(b).tolist() == [0, 1, 2]
    a, b = cache.receive(1, [[[3], []]])
    assert np.flatnonzero(b).tolist() == [3]
    with pytest.raises(ValueError):
        cache.receive(0, [])
    score = score_returned([[2, 3], []], [[0, 2], []], np.array([False, False, True, True, False]))
    assert score['recall'] == .5 and score['empty_reference_categories'] == 1
    assert score['unavailable_returned_items'] == 1


def test_union_cache_monotone_recall_after_local_top_five_ranking():
    rn, context, _ = fixture()
    ranking = RankedRoadPois(LanePoiService(rn, list(context.pois)))
    world = AvailabilityWorld(ranking.n, 78)
    cache = EpochResponseCache(ranking.n)
    for step, state in enumerate(range(len(rn))):
        epoch = world.epoch(step*20)
        available = world.at_epoch(epoch)
        reply = ranking.top(state, available, 2)
        current, history = cache.receive(epoch, [reply])
        for truth in range(len(rn)):
            target = ranking.top(truth, available, 5)
            before = score_returned(target, ranking.top(truth, current, 5), available)['recall']
            after = score_returned(target, ranking.top(truth, history, 5), available)['recall']
            assert before is None or after >= before - 1e-12
