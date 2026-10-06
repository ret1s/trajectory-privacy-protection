import numpy as np
import pytest

from benchmark.paper_comparators import PublicHistory
from evaluation.ordered_endpoint_attacks import canonical_tracks, ordered_arrays, ordered_endpoint_features
from experiments.endpoint_order_challenge import bank_group, permutation, publication_events, validate_view
from tests.test_belief_lane import fixture


def test_canonical_hungarian_tracks_are_invariant_to_per_event_permutation():
    points = np.asarray([[[0., 0.], [10., 0.], [20., 0.]],
                         [[2., 0.], [12., 0.], [22., 0.]],
                         [[4., 0.], [14., 0.], [24., 0.]]])
    shuffled = np.asarray([p[order] for p, order in zip(points, ([2, 0, 1], [1, 2, 0], [0, 2, 1]))])
    for velocity in (False, True):
        assert np.array_equal(canonical_tracks(points, [0., 60., 120.], velocity=velocity),
                              canonical_tracks(shuffled, [0., 60., 120.], velocity=velocity))


def test_velocity_linker_can_follow_crossing_tracks_without_internal_ids():
    points = np.asarray([[[0., 0.], [10., 0.]], [[4., 0.], [6., 0.]], [[8., 0.], [2., 0.]]])
    nearest = canonical_tracks(points, [0., 1., 2.])
    velocity = canonical_tracks(points, [0., 1., 2.], velocity=True)
    assert nearest[-1, 0, 0] == 2.
    assert velocity[-1, 0, 0] == 8.


def test_publication_preserves_duplicates_clock_and_input_and_drops_private_fields():
    events = [dict(timestamp_s=i*60., coordinates=[[1., 2.], [3., 4.], [1., 2.]], candidate_ids=['a', 'b', 'c'])
              for i in range(8)]
    item = dict(session_id='trip', seed=0, events=events, target_xy_evaluator_only=[5., 6.])
    published = publication_events(item, 'shuffled', 'ab'*32)
    validate_view(events, published)
    assert all(len(e['coordinates']) == 3 and e['coordinates'].count([1., 2.]) == 2 for e in published)
    assert all(set(e) == {'timestamp_s', 'coordinates'} for e in published)
    assert events[0]['candidate_ids'] == ['a', 'b', 'c']
    assert published == publication_events(item, 'shuffled', 'ab'*32)


def test_private_shuffle_substreams_depend_on_session_rep_event_and_master():
    inputs = [('ab'*32, 'one', 0, 0), ('ab'*32, 'two', 0, 0), ('ab'*32, 'one', 1, 0),
              ('ab'*32, 'one', 0, 1), ('cd'*32, 'one', 0, 0)]
    outputs = [tuple(permutation(*key, 20)) for key in inputs]
    assert len(set(outputs)) == len(inputs)
    assert outputs[0] == tuple(permutation('ab'*32, 'one', 0, 0, 20))
    with pytest.raises(ValueError):
        permutation('ab', 'one', 0, 0, 5)


def test_ordered_features_observe_positions_but_geometry_survives_shuffle_and_truth_pollution():
    rn, _, _ = fixture()
    history = PublicHistory(rn, [list(range(15))])
    events = [dict(timestamp_s=i*20., coordinates=[rn.latlon(i+j) for j in range(3)]) for i in range(6)]
    item = dict(session_id='trip', seed=0, events=events)
    shuffled = publication_events(item, 'shuffled', 'ab'*32)
    polluted = [dict(e, candidate_ids=['true', 'fake'], private_gps=[45., 90.], target_xy_evaluator_only=[1., 2.]) for e in events]
    a, predictions_a = ordered_endpoint_features(events, 'S10', rn, history, observable_close_s=100.)
    b, predictions_b = ordered_endpoint_features(shuffled, 'S10', rn, history, observable_close_s=100.)
    c, _ = ordered_endpoint_features(polluted, 'S10', rn, history, observable_close_s=100.)
    assert not np.array_equal(a['observed_slots'], b['observed_slots'])
    for channel in ('geometry_nearest', 'geometry_velocity', 'invariant_aggregate', 'invariant_sequence'):
        assert np.array_equal(a[channel], b[channel])
    for channel in a:
        assert np.array_equal(a[channel], c[channel])
    for name in predictions_a:
        if not name.startswith('observed_slots_'):
            assert np.array_equal(predictions_a[name], predictions_b[name])


def test_ordered_arrays_retain_public_multiplicity_and_require_fixed_k():
    rn, _, _ = fixture()
    points = [rn.latlon(0), rn.latlon(0), rn.latlon(1)]
    events = [dict(timestamp_s=i*20., coordinates=points) for i in range(2)]
    xy, times = ordered_arrays(events, rn)
    assert xy.shape == (2, 3, 2) and np.array_equal(xy[:, 0], xy[:, 1])
    with pytest.raises(ValueError, match='fixed public K'):
        ordered_arrays([events[0], dict(timestamp_s=20., coordinates=points[:2])], rn)


def test_geometry_group_never_uses_observed_slot_labels_and_invariant_control_is_retained():
    rows = [dict(errors={'centroid': 5., 'observed_slots_slot0_endpoint': 2.,
                         'geometry_nearest_slot0_endpoint': 3., 'geometry_velocity_tree': 4.})]
    assert set(bank_group(rows, 'invariant')[0]['errors']) == {'centroid'}
    assert set(bank_group(rows, 'geometry_associated')[0]['errors']) == {'centroid', 'geometry_nearest_slot0_endpoint', 'geometry_velocity_tree'}
    assert set(bank_group(rows, 'observed_slots')[0]['errors']) == {'centroid', 'observed_slots_slot0_endpoint'}
    assert set(bank_group(rows, 'full')[0]['errors']) == set(rows[0]['errors'])
