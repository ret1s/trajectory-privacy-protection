import numpy as np
import pytest

from benchmark.paper_comparators import PublicHistory
from evaluation.endpoint_noise_attacks import EndpointShadowBank, endpoint_features
from experiments.endpoint_generalization import (
    LOCKED, fresh_dataset, linked_groups, linked_rows, session_seed,
)
from tests.test_belief_lane import fixture
from experiments.endpoint_generalization_readout import proportions_interval


def test_session_rng_is_independent_order_invariant_and_matched_across_methods():
    master = 'ab'*32
    assert session_seed(master, 'one', 0) == session_seed(master, 'one', 0)
    assert session_seed(master, 'one', 0) != session_seed(master, 'two', 0)
    assert session_seed(master, 'one', 0) != session_seed(master, 'one', 1)
    assert session_seed(master, 'one', 0) != session_seed('cd'*32, 'one', 0)
    forward = {sid: session_seed(master, sid, 0) for sid in ('one', 'two', 'three')}
    reverse = {sid: session_seed(master, sid, 0) for sid in ('three', 'two', 'one')}
    assert forward == reverse
    assert len({session_seed(master, f'sid-{i}', rep) for i in range(100) for rep in (0, 1)}) == 200
    # Methods do not appear in this API: matching remains common only for
    # the same session/rep, never for unrelated trip resets.


@pytest.mark.parametrize('master,sid,rep', [('ab', 'one', 0), ('ab'*32, '', 0),
                                        ('ab'*32, 'one', -.5), ('ab'*32, 'one', -1)])
def test_invalid_randomization_keys_rejected(master, sid, rep):
    with pytest.raises(ValueError):
        session_seed(master, sid, rep)


def test_unused_test_labels_require_frozen_selection(tmp_path):
    with pytest.raises(RuntimeError, match='Freeze attacker selection'):
        fresh_dataset(tmp_path)
    assert LOCKED == 'scale025_L20'


def test_linked_inference_keeps_each_trip_target_instead_of_common_destination():
    rn, _, _ = fixture()
    history = PublicHistory(rn, [list(range(15))])
    items, x, seq, y = [], [], [], []
    for j in (0, 1):
        events = [{'timestamp_s': i*20., 'coordinates': [rn.latlon(i+j)]} for i in range(6)]
        target = list(rn.point_xy(*events[-1]['coordinates'][0]))
        item = dict(family_id='family', session_id=f'trip-{j}', seed=0, method='raw',
            events=events, close_s=100., target_xy_evaluator_only={'S10': target})
        items.append(item)
        aggregate, sequence, _ = endpoint_features(events, 'S10', rn, history, observable_close_s=100.)
        x.append(aggregate[0]); seq.append(sequence[0]); y.append(target)
    groups = linked_groups(items, 'S10', rn, history)
    assert len(groups) == 1
    assert groups[0]['endpoint_spread_m'] > 0
    assert np.array_equal(groups[0]['targets'][0], y[0])
    assert np.array_equal(groups[0]['targets'][1], y[1])
    single = EndpointShadowBank(x, seq, y)
    linked = EndpointShadowBank(groups[0]['aggregates'], groups[0]['sequences'], y)
    rows = linked_rows(items, 'S10', single, linked, rn, history)
    assert {row['session_id'] for row in rows} == {'trip-0', 'trip-1'}
    assert all(row['errors']['linked_individual_centroid'] == pytest.approx(0.) for row in rows)
    assert all(row['errors']['linked_mean_centroid'] > 0 for row in rows)


def test_private_seed_material_and_truth_are_not_public_attack_features():
    rn, _, _ = fixture()
    history = PublicHistory(rn, [list(range(15))])
    events = [{'timestamp_s': i*20., 'coordinates': [rn.latlon(i)]} for i in range(6)]
    polluted = [dict(event, master_hex='ab'*32, rng_seed_evaluator_only=12345,
                     private_gps=[45., 90.], target_xy_evaluator_only=[100., 200.]) for event in events]
    a, sequence_a, _ = endpoint_features(events, 'S10', rn, history, observable_close_s=100.)
    b, sequence_b, _ = endpoint_features(polluted, 'S10', rn, history, observable_close_s=100.)
    assert np.array_equal(a, b)
    assert np.array_equal(sequence_a, sequence_b)


def test_zero_hit_caution_counts_families_instead_of_correlated_repetitions():
    caution = proportions_interval(0, 28)
    assert caution['one_sided95_zero_success_upper'] == pytest.approx(1-.05**(1/28))
    assert caution['exact95_high'] > caution['one_sided95_zero_success_upper']
    assert caution['families'] == 28
    with pytest.raises(ValueError):
        proportions_interval(0, 0)
