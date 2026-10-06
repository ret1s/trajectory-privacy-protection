import numpy as np
import pytest

from benchmark.anchor_belief import PublicAnchorModel
from benchmark.engines.endpoint_noise import EndpointNoiseProgressLaneDummy
from benchmark.engines.endpoint_phase_noise import EndpointPhaseNoiseProgressLaneDummy
from benchmark.engines.paced_slack import PacedSlackProgressLaneDummy
from core.demo_protocol import TrajectoryPoint
from tests.test_belief_lane import fixture
from benchmark.paper_comparators import PublicHistory
from evaluation.endpoint_noise_attacks import endpoint_features, select_attackers


def make(scale=.5, seed=73):
    rn, context, base = fixture()
    belief = PublicAnchorModel(rn, context, np.linspace(1, 2, len(rn)),
        spacing_m=2., epsilon_release=.01*scale, epsilon_test=.01*scale)
    return EndpointNoiseProgressLaneDummy(rn, belief_model=belief, privacy_scale=scale,
        k=3, budget=.24, horizon=12, utility_slack=.03,
        rng=np.random.default_rng(seed))


def test_all_endpoint_times_preserved_prefix_causal_and_no_private_diagnostics():
    points = tuple(TrajectoryPoint(i*20., 0., .0001+(i%14)*.0001) for i in range(32))
    model = make()
    full = model.protect_run(points).to_attacker_dict()
    prefix = make().protect_run(points[:7]).to_attacker_dict()
    assert prefix['events'] == full['events'][:7]
    assert [e['timestamp_s'] for e in full['events']] == [p.timestamp_s for p in points]
    assert len(full['events']) == len(points)
    assert 'evaluator_ledger' not in str(full) and 'evaluator_anchors' not in str(full)
    assert full['public_parameters']['publication_delay_s'] == 0


def test_scale_one_is_exact_parent_and_mismatched_emission_rejected():
    rn, context, belief = fixture()
    points = tuple(TrajectoryPoint(i*20., 0., .001) for i in range(14))
    common = dict(belief_model=belief, k=3, budget=.24, horizon=12, utility_slack=.03)
    a = EndpointNoiseProgressLaneDummy(rn, privacy_scale=1, rng=np.random.default_rng(8), **common)
    b = PacedSlackProgressLaneDummy(rn, rng=np.random.default_rng(8), **common)
    assert a.protect_run(points).to_attacker_dict()['events'] == b.protect_run(points).to_attacker_dict()['events']
    with pytest.raises(ValueError, match='match the actual'):
        EndpointNoiseProgressLaneDummy(rn, privacy_scale=.5, **common)


def test_scaled_ledger_reachable_and_no_read_between_clock_or_after_filter_stop():
    model = make(.25)
    for i in range(90):
        # Large displacement forces refresh until cap is spent. Reads between
        # public ticks must tolerate absent GPS coordinates entirely.
        if i%3:
            model.protect_step(float('nan'), float('nan'), i*20.)
        else:
            model.protect_step(45., 90., i*20.)
    assert model.spent_bound <= .23*.25+1e-12
    assert sum(r['cost_units'] for r in model.evaluator_ledger) == model.spent_units
    assert any(r['branch']=='postprocess' for r in model.evaluator_ledger)
    model.protect_step(float('nan'), float('nan'), 1800.)
    assert not model.privacy_read_this_step
    for a, b in zip(model.evaluator_states, model.evaluator_states[1:]):
        assert all(v in model.travel.reachable(u, 20.) for u,v in zip(a,b))
    assert model.anchor.epsilon == pytest.approx(.0025)
    assert model.anchor.eps_test == pytest.approx(.0025)


@pytest.mark.parametrize('scale', [0, -.5, 1.01, float('nan'), float('inf'), True])
def test_invalid_scale_rejected(scale):
    rn, _, belief = fixture()
    with pytest.raises(ValueError, match='privacy_scale'):
        EndpointNoiseProgressLaneDummy(rn, privacy_scale=scale, belief_model=belief)


def test_attack_features_ignore_candidate_order_and_evaluator_truth():
    rn, _, _ = fixture()
    history = PublicHistory(rn, [list(range(10))])
    events = [{'timestamp_s': i*20., 'coordinates': [rn.latlon(i), rn.latlon(i+4)]}
              for i in range(8)]
    changed = [dict(e, coordinates=list(reversed(e['coordinates'])),
                    evaluator_truth=[45., 90.], candidate_ids=['real', 'fake']) for e in events]
    a, sequence_a, geometry_a = endpoint_features(events, 'S9', rn, history, observable_close_s=200.)
    b, sequence_b, geometry_b = endpoint_features(changed, 'S9', rn, history, observable_close_s=200.)
    assert np.array_equal(a, b)
    assert np.array_equal(sequence_a, sequence_b)
    assert all(np.array_equal(geometry_a[name], geometry_b[name]) for name in geometry_a)
    with pytest.raises(ValueError, match='close'):
        endpoint_features(events, 'S10', rn, history, observable_close_s=100.)


def test_attacker_selection_balances_families_and_can_select_different_losses():
    # Ten copies from one family must not outweigh a second family. B hits
    # exactly but misses badly otherwise; A has lower MAE but no Hit100.
    rows = [{'family_id': 'one', 'errors': {'a': 110., 'b': 0.}} for _ in range(10)]
    rows += [{'family_id': 'two', 'errors': {'a': 110., 'b': 1000.}}]
    choice = select_attackers(rows)
    assert choice['mae'] == 'a'
    assert choice['hit100'] == 'b'


def test_initial_phase_noise_has_matching_emission_causal_slack_and_accounting():
    rn, context, base = fixture()
    early = PublicAnchorModel(rn, context, np.linspace(1, 2, len(rn)), spacing_m=2.,
                              epsilon_release=.0025, epsilon_test=.0025)
    def phase():
        return EndpointPhaseNoiseProgressLaneDummy(rn, belief_model=base,
            early_belief_model=early, guard_seconds=60., k=3, horizon=12, budget=.24,
            utility_slack=.03, read_interval_s=60., rng=np.random.default_rng(73))
    points = tuple(TrajectoryPoint(i*20., 0., .001) for i in range(40))
    model = phase()
    full = model.protect_run(points).to_attacker_dict()
    assert phase().protect_run(points[:5]).to_attacker_dict()['events'] == full['events'][:5]
    assert [e['timestamp_s'] for e in full['events']] == [p.timestamp_s for p in points]
    assert model.evaluator_ledger[0]['phase_unit_cost'] == 1
    assert model.evaluator_ledger[3]['phase_unit_cost'] == 4
    assert sum(row['cost_units'] for row in model.evaluator_ledger) == model.spent_units
    assert model.spent_bound == pytest.approx(model.spent_units*.0025)
    assert model.spent_bound <= .23+1e-12
    for row in model.evaluator_objective[1:]:
        assert row['objective_loss'] <= .03+1e-12
    assert full['public_parameters']['destination_guard'] == 'ordinary_GeoI_no_future_end_window_detection'
