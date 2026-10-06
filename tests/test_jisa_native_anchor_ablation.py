"""Configuration and causal supplier checks for the publication development run."""
import numpy as np
import pytest

from benchmark.anchor_belief import PublicAnchorModel
from benchmark.response_aware_belief import ResponseAwareAnchorModel
from benchmark.public_poi_context import PublicPoiContext
from benchmark.planar_anchor import PlanarAnchorModel
from core.session_budget import PersistentEpochBudget, FixedEpochProtectedSessions
from evaluation.lane_travel import LanePoiService
from experiments.jisa_native_anchor_ablation_20261006 import policy, make_engine, configuration
from tests.test_lane_comparison import road


def fixture():
    rn = road()
    pois = [{'id': str(i), 'lat': 0., 'lon': i*.0001, 'category': 'cafe'} for i in range(1, 28, 3)]
    reference = PublicPoiContext(LanePoiService(rn, pois, k=5))
    reply = PublicPoiContext(LanePoiService(rn, pois, k=10))
    base = PublicAnchorModel(rn, reference, np.ones(len(rn)), spacing_m=20.,
                            epsilon_release=.00125, epsilon_test=.00125)
    return rn, ResponseAwareAnchorModel(base, reply)


def test_matched_epoch_and_continuous_emission_configuration(tmp_path):
    rn, belief = fixture(); p = policy()
    assert p.unit_epsilon_per_m == .00125 and p.nominal_session_budget_per_m == .03
    assert p.session_slots*p.effective_session_cap_per_m == .23
    engines = {}
    for method in ('rem_epoch8', 'planar_epoch8'):
        ledger = PersistentEpochBudget(p, tmp_path/f'{method}.sqlite', private_key=b'k'*32)
        allocation = ledger.reserve('s0', 0.)
        engine = make_engine(method, rn, belief, allocation, ledger.private_rng_streams(allocation))
        engines[method] = engine
        assert engine.max_units == 23 and engine.k == 5 and engine.unit_epsilon == .00125
        assert engine.read_interval_s == 60. and engine.belief_model.context.k == 10
        ledger.close()
    assert isinstance(engines['planar_epoch8'].belief_model, PlanarAnchorModel)
    assert not isinstance(engines['rem_epoch8'].belief_model, PlanarAnchorModel)
    assert engines['planar_epoch8'].anchor.name == 'predictive_planar_laplace'
    assert engines['rem_epoch8'].anchor.name == 'pr_sm_rem'
    assert configuration()['reply_depth_L'] == 20 and configuration()['planner_reply_depth_L'] == 10


def test_wrong_emission_epsilon_and_map_rejected_before_private_read(tmp_path):
    rn, belief = fixture()
    ledger = PersistentEpochBudget(policy(), tmp_path/'x.sqlite', private_key=b'k'*32)
    allocation = ledger.reserve('s', 0.); streams = ledger.private_rng_streams(allocation)
    # A publicly mismatched model must not silently inherit the correct engine epsilon.
    wrong = ResponseAwareAnchorModel(PublicAnchorModel(rn, belief.base.context, np.ones(len(rn)),
        spacing_m=20., epsilon_release=.01, epsilon_test=.01), belief.context)
    with pytest.raises(ValueError, match='matched emission'):
        make_engine('planar_epoch8', rn, wrong, allocation, streams)
    with pytest.raises(ValueError, match='matched emission'):
        make_engine('rem_epoch8', road(), belief, allocation, streams)
    ledger.close()


@pytest.mark.parametrize('method', ['rem_epoch8', 'planar_epoch8'])
def test_public_clock_and_stopped_cap_never_read_gps(method, tmp_path):
    rn, belief = fixture(); ledger = PersistentEpochBudget(policy(), tmp_path/'x.sqlite', private_key=b'k'*32)
    client = FixedEpochProtectedSessions(ledger, lambda a, s: make_engine(method, rn, belief, a, s))
    assert client.start_session('s0', 0.)
    reads = []
    def gps(): reads.append(True); return rn.latlon(10)
    assert len(client.protect_step(0., gps)) == 5
    assert len(client.protect_step(20., lambda: (_ for _ in ()).throw(AssertionError('unscheduled GPS')))) == 5
    assert len(client.protect_step(60., gps)) == 5 and len(reads) == 2
    client._engine.spent_units = client._engine.max_units
    client._engine.spent_bound = client._engine.spent_units*client._engine.unit_epsilon
    assert len(client.protect_step(120., lambda: (_ for _ in ()).throw(AssertionError('GPS after cap')))) == 5
    assert client.evaluator_current_session()['spent_per_m'] <= .02875
    client.close_session(600.); ledger.close()


def test_cold_subset_denominator_excludes_unrepresented_sessions():
    from experiments.verify_jisa_native_anchor_ablation_20261006 import summarize
    def row(family, value):
        return {'family_id': family, 'slot': 0, 'recall5': value, 'cached_poi_count': 2,
                'reply_bytes': 20, 'requests': 5}
    result = summarize([row('a', .5), row('a', 1.), row('b', None), row('b', None)])
    assert result['represented_session_count'] == 2 and result['undefined_session_count'] == 1
    assert result['reference_defined_windows'] == 2 and result['total_windows'] == 4
    assert result['reference_coverage'] == .5 and result['family_macro_recall5'] == .75
    assert result['minimum_session'] == {'family_id': 'a', 'slot': 0, 'recall5': .75}


def test_independent_probability_metrics_keep_raw_fork_ambiguity():
    from experiments.verify_jisa_native_anchor_ablation_20261006 import metrics
    rows = [{'family_id': 'a', 'choice_index': i, 'destination_role': role,
             'destination_xy': [200*i, 0], 'public_context': {'choices': [
                 {'destination_xy': [0, 0]}, {'destination_xy': [200, 0]}]}}
            for i, role in enumerate(('routine', 'rare'))]
    unknown = metrics(rows, [[.5, .5], [.5, .5]])
    assert unknown['exact_candidate_edge_accuracy'] == .5 and unknown['destination_mae_m'] == 100.
    assert unknown['routine_accuracy'] == 1. and unknown['rare_accuracy'] == 0.
    assert unknown['brier'] == .25 and unknown['log_loss'] == pytest.approx(np.log(2))
    signal = metrics(rows, [[1., 0.], [0., 1.]])
    assert signal['exact_candidate_edge_accuracy'] == 1. and signal['destination_mae_m'] == 0.


def test_descriptive_bootstrap_pairs_ids_and_retains_undefined_coverage():
    from experiments.jisa_anchor_paired_readout_20261006 import paired
    result = paired({'a': .2, 'b': .8, 'c': None}, {'b': .8, 'c': .7, 'a': .2})
    assert result['family_ids'] == ['a', 'b'] and result['excluded_undefined_family_pairs'] == 1
    assert result['mean_REM_minus_Planar'] == 0 and result['percentile95_family_bootstrap'] == [0., 0.]
