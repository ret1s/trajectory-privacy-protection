from types import SimpleNamespace
import json
import tempfile

import numpy as np
import pytest

from benchmark.anchor_belief import PublicAnchorModel
from benchmark.engines.paced_slack import PacedSlackProgressLaneDummy
from benchmark.engines.public_service_planner import MODES, PublicPurposePacedLaneDummy, make_service_planner_engine
from benchmark.public_poi_context import PublicPoiContext
from benchmark.public_service_profiles import PublicServiceProfiles, build_public_service_profiles
from benchmark.query_purpose import MultiPurposeRoadRanking, QueryPurpose, QuerySpec
from benchmark.response_aware_belief import ResponseAwareAnchorModel
from benchmark.risk_aware_cover import TailAwareCoverObjective
from core.demo_protocol import TrajectoryPoint
from core.session_budget import FixedEpochPolicy, PersistentEpochBudget, FixedEpochProtectedSessions
from evaluation.lane_travel import LanePoiService
from tests.test_lane_comparison import road
from tests.test_query_purpose import fixture as variable_speed_ranking


def fixture():
    rn = road()
    pois = [dict(id=f'p{i:03d}', category='cafe', lat=0., lon=i*.0001) for i in range(30)]
    contexts = [PublicPoiContext(LanePoiService(rn, pois, k=k)) for k in (5, 10, 20)]
    base = PublicAnchorModel(rn, contexts[0], np.ones(len(rn)), spacing_m=20.,
                             epsilon_release=.01, epsilon_test=.01)
    profiles = build_public_service_profiles(base, contexts[2], public_destination_states=[2, 27], public_radius_m=15.)
    return rn, base, contexts[1], contexts[2], profiles


def make(mode, resources, **extra):
    rn, base, legacy, reply, profiles = resources
    engine = make_service_planner_engine(mode, rn, base, reply, profiles, legacy_context=legacy,
        k=2, budget=.24, horizon=12, theta_m=200., read_interval_s=60., utility_slack=.03,
        rng=np.random.default_rng(41), **extra)
    engine.anchor_rng = np.random.default_rng(73)
    engine.dummy_rng = np.random.default_rng(97)
    engine.reset()
    return engine


def test_public_profiles_match_all_four_exact_local_objectives_and_na_mass():
    ranking = variable_speed_ranking()
    rn = ranking.rn; rn.catalogue_sha256 = 'public-variable-speed-test'
    reference = PublicPoiContext(SimpleNamespace(rn=rn, pois=ranking.pois, categories=ranking.categories, k=1))
    reply = PublicPoiContext(SimpleNamespace(rn=rn, pois=ranking.pois, categories=ranking.categories, k=2))
    base = PublicAnchorModel(rn, reference, np.ones(len(rn)), spacing_m=2.)
    profiles = build_public_service_profiles(base, reply, public_destination_states=[3], public_radius_m=90., reference_k=1)
    oracle = [QuerySpec(QueryPurpose.NEAREST, 'cafe', k=1),
              QuerySpec(QueryPurpose.FASTEST, 'cafe', k=1),
              QuerySpec(QueryPurpose.WITHIN_RADIUS, 'cafe', k=1, radius_m=90.),
              QuerySpec(QueryPurpose.MIN_DETOUR, 'cafe', k=1, destination_state=3)]
    for latent, state in enumerate(base.state_ids):
        accesses = reference.access[state]
        expected = {}
        for query in oracle:
            ids = tuple(sorted(ranking.top(accesses, np.ones(ranking.n, dtype=bool), query)))
            expected[ids] = expected.get(ids, 0.)+.25
        actual = {profile: value for profile, value in zip(profiles.reference_profiles,
                    profiles.profile_mass.getrow(latent).toarray().ravel()) if value}
        assert actual == pytest.approx(expected)
    latent = next(i for i, state in enumerate(base.state_ids) if reference.access[state] == 0)
    weights = np.zeros(len(base.state_ids)); weights[latent] = 1.
    objective = profiles.objective(weights)
    assert objective.empty_profile_mass == .25  # Radius90 is N/A, not recall0.
    assert {(), (0,), (1,)}.issubset(set(profiles.reference_profiles))
    assert profiles.metadata['reply_l'] == 2 and profiles.metadata['reference_k'] == 1


def test_vectorized_discrete_cvar_matches_independent_set_oracle_for_every_proposal():
    rn, base, legacy, reply, profiles = fixture()
    weights = np.arange(1, len(base.state_ids)+1, dtype=float); weights /= weights.sum()
    objective = profiles.objective(weights, tail_mass=.31, risk_weight=.75)
    signatures = {i: tuple(int(p) for p in reply.query_indices(i).ravel() if p >= 0) for i in range(len(rn))}
    mass = np.asarray(weights @ profiles.profile_mass).ravel()
    oracle = TailAwareCoverObjective(profiles.reference_profiles, mass, signatures, tail_mass=.31, risk_weight=.75)
    means, tails, values = objective.scores_replacing(np.arange(len(rn)), [13], batch_size=7)
    for state in range(len(rn)):
        expected = oracle.score((state, 13))
        actual = objective.score((state, 13))
        assert actual.mean == pytest.approx(expected.mean, abs=1e-13)
        assert actual.lower_tail_cvar == pytest.approx(expected.lower_tail_cvar, abs=1e-13)
        assert actual.objective == pytest.approx(expected.objective, abs=1e-13)
        assert (means[state], tails[state], values[state]) == pytest.approx((expected.mean, expected.lower_tail_cvar, expected.objective))


def test_integrated_risk_exchange_accepts_tail_gain_only_within_mean_floor():
    references = [tuple(range(5*j,5*(j+1))) for j in range(4)]
    signatures = np.full((2,1,17),-1,dtype=int)
    signatures[0,0] = list(range(17))  # Recall1,1,1,.4; mean.85, tail.4.
    signatures[1,0,:16] = [poi for profile in references for poi in profile[:4]]
    context = SimpleNamespace(rn=[0,1],pois=list(range(20)),access=np.arange(2),signatures=signatures)
    profiles = PublicServiceProfiles(SimpleNamespace(state_ids=np.array([0])),context,
        references,np.array([[.25]*4]),{'reference_k':5})
    objective = profiles.objective([1.],tail_mass=.25,risk_weight=.5)
    engine = PublicPurposePacedLaneDummy.__new__(PublicPurposePacedLaneDummy)
    engine.planner_mode='risk_multi';engine.risk_weight=.5;engine.max_risk_exchanges=3
    engine.mean_slack=.05
    selected,history,floor,evaluations,termination=engine._risk_refine(objective,[np.array([1,0])],[0])
    assert selected==[1] and len(history)==2 and evaluations==4
    assert termination=='no_improving_exchange'
    assert history[0]['mean']==pytest.approx(.85) and floor==pytest.approx(.8)
    assert history[1]['mean']==history[1]['lower_tail_cvar']==pytest.approx(.8)
    assert history[1]['objective']>history[0]['objective']
    engine.mean_slack=.01
    rejected=engine._risk_refine(objective,[np.array([0,1])],[0])
    assert rejected[0]==[0] and len(rejected[1])==1
    assert rejected[4]=='no_improving_exchange'
    with pytest.raises(ValueError,match='public response'):
        objective.scores_replacing([-1],[])
    with pytest.raises(ValueError,match='batch size'):
        objective.scores_replacing([0],[],batch_size=0)


def test_all_modes_pair_original_geo_i_anchors_ledgers_and_private_read_flags():
    resources = fixture()
    engines = {mode: make(mode, resources) for mode in MODES}
    points = [TrajectoryPoint(i*20., 0., .0001+(i%20)*.0001) for i in range(45)]
    for point in points:
        for engine in engines.values():
            engine.protect_step(point.lat, point.lon, point.timestamp_s)
    original = engines['legacy_l10']
    for engine in engines.values():
        assert engine.evaluator_anchors == original.evaluator_anchors
        assert engine.evaluator_ledger == original.evaluator_ledger
        assert engine.spent_bound == original.spent_bound
    rn, base, legacy, reply, _ = resources
    old = PacedSlackProgressLaneDummy(rn, belief_model=ResponseAwareAnchorModel(base, legacy),
        k=2, budget=.24, horizon=12, theta_m=200., read_interval_s=60., utility_slack=.03,
        rng=np.random.default_rng(41))
    old.anchor_rng=np.random.default_rng(73); old.dummy_rng=np.random.default_rng(97); old.reset()
    for point in points: old.protect_step(point.lat, point.lon, point.timestamp_s)
    assert old.evaluator_states == original.evaluator_states
    assert old.evaluator_anchors == original.evaluator_anchors


def test_risk_zero_replays_aligned_mean_and_risk_floor_feasibility_prefix_hold():
    resources = fixture()
    points = tuple(TrajectoryPoint(i*20., 0., .0001+(i%20)*.0001) for i in range(12))
    mean = make('aligned_mean_multi', resources)
    zero = make('risk_multi', resources, risk_weight=0.)
    assert mean.protect_run(points).to_attacker_dict()['events'] == zero.protect_run(points).to_attacker_dict()['events']
    risk = make('risk_multi', resources, risk_weight=.75)
    public = risk.protect_run(points).to_attacker_dict()
    assert public['events'][:5] == make('risk_multi', resources, risk_weight=.75).protect_run(points[:5]).to_attacker_dict()['events']
    for row in risk.evaluator_objective:
        assert row['mean_value'] >= row['risk_mean_floor']-1e-12
        assert all(b['objective'] > a['objective']+1e-12 for a, b in zip(row['risk_history'], row['risk_history'][1:]))
    for earlier, later in zip(risk.evaluator_states, risk.evaluator_states[1:]):
        assert all(b in risk.travel.reachable(a,20.) for a,b in zip(earlier,later))
    assert 'risk_history' not in json.dumps(public) and 'mean_baseline_states' not in json.dumps(public)


def test_public_profiles_and_aligned_reply_do_not_change_primitive_or_context_arrays():
    rn, base, legacy, reply, profiles = fixture()
    originals = [base.prior.copy(), base.log_normalizers.copy(), base.poi_weights.toarray().copy(), reply.signatures.copy()]
    again = build_public_service_profiles(base, reply, public_destination_states=[27,2], public_radius_m=15.)
    assert again.sha256 == profiles.sha256
    for mode in MODES:
        engine = make(mode, (rn,base,legacy,reply,profiles))
        assert engine.belief_model.emission(rn.latlon(2)).tolist() == base.emission(rn.latlon(2)).tolist()
    for actual, before in zip((base.prior, base.log_normalizers, base.poi_weights.toarray(), reply.signatures), originals):
        assert np.array_equal(actual,before)
    with pytest.raises(ValueError,match='PUBLIC'):
        build_public_service_profiles(base,reply,public_destination_states=[True])
    with pytest.raises(ValueError,match='Reference'):
        build_public_service_profiles(base,reply,public_destination_states=[1],reference_k=3)


def test_integrated_budget_supplier_has_no_extra_gps_after_cap_or_public_skip():
    rn, _, legacy, reply, _ = fixture()
    base = PublicAnchorModel(rn, PublicPoiContext(LanePoiService(rn,list(reply.pois),k=5)), np.ones(len(rn)),
                             spacing_m=20.,epsilon_release=.01,epsilon_test=.01)
    profiles=build_public_service_profiles(base,reply,public_destination_states=[2,27])
    policy=FixedEpochPolicy('planner-boundary',0.,1000.,total_effective_epsilon_per_m=.05,
                            session_slots=1,horizon=3,read_interval_s=60.)
    with tempfile.TemporaryDirectory(prefix='jisa-planner-test-',dir='/private/tmp') as directory:
        clients=[];calls={mode:[] for mode in MODES}
        for mode in MODES:
            ledger=PersistentEpochBudget(policy,f'{directory}/{mode}.sqlite',private_key=b'x'*32)
            def factory(allocation,streams,m=mode):
                engine=make_service_planner_engine(m,rn,base,reply,profiles,legacy_context=legacy,k=2,
                    budget=allocation.nominal_budget_per_m,horizon=allocation.horizon,
                    read_interval_s=allocation.read_interval_s,rng=streams.initialization)
                engine.anchor_rng=streams.anchor;engine.dummy_rng=streams.dummy;engine.reset();return engine
            client=FixedEpochProtectedSessions(ledger,factory);client.start_session('s',0.)
            clients.append((mode,client,ledger))
        for t in range(0,481,20):
            for mode,client,_ in clients:
                def supplier(m=mode,clock=t):calls[m].append(clock);return rn.latlon(clock//20%20)
                assert len(client.protect_step(t,supplier))==2
        assert all(clocks==calls['legacy_l10'] for clocks in calls.values())
        assert all(b-a>=60 for a,b in zip(calls['legacy_l10'],calls['legacy_l10'][1:]))
        assert len(calls['legacy_l10'])<=4 and calls['legacy_l10'][-1]<480
        for _,client,ledger in clients:
            assert client.evaluator_summary()['spent_per_m']<=.05+1e-12
            ledger.close()
