"""Independent local-query oracle for purpose/case/category normalization.

Fixtures are public toy graphs; no study output or holdout label is read.
"""
from collections import defaultdict

import networkx as nx
import numpy as np
import pytest

from benchmark.anchor_belief import PublicAnchorModel
from benchmark.engines.public_service_planner_v2 import make_service_planner_engine_v2
from benchmark.public_poi_context import PublicPoiContext
from benchmark.public_service_profiles_v2 import PURPOSES, build_public_service_profiles_v2
from benchmark.query_purpose import MultiPurposeRoadRanking, QueryPurpose, QuerySpec
from core.road_network import RoadNetwork
from evaluation.lane_travel import LanePoiService
from experiments.qplanner_study_20261006_v2 import METHOD_CONFIGS


def resources():
    # B(0) sees six valid categories. A(1) sees only its own cafe.
    # B's nearest cafe is7; A's cafe is1, so Q=A does not cover B's references.
    graph = nx.DiGraph(schema='sumo-lane-progress-v1', spacing_m=20.)
    for node in range(8):
        graph.add_node(node, x=node*.0001, y=0.)
    for node in range(1, 8):
        graph.add_edge(0, node, length=10. if node == 7 else 100.+node, speed=5.)
    rn = RoadNetwork(graph)
    rn.catalogue_sha256 = 'independent-category-gap-graph'
    records = [dict(id=f'poi{i}', category='cafe' if i in (1, 7) else f'c{i}',
                    lat=rn.latlon(i)[0], lon=rn.latlon(i)[1]) for i in range(1, 8)]
    contexts = [PublicPoiContext(LanePoiService(rn, records, k=k)) for k in (1, 10, 20)]
    base = PublicAnchorModel(rn, contexts[0], np.ones(len(rn)), spacing_m=2.,
                             epsilon_release=.01, epsilon_test=.01)
    destinations = [1, 2, 6]
    profiles = build_public_service_profiles_v2(base, contexts[2],
        public_destination_states=destinations, public_radius_m=5., reference_k=1)
    return rn, base, contexts[1], contexts[2], profiles, destinations


def belief_at(base, weights):
    result = np.zeros(len(base.state_ids))
    for state, weight in weights.items():
        indices = np.flatnonzero(base.state_ids == state)
        assert len(indices) == 1
        result[indices[0]] = weight
    return result


def local_query_oracle(base, reply, destinations, belief):
    """Literal expected per-window valid-category mean, with declared priors."""
    ranking = MultiPurposeRoadRanking(reply)
    masses = {}
    for purpose in QueryPurpose:
        purpose_mass = defaultdict(float)
        case_mass = 0.
        prototypes = destinations if purpose == QueryPurpose.MIN_DETOUR else [None]
        for latent, probability in enumerate(belief/belief.sum()):
            if not probability:
                continue
            state = base.context.access[base.state_ids[latent]]
            for destination in prototypes:
                refs = []
                for category in ranking.categories:
                    spec = QuerySpec(purpose, category, k=1,
                        radius_m=5. if purpose == QueryPurpose.WITHIN_RADIUS else None,
                        destination_state=destination)
                    ref = tuple(sorted(ranking.top(state, np.ones(ranking.n, dtype=bool), spec)))
                    if ref:
                        refs.append(ref)
                if refs:
                    weight = probability/len(prototypes)
                    case_mass += weight
                    for ref in refs:
                        purpose_mass[ref] += weight/len(refs)
        if case_mass:
            masses[purpose.value] = {r: p/case_mass for r, p in purpose_mass.items()}
    combined = defaultdict(float)
    for distribution in masses.values():
        for ref, probability in distribution.items():
            combined[ref] += probability/len(masses)
    return dict(combined), masses


def weighted_lower_tail(items, mass):
    remaining, total = mass, 0.
    for value, probability in sorted(items):
        take = min(remaining, probability)
        total += take*value
        remaining -= take
        if remaining <= 1e-14:
            break
    assert remaining <= 1e-12
    return total/mass


def test_one_vs_six_valid_categories_keep_equal_case_mass_not_one_seventh():
    rn, base, _, reply, profiles, _ = resources()
    belief = belief_at(base, {0: .5, 1: .5})
    ids = np.flatnonzero(base.state_ids == 0)
    assert profiles.case_valid_category_counts[PURPOSES[0]][ids[0], 0] == 6
    objective = profiles.objective(belief)
    # Direct per-purpose profile mass, before equal-purpose mixing.
    nearest = np.asarray(belief @ profiles.purpose_masses[PURPOSES[0]]).ravel()
    received = set(int(x) for x in reply.query_indices(1).ravel() if x >= 0)
    score = sum(probability*len(received.intersection(ref))/len(ref)
                for ref, probability in zip(profiles.reference_profiles, nearest) if ref)
    assert score == pytest.approx(.5)
    assert score != pytest.approx(1/7)
    assert objective.normalization_diagnostics['purpose_valid_case_mass']['within_radius'] == .5


@pytest.mark.parametrize('weights', [{0: .5, 1: .5}, {0: 1.}, {0: .2, 1: .3, 4: .5}])
def test_full_normalized_mass_and_cvar_match_independent_local_query_oracle(weights):
    rn, base, _, reply, profiles, destinations = resources()
    belief = belief_at(base, weights)
    expected, by_purpose = local_query_oracle(base, reply, destinations, belief)
    actual, diagnostics = profiles.normalized_mass(belief)
    actual = {ref: probability for ref, probability in zip(profiles.reference_profiles, actual) if probability}
    assert actual == pytest.approx(expected, abs=1e-13)
    assert set(diagnostics['defined_purposes']) == set(by_purpose)
    if weights == {0: 1.}:
        assert diagnostics['undefined_purposes'] == ['within_radius']
        assert diagnostics['effective_purpose_weights']['within_radius'] == 0.
        assert diagnostics['effective_purpose_weights']['nearest_distance'] == pytest.approx(1/3)
    objective = profiles.objective(belief, tail_mass=.31, risk_weight=.25)
    means, tails, totals = objective.scores_replacing(np.arange(len(rn)), [], batch_size=3)
    for state in range(len(rn)):
        received = set(int(x) for x in reply.query_indices(state).ravel() if x >= 0)
        outcomes = [(len(received.intersection(ref))/len(ref), probability)
                    for ref, probability in expected.items()]
        mean = sum(value*probability for value, probability in outcomes)
        tail = weighted_lower_tail(outcomes, .31)
        assert (means[state], tails[state], totals[state]) == pytest.approx(
            (mean, tail, .75*mean+.25*tail), abs=1e-13)


def test_actual_runner_method_arguments_match_named_public_arms():
    rn, base, legacy, reply, profiles, _ = resources()
    for name, configuration in METHOD_CONFIGS.items():
        engine = make_service_planner_engine_v2(configuration['mode'], rn, base, reply,
            profiles, legacy_context=legacy, risk_weight=configuration.get('risk_weight', .5),
            tail_mass=.25, mean_slack=.01, max_risk_exchanges=3, k=2,
            budget=.24, horizon=12, theta_m=200., read_interval_s=60.,
            utility_slack=configuration['utility_slack'], rng=np.random.default_rng(13))
        assert engine.utility_slack == configuration['utility_slack']
        if name.startswith('normalized'):
            assert engine.risk_weight == configuration['risk_weight']
            assert np.array_equal(engine.belief_model.emission(rn.latlon(0)), base.emission(rn.latlon(0)))


def test_every_public_case_count_and_na_metadata_match_local_query_eligibility():
    _, base, _, reply, profiles, destinations = resources()
    ranking = MultiPurposeRoadRanking(reply)
    empty_category_count = 0
    for purpose in QueryPurpose:
        prototypes = destinations if purpose == QueryPurpose.MIN_DETOUR else [None]
        expected = np.zeros((len(base.state_ids), len(prototypes)), dtype=int)
        for latent, state in enumerate(base.state_ids):
            state = base.context.access[state]
            for prototype, destination in enumerate(prototypes):
                for category in ranking.categories:
                    spec = QuerySpec(purpose, category, k=1,
                        radius_m=5. if purpose == QueryPurpose.WITHIN_RADIUS else None,
                        destination_state=destination)
                    expected[latent, prototype] += bool(ranking.top(
                        state, np.ones(ranking.n, dtype=bool), spec))
                empty_category_count += len(ranking.categories)-expected[latent, prototype]
        assert np.array_equal(profiles.case_valid_category_counts[purpose.value], expected)
        assert profiles.metadata['all_empty_case_count_by_purpose'][purpose.value] == int((expected == 0).sum())
    assert profiles.metadata['empty_input_category_profile_count'] == empty_category_count
