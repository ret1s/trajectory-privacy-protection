"""Contracts for the second research cycle, independent of large cached runs."""
import json
import numpy as np
import pytest

from tests.test_report_demo import road, event
from benchmark.engines.budgeted import BudgetedReachableDummy
from core.demo_protocol import TrajectoryPoint
from data.sumo_demo import FCDSample
from data.paper_scenarios import records_for_study, study_catalogue
from evaluation.scenario_metrics import RoadTravelTimes, PoiService, stable_track_validity
from evaluation.research_protocol import attack_candidates, full_path_estimates, target_errors, utility_metrics
from experiments.run_paper_benchmark import choose_mechanism, generate


@pytest.mark.parametrize("mode", ["fresh", "private_reuse"])
def test_stream_batch_prefix_and_budget(mode):
    rn = road()
    points = tuple(TrajectoryPoint(i * 20, *rn.latlon(i % len(rn))) for i in range(16))
    def make():
        return BudgetedReachableDummy(rn, horizon=12, anchor_mode=mode, rng=np.random.default_rng(19))
    whole_model = make()
    whole = whole_model.protect_run(points).to_attacker_dict()
    short = make().protect_run(points[:5]).to_attacker_dict()
    assert whole['events'][:5] == short['events']
    stream = make()
    for point, expected in zip(points, whole['events']):
        output = stream.protect_step(point.lat, point.lon, point.timestamp_s)
        assert list(output) == [(c['lat'], c['lon']) for c in expected['candidates']]
    assert stream.spent_bound == pytest.approx(.24 if mode == 'fresh' else .23)
    assert stream.spent_bound <= stream.budget + 1e-12
    # Once exhausted, new private input must not be read (NaN is intentional).
    a = stream.protect_step(float('nan'), float('nan'), 320)
    b = whole_model.protect_step(0, .004, 320)
    assert a == b
    assert stable_track_validity(whole, rn, RoadTravelTimes(rn)) == 1.
    assert 'last_anchor' not in json.dumps(whole)
    with pytest.raises(ValueError):
        stream.protect_step(0, 0, 320)


def test_scc_excludes_one_way_dead_end():
    rn = road()
    rn.graph.remove_edge(5, 4)
    model = BudgetedReachableDummy(rn, rng=np.random.default_rng(4))
    assert not model.viable[5]
    assert model.viable[:5].all()


def test_full_sequence_attack_and_intersection_fallback():
    rn = road()
    public = {'output_kind': 'real_plus_dummies', 'events': [
        event([(0, .001), (0, .005)], 0), event([(0, .001), (0, 0)], 20)]}
    prior = np.ones(len(rn)) / len(rn)
    result = attack_candidates(public, rn, 'S2', RoadTravelTimes(rn), prior)
    assert np.allclose(result['stationary_intersection'], [rn.xy[1], rn.xy[1]])
    assert full_path_estimates(public, rn, RoadTravelTimes(rn), prior).shape == (2, 2)
    public['events'][1] = event([(0, .002)], 20)
    result = attack_candidates(public, rn, 'S2', RoadTravelTimes(rn), prior)
    assert 'stationary_intersection' in result  # same selection denominator
    assert np.allclose(result['stationary_intersection'], result['centroid'])


def test_endpoint_targets_and_completed_trip_filter(tmp_path):
    rn = road()
    trace = tuple(FCDSample(t, 0, 0, 0., 'e', 'e_0') for t in range(181))
    trace += tuple(FCDSample(t, 0, (t - 180) * .00001, 1., 'e', 'e_0') for t in range(181, 501))
    xml = tmp_path / 'routes.xml'
    xml.write_text('<routes><vehicle id="u" arrival="501"/><vehicle id="v" arrival="-1"/></routes>')
    records, _ = records_for_study({'u': trace, 'v': trace}, rn, xml)
    endpoints = [r for r in records if r['scenario'] in {'S9', 'S10'}]
    assert len(endpoints) == 2 and all(r['vehicle_id'] == 'u' for r in endpoints)
    for r in endpoints:
        index = 0 if r['scenario'] == 'S9' else -1
        assert abs(r['times'][index] - r['hidden_target_fcd']['timestamp_s']) >= 60
        prediction = np.array([rn.point_xy(*r['hidden_target'])] * 12)
        assert target_errors(prediction, r, rn).tolist() == [0.]
    assert sum(c['status'] == 'runnable' for c in study_catalogue()) == 5


def test_poi_gap_and_mechanism_selection():
    rn = road()
    service = PoiService(rn, [{'id': str(i), 'lat': 0, 'lon': i * .001, 'category': 'cafe'} for i in range(6)], k=5)
    metrics = utility_metrics(service, {'events': [event([(0, .005)])]}, [(0, 0)])
    assert metrics['poi_recall_at_5'] == .8
    assert metrics['poi_complete_rate'] == 1
    assert metrics['poi_extra_distance_m'] == 100
    options = [{'spec': {'id': 'a'}, 'min_case_recall': .89, 'macro_hit100': 0},
               {'spec': {'id': 'b'}, 'min_case_recall': .91, 'macro_hit100': .4}]
    assert choose_mechanism(options) == (options[1], True)
    assert choose_mechanism(options[:1]) == (options[0], False)


def test_generation_hides_endpoint_and_measures_steps():
    rn = road()
    record = {'record_id': 'u/S9', 'scenario': 'S9', 'times': [60, 80],
              'points_input': [(0, .001), (0, .002)], 'hidden_target': (0, 0)}
    result = generate({'id': 'br_private'}, record, rn, [], 81, 3)
    assert result['status'] == 'ok'
    assert result['metrics']['generation_p95_ms'] >= 0
    assert 'hidden_target' not in json.dumps(result['public'])
