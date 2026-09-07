"""Scientific contract regression checks for the report pilot."""
import json
import hashlib
from pathlib import Path

import networkx as nx
import numpy as np
import pytest

from core.road_network import RoadNetwork
from data.sumo_demo import FCDSample
from data.threat_scenarios import catalogue, records_from_fcd
from evaluation.scenario_metrics import PoiService, ShadowKnnAttack, attack_scores, public_features, read_osm_pois, continuity_estimates
from experiments.run_report_demo import protect, summary_rows


def road():
    graph = nx.DiGraph()
    for i in range(6):
        graph.add_node(i, x=i * .001, y=0.)
        if i:
            graph.add_edge(i - 1, i, length=100)
            graph.add_edge(i, i - 1, length=100)
    return RoadNetwork(graph)


def event(points, t=0):
    return {"event_id": str(t), "timestamp_s": t,
            "candidates": [{"candidate_id": str(i), "lat": p[0], "lon": p[1]}
                           for i, p in enumerate(points)]}


def test_catalogue_has_ten_distinct_targets_and_honest_status():
    c = catalogue()
    assert [s["id"] for s in c] == [f"S{i}" for i in range(1, 11)]
    assert [s["id"] for s in c if s["status"] == "runnable"] == ["S1", "S2", "S3"]


def test_scenario_eligibility_requires_actual_dwell_and_movement():
    rn = road()
    trace = tuple(FCDSample(t, 0, 0, 0., "e", "e_0") for t in range(181))
    trace += tuple(FCDSample(t, 0, (t - 180) * .00001, 1., "e", "e_0")
                   for t in range(181, 300))
    records, rejected = records_from_fcd({"u": trace}, rn)
    assert not rejected
    assert {r["scenario"] for r in records} == {"S1", "S2", "S3"}
    dwell = next(r for r in records if r["scenario"] == "S2")
    assert dwell["checks"]["span_s"] >= 120
    assert dwell["checks"]["max_radius_from_first_m"] == 0
    assert len(next(r for r in records if r["scenario"] == "S1")["points"]) == 1
    assert records_from_fcd({"u": trace[181:]}, rn)[0] == []


def test_poi_recall_is_after_union_dedup_and_real_local_ranking():
    rn = road()
    pois = [{"id": str(i), "lat": 0, "lon": i * .001, "category": "cafe"}
            for i in range(6)]
    service = PoiService(rn, pois, k=2)
    public = {"events": [event([(0, 0), (0, .005), (0, 0)])]}
    result = service.evaluate(public, [(0, 0)], "cafe")
    assert result["poi_recall_at_k"] == 1
    assert result["poi_rows"][0]["returned"] == ["0", "1"]
    remote = service.evaluate({"events": [event([(0, .005)])]}, [(0, 0)], "cafe")
    assert remote["poi_recall_at_k"] == 0
    assert service.evaluate(public, [(0, 0)], "pharmacy")["poi_recall_at_k"] is None
    with pytest.raises(ValueError):
        service.evaluate(public, [], "cafe")


def test_directed_unreachable_pois_are_not_nearest():
    rn = road()
    rn.graph.remove_edge(0, 1)
    service = PoiService(rn, [{"id": "x", "lat": 0, "lon": .001, "category": "cafe"}])
    assert service.query((0, 0), "cafe") == []


def test_poi_parser_does_not_invent_or_read_geolife(tmp_path):
    source = tmp_path / "map.osm"
    source.write_text('<osm><node id="1" lat="0" lon="0"><tag k="amenity" v="cafe"/></node>'
                      '<node id="2" lat="20" lon="20"><tag k="amenity" v="cafe"/></node></osm>')
    pois = read_osm_pois(source, (-1, -1, 1, 1))
    assert [p["id"] for p in pois] == ["osm/node/1"]


def test_public_feature_prefix_and_estimator_no_truth_parameter():
    rn = road()
    p = {"events": [event([(0, .001)], 0), event([(0, .002)], 20)]}
    for scenario in ("S1", "S2", "S3"):
        assert np.allclose(public_features(p, rn, scenario)[:1],
                           public_features({"events": p["events"][:1]}, rn, scenario))
    decoder = ShadowKnnAttack([[0, 0], [1, 1]], [[5, 6], [7, 8]], neighbors=1)
    assert decoder.predict([[0, 0]]).tolist() == [[5, 6]]
    with pytest.raises(ValueError):
        ShadowKnnAttack([], [])


def test_errors_units_denominators_and_failed_rows():
    result = attack_scores([[3, 4], [0, 0]], [[0, 0], [0, 0]])
    assert result["location_mae_m"] == 2.5
    assert result["location_hit_100m"] == 1
    with pytest.raises(ValueError):
        attack_scores([], [])
    summaries = summary_rows([
        {"scenario": "S1", "method": "x", "k": 3, "status": "failed", "metrics": {}},
        {"scenario": "S1", "method": "x", "k": 3, "status": "ok",
         "metrics": {"poi_recall_at_k": 0.5}},
    ])
    assert summaries[0]["attempted"] == 2
    assert summaries[0]["failures"] == 1
    assert summaries[0]["poi_recall_at_k_n"] == 1


def test_uniform_control_prefix_and_public_separation():
    from core.demo_protocol import TrajectoryPoint
    rn = road()
    points = tuple(TrajectoryPoint(i * 20, *rn.latlon(i)) for i in range(4))
    a = protect("uniform_dummy", points, rn, None, 7, 3, .02)
    b = protect("uniform_dummy", points[:2], rn, None, 7, 3, .02)
    assert a["events"][:2] == b["events"]
    assert "truth" not in a
    assert "real_candidate_ids" not in json.dumps(a)


def test_continuity_attack_uses_no_future_and_finds_stationary_candidate():
    rn = road()
    p = {"events": [event([(0, .001), (0, .005)], 0),
                    event([(0, .001), (0, 0)], 20),
                    event([(0, .001), (0, .004)], 40)]}
    result = continuity_estimates(p, rn)
    assert np.allclose(result[1:], [rn.point_xy(0, .001)] * 2)
    assert np.allclose(result[:2], continuity_estimates({"events": p["events"][:2]}, rn))


def test_failed_service_is_zero_but_empty_reference_not_applicable():
    rows = [{"scenario": "S1", "method": "x", "k": 3, "status": "failed",
             "poi_reference_evaluable": True, "metrics": {}},
            {"scenario": "S1", "method": "x", "k": 3, "status": "ok",
             "poi_reference_evaluable": True, "metrics": {"poi_recall_at_k": 1}},
            {"scenario": "S1", "method": "x", "k": 3, "status": "failed",
             "poi_reference_evaluable": False, "metrics": {}}]
    assert summary_rows(rows)[0]["poi_delivered_recall_at_k"] == .5
    assert summary_rows(rows)[0]["poi_delivered_recall_at_k_n"] == 2


def test_report_web_separates_truth_and_fails_closed_on_checksum(tmp_path):
    from web.benchmark_app import create_app
    path = tmp_path / "results.json"
    value = {"schema": "report-demo-v1", "scenario_catalogue": [], "limitations": [],
             "roads": [], "summary": [], "causal_checks": [], "manifests": [],
             "rows": [{"id": "run_00000", "scenario": "S1", "method": "x", "k": 3,
                       "status": "ok", "public": {"events": []},
                       "truth": [[1, 2]], "record_id": "secret", "seed": 71}]}
    path.write_text(json.dumps(value))
    path.with_suffix(".sha256").write_text(hashlib.sha256(path.read_bytes()).hexdigest())
    app = create_app({"TESTING": True, "REPORT_DEMO_RESULTS_PATH": str(path)})
    client = app.test_client()
    assert client.get("/report-demo").status_code == 200
    assert client.get("/api/report-demo/evaluation").status_code == 404
    public = client.get("/api/report-demo/runs/run_00000")
    assert set(public.json) == {"id", "scenario", "method", "k", "status", "public"}
    assert public.headers["X-Benchmark-Visibility"] == "public-transcript"
    assert "secret" not in client.get("/api/report-demo").text
    assert client.get("/api/report-demo/runs/missing").status_code == 404
    app.config["BENCHMARK_ENABLE_EVALUATOR_VIEW"] = True
    assert client.get("/api/report-demo/evaluation").json["rows"][0]["truth"] == [[1, 2]]
    path.write_text("{}")
    assert client.get("/api/report-demo").status_code == 503


def test_dls_pool_entropy_truth_separation_and_causality():
    from benchmark.engines.dls import DLSGraph, entropy
    from core.demo_protocol import TrajectoryPoint
    rn = road()
    q = np.arange(1, len(rn) + 1, dtype=float)
    for true in range(len(rn)):
        model = DLSGraph(rn, q, k=3, rng=np.random.default_rng(4))
        pool = model.candidate_pool(true)
        assert true not in pool and len(set(pool)) == len(rn) - 1
        ids, score = model.select(true)
        assert true in ids and len(set(ids)) == 3
        assert 0 <= score <= np.log2(3)
    assert entropy([1, 1, 1]) == pytest.approx(np.log2(3))
    points = tuple(TrajectoryPoint(i * 20, *rn.latlon(i)) for i in range(4))
    whole = DLSGraph(rn, q, rng=np.random.default_rng(7)).protect_run(points).to_attacker_dict()
    prefix = DLSGraph(rn, q, rng=np.random.default_rng(7)).protect_run(points[:2]).to_attacker_dict()
    assert whole["events"][:2] == prefix["events"]
    assert "true" not in json.dumps(whole)
    with pytest.raises(ValueError):
        DLSGraph(rn, np.zeros(len(rn)))


def test_streaming_equals_batch_and_directed_road_constraint():
    from benchmark.engines.proposed import GeoIAnchoredDummyEngine
    from evaluation.scenario_metrics import RoadTravelTimes, stable_track_validity
    from core.demo_protocol import TrajectoryPoint
    rn = road()
    for i in range(1, 6):
        rn.graph.remove_edge(i, i - 1)
    points = [rn.latlon(i) for i in range(4)]
    times = [0, 20, 40, 60]
    def model():
        return GeoIAnchoredDummyEngine(.02, rn, road_constrained=True, rng=np.random.default_rng(3))
    batch = model().protect_trajectory(points, times)
    streaming = model()
    steps = [streaming.protect_step(*p, t) for p, t in zip(points, times)]
    assert tuple(s[0] for s in steps) == batch.anchors
    assert tuple(tuple(s[1][j] for s in steps) for j in range(3)) == batch.trajectories
    assert model().protect_trajectory(points[:2], times[:2]).trajectories == tuple(t[:2] for t in batch.trajectories)
    public = model().protect_run([TrajectoryPoint(t, *p) for t, p in zip(times, points)]).to_attacker_dict()
    assert stable_track_validity(public, rn, RoadTravelTimes(rn)) == 1.
    with pytest.raises(ValueError):
        streaming.protect_step(*points[-1], 60)


def test_road_attack_is_causal_and_rejects_reverse_path():
    from evaluation.scenario_metrics import RoadTravelTimes, road_filter_estimates, stable_track_validity
    rn = road()
    for i in range(1, 6):
        rn.graph.remove_edge(i, i - 1)
    travel = RoadTravelTimes(rn)
    assert rn.node_ids[0] not in travel.reachable(1, 100)
    p = {"output_kind": "real_plus_dummies", "events": [event([(0, .005)], 0), event([(0, 0)], 20)]}
    estimates, restarts = road_filter_estimates(p, rn, travel, np.ones(len(rn)) / len(rn))
    prefix, _ = road_filter_estimates({**p, "events": p["events"][:1]}, rn, travel, np.ones(len(rn)) / len(rn))
    assert np.array_equal(prefix, estimates[:1])
    assert restarts == 1
    assert stable_track_validity(p, rn, travel) == 0
    p["events"][1]["candidates"][0]["candidate_id"] = "unlinked"
    assert stable_track_validity(p, rn, travel) is None


def test_invalid_fcd_and_sampling_window_are_rejected():
    rn = road()
    with pytest.raises(ValueError):
        records_from_fcd({}, rn, max_events=3)
    trace = [FCDSample(0, 0, 0, 0., "e", "e_0"), FCDSample(180, 0, 0, 0., "e", "e_0")]
    assert records_from_fcd({"u": trace}, rn)[1][0]["reason"] == "non_contiguous_1hz_fcd"


def test_not_applicable_is_not_execution_failure_or_zero_utility():
    row = {"scenario": "S2", "method": "offline", "k": 3, "status": "not_applicable",
           "poi_reference_evaluable": True, "metrics": {}}
    summary = summary_rows([row])[0]
    assert summary["not_applicable"] == 1 and summary["failures"] == 0
    assert summary["poi_delivered_recall_at_k"] is None
