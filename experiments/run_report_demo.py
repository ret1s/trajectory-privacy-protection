"""Reproducible S1/S2/S3 pilot: actual SUMO, OSM POIs, shadow attacks.

Run: python -m experiments.run_report_demo --seeds 71 72 73
This release is a bounded pilot, never a paper-equivalent SOTA leaderboard.
"""
import argparse
from collections import defaultdict
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import time

import numpy as np
from benchmark.engines.dls import DLSGraph

from benchmark.methods import (AnotherMeAdaptation, GeoIAnchoredDummyTrajectories,
                               SemanticCorrelationComparator, TransProtectAdaptation)
from core.demo_protocol import (TrajectoryPoint, make_replacement_run,
                                make_real_plus_dummies_run)
from core.mechanisms import RoadExponential
from data.sumo_demo import (SumoSmokeConfig, run_sumo_smoke_demo, parse_fcd,
                           load_sumo_road_network, _existing_default_osm)
from data.threat_scenarios import TAXONOMY_VERSION, catalogue, records_from_fcd
from evaluation.scenario_metrics import (PoiService, read_osm_pois, public_features,
                                         ShadowKnnAttack, attack_scores, continuity_estimates,
                                         RoadTravelTimes, road_filter_estimates, stable_track_validity)
from experiments.rng_util import rng_from_key

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "artifacts/benchmarks/report_demo"
METHODS = ("unprotected", "uniform_dummy", "dls_graph_adaptation", "rem_anchor_only",
           "transprotect_adaptation", "anotherme_adaptation",
           "semantic_correlation_local_adaptation", "geo_i_anchored_dummy",
           "geo_i_anchored_dummy_road")


def population_prior(rn, training):
    counts = np.ones(len(rn))  # Laplace smoothing over the fixed public catalog.
    for trajectory in training:
        for point in trajectory["points"]:
            counts[rn.nearest(*point)[0]] += 1
    return counts / counts.sum()


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def make_model(name, rn, training, seed, k, epsilon):
    rng = rng_from_key(seed, name, k, schema="report-demo-v1")
    if name == "dls_graph_adaptation":
        return DLSGraph(rn, population_prior(rn, training), k=k, rng=rng)
    if name == "transprotect_adaptation":
        return TransProtectAdaptation.from_road_network(
            rn, training_trajectories=training, candidate_k=10, target_count=8,
            alpha=10000, epsilon=0.005, rng=rng)
    if name == "anotherme_adaptation":
        return AnotherMeAdaptation(rn, rng=rng)
    if name == "semantic_correlation_local_adaptation":
        return SemanticCorrelationComparator(rn, k=k, rng=rng)
    if name in {"geo_i_anchored_dummy", "geo_i_anchored_dummy_road"}:
        return GeoIAnchoredDummyTrajectories(
            epsilon, rn, k=k, road_constrained=name.endswith("_road"), rng=rng)
    return None


def protect(name, points, rn, model, seed, k, epsilon):
    if model is not None:
        return model.protect_run(points).to_attacker_dict()
    rng = rng_from_key(seed, name, k, schema="report-demo-baseline-v1")
    if name in {"unprotected", "rem_anchor_only"}:
        released = points
        if name == "rem_anchor_only":
            mechanism = RoadExponential(epsilon, rn, rng=rng)
            released = tuple(TrajectoryPoint(p.timestamp_s, *mechanism.perturb(p.lat, p.lon))
                             for p in points)
        return make_replacement_run(name, points, released).to_attacker_dict()
    # Foundational uniform dummy control, deliberately NOT labelled DLS/RDG.
    real_slot = int(rng.integers(k))
    tracks = {f"candidate_{j:04d}": [] for j in range(k)}
    for point in points:
        true_vertex = rn.nearest(point.lat, point.lon)[0]
        ids = rng.choice(np.delete(np.arange(len(rn)), true_vertex), k - 1, replace=False)
        locations = iter(rn.latlon(i) for i in ids)
        for j, track in enumerate(tracks.values()):
            track.append(point if j == real_slot else
                         TrajectoryPoint(point.timestamp_s, *next(locations)))
    return make_real_plus_dummies_run(name, points, tracks,
                                     f"candidate_{real_slot:04d}").to_attacker_dict()


def prefix_check(name, points, rn, training, seed, k, epsilon):
    if len(points) < 3:
        return None
    n = max(1, len(points) // 2)
    whole = protect(name, points, rn, make_model(name, rn, training, seed, k, epsilon),
                    seed, k, epsilon)
    prefix = protect(name, points[:n], rn, make_model(name, rn, training, seed, k, epsilon),
                     seed, k, epsilon)
    return whole["events"][:n] == prefix["events"]


def summary_rows(rows):
    groups = defaultdict(list)
    for row in rows:
        groups[(row["scenario"], row["method"], row["k"])].append(row)
    result = []
    for (scenario, method, k), group in sorted(groups.items()):
        good = [r for r in group if r["status"] == "ok"]
        item = {"scenario": scenario, "method": method, "k": k,
                "attempted": len(group), "completed": len(good),
                "failures": sum(r["status"] == "failed" for r in group),
                "not_applicable": sum(r["status"] == "not_applicable" for r in group)}
        delivered = [r["metrics"].get("poi_recall_at_k", 0.) if r["status"] == "ok" else 0.
                     for r in group if r.get("poi_reference_evaluable", False)
                     and r["status"] != "not_applicable"]
        delivered = [v for v in delivered if v is not None]
        item["poi_delivered_recall_at_k"] = float(np.mean(delivered)) if delivered else None
        item["poi_delivered_recall_at_k_n"] = len(delivered)
        for field in ("location_mae_m", "location_hit_100m", "prior_mae_m",
                      "centroid_mae_m", "poi_recall_at_k", "coordinates_per_request",
                      "amortized_generation_ms", "payload_json_bytes_per_request",
                      "directed_track_validity", "represented_location_mae_m",
                      "model_initialization_ms", "quantization_mean_m"):
            vals = [r["metrics"][field] for r in good if r["metrics"].get(field) is not None]
            item[field] = float(np.mean(vals)) if vals else None
            item[field + "_n"] = len(vals)
        errors = [r["metrics"]["location_mae_m"] for r in good
                  if r["metrics"].get("location_mae_m") is not None]
        item["location_mae_m_sd"] = float(np.std(errors, ddof=1)) if len(errors) > 1 else None
        result.append(item)
    return result


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", nargs="+", type=int, default=[71, 72, 73])
    parser.add_argument("--k", nargs="+", type=int, default=[3, 5])
    parser.add_argument("--epsilon", type=float, default=0.02)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    args = parser.parse_args(argv)
    if (len(set(args.seeds)) != len(args.seeds) or len(set(args.k)) != len(args.k)
            or any(k < 2 for k in args.k) or not np.isfinite(args.epsilon) or args.epsilon <= 0):
        parser.error("unique seeds/K, K >= 2 and finite positive epsilon required")
    destination = args.output.resolve()
    destination.mkdir(parents=True, exist_ok=True)
    all_rows, manifests, samples, causal = [], [], [], []
    source = _existing_default_osm()
    pois = read_osm_pois(source, SumoSmokeConfig().bbox)
    roads = None
    for seed in args.seeds:
        print(f"SUMO seed {seed}", flush=True)
        config = SumoSmokeConfig(route_seed=seed, simulation_seed=seed,
                                 demand_end_s=180, simulation_end_s=1400,
                                 max_points=1000, planned_stop_duration_s=180)
        work = ROOT / "cache/report_demo" / f"seed_{seed}"
        simulation = run_sumo_smoke_demo(workdir=work, config=config)
        rn = load_sumo_road_network(simulation.network_path)
        travel = RoadTravelTimes(rn)
        service = PoiService(rn, pois, k=5)
        if not service.categories:
            raise RuntimeError("No qualifying OSM POIs: no synthetic fallback")
        traces = parse_fcd(work / "beijing_smoke.fcd.xml")
        records, rejected = records_from_fcd(traces, rn)
        users = sorted({r["vehicle_id"] for r in records})
        if len(users) < 9:
            raise RuntimeError("Need >=9 qualifying SUMO vehicles for held-out pilot")
        # Fixed split before model or attack results are observed.
        test_ids, shadow_ids, validation_ids, model_ids = users[:2], users[2:6], users[6:8], users[8:]
        if not model_ids:
            raise RuntimeError("No disjoint population training users")
        training = [{"points": r["points"], "times": r["times"]}
                    for r in records if r["vehicle_id"] in model_ids and r["scenario"] == "S3"]
        population = population_prior(rn, training)
        prior_center = np.average(rn.xy, axis=0, weights=population)
        manifests.append({
            "seed": seed, "config": asdict(config),
            "projection_lat0": rn.proj.lat0,
            "provenance": simulation.provenance.to_dict(),
            "split": {"test": test_ids, "shadow": shadow_ids, "validation": validation_ids,
                      "model_train": model_ids},
            "rejected": rejected,
            "pois": {"accepted": len(service.pois), "excluded_access_distance": len(service.excluded),
                     "categories": service.categories, "node_amenities_only": True,
                     "max_access_m": 250, "distance": "directed shortest road length; nearest vertex approximation"},
        })
        if roads is None:
            roads = [[list(x) for x in edge["geometry"].coords]
                     for _, _, edge in rn.graph.edges(data=True)]
        for record in records:
            record["points_input"] = [rn.latlon(rn.nearest(*p)[0]) for p in record["points"]]
            record["quantization_mean_m"] = float(np.mean([
                rn.nearest(*p)[1] for p in record["points"]]))
            if record["vehicle_id"] in test_ids:
                samples.append({"seed": seed, **record})
        for k in args.k:
            for name in METHODS:
                print(f"  K={k} {name}", flush=True)
                shadow = defaultdict(lambda: [[], []])
                for record in records:
                    if record["vehicle_id"] not in shadow_ids:
                        continue
                    real = tuple(TrajectoryPoint(t, *p) for p, t in zip(record["points_input"], record["times"]))
                    try:
                        shadow_seed = int(rng_from_key(seed, record["record_id"], name, k,
                                                       schema="shadow-row-v1").integers(0, 2**31))
                        model = make_model(name, rn, training, shadow_seed, k, args.epsilon)
                        public = protect(name, real, rn, model, shadow_seed, k, args.epsilon)
                    except (ValueError, RuntimeError, IndexError) as exc:
                        # Persist failures in manifest, not silently fabricate training.
                        manifests[-1].setdefault("shadow_failures", []).append(
                            {"method": name, "k": k, "record": record["record_id"], "error": str(exc)})
                        continue
                    features = public_features(public, rn, record["scenario"])
                    truth_xy = [rn.point_xy(*p) for p in record["points"]]
                    shadow[record["scenario"]][0].extend(features.tolist())
                    shadow[record["scenario"]][1].extend(truth_xy)
                attacks = {scenario: ShadowKnnAttack(x, y, neighbors=3)
                           for scenario, (x, y) in shadow.items() if x}
                validation_losses = defaultdict(lambda: defaultdict(list))
                for record in records:
                    if record["vehicle_id"] not in validation_ids:
                        continue
                    real = tuple(TrajectoryPoint(t, *p) for p, t in zip(record["points_input"], record["times"]))
                    validation_seed = int(rng_from_key(seed, record["record_id"], name, k,
                                                       schema="validation-row-v1").integers(0, 2**31))
                    try:
                        public = protect(name, real, rn,
                                         make_model(name, rn, training, validation_seed, k, args.epsilon),
                                         validation_seed, k, args.epsilon)
                    except (ValueError, RuntimeError, IndexError):
                        continue
                    scenario = record["scenario"]
                    truth_xy = np.array([rn.point_xy(*p) for p in record["points"]])
                    alternatives = {"centroid": public_features(public, rn, "S1"),
                                    "continuity": continuity_estimates(public, rn),
                                    "road_filter": road_filter_estimates(public, rn, travel, population)[0],
                                    "prior": np.tile(prior_center, (len(real), 1))}
                    if scenario == "S2":
                        alternatives["running_mean"] = public_features(public, rn, "S2")
                    if scenario in attacks:
                        alternatives["shadow_knn"] = attacks[scenario].predict(public_features(public, rn, scenario))
                    for attack_name, prediction in alternatives.items():
                        validation_losses[scenario][attack_name].append(
                            attack_scores(prediction, truth_xy)["location_mae_m"])
                chosen = {
                    scenario: min(losses, key=lambda key: (np.mean(losses[key]), key))
                    for scenario, losses in validation_losses.items() if losses
                }
                manifests[-1].setdefault("attacker_selection", []).append(
                    {"method": name, "k": k, "selected": chosen,
                     "validation_mae_m": {s: {a: float(np.mean(v)) for a, v in losses.items()}
                                          for s, losses in validation_losses.items()}})
                # Fresh instance for test; methods may otherwise retain private history.
                for record in records:
                    if record["vehicle_id"] not in test_ids:
                        continue
                    slot = len(all_rows)
                    row = {"id": f"run_{slot:05d}", "seed": seed, "method": name, "k": k,
                           "record_id": record["record_id"], "scenario": record["scenario"],
                           "status": "failed", "metrics": {}, "checks": record["checks"]}
                    real = tuple(TrajectoryPoint(t, *p) for p, t in zip(record["points_input"], record["times"]))
                    category = service.categories[seed % len(service.categories)]
                    row["poi_reference_evaluable"] = any(service.query(p, category) for p in record["points_input"])
                    # AnotherMe is an offline complete-route method. A single
                    # observation or a parked interval is outside its contract.
                    if name == "anotherme_adaptation" and record["scenario"] != "S3":
                        row.update(status="not_applicable", error="Offline complete moving route required")
                        all_rows.append(row)
                        continue
                    try:
                        row_seed = int(rng_from_key(seed, record["record_id"], name, k,
                                                   schema="test-row-v1").integers(0, 2**31))
                        init_started = time.perf_counter()
                        model = make_model(name, rn, training, row_seed, k, args.epsilon)
                        init_ms = (time.perf_counter() - init_started) * 1000
                        started = time.perf_counter()
                        public = protect(name, real, rn, model, row_seed, k, args.epsilon)
                        elapsed = (time.perf_counter() - started) * 1000
                        public["query_category"] = category
                        features = public_features(public, rn, record["scenario"])
                        sx, sy = shadow[record["scenario"]]
                        attack_name = chosen.get(record["scenario"], "centroid")
                        estimates = public_features(public, rn, "S1")
                        if attack_name == "shadow_knn":
                            estimates = attacks[record["scenario"]].predict(features)
                        elif attack_name == "running_mean":
                            estimates = public_features(public, rn, "S2")
                        elif attack_name == "continuity":
                            estimates = continuity_estimates(public, rn)
                        elif attack_name == "road_filter":
                            estimates = road_filter_estimates(public, rn, travel, population)[0]
                        elif attack_name == "prior":
                            estimates = np.tile(prior_center, (len(real), 1))
                        truth_xy = np.array([rn.point_xy(*p) for p in record["points"]])
                        prior = np.tile(prior_center, (len(real), 1))
                        raw_centers = public_features(public, rn, "S1")
                        # Identity decoder reads PUBLIC coordinates, never truth.
                        if name == "unprotected":
                            estimates = raw_centers.copy()
                        utility = service.evaluate(public, record["points"], category)
                        metrics = attack_scores(estimates, truth_xy)
                        metrics.update({
                            "represented_location_mae_m": attack_scores(
                                estimates, np.array([rn.point_xy(*p) for p in record["points_input"]]))["location_mae_m"],
                            "quantization_mean_m": record["quantization_mean_m"],
                            "directed_track_validity": stable_track_validity(public, rn, travel),
                            "model_initialization_ms": init_ms,
                            "prior_mae_m": attack_scores(prior, truth_xy)["location_mae_m"],
                            "centroid_mae_m": attack_scores(raw_centers, truth_xy)["location_mae_m"],
                            "poi_recall_at_k": utility["poi_recall_at_k"],
                            "coordinates_per_request": float(np.mean([len(e["candidates"]) for e in public["events"]])),
                            "requests_per_real_query": 1.0,
                            "amortized_generation_ms": elapsed / len(real),
                            "payload_json_bytes_per_request": (
                                len(json.dumps({"events": public["events"], "query_category": category},
                                               separators=(",", ":")).encode())
                                + sum(p["response_json_bytes"] for p in utility["poi_rows"])) / len(real),
                        })
                        row.update({"status": "ok", "metrics": metrics,
                                    "public": public, "truth": record["points"],
                                    "truth_input": record["points_input"],
                                    "attack_xy": estimates.tolist(), "utility": utility,
                                    "query_category": category,
                                    "attacker_selected": attack_name,
                                    "attacker_selection_status": ("validation_selected" if record["scenario"] in chosen
                                                                  else "fixed_centroid_no_valid_validation_record"),
                                    "quantization_mean_m": record["quantization_mean_m"]})
                        if record["vehicle_id"] == test_ids[0]:
                            try:
                                passes = prefix_check(name, real, rn, training, seed, k, args.epsilon)
                            except (ValueError, RuntimeError, IndexError):
                                passes = False
                            causal.append({"method": name, "scenario": record["scenario"],
                                           "seed": seed, "k": k,
                                           "prefix_invariance": passes})
                    except (ValueError, RuntimeError, IndexError) as exc:
                        row["error"] = f"{type(exc).__name__}: {exc}"
                    all_rows.append(row)
    # Source digest makes rerun provenance independent of an uncommitted worktree.
    paths = sorted({*ROOT.glob("benchmark/**/*.py"), *ROOT.glob("core/*.py"),
                    *ROOT.glob("data/*.py"), *ROOT.glob("evaluation/*.py"),
                    Path(__file__).resolve(), ROOT / "experiments/rng_util.py"})
    source_hashes = {str(p.relative_to(ROOT)): digest(p) for p in paths}
    payload = {
        "schema": "report-demo-v1", "taxonomy_version": TAXONOMY_VERSION,
        "status": "controlled_pilot_not_sota_reproduction",
        "scenario_catalogue": catalogue(), "seeds": args.seeds, "k_values": args.k,
        "epsilon_per_m": args.epsilon, "source_sha256": source_hashes,
        "manifests": manifests, "dataset_records": samples, "rows": all_rows,
        "summary": summary_rows(all_rows), "causal_checks": causal, "roads": roads,
        "limitations": [
            "Chỉ S1–S3 đã chạy; S4–S10 mới có đặc tả, chưa được thực nghiệm.",
            "Ba đối chứng từ bài báo là bản thích nghi cục bộ, không phải bảng xếp hạng SOTA tái lập.",
            "Chọn đối thủ trên validation: tâm tập điểm, trung bình tích luỹ, liên tục, mạng đường, prior hoặc 3-NN; chưa phải đối thủ tối ưu.",
            "S3 hiện đo bộ lọc nhân quả; chưa kiểm tra đối thủ dùng toàn bộ đoạn để suy luận ngược quá khứ.",
            "Sai số chính so với FCD thật; đầu vào cơ chế chiếu tới nút giao. Chỉ số rời rạc và sai số lượng tử hoá lưu riêng.",
            "Ràng buộc đường có hướng chỉ kiểm tra thời gian tự do giữa nút giao, chưa đủ cấm rẽ, gia tốc và đèn tín hiệu.",
            "Mẫu thử nhỏ; chưa so tại cùng chất lượng dịch vụ, chưa kết luận ưu thế thống kê.",
            "POI dạng node OSM; chiếu tới đỉnh gần nhất và đồ thị nút giao chỉ xấp xỉ đường đi, chưa xét đủ cấm rẽ.",
            "Điểm đỗ do SUMO cho phép trong mô phỏng, không xác nhận được phép đỗ ngoài thực tế.",
            "Thời gian là trung bình xử lý cả đoạn, không phải độ trễ p95 của hệ trực tuyến.",
            "Dung lượng JSON chỉ ước tính tải dữ liệu, không phải lưu lượng mạng đo thực tế.",
        ],
    }
    result_path = destination / "results.json"
    result_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False))
    (destination / "results.sha256").write_text(digest(result_path) + "\n")
    print(f"Wrote {result_path}: {len(all_rows)} attempts", flush=True)
    return payload


if __name__ == "__main__":
    main()
