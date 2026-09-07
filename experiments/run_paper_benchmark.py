"""Second research cycle: held-out SUMO seeds, stronger attacks, budgeted dummies.

Five disjoint vehicle splits: model prior, shadow, attack selection, mechanism
selection, test. Selection never reads test labels. V1 results remain intact.
"""
import argparse
from collections import defaultdict
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import platform
import time

import numpy as np

from benchmark.engines.budgeted import BudgetedReachableDummy
from core.demo_protocol import TrajectoryPoint
from data.sumo_demo import (SumoSmokeConfig, run_sumo_smoke_demo, parse_fcd,
                           load_sumo_road_network, _existing_default_osm)
from data.paper_scenarios import records_for_study, study_catalogue
from evaluation.scenario_metrics import PoiService, read_osm_pois, RoadTravelTimes, ShadowKnnAttack, stable_track_validity
from evaluation.research_protocol import full_features, attack_candidates, target_errors, utility_metrics
from experiments.run_report_demo import METHODS, make_model, protect, population_prior, digest
from experiments.rng_util import rng_from_key

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "artifacts/benchmarks/paper_benchmark"
CASES = ("S1", "S2", "S3", "S9", "S10")
BASE = [{"id": name} for name in METHODS]
BR_BASE = [{"id": "br_fresh", "anchor_mode": "fresh", "theta_m": 200., "offset_m": 80.},
           {"id": "br_private", "anchor_mode": "private_reuse", "theta_m": 200., "offset_m": 80.}]
GRID = [{"id": f"br_p{theta}_o{offset}", "anchor_mode": "private_reuse", "theta_m": float(theta), "offset_m": float(offset)}
        for theta in (100, 200, 400) for offset in (40, 120)]


def generate(spec, record, rn, training, seed, k):
    real = tuple(TrajectoryPoint(t, *p) for p, t in zip(record["points_input"], record["times"]))
    if spec["id"] == "anotherme_adaptation" and record["scenario"] != "S3":
        return {"status": "not_applicable", "error": "Complete moving-route contract required"}
    keyed = int(rng_from_key(seed, record["record_id"], spec["id"], k, schema="paper-v2-row").integers(0, 2**31))
    started = time.perf_counter()
    try:
        if spec["id"].startswith("br_"):
            model = BudgetedReachableDummy(rn, budget=.24, horizon=12, k=k,
                                          rng=rng_from_key(keyed, schema="br-dummy-v2"),
                                          **{a: b for a, b in spec.items() if a != "id"})
        else:
            model = make_model(spec["id"], rn, training, keyed, k, .02)
        init_ms = (time.perf_counter() - started) * 1000
        started = time.perf_counter()
        public = protect(spec["id"], real, rn, model, keyed, k, .02)
        elapsed = (time.perf_counter() - started) * 1000
        # Fixed six-category service contract, public for all methods.
        public["query_categories"] = ["cafe", "clinic", "fuel", "hospital", "pharmacy", "restaurant"]
        metrics = {"model_initialization_ms": init_ms, "amortized_generation_ms": elapsed / len(real),
                   "generation_p50_ms": float(np.median(model.step_ms)) if isinstance(model, BudgetedReachableDummy) else None,
                   "generation_p95_ms": float(np.quantile(model.step_ms, .95)) if isinstance(model, BudgetedReachableDummy) else None,
                   "privacy_budget_bound": model.spent_bound if isinstance(model, BudgetedReachableDummy) else None}
        return {"status": "ok", "public": public, "metrics": metrics}
    except (ValueError, RuntimeError, IndexError) as exc:
        return {"status": "failed", "error": f"{type(exc).__name__}: {exc}"}


def fit_attacks(spec, records, splits, rn, training, seed, k, travel, population):
    fitted, outcomes = {}, {}
    for scenario in CASES:
        x, y = [], []
        for r in records:
            if r["scenario"] != scenario or r["vehicle_id"] not in splits["shadow"]:
                continue
            out = generate(spec, r, rn, training, seed, k)
            if out["status"] != "ok":
                continue
            x.extend(full_features(out["public"], rn))
            y.extend([rn.point_xy(*r["hidden_target"])] * len(r["points"]) if "hidden_target" in r
                     else [rn.point_xy(*p) for p in r["points"]])
        fitted[scenario] = ShadowKnnAttack(x, y, neighbors=3) if x else None
        losses, hits = defaultdict(list), defaultdict(list)
        for r in records:
            if r["scenario"] != scenario or r["vehicle_id"] not in splits["attack_validation"]:
                continue
            out = generate(spec, r, rn, training, seed, k)
            if out["status"] != "ok":
                continue
            for name, prediction in attack_candidates(out["public"], rn, scenario, travel, population, fitted[scenario]).items():
                error = target_errors(prediction, r, rn)
                losses[name].append(float(error.mean()))
                hits[name].append(float((error <= 100).mean()))
        outcomes[scenario] = {
            "mae_attack": min(losses, key=lambda key: (np.mean(losses[key]), key)) if losses else "centroid",
            "hit_attack": min(hits, key=lambda key: (-np.mean(hits[key]), key)) if hits else "centroid",
            "validation_mae": {n: float(np.mean(v)) for n, v in losses.items()},
            "validation_hit100": {n: float(np.mean(v)) for n, v in hits.items()},
            "valid_validation_records": max(map(len, losses.values()), default=0)}
    return fitted, outcomes


def evaluate(spec, record, rn, training, seed, k, travel, population, fitted, selection, service):
    out = generate(spec, record, rn, training, seed, k)
    row = {"seed": seed, "method": spec["id"], "k": k, "scenario": record["scenario"],
           "record_id": record["record_id"], "status": out["status"], "metrics": {},
           "poi_reference_evaluable": any(service.query(p, c) for p in record["points"] for c in service.categories)}
    if out["status"] != "ok":
        return {**row, "error": out["error"]}
    public = out["public"]
    scenario = record["scenario"]
    predictions = attack_candidates(public, rn, scenario, travel, population, fitted[scenario])
    choice = selection[scenario]
    main_attack, hit_attack = choice["mae_attack"], choice["hit_attack"]
    estimates = predictions[main_attack]
    errors = target_errors(estimates, record, rn)
    hit_errors = target_errors(predictions[hit_attack], record, rn)
    utility = utility_metrics(service, public, record["points"])
    metrics = {**out["metrics"], "location_mae_m": float(errors.mean()),
               "location_median_error_m": float(np.median(errors)), "location_p90_error_m": float(np.quantile(errors, .9)),
               "location_hit_100m": float((hit_errors <= 100).mean()),
               "location_hit_50m": float((hit_errors <= 50).mean()), "location_hit_200m": float((hit_errors <= 200).mean()),
               "per_event_error_m": errors.tolist(), "hit_decoder_errors_m": hit_errors.tolist(),
               "poi_recall_at_k": utility["poi_recall_at_5"], "poi_complete_rate": utility["poi_complete_rate"],
               "poi_extra_distance_m": utility["poi_extra_distance_m"],
               "directed_track_validity": stable_track_validity(public, rn, travel),
               "coordinates_per_request": float(np.mean([len(e["candidates"]) for e in public["events"]])),
               "unique_coordinate_ratio": float(np.mean([len({(c['lat'], c['lon']) for c in e['candidates']}) / len(e['candidates']) for e in public['events']])),
               "payload_json_bytes_per_request": (len(json.dumps({"events": public["events"], "query_categories": public["query_categories"]}, separators=(",", ":")).encode())
                    + sum(u["response_json_bytes"] for u in utility["poi_rows"])) / len(record["points"]),
               "prior_mae_m": float(target_errors(predictions["prior"], record, rn).mean())}
    row.update(public=public, metrics=metrics, truth=record["points"], truth_input=record["points_input"],
               attack_xy=estimates.tolist(), hit_attack_xy=predictions[hit_attack].tolist(),
               attacker_selected=main_attack, hit_attacker_selected=hit_attack, utility=utility,
               hidden_target=record.get("hidden_target"), checks=record["checks"])
    return row


def summarize(rows):
    grouped = defaultdict(list)
    for r in rows:
        grouped[r["scenario"], r["method"], r["k"]].append(r)
    results = []
    for (case, method, k), group in sorted(grouped.items()):
        good = [r for r in group if r["status"] == "ok"]
        s = {"scenario": case, "method": method, "k": k, "attempted": len(group), "completed": len(good),
             "not_applicable": sum(r["status"] == "not_applicable" for r in group), "failures": sum(r["status"] == "failed" for r in group)}
        fields = ["location_mae_m", "location_median_error_m", "location_hit_100m", "location_hit_50m", "location_hit_200m",
                  "poi_recall_at_k", "poi_extra_distance_m", "poi_complete_rate", "coordinates_per_request",
                  "directed_track_validity", "amortized_generation_ms", "generation_p50_ms", "generation_p95_ms",
                  "payload_json_bytes_per_request", "unique_coordinate_ratio", "privacy_budget_bound"]
        for field in fields:
            values = [r["metrics"][field] for r in good if r["metrics"].get(field) is not None]
            s[field] = float(np.mean(values)) if values else None
            s[field + "_n"] = len(values)
        # Equal trip weights; no pseudo-replication across consecutive events.
        s["location_mae_sd_m"] = float(np.std([r["metrics"]["location_mae_m"] for r in good], ddof=1)) if len(good) > 1 else None
        delivered = [r["metrics"]["poi_recall_at_k"] if r["status"] == "ok" else 0.
                     for r in group if r["status"] != "not_applicable" and r["poi_reference_evaluable"]]
        delivered = [v for v in delivered if v is not None]
        s["poi_delivered_recall_at_k"] = float(np.mean(delivered)) if delivered else None
        results.append(s)
    return results


def choose_mechanism(options):
    feasible = [o for o in options if o["min_case_recall"] >= .9]
    if feasible:
        return min(feasible, key=lambda o: (o["macro_hit100"], -o["min_case_recall"], o["spec"]["id"])), True
    return min(options, key=lambda o: (-o["min_case_recall"], o["macro_hit100"], o["spec"]["id"])), False


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", nargs="+", type=int, default=[81, 82, 83])
    parser.add_argument("--k", nargs="+", type=int, default=[3, 5])
    parser.add_argument("--output", type=Path, default=OUTPUT)
    args = parser.parse_args(argv)
    if len(set(args.seeds)) != len(args.seeds) or len(set(args.k)) != len(args.k) or any(k < 2 for k in args.k):
        parser.error("Unique seeds and K >=2 required")
    rows, manifests, samples, causal = [], [], [], []
    roads = None
    for seed in args.seeds:
        print(f"SUMO held-out seed {seed}", flush=True)
        config = SumoSmokeConfig(route_seed=seed, simulation_seed=seed, demand_end_s=480,
                                 simulation_end_s=1800, max_points=2000, planned_stop_duration_s=180)
        work = ROOT / "cache/paper_benchmark" / f"seed_{seed}"
        simulation = run_sumo_smoke_demo(workdir=work, config=config)
        rn = load_sumo_road_network(simulation.network_path)
        travel = RoadTravelTimes(rn)
        service = PoiService(rn, read_osm_pois(_existing_default_osm(), config.bbox), k=5)
        records, rejected = records_for_study(parse_fcd(work / "beijing_smoke.fcd.xml"), rn, work / "beijing_smoke.vehroute.xml")
        users = sorted({r["vehicle_id"] for r in records if r["scenario"] == "S3"})
        if len(users) < 24:
            raise RuntimeError(f"Need 24 qualifying users, got {len(users)}; do not silently change split")
        splits = {"test": users[:4], "shadow": users[4:10], "attack_validation": users[10:14],
                  "defense_validation": users[14:18], "model_train": users[18:]}
        training = [{"points": r["points"], "times": r["times"]} for r in records if r["scenario"] == "S3" and r["vehicle_id"] in splits["model_train"]]
        population = population_prior(rn, training)
        for r in records:
            r["points_input"] = [rn.latlon(rn.nearest(*p)[0]) for p in r["points"]]
            if r["vehicle_id"] in splits["test"]:
                samples.append({"seed": seed, **r})
        manifest = {"seed": seed, "projection_lat0": rn.proj.lat0, "config": asdict(config), "split": splits,
                    "provenance": simulation.provenance.to_dict(), "rejected": rejected, "attacker_selection": [],
                    "defender_selection": [], "pois": {"accepted": len(service.pois), "excluded": len(service.excluded), "categories": service.categories}}
        manifests.append(manifest)
        if roads is None:
            roads = [[list(p) for p in edge["geometry"].coords] for _, _, edge in rn.graph.edges(data=True)]
        for k in args.k:
            fitted_specs, options = {}, []
            for spec in BASE + BR_BASE + GRID:
                print(f"  K={k} fit/select {spec['id']}", flush=True)
                fitted, selected = fit_attacks(spec, records, splits, rn, training, seed, k, travel, population)
                fitted_specs[spec["id"]] = fitted, selected
                manifest["attacker_selection"].append({"spec": spec, "k": k, "cases": selected})
                if spec in GRID:
                    validation_rows = [evaluate(spec, r, rn, training, seed, k, travel, population, fitted, selected, service)
                                       for r in records if r["vehicle_id"] in splits["defense_validation"]]
                    summaries = summarize(validation_rows)
                    if len(summaries) != len(CASES) or any(s["failures"] or s["completed"] == 0 for s in summaries):
                        raise RuntimeError("Incomplete mechanism-selection coverage")
                    options.append({"spec": spec, "min_case_recall": min(s["poi_delivered_recall_at_k"] for s in summaries),
                                    "macro_hit100": float(np.mean([s["location_hit_100m"] for s in summaries])),
                                    "validation_summary": summaries})
            winner, feasible = choose_mechanism(options)
            manifest["defender_selection"].append({"k": k, "candidates": options, "selected": winner["spec"], "utility_feasible": feasible,
                                                   "rule": "min case Recall>=.9 then minimize macro Hit100; else max min-case Recall"})
            for spec in BASE + BR_BASE + [winner["spec"]]:
                fitted, selected = fitted_specs[spec["id"]]
                label = "br_selected" if spec is winner["spec"] else spec["id"]
                print(f"  K={k} TEST {label}", flush=True)
                for record in records:
                    if record["vehicle_id"] not in splits["test"]:
                        continue
                    row = evaluate(spec, record, rn, training, seed, k, travel, population, fitted, selected, service)
                    row.update(id=f"run_{len(rows):05d}", method=label, selected_spec=spec)
                    rows.append(row)
                    if record["vehicle_id"] == splits["test"][0] and len(record["points"]) > 2 and row["status"] == "ok":
                        n = len(record["points"]) // 2
                        prefix_record = {**record, "points": record["points"][:n], "points_input": record["points_input"][:n], "times": record["times"][:n]}
                        prefix = generate(spec, prefix_record, rn, training, seed, k)
                        causal.append({"method": label, "scenario": record["scenario"], "seed": seed, "k": k,
                                       "prefix_invariance": prefix["status"] == "ok" and row["public"]["events"][:n] == prefix["public"]["events"]})
    paths = sorted({*ROOT.glob("benchmark/**/*.py"), *ROOT.glob("core/*.py"), *ROOT.glob("data/*.py"),
                    *ROOT.glob("evaluation/*.py"), ROOT / "experiments/run_report_demo.py",
                    ROOT / "experiments/rng_util.py", Path(__file__).resolve()})
    payload = {"schema": "report-demo-v1", "protocol_version": "paper-v2", "scenario_catalogue": study_catalogue(),
               "status": "controlled_comparative_study_not_sota_reproduction", "seeds": args.seeds, "k_values": args.k,
               "runtime": {"python": platform.python_version(), "platform": platform.platform(), "machine": platform.machine()},
               "source_sha256": {str(p.relative_to(ROOT)): digest(p) for p in paths}, "manifests": manifests,
               "rows": rows, "dataset_records": samples, "summary": summarize(rows), "roads": roads, "causal_checks": causal,
               "limitations": ["S1–S3 và S9–S10 có thực nghiệm; S4–S8 chưa có kết quả.",
                   "S9/S10: điểm FCD đầu/cuối của chuyến SUMO hoàn tất, không phải nhà hay danh tính thật.",
                   "Đối thủ MAE và Hit được chọn riêng trên tập chọn đối thủ; không phải Bayes tối ưu.",
                   "Bộ lọc toàn chuỗi vẫn dùng mô hình quan sát gần đúng, chưa xét đủ ngữ nghĩa/lịch sử nhiều ngày.",
                   "Ràng buộc đồ thị nút giao không thay thế ràng buộc cạnh/làn/cấm rẽ của SUMO.",
                   "Ba comparator vẫn là bản thích nghi, không phải tái lập kết quả paper.",
                   "Ngưỡng Recall .9 là yêu cầu thử nghiệm của luận văn; cần báo cả trường hợp không đạt trên dữ liệu mới.",
                   "p95 chỉ đo từng bước của BR-Dummy trên máy phát triển; không phải p95 mạng hay điện thoại."]}
    args.output.mkdir(parents=True, exist_ok=True)
    target = args.output / "results.json"
    target.write_text(json.dumps(payload, ensure_ascii=False, allow_nan=False, separators=(",", ":")))
    target.with_suffix(".sha256").write_text(digest(target) + "\n")
    print(f"Wrote {target}: {len(rows)} rows", flush=True)
    return payload


if __name__ == "__main__":
    main()
