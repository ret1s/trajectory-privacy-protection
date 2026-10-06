"""Read frozen evaluations with native metric formulas; never refit mechanisms.

All source artifacts remain immutable. Incompatible/absent metric inputs stay
N/A. Cohorts with different clocks, attackers or service protocols are reported
separately. This adds accounting evidence, not a new superiority experiment.
"""
import argparse
from collections import defaultdict
import csv
import gzip
import hashlib
import json
import math
from pathlib import Path

from evaluation.native_comparator_metrics import comparator_native_metrics, inference_error_m


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "artifacts/benchmarks/native_metrics_20261005"
ACTIVE = {"unprotected", "dls_graph_adaptation", "transprotect_adaptation",
          "semantic_correlation_local_adaptation", "geo_i_anchored_dummy",
          "geo_i_anchored_dummy_road", "br_private", "anotherme_adaptation"}
COMMON = {"raw", "dls", "rdg", "transprotect_markov", "semantic_poi",
          "fake_queries", "anotherme_offline"}
KINDS = {"raw": "replacement_trajectory", "transprotect_markov": "replacement_trajectory",
         "anotherme_offline": "replacement_trajectory", "dls": "real_plus_dummies",
         "rdg": "real_plus_dummies", "semantic_poi": "real_plus_dummies",
         "fake_queries": "real_plus_dummies"}
SCENARIOS = ("S1", "S2", "S3", "S9", "S10")


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def projected(point, lat0):
    scale = math.pi / 180. * 6371000.
    lat, lon = point
    return lon * scale * math.cos(math.radians(lat0)), lat * scale


def paper_rows(source, k):
    lat0 = {m["seed"]: m["projection_lat0"] for m in source["manifests"]}
    rows = []
    for row in source["rows"]:
        if row["k"] != k or row["method"] not in ACTIVE:
            continue
        result = {"cohort": "paper_v2_test", "method": row["method"], "k": k,
                  "scenario": row["scenario"], "case_id": row["scenario"],
                  "record_id": row["record_id"], "family_id": f"seed-{row['seed']}",
                  "rep": 0, "status": row["status"]}
        if row["status"] != "ok":
            result["unavailable_reason"] = row.get("error", "no successful output")
            rows.append(result)
            continue
        kind = row["public"]["output_kind"]
        prediction = row["attack_xy"]
        if row["scenario"] in ("S9", "S10"):
            prediction = [prediction[0 if row["scenario"] == "S9" else -1]]
            targets = [row["hidden_target"]]
        else:
            targets = row["truth"]
        if len(prediction) != len(targets):
            raise ValueError("frozen attack/target alignment mismatch")
        errors = [math.dist(p, projected(t, lat0[row["seed"]])) for p, t in zip(prediction, targets)]
        archived = row["metrics"]["per_event_error_m"]
        if len(errors) != len(archived) or any(not math.isclose(a, b, rel_tol=1e-9, abs_tol=1e-7)
                                            for a, b in zip(errors, archived)):
            raise ValueError("frozen attack errors do not match prediction/target coordinates")
        events = row["public"]["events"]
        displacement = None
        if kind == "replacement_trajectory":
            if len(events) != len(row["truth"]) or any(len(e["candidates"]) != 1 for e in events):
                raise ValueError("replacement trajectory has an incompatible event contract")
            displacement = [math.dist(projected(t, lat0[row["seed"]]),
                                      projected((e["candidates"][0]["lat"], e["candidates"][0]["lon"]),
                                                lat0[row["seed"]]))
                            for t, e in zip(row["truth"], events)]
        result.update(output_kind=kind, fixed_attacker=row["attacker_selected"],
                      target_count=len(errors), eligible_release_count=len(events),
                      metrics=comparator_native_metrics(
                          kind, attack_errors_m=errors, release_errors_m=displacement,
                          generation_ms=row["metrics"]["amortized_generation_ms"] * len(events),
                          generation_event_count=len(events)))
        if not math.isclose(result["metrics"]["eie_point_estimate_m"]["value"],
                            row["metrics"]["location_mae_m"], rel_tol=1e-9, abs_tol=1e-7):
            raise ValueError("empirical point-estimate EIE differs from existing MAE")
        rows.append(result)
    return rows


def common_rows(source):
    rows = []
    for row in source["rows"]:
        if row["method"] not in COMMON or row["scenario"] not in SCENARIOS:
            continue
        result = {key: row[key] for key in ("method", "scenario", "case_id", "record_id", "family_id", "rep", "status")}
        result.update(cohort="common_live_" + row["split"], k=5)
        if row["status"] != "ok":
            result["unavailable_reason"] = "Frozen execution failed; not counted as private."
        else:
            decoder = source["selection"][row["method"] + "/" + row["case_id"]]["selected"]["mae"]
            errors = row["errors"][decoder]
            result.update(output_kind=KINDS[row["method"]], fixed_attacker=decoder,
                          target_count=len(errors),
                          metrics=comparator_native_metrics(KINDS[row["method"]], attack_errors_m=errors))
        rows.append(result)
    return rows


def expanded_rows(source):
    rows = []
    for row in source["rows"]:
        # Preserve the current presentation's endpoint scope, not the older
        # hidden-destination development diagnostic that shares an S10 label.
        if row["case_id"] == "S10.B":
            continue
        kind = "replacement_trajectory" if row["method"] == "raw" else "dummy_only"
        decoder = row["selected_attack"]["mae"]
        errors = row["errors"][decoder]
        result = {key: row[key] for key in ("method", "case_id", "record_id", "family_id", "rep")}
        result.update(cohort="geoi_expanded_development", k=5, scenario=row["case_id"].split(".")[0],
                      status="ok", output_kind=kind, fixed_attacker=decoder, target_count=len(errors),
                      metrics=comparator_native_metrics(kind, attack_errors_m=errors))
        rows.append(result)
    return rows


def summaries(rows):
    metric_inventory = comparator_native_metrics("replacement_trajectory")
    groups = defaultdict(list)
    for row in rows:
        groups[row["cohort"], row["method"], row["scenario"], row["k"]].append(row)
    result = []
    for (cohort, method, scenario, k), items in sorted(groups.items()):
        summary = {"cohort": cohort, "method": method, "scenario": scenario, "k": k,
                   "attempted_rows": len(items), "completed_rows": sum(r["status"] == "ok" for r in items),
                   "unavailable_rows": sum(r["status"] != "ok" for r in items)}
        metric_names = sorted(metric_inventory)
        summary["metrics"] = {}
        for name in metric_names:
            record = defaultdict(list)
            eligible = [row for row in items if row.get("metrics", {}).get(name, {}).get("value") is not None]
            for row in eligible:
                record[row["case_id"], row["family_id"], row["record_id"]].append(row["metrics"][name]["value"])
            family = defaultdict(list)
            for (case, group, _), values in record.items():
                family[case, group].append(inference_error_m(values))
            case = defaultdict(list)
            for (condition, _), values in family.items():
                case[condition].append(inference_error_m(values))
            scores = [inference_error_m(values) for values in case.values()]
            summary["metrics"][name] = {"value": inference_error_m(scores) if scores else None,
                                       "eligible_rows": len(eligible),
                                       "unavailable_rows": len(items) - len(eligible),
                                       "family_count": len({row["family_id"] for row in eligible}),
                                       "unit": eligible[0]["metrics"][name]["unit"] if eligible else metric_inventory[name]["unit"],
                                       "status": "computed" if len(eligible) == len(items) else "conditional" if eligible else "not_available",
                                       "reasons": sorted({row.get("metrics", {}).get(name, {}).get("reason", row.get("unavailable_reason", "Missing evidence"))
                                                          for row in items if row not in eligible}),
                                       "aggregation": "rep -> record -> family -> source condition -> scenario; equal means"}
        result.append(summary)
    return result


def write_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, allow_nan=False, indent=2) + "\n")


def run(output=OUT, k=5):
    output = Path(output)
    if output.exists():
        raise FileExistsError(f"New readout directory required; refusing to overwrite {output}")
    source_paths = [ROOT / "artifacts/benchmarks/paper_benchmark/results.json",
                    ROOT / "artifacts/benchmarks/live_paper_comparison_v1/privacy.json.gz",
                    ROOT / "artifacts/benchmarks/research_loop/iteration18_expanded_attacks.json",
                    ROOT / "artifacts/benchmarks/dummy_benchmark_results.json"]
    hashes = {str(path.relative_to(ROOT)): sha(path) for path in source_paths}
    loaded = [json.loads(gzip.decompress(p.read_bytes()) if p.suffix == ".gz" else p.read_text()) for p in source_paths]
    rows = paper_rows(loaded[0], k) + common_rows(loaded[1]) + expanded_rows(loaded[2])
    aggregate = summaries(rows)
    retained = []
    for r in loaded[3]["runs"]:
        if r["paper_metrics"].get("expected_travel_cost_loss_m") is not None:
            retained.append({"cohort": "historical_8_event_smoke_not_common_comparison",
                             "method": r["mechanism"], "status": "archived_reported_not_recomputed",
                             "metric": "transprotect_delta_cost", "value": r["paper_metrics"]["expected_travel_cost_loss_m"],
                             "unit": "m", "source": str(source_paths[3].relative_to(ROOT)),
                             "reason": "Frozen scalar exists; target-cost table/network absent for independent recomputation."})
    output.mkdir(parents=True)
    (output / "rows.json.gz").write_bytes(gzip.compress(json.dumps(rows, ensure_ascii=False, allow_nan=False,
                                                                 separators=(",", ":")).encode(), mtime=0))
    write_json(output / "readout.json", {"schema": "native-comparator-metrics-readout-v1", "date": "2026-10-05",
        "scope": "frozen local adaptations; original metric formulas where inputs exist; no new fit or confirmation",
        "source_sha256": hashes,
        "code_sha256": {str(p.relative_to(ROOT)): sha(p) for p in [Path(__file__), ROOT / "evaluation/native_comparator_metrics.py"]},
        "rows": len(rows), "summaries": aggregate, "historical_reported_metrics": retained,
        "limitations": ["EIE of a fixed point estimator is mathematically the existing MAE; this adds provenance, not new privacy evidence.",
                        "No VehiTrack or paper-calibrated attacker parity is claimed.",
                        "Eq13 tables and original road cache are absent in comparative archives; do not substitute straight-line costs.",
                        "DLS/RDG posterior meanings, ASR/DER and fake-path similarity cannot be coerced onto dummy-only output.",
                        "Different cohorts/protocols remain separate; offline AnotherMe failures and conditional coverage retained.",
                        "Source conditions remain accounting keys only; tables present scenario-level results."]})
    with (output / "summary.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["cohort", "method", "scenario", "k", "attempted_rows", "completed_rows",
                                                   "metric", "value", "unit", "status", "eligible_rows", "unavailable_rows", "family_count"])
        writer.writeheader()
        for summary in aggregate:
            for metric, value in summary["metrics"].items():
                writer.writerow({**{key: summary[key] for key in writer.fieldnames if key in summary and key != "status"}, "metric": metric,
                                 **{key: value[key] for key in value if key in writer.fieldnames}})
    assert all(sha(ROOT / name) == digest for name, digest in hashes.items()), "Source artifact changed"
    write_json(output / "verification.json", {"status": "passed", "source_artifacts_unchanged": True,
        "row_count": len(rows), "summary_count": len(aggregate),
        "paper_eie_independent_coordinate_checks": sum(r["cohort"] == "paper_v2_test" and r["status"] == "ok" for r in rows),
        "n_a_remains_json_null": True, "new_attacker_or_defender_fits": 0,
        "outputs": {p.name: sha(p) for p in output.iterdir() if p.is_file()}})
    print(json.dumps({"output": str(output), "rows": len(rows), "summaries": len(aggregate)}, ensure_ascii=False))
    return rows, aggregate


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=OUT)
    parser.add_argument("--k", type=int, default=5)
    args = parser.parse_args()
    run(args.output, args.k)
