"""Export thesis tables from retained October evidence; never regenerate scores.

Run from any directory with ``python -m experiments.export_thesis_evaluation_20261006``.
``--check`` authenticates the existing source/output manifest without writing.
Only thesis/current_evaluation_generated is authored. No private key is read.
"""
import argparse
import hashlib
import json
from pathlib import Path
import statistics

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "thesis/current_evaluation_generated"
B = "artifacts/benchmarks/"
FRESH = B + "qplanner_response_depth_generalization_20261006_v1/"
DYNAMIC = B + "dynamic_provider_status_20261006_v1/"
SOURCES = {
    "fresh_configuration": FRESH + "recommended_configuration.json",
    "fresh_paired": FRESH + "paired_readout.json",
    "fresh_protocol": FRESH + "protocol.json",
    "fresh_validation": FRESH + "validation.json",
    "fresh_verification_protocol": FRESH + "verification_protocol.json",
    "paper_development": B + "paper_benchmark/results.json",
    "native_metrics": B + "native_metrics_20261005/readout.json",
    "native_future": B + "future_native_20261005_v1/results.json",
    "native_future_validation": B + "future_native_20261005_v1/validation.json",
    "matched_anchor": B + "jisa_native_anchor_ablation_20261006_v1/derived_readout_v2.json",
    "matched_anchor_results": B + "jisa_native_anchor_ablation_20261006_v1/results.json",
    "matched_anchor_validation": B + "jisa_native_anchor_ablation_20261006_v1/validation.json",
    "bulk": B + "jisa_static_catalogue_control_20261006_v1/results.json",
    "bulk_validation": B + "jisa_static_catalogue_control_20261006_v1/validation.json",
    "bulk_cost_rows": B + "jisa_static_catalogue_control_20261006_v1/cost_rows.json",
    "endpoint_development": B + "endpoint_robust_selection_20261006/readout.json",
    "identity_development": B + "identity_future_20261005/refinement_results.json",
    "purpose_development": B + "query_purpose_20261005/sequence/readout.json",
    "planner_round_one": B + "qplanner_development_20261006_v2/defense_selection.json",
    "planner_round_two": B + "qplanner_development_20261006_v3/defense_selection.json",
    "depth_development": B + "qplanner_response_depth_development_20261006_v1/depth_selection.json",
    "figure_provenance": FRESH + "figures_v2/figure_provenance.json",
    "thesis_figure": "artifacts/reports/thesis_figures_20261006/source.json",
    "dynamic_protocol": DYNAMIC + "protocol.json",
    "dynamic_readout": DYNAMIC + "readout.json",
    "dynamic_validation": DYNAMIC + "validation.json",
    "dynamic_verification_protocol": DYNAMIC + "verification_protocol.json",
    "dynamic_verification_review": DYNAMIC + "verification_review_protocol_v2.json",
    "dynamic_failed_verification": DYNAMIC + "validation_attempt_v1_failure.json",
}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def vn(value, digits=2):
    return f"{value:.{digits}f}".replace(".", ",")


def integer(value):
    return f"{value:,}".replace(",", r"\,")


def render():
    data = {name: json.loads((ROOT / path).read_text()) for name, path in SOURCES.items()}
    source_pins = {path: sha(ROOT / path) for path in SOURCES.values()}
    config, paired = data["fresh_configuration"], data["fresh_paired"]
    for path, pin in config["provenance_sha256"].items():
        assert sha(ROOT / FRESH / path) == pin, path
    cert = data["fresh_validation"]
    assert cert["status"] == "pass" and paired["primary_criterion_result"]["passes"] is True
    assert cert["protocol_sha256"] == source_pins[SOURCES["fresh_protocol"]]
    assert cert["paired_readout_sha256"] == source_pins[SOURCES["fresh_paired"]]
    assert cert["verification_protocol_sha256"] == source_pins[SOURCES["fresh_verification_protocol"]]
    for path, pin in data["fresh_verification_protocol"]["source_sha256"].items():
        assert sha(ROOT / path) == pin, path
    assert config["configuration"]["server_max_POIs_per_category_per_Q_L"] == 30
    assert config["certified_primary"]["frozen_criterion_passes"] is True
    matched = data["matched_anchor"]
    assert matched["source_results_sha256"] == source_pins[SOURCES["matched_anchor_results"]]
    assert data["native_future_validation"]["status"] == "passed"
    assert data["bulk_validation"]["status"] == "pass"
    matched_cert = data["matched_anchor_validation"]
    assert matched_cert["same_cap_map_clock_K_L"] is True
    assert matched_cert["old_source_outputs_unchanged"] is True
    assert matched_cert["derived_readout_v2_sha256"] == source_pins[SOURCES["matched_anchor"]]
    assert data["bulk"]["protected_draws_generated"] == 0 and data["bulk"]["timing_measured"] is False
    for key in ["planner_round_one", "planner_round_two"]:
        assert data[key]["selected"] is None and data[key]["fresh_test_scores_viewed"] is False
    assert data["depth_development"]["selected_depth"] == 30
    assert all(data["endpoint_development"]["original_controls_recovered"].values())
    for name, pin in data["figure_provenance"]["output_sha256"].items():
        assert sha(ROOT / FRESH / "figures_v2" / name) == pin
    assert data["thesis_figure"]["builder_sha256"] == sha(ROOT / "experiments/plot_thesis_depth_20261006.py")
    for path, pin in data["thesis_figure"]["source_sha256"].items():
        assert sha(ROOT / path) == pin, path
    for name, pin in data["thesis_figure"]["output_sha256"].items():
        assert sha(ROOT / "artifacts/reports/thesis_figures_20261006" / name) == pin, name
    dynamic = data["dynamic_readout"]
    dynamic_cert = data["dynamic_validation"]
    assert dynamic_cert["status"] == "pass" and dynamic_cert["blocks"] == 72
    assert dynamic_cert["readout_sha256"] == source_pins[SOURCES["dynamic_readout"]]
    assert dynamic_cert["protocol_sha256"] == source_pins[SOURCES["dynamic_protocol"]]
    assert dynamic_cert["verification_protocol_sha256"] == source_pins[SOURCES["dynamic_verification_protocol"]]
    assert dynamic_cert["verification_review_protocol_v2_sha256"] == source_pins[SOURCES["dynamic_verification_review"]]
    assert dynamic_cert["post_score_checker_row_order_correction"] is True
    assert dynamic_cert["fresh_confirmation"] is False and dynamic_cert["privacy_theorem_certified"] is False
    assert data["dynamic_failed_verification"]["status"] == "failed"
    for source in [data["dynamic_protocol"], data["dynamic_verification_review"]]:
        for path, pin in source["source_sha256"].items():
            assert sha(ROOT / path) == pin, path
    for path, pin in data["dynamic_verification_review"]["completed_inputs_sha256"].items():
        assert sha(ROOT / DYNAMIC / path) == pin, path
    for path, pin in dynamic["block_files_sha256"].items():
        assert sha(ROOT / DYNAMIC / "blocks" / path) == pin, path
    assert dynamic["protected_Q_clocks_ledger_unchanged"] is True
    assert dynamic["source_already_inspected_secondary_only"] is True
    prefix = "% Generated by experiments/export_thesis_evaluation_20261006.py; do not hand-edit.\n"
    outputs = {}
    primary = paired["contrasts"][paired["primary_criterion_result"]["primary_key"]]
    purposes = [("nearest_distance", "Gần nhất"), ("fastest_travel", "Nhanh nhất"),
                ("within_radius", "Trong bán kính"), ("minimum_detour", "Ít vòng đường nhất"),
                ("equal_purpose_macro", "Trung bình đều purpose")]
    rows = []
    for purpose, label in purposes:
        means = [statistics.mean(paired["conditional_family_cells"][f"service_l{L}--test--current--all"][purpose]["family_values"].values()) for L in [20, 30]]
        contrast = paired["contrasts"][f"service_l30--minus--service_l20--test--current--all--{purpose}"]
        assert abs(means[1] - means[0] - contrast["mean_difference"]) < 1e-12
        lo, hi = contrast["percentile95_family_bootstrap"]
        rows.append(f"{label} & {vn(100*means[0],3)} & {vn(100*means[1],3)} & +{vn(100*contrast['mean_difference'],3)} & [{vn(100*lo,3)}; {vn(100*hi,3)}]" + r" \\")
    outputs["fresh_purposes.tex"] = prefix + "\n".join(rows) + "\n"
    rows = []
    for L in [20, 30]:
        cost = config["actual_TEST_cost"][f"L{L}"]
        rows.append(f"L{L} & {integer(cost['requests'])} & {integer(cost['request_bytes'])} & {integer(cost['reply_bytes'])}" + r" \\")
    outputs["fresh_cost.tex"] = prefix + "\n".join(rows) + "\n"
    rows = []
    for method, label in [("raw", "GPS thật"), ("rem_epoch8", "REM--Geo-I"), ("planar_epoch8", "Planar--Geo-I")]:
        utility = matched["utility"][method]["test"]
        current = utility["current_only"]["all_0_600"]["family_macro_recall5"]
        cached = utility["versioned_epoch8"]["all_0_600"]["family_macro_recall5"]
        S5 = matched["attacks"][f"{method}--turn_visible--S5_next_edge"]["test"]["exact_candidate_edge_accuracy"]
        S6 = matched["attacks"][f"{method}--turn_visible--S6_history_destination"]["test"]["destination_hit100"]
        rows.append(f"{label} & {vn(100*S5,1)} & {vn(100*S6,1)} & {vn(100*current,3)} & {vn(100*cached,3)}" + r" \\")
    outputs["matched_anchor.tex"] = prefix + "\n".join(rows) + "\n"
    rows = []
    for method, label in [("raw", "GPS thật"), ("scale100_L20", "GeoI-Slack L20"), ("scale025_L20", "GeoI-Endpoint20")]:
        scenarios = data["endpoint_development"]["results"][method]["robust_crossfit"]
        cells = []
        for S in ["S9", "S10"]:
            row = scenarios[S]
            assert row["families"] == 28 and row["observations"] == 112
            cells.append(f"{vn(100*row['hit100'])} / {vn(row['mae_m'],1)} / {vn(100*row['hit500'])}")
        rows.append(label + " & " + " & ".join(cells) + r" \\")
    outputs["endpoint.tex"] = prefix + "\n".join(rows) + "\n"
    rows = []
    methods = [("unprotected", "GPS thật"), ("dls_graph_adaptation", "DLS*"),
               ("transprotect_adaptation", "TransProtect*"),
               ("semantic_correlation_local_adaptation", "Semantic*"), ("br_private", "BR--Geo-I lịch sử")]
    for method, label in methods:
        cells = []
        for S in ["S1", "S2", "S3"]:
            row, = [r for r in data["paper_development"]["summary"] if r["method"] == method and r["scenario"] == S and r["k"] == 5]
            assert row["attempted"] == row["completed"] == 12 and row["failures"] == row["not_applicable"] == 0
            cells.append(f"{vn(100*row['location_hit_100m'])} / {vn(row['location_mae_m'],0)} / {vn(100*row['poi_recall_at_k'])}")
        rows.append(label + " & " + " & ".join(cells) + r" \\")
    outputs["historical_location.tex"] = prefix + "\n".join(rows) + "\n"
    rows = []
    for round_name, key in [("Vòng 1", "planner_round_one"), ("Vòng 2", "planner_round_two")]:
        for name, row in data[key]["records"].items():
            label = name.replace("_", r"\_")
            reason = "Thiếu gain" + ("; không qua S5/S6" if "S5 future guard failed" in row["reasons"] else "")
            rows.append(f"{round_name} & \\texttt{{{label}}} & {vn(100*row['macro_gain'],3)} & {reason}" + r" \\")
    outputs["rejected_planners.tex"] = prefix + "\n".join(rows) + "\n"
    rows = []
    dynamic_arms = [
        ("geoi_l20_current", "Geo-I L20, hiện tại"),
        ("geoi_l20_epoch60", "Geo-I L20, cache 60 s"),
        ("geoi_l30_current", "Geo-I L30, hiện tại"),
        ("geoi_l30_epoch60", "Geo-I L30, cache 60 s"),
        ("raw_l20_current", "GPS thật L20"),
        ("raw_l30_current", "GPS thật L30"),
        ("static_catalogue_stale_safe", "Catalogue cũ, loại trạng thái hết hạn"),
        ("current_catalogue_bulk_oracle", "Catalogue đầy đủ, cập nhật hiện tại"),
        ("static_catalogue_stale_unsafe", r"\textbf{Catalogue cũ, không hợp lệ*}"),
    ]
    for arm, label in dynamic_arms:
        cells = []
        for window in ["all", "cold", "temporal_tail_400_600"]:
            summary = dynamic["summary"][arm][window]["equal_purpose_macro"]
            assert abs(statistics.mean(summary["family_values"].values()) - summary["family_mean"]) < 1e-12
            cells.append(vn(100 * summary["family_mean"]))
        cost = dynamic["cost"][arm]
        payload = sum(cost[key] for key in ["static_request_bytes", "status_upload_bytes", "static_reply_bytes", "status_download_bytes"])
        rows.append(label + " & " + " & ".join(cells) + " & " + vn(payload / 1e6) + r" \\")
    outputs["dynamic_status.tex"] = prefix + "\n".join(rows) + "\n"
    macros = {
        "EvalFreshRecallBefore": vn(100*statistics.mean(paired["conditional_family_cells"]["service_l20--test--current--all"]["equal_purpose_macro"]["family_values"].values()),3),
        "EvalFreshRecallAfter": vn(100*statistics.mean(paired["conditional_family_cells"]["service_l30--test--current--all"]["equal_purpose_macro"]["family_values"].values()),3),
        "EvalFreshGain": vn(100*primary["mean_difference"],3),
        "EvalFreshLow": vn(100*primary["percentile95_family_bootstrap"][0],3),
        "EvalFreshHigh": vn(100*primary["percentile95_family_bootstrap"][1],3),
        "EvalReplyGrowth": vn(100*(config["actual_TEST_cost"]["reply_byte_ratio"]-1),3),
        "EvalTailLowBefore": vn(100*primary["right_lower_tail_mean"],3),
        "EvalTailLowAfter": vn(100*primary["left_lower_tail_mean"],3),
        "EvalTailGain": vn(100*primary["lower_tail_mean_difference"],3),
        "EvalDrawOne": vn(100*primary["within_draw"]["draws"]["1"]["paired_mean_difference"],3),
        "EvalDrawTwo": vn(100*primary["within_draw"]["draws"]["2"]["paired_mean_difference"],3),
        "EvalDrawThree": vn(100*primary["within_draw"]["draws"]["3"]["paired_mean_difference"],3),
    }
    radius = paired["contrasts"]["service_l30--minus--service_l20--test--current--all--within_radius"]["left_reference_coverage"]
    for field, name in [("defined_categories", "EvalRadiusDefinedCategories"), ("total_categories", "EvalRadiusTotalCategories"), ("defined_windows", "EvalRadiusDefinedWindows"), ("total_windows", "EvalRadiusTotalWindows")]:
        macros[name] = integer(radius[field])
    dynamic_means = {arm: dynamic["summary"][arm]["all"]["equal_purpose_macro"]["family_mean"] for arm, _ in dynamic_arms}
    dynamic_bytes = {arm: sum(value for key, value in dynamic["cost"][arm].items() if key != "requests") for arm, _ in dynamic_arms}
    macros.update({
        "EvalDynamicLThirtyCurrent": vn(100 * dynamic_means["geoi_l30_current"]),
        "EvalDynamicLThirtyCached": vn(100 * dynamic_means["geoi_l30_epoch60"]),
        "EvalDynamicGain": vn(100 * (dynamic_means["geoi_l30_current"] - dynamic_means["geoi_l20_current"])),
        "EvalDynamicCostGrowth": vn(100 * (dynamic_bytes["geoi_l30_current"] / dynamic_bytes["geoi_l20_current"] - 1)),
        "EvalDynamicRadiusDefined": integer(dynamic["summary"]["geoi_l30_current"]["all"]["within_radius"]["explicit_denominators"]["defined_events"]),
    })
    outputs["numbers.tex"] = prefix + "\n".join(f"\\newcommand{{\\{k}}}{{{v}}}" for k, v in macros.items()) + "\n"
    # Load rows outside alignments: LaTeX's protected \input leaves tokens that
    # can put a following booktabs rule in a cell (Misplaced \noalign).
    commands = {
        "fresh_purposes.tex": "EvalFreshPurposeRows",
        "fresh_cost.tex": "EvalFreshCostRows",
        "matched_anchor.tex": "EvalMatchedAnchorRows",
        "endpoint.tex": "EvalEndpointRows",
        "historical_location.tex": "EvalHistoricalLocationRows",
        "rejected_planners.tex": "EvalRejectedPlannerRows",
        "dynamic_status.tex": "EvalDynamicRows",
    }
    for name, command in commands.items():
        rows = outputs[name][len(prefix):]
        outputs[name] = prefix + "\\newcommand{\\" + command + "}{%\n" + rows + "}\n"
    manifest = {
        "schema": "thesis-current-evaluation-source-manifest-v1", "as_of": "2026-10-06",
        "exporter_sha256": sha(__file__), "source_sha256": source_pins,
        "generated_sha256": {name: hashlib.sha256(text.encode()).hexdigest() for name, text in outputs.items()},
        "derivation": "Formatting retained JSON summaries; primary means independently checked against family cells; no new fitting, scoring, selection, bootstrap or private-key read.",
        "numeric_scope": "Separate cohorts/protocols remain separate; paper br_private, matched native anchor pilot, endpoint inspected development and frozen fresh depth confirmation are not pooled. Dynamic status is secondary inspected single-world sensitivity; original checker failure and post-score serialization-only correction are retained.",
    }
    outputs["sources.json"] = json.dumps(manifest, indent=2, ensure_ascii=False, allow_nan=False) + "\n"
    return outputs


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    outputs = render()
    if args.check:
        for name, text in outputs.items():
            assert (OUTPUT / name).read_text() == text, name
        print(f"PASS: {len(SOURCES)} source JSON files; {len(outputs)-1} deterministic TeX exports; no scores regenerated")
    else:
        OUTPUT.mkdir(parents=True, exist_ok=True)
        for name, text in outputs.items():
            (OUTPUT / name).write_text(text)
        print(f"Exported {len(outputs)-1} TeX fragments and source manifest to {OUTPUT}")


if __name__ == "__main__":
    main()
