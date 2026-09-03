"""Top-level orchestration and artifact assembly for the benchmark."""

from __future__ import annotations

import json
import os
from collections.abc import Callable

from benchmark.methods import (
    AnotherMeAdaptation,
    GeoIAnchoredDummyTrajectories,
    SemanticCorrelationComparator,
    TransProtectAdaptation,
)
from benchmark.registry import method_inventory, require_faithful_sota
from core.demo_protocol import OutputKind
from core.road_network import RoadNetwork
from data.sumo_demo import load_sumo_road_network, road_network_semantic_sha256
from experiments.provenance import (
    GRAPH_PKL,
    assert_graph_matches_manifest,
    begin_run,
    provenance,
    sha256_file,
)

from .constants import BASE_DISCLAIMER, BENCHMARK_SCHEMA, DISCLAIMER
from .results import round_value


def run_benchmark(
    args,
    *,
    load_records: Callable,
    run_models: Callable,
    diagnostics: Callable,
    aggregate: Callable,
    print_summary: Callable,
    write_map: Callable,
    write_preview: Callable,
) -> dict:
    """Execute the benchmark using helpers supplied by the stable facade.

    Supplying these boundaries explicitly keeps legacy monkey-patch points in
    ``experiments.run_dummy_benchmark`` effective after the implementation was
    split into focused modules.
    """

    if getattr(args, "require_faithful_sota", False):
        require_faithful_sota(
            [
                TransProtectAdaptation.name,
                AnotherMeAdaptation.name,
                SemanticCorrelationComparator.name,
            ]
        )
    run_context = begin_run()
    if args.mobility_source == "sumo":
        print("Running SUMO and loading its passenger road graph...", flush=True)
        records, mobility = load_records(args)
        network_path = mobility.get("_candidate_network_path")
        if not network_path:
            raise RuntimeError("SUMO record did not expose its generated network")
        rn = load_sumo_road_network(network_path)
        print(
            f"  {len(rn)} passenger-network vertices "
            f"({rn.graph.number_of_edges()} directed edges)",
            flush=True,
        )
    else:
        print("Loading the pinned Beijing road graph...", flush=True)
        rn = RoadNetwork.from_pickle(GRAPH_PKL)
        assert_graph_matches_manifest(rn)
        print(f"  {len(rn)} road vertices (manifest verified)", flush=True)
        records, mobility = load_records(args)
    print(
        f"Loaded {len(records)} {mobility['label']} "
        f"record(s), {sum(len(record['points']) for record in records)} points total.",
        flush=True,
    )

    result_rows = []
    first_runs = None
    for record in records:
        runs = run_models(
            rn,
            record["points"],
            record["times"],
            record["record_id"],
            training_trajectories=mobility.get("_model_training_records", ()),
            epsilon=args.epsilon,
            transprotect_epsilon_per_km=args.transprotect_epsilon_per_km,
            transprotect_k=args.transprotect_k,
            transprotect_target_count=args.transprotect_target_count,
            transprotect_alpha=args.transprotect_alpha,
            transprotect_probability_smoothing=(
                args.transprotect_probability_smoothing
            ),
            transprotect_probability_backoff_weight=(
                args.transprotect_probability_backoff_weight
            ),
            k=args.k,
            seed=args.seed,
        )
        if first_runs is None:
            first_runs = runs
        for protected, card, runtime in runs:
            evaluator = protected.to_evaluator_dict()
            row = {
                "record_id": record["record_id"],
                "mechanism": protected.transcript.mechanism,
                "source_method": card.source.citation,
                "implementation": card.to_dict(),
                "output_kind": protected.transcript.output_kind.value,
                # Compatibility scalar now means full model setup + inference.
                "runtime_ms": round_value(runtime.end_to_end_runtime_ms),
                "runtime_breakdown_ms": runtime.to_dict(),
                "paper_metrics": dict(runtime.paper_metrics),
                "metrics": {
                    key: round_value(value)
                    for key, value in diagnostics(
                        protected, rn, args.qos_radius_m
                    ).items()
                },
                # The split is deliberate: attack code should receive only the
                # first object. Truth is serialized solely for offline scoring.
                "attacker_view": evaluator["public"],
                "evaluator_truth": evaluator["truth"],
            }
            result_rows.append(row)

    summaries = aggregate(result_rows)
    if mobility["source"] == "sumo":
        artifact_disclaimer = DISCLAIMER
    else:
        artifact_disclaimer = (
            BASE_DISCLAIMER
            + " GeoLife is an explicitly selected validation source; this run "
            "does not execute SUMO."
        )
    print_summary(summaries, artifact_disclaimer)

    if mobility["source"] == "sumo":
        dataset = {
            "name": "Eclipse SUMO controlled Beijing passenger smoke scenario",
            "raw_bytes_in_git": False,
            "note": (
                "Movement is generated at run time from the local Beijing OSM "
                "extract. Exact SUMO commands, versions, and input/output SHA-256 "
                "digests are recorded in sumo_runs. This is not a calibrated "
                "Beijing population model."
            ),
        }
    else:
        dataset = {
            "name": "GeoLife v1.3 (explicit optional validation mode)",
            "raw_bytes_in_git": False,
            "note": (
                "Raw GeoLife data are gitignored; selected record IDs are pinned "
                "below. This mode was explicitly requested and is never a fallback."
            ),
        }
    if mobility["source"] == "sumo":
        sumo_hashes = mobility["sumo_runs"][0]["sha256"]
        graph_provenance = {
            "graph_sha256": sumo_hashes["network"],
            "graph_raw_file_sha256": sumo_hashes["network"],
            "graph_semantic_sha256": road_network_semantic_sha256(rn),
            "graph_semantic_hash_schema": "sumo-passenger-road-graph-v1",
            "graph_source_sha256": sumo_hashes["osm"],
            "graph_network_profile": "SUMO passenger-only, largest component",
            "graph_origin": "generated SUMO .net.xml used by the same FCD run",
            "graph_edges": rn.graph.number_of_edges(),
        }
    else:
        graph_provenance = {
            "graph_edges": rn.graph.number_of_edges(),
            "graph_origin": "pinned OSMnx pickle",
        }
    prov = provenance(
        rn,
        [args.epsilon],
        root_seeds=[args.seed],
        quick=args.quick,
        begin=run_context,
        extra={
            "artifact_schema": BENCHMARK_SCHEMA,
            "artifact_status": (
                "executable clean-room benchmark comparators; unavailable "
                "paper assets use explicit local adapters, so these are not "
                "reproduced SOTA results"
            ),
            "dataset": dataset,
            "mobility_source": mobility["source"],
            "mobility_label": mobility["label"],
            "sumo_runs": mobility["sumo_runs"],
            "candidate_graph_compatibility": mobility[
                "candidate_graph_compatibility"
            ],
            "model_training_context": mobility["model_training_context"],
            "selected_record_ids": [record["record_id"] for record in records],
            "n_points_per_record": [len(record["points"]) for record in records],
            "k": args.k,
            "privacy_parameters": {
                "thesis_epsilon_per_m": args.epsilon,
                "transprotect_epsilon_per_km": (
                    args.transprotect_epsilon_per_km
                ),
                "transprotect_epsilon_per_m_internal": (
                    args.transprotect_epsilon_per_km / 1_000.0
                ),
            },
            "method_parameters": {
                TransProtectAdaptation.name: {
                    "candidate_k": args.transprotect_k,
                    "target_count": args.transprotect_target_count,
                    "alpha": args.transprotect_alpha,
                    "probability_smoothing": (
                        args.transprotect_probability_smoothing
                    ),
                    "probability_backoff_weight": (
                        args.transprotect_probability_backoff_weight
                    ),
                    "epsilon_per_km": args.transprotect_epsilon_per_km,
                },
                AnotherMeAdaptation.name: {
                    "rng_invariant_to_thesis_epsilon": True,
                },
                SemanticCorrelationComparator.name: {
                    "k": args.k,
                    "candidate_linkage": "event_local_unlinked_sets",
                    "rng_invariant_to_thesis_epsilon": True,
                },
                GeoIAnchoredDummyTrajectories.name: {
                    "k": args.k,
                    "epsilon_per_m": args.epsilon,
                },
            },
            "qos_radius_m": args.qos_radius_m,
            "cross_contract_comparison_allowed": False,
            **graph_provenance,
        },
    )
    artifact = {
        "schema": BENCHMARK_SCHEMA,
        "status": "EXECUTABLE_CLEAN_ROOM_ADAPTATIONS",
        "disclaimer": artifact_disclaimer,
        "method_inventory": method_inventory(),
        "provenance": prov,
        # Route, speed, edge, lane, and simulator vehicle ID are privileged
        # ground truth for offline evaluation. They are intentionally absent
        # from every run's attacker_view below.
        "mobility_evaluator_only": mobility["evaluator_only"],
        "summaries": summaries,
        "runs": result_rows,
    }
    visual_artifacts = None
    if not args.no_map and first_runs is not None:
        write_map(
            args.map_output,
            records[0],
            first_runs,
            rn,
            mobility_source=mobility["source"],
        )
        print(f"Saved interactive map: {args.map_output}")
        write_preview(
            args.preview_output,
            records[0],
            first_runs,
            rn,
            mobility_source=mobility["source"],
        )
        print(f"Saved static preview: {args.preview_output}")
        visual_artifacts = {
            "evaluator_only": True,
            "contains_ground_truth": True,
            "interactive_map": {
                "path": args.map_output,
                "sha256": sha256_file(args.map_output),
                "embedded_local_roads": True,
                "online_tiles_enabled_by_default": False,
                "runtime_assets_may_require_network_or_browser_cache": True,
                "byte_reproducible": False,
            },
            "static_preview": {
                "path": args.preview_output,
                "sha256": sha256_file(args.preview_output),
                "embedded_local_roads": True,
                "byte_reproducible_across_platforms": False,
            },
        }
    artifact["visual_artifacts"] = visual_artifacts
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as handle:
        json.dump(artifact, handle, indent=2, ensure_ascii=False, allow_nan=False)
    print(f"Saved evaluator JSON: {args.output}")
    return artifact
