"""Run source-mapped clean-room comparators beside the thesis candidate.

This module is the stable command-line and import facade for the executable
dummy-generation benchmark harness. Its implementation lives in the focused
modules under :mod:`experiments.dummy_benchmark`; keeping this facade preserves
existing scripts, imports, and test patch points.

The three paper comparators implement every locally reproducible stage and
expose unavailable paper dependencies as explicit adapters or blockers. They
are not official or paper-equivalent reproductions. Results MUST NOT be
presented as reproduced SOTA results or as a cross-contract leaderboard until
the faithful gate passes.

Usage (from the repository root)::

    python -m experiments.run_dummy_benchmark
    python -m experiments.run_dummy_benchmark --quick
    python -m experiments.run_dummy_benchmark --mobility-source geolife --no-map

The default run executes a controlled Eclipse SUMO scenario over the local
Beijing OpenStreetMap extract and writes JSON, interactive-map, and offline-PNG
artifacts under ``artifacts/benchmarks/``. GeoLife remains an explicit real-data validation
option; neither source silently falls back to other mobility data.
"""

from __future__ import annotations

# Historical public imports are retained during this compatibility phase. A
# few external notebooks imported these transitively before the runner became
# a facade; removing them belongs in an explicitly versioned cleanup.
import argparse
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
import json
import os
import time
from typing import Iterable, Mapping, Sequence

import numpy as np

# These source functions deliberately remain module globals. Existing tests and
# review scripts patch these names, while the wrappers below inject their
# current values into the extracted source adapters.
from data.geolife import load_trajectories
from data.sumo_demo import (
    DEFAULT_ROUTE_SEED,
    DEFAULT_SIM_SEED,
    DEFAULT_WORKDIR,
    SumoSmokeConfig,
    load_sumo_road_network,
    road_network_semantic_sha256,
    run_sumo_smoke_demo,
)

# Domain imports retained for compatibility with earlier direct imports from
# this module. New code should import these from their defining packages.
from benchmark.contracts import MethodCard
from benchmark.methods import (
    AnotherMeAdaptation,
    GeoIAnchoredDummyTrajectories,
    SemanticCorrelationComparator,
    TransProtectAdaptation,
)
from benchmark.registry import method_inventory, require_faithful_sota
from core.demo_protocol import OutputKind, ProtectedRun, TrajectoryPoint
from core.road_network import RoadNetwork
from evaluation import metrics as legacy_metrics
from experiments.provenance import (
    GRAPH_PKL,
    assert_graph_matches_manifest,
    begin_run,
    provenance,
    sha256_file,
)
from experiments.rng_util import rng_from_key

from experiments.dummy_benchmark.config import parse_args
from experiments.dummy_benchmark.constants import (
    BASE_DISCLAIMER,
    BENCHMARK_SCHEMA,
    DEFAULT_MAP_OUTPUT,
    DEFAULT_OUTPUT,
    DEFAULT_PREVIEW_OUTPUT,
    DEMO_SCHEMA,
    DISCLAIMER,
    MAJOR_HIGHWAYS as _MAJOR_HIGHWAYS,
    MAP_LABELS as _MAP_LABELS,
    SUMO_GRAPH_COMPATIBILITY_NOTE,
)
from experiments.dummy_benchmark.execution import (
    RuntimeEvidence,
    contract_trajectory as _contract_trajectory,
    model_rng as _model_rng,
    run_models as _run_models,
    timestamp_s as _timestamp_s,
)
from experiments.dummy_benchmark.renderers import (
    display_bounds as _display_bounds,
    display_points as _display_points,
    edge_coordinates as _edge_coordinates,
    embedded_road_geojson as _embedded_road_geojson,
    ground_truth_label as _ground_truth_label,
    line_parts as _line_parts,
    run_tracks as _run_tracks,
    write_map as _write_map,
    write_preview as _write_preview,
)
from experiments.dummy_benchmark.results import (
    aggregate as _aggregate,
    diagnostics as _diagnostics,
    event_candidate_points as _event_candidate_points,
    format_number as _format_number,
    latlon as _latlon,
    print_summary as _print_summary,
    print_table as _print_table,
    public_tracks as _public_tracks,
    replacement_diagnostics as _replacement_diagnostics,
    round_value as _round_value,
    set_diagnostics as _set_diagnostics,
)
from experiments.dummy_benchmark.runner import run_benchmark as _run_benchmark
from experiments.dummy_benchmark.sources import (
    load_geolife_records as _load_geolife_records_impl,
    load_sumo_record as _load_sumo_record_impl,
)


def _load_sumo_record(args):
    """Compatibility wrapper retaining the historical SUMO patch boundary."""

    return _load_sumo_record_impl(
        args, run_sumo_smoke_demo=run_sumo_smoke_demo
    )


def _load_geolife_records(args):
    """Compatibility wrapper retaining the historical GeoLife patch boundary."""

    return _load_geolife_records_impl(
        args, load_trajectories=load_trajectories
    )


def _load_records(args):
    """Dispatch through facade wrappers so patched loaders remain effective."""

    if args.mobility_source == "sumo":
        return _load_sumo_record(args)
    if args.mobility_source == "geolife":
        return _load_geolife_records(args)
    raise ValueError(f"unsupported mobility source: {args.mobility_source!r}")


def run(args) -> dict:
    """Run the benchmark through the extracted implementation modules."""

    return _run_benchmark(
        args,
        load_records=_load_records,
        run_models=_run_models,
        diagnostics=_diagnostics,
        aggregate=_aggregate,
        print_summary=_print_summary,
        write_map=_write_map,
        write_preview=_write_preview,
    )


if __name__ == "__main__":
    run(parse_args())
