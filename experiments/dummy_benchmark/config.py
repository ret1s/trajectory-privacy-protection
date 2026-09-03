"""Command-line configuration for the dummy-generation benchmark."""

from __future__ import annotations

import argparse

from data.sumo_demo import DEFAULT_ROUTE_SEED, DEFAULT_SIM_SEED, DEFAULT_WORKDIR

from .constants import DEFAULT_MAP_OUTPUT, DEFAULT_OUTPUT, DEFAULT_PREVIEW_OUTPUT


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description=(
            "Run three source-mapped clean-room paper comparators and the "
            "thesis candidate."
        )
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help="use at most 8 points (the mobility source remains SUMO by default)",
    )
    parser.add_argument(
        "--mobility-source",
        choices=("sumo", "geolife"),
        default="sumo",
        help=(
            "mobility ground truth: controlled SUMO simulation (default), or "
            "an explicitly requested local GeoLife validation trace"
        ),
    )
    parser.add_argument(
        "--n-trajectories",
        type=int,
        default=1,
        help="number of records in GeoLife mode; current SUMO pilot emits exactly 1",
    )
    parser.add_argument("--max-points", type=int, default=12)
    parser.add_argument("--interval-s", type=int, default=20)
    parser.add_argument(
        "--sumo-osm-path",
        default=None,
        help="Beijing .osm or .osm.gz input (default: local pinned extract)",
    )
    parser.add_argument(
        "--sumo-workdir",
        default=str(DEFAULT_WORKDIR),
        help="directory for generated SUMO network, routes, and FCD files",
    )
    parser.add_argument(
        "--sumo-route-seed",
        type=int,
        default=DEFAULT_ROUTE_SEED,
        help="randomTrips.py demand/route seed",
    )
    parser.add_argument(
        "--sumo-simulation-seed",
        type=int,
        default=DEFAULT_SIM_SEED,
        help="SUMO simulation seed",
    )
    parser.add_argument(
        "--sumo-min-trip-distance-m",
        type=float,
        default=2500.0,
        help=(
            "minimum straight-line origin/destination separation requested "
            "from randomTrips.py"
        ),
    )
    parser.add_argument(
        "--sumo-demand-end-s",
        type=float,
        default=300.0,
        help="last time at which random demand may depart",
    )
    parser.add_argument(
        "--sumo-simulation-end-s",
        type=float,
        default=900.0,
        help="SUMO simulation end time",
    )
    parser.add_argument(
        "--epsilon",
        type=float,
        default=0.02,
        help="thesis-candidate Geo-I budget in m^-1",
    )
    parser.add_argument(
        "--transprotect-epsilon-per-km",
        type=float,
        default=5.0,
        help=(
            "TransProtect budget in km^-1 (paper sweep: 5, 7.5, 10); "
            "converted to m^-1 internally"
        ),
    )
    parser.add_argument(
        "--transprotect-k",
        type=int,
        default=10,
        help="TransProtect top-K candidate-set size (independent of thesis K)",
    )
    parser.add_argument(
        "--transprotect-target-count",
        type=int,
        default=8,
        help="number of disjoint-background target proxies for Equation (13)",
    )
    parser.add_argument(
        "--transprotect-alpha",
        type=float,
        default=10_000.0,
        help="utility weighting alpha in the TransProtect top-K score",
    )
    parser.add_argument(
        "--transprotect-probability-smoothing",
        type=float,
        default=1e-6,
        help="additive smoothing for the explicitly labelled local Markov proxy",
    )
    parser.add_argument(
        "--transprotect-probability-backoff-weight",
        type=float,
        default=0.1,
        help="global-frequency backoff weight for the local Markov proxy",
    )
    parser.add_argument("--k", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--qos-radius-m", type=float, default=200.0)
    parser.add_argument(
        "--require-faithful-sota",
        "--require-reproduced-sota",
        dest="require_faithful_sota",
        action="store_true",
        help=(
            "fail closed unless every paper comparator is official or a faithful "
            "validated reimplementation (currently unavailable)"
        ),
    )
    parser.add_argument("--output", default=DEFAULT_OUTPUT)
    parser.add_argument("--map-output", default=DEFAULT_MAP_OUTPUT)
    parser.add_argument(
        "--preview-output",
        default=DEFAULT_PREVIEW_OUTPUT,
        help="offline PNG preview written whenever map generation is enabled",
    )
    parser.add_argument(
        "--no-map",
        action="store_true",
        help="disable both the interactive HTML map and static PNG preview",
    )
    args = parser.parse_args(argv)
    if args.quick:
        args.n_trajectories = 1
        args.max_points = min(args.max_points, 8)
    if args.n_trajectories < 1:
        parser.error("--n-trajectories must be at least 1")
    if args.mobility_source == "sumo" and args.n_trajectories != 1:
        parser.error(
            "the current SUMO pilot emits exactly one selected trajectory; "
            "use --n-trajectories 1"
        )
    if args.max_points < 2:
        parser.error("--max-points must be at least 2")
    if args.interval_s < 1:
        parser.error("--interval-s must be at least 1")
    if args.epsilon <= 0:
        parser.error("--epsilon must be positive")
    if args.transprotect_epsilon_per_km <= 0:
        parser.error("--transprotect-epsilon-per-km must be positive")
    if args.transprotect_k < 1:
        parser.error("--transprotect-k must be at least 1")
    if args.transprotect_target_count < 1:
        parser.error("--transprotect-target-count must be at least 1")
    if args.transprotect_alpha <= 0:
        parser.error("--transprotect-alpha must be positive")
    if args.transprotect_probability_smoothing <= 0:
        parser.error("--transprotect-probability-smoothing must be positive")
    if not 0 <= args.transprotect_probability_backoff_weight <= 1:
        parser.error(
            "--transprotect-probability-backoff-weight must be in [0, 1]"
        )
    if args.k < 2:
        parser.error("--k must be at least 2")
    if args.qos_radius_m <= 0:
        parser.error("--qos-radius-m must be positive")
    if args.sumo_route_seed < 0:
        parser.error("--sumo-route-seed must be non-negative")
    if args.sumo_simulation_seed < 0:
        parser.error("--sumo-simulation-seed must be non-negative")
    if args.sumo_min_trip_distance_m < 0:
        parser.error("--sumo-min-trip-distance-m must be non-negative")
    if args.sumo_demand_end_s <= 0:
        parser.error("--sumo-demand-end-s must be positive")
    if args.sumo_simulation_end_s < args.sumo_demand_end_s:
        parser.error(
            "--sumo-simulation-end-s must be at least --sumo-demand-end-s"
        )
    return args
