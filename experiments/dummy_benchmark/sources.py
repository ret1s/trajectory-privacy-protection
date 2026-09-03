"""Mobility-source adapters and evaluator-only metadata separation."""

from __future__ import annotations

from collections.abc import Callable

from data.sumo_demo import SumoSmokeConfig

from .constants import SUMO_GRAPH_COMPATIBILITY_NOTE


def load_sumo_record(args, *, run_sumo_smoke_demo: Callable):
    """Run SUMO once and preserve privileged simulator state for evaluation.

    Only ``record.to_mechanism_input()`` is returned to the model loop. Vehicle
    identity, route, speed, edge, and lane data live in the separate
    ``evaluator_only`` object and are never merged into an attacker transcript.

    The simulator callable is injected so the stable compatibility module can
    retain its long-standing monkey-patch point.
    """

    config = SumoSmokeConfig(
        route_seed=args.sumo_route_seed,
        simulation_seed=args.sumo_simulation_seed,
        demand_end_s=args.sumo_demand_end_s,
        simulation_end_s=args.sumo_simulation_end_s,
        resample_interval_s=args.interval_s,
        max_points=args.max_points,
        min_points=min(8, args.max_points),
        min_trip_distance_m=args.sumo_min_trip_distance_m,
    )
    record = run_sumo_smoke_demo(
        osm_path=args.sumo_osm_path,
        workdir=args.sumo_workdir,
        config=config,
    )
    mobility = {
        "source": "sumo",
        "label": "Eclipse SUMO controlled Beijing passenger smoke scenario",
        "sumo_runs": [record.provenance.to_dict()],
        "evaluator_only": [
            {
                "record_id": record.record_id,
                **record.evaluator_only.to_dict(),
            }
        ],
        "candidate_graph_compatibility": SUMO_GRAPH_COMPATIBILITY_NOTE,
        "model_training_context": {
            "source": "held-out SUMO background vehicles",
            "held_out_selected_vehicle": True,
            "trajectory_count": len(
                getattr(record, "background_trajectories", ())
            ),
        },
        "_model_training_records": [
            trajectory.to_model_input()
            for trajectory in getattr(record, "background_trajectories", ())
        ],
        "_candidate_network_path": getattr(record, "network_path", None),
    }
    return [record.to_mechanism_input()], mobility


def load_geolife_records(args, *, load_trajectories: Callable):
    """Load explicit real-data validation records, failing closed if absent.

    The loader callable is injected for the same stable patch boundary as the
    SUMO adapter.
    """

    loaded = load_trajectories(
        n_trajectories=args.n_trajectories,
        min_points=min(8, args.max_points),
        max_points=args.max_points,
        interval_s=args.interval_s,
        min_span_m=200,
    )
    if not loaded:
        raise FileNotFoundError(
            "--mobility-source geolife requested, but no qualifying local "
            "GeoLife trace was found. Install the dataset or choose SUMO; no "
            "synthetic fallback is available."
        )

    records = []
    evaluator_only = []
    for index, record in enumerate(loaded):
        record_id = f"geolife/evaluation_{index:04d}"
        records.append(
            {
                "record_id": record_id,
                "points": record["points"],
                "times": record["times"],
            }
        )
        evaluator_only.append(
            {
                "record_id": record_id,
                "source_user": record.get("user"),
                "source_file": record.get("file"),
            }
        )
    mobility = {
        "source": "geolife",
        "label": "Microsoft GeoLife v1.3 real-data validation",
        "sumo_runs": [],
        "evaluator_only": evaluator_only,
        "candidate_graph_compatibility": (
            "GeoLife coordinates are evaluated against the pinned OSMnx graph; "
            "this optional mode does not execute SUMO."
        ),
        "model_training_context": {
            "source": "road-topology proxy",
            "held_out_selected_vehicle": True,
            "trajectory_count": 0,
        },
        "_model_training_records": [],
    }
    return records, mobility


def load_records(
    args,
    *,
    run_sumo_smoke_demo: Callable,
    load_trajectories: Callable,
):
    if args.mobility_source == "sumo":
        return load_sumo_record(args, run_sumo_smoke_demo=run_sumo_smoke_demo)
    if args.mobility_source == "geolife":
        return load_geolife_records(args, load_trajectories=load_trajectories)
    raise ValueError(f"unsupported mobility source: {args.mobility_source!r}")
