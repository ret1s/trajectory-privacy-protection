"""Mechanism construction and execution for one benchmark record."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import time
from typing import Mapping, Sequence

import numpy as np

from benchmark.contracts import MethodCard
from benchmark.methods import (
    AnotherMeAdaptation,
    GeoIAnchoredDummyTrajectories,
    SemanticCorrelationComparator,
    TransProtectAdaptation,
)
from core.demo_protocol import ProtectedRun, TrajectoryPoint
from core.road_network import RoadNetwork
from experiments.rng_util import rng_from_key

from .constants import BENCHMARK_SCHEMA
from .results import round_value


@dataclass(frozen=True)
class RuntimeEvidence:
    """Separate per-method construction cost from online protection cost."""

    setup_runtime_ms: float
    inference_runtime_ms: float
    paper_metrics: Mapping[str, object]

    @property
    def end_to_end_runtime_ms(self) -> float:
        return self.setup_runtime_ms + self.inference_runtime_ms

    def to_dict(self) -> dict[str, float]:
        return {
            "setup_runtime_ms": round_value(self.setup_runtime_ms),
            "inference_runtime_ms": round_value(self.inference_runtime_ms),
            "end_to_end_runtime_ms": round_value(self.end_to_end_runtime_ms),
        }


def timestamp_s(value, fallback: float) -> float:
    if value is None:
        return float(fallback)
    if isinstance(value, datetime):
        if value.tzinfo is None:
            value = value.replace(tzinfo=timezone.utc)
        return float(value.timestamp())
    try:
        return float(value)
    except (TypeError, ValueError):
        return float(fallback)


def contract_trajectory(
    points: Sequence, times: Sequence
) -> tuple[TrajectoryPoint, ...]:
    if len(points) != len(times):
        raise ValueError("points and times must have the same length")
    return tuple(
        TrajectoryPoint(timestamp_s(t, i * 60.0), float(lat), float(lon))
        for i, ((lat, lon), t) in enumerate(zip(points, times))
    )


def model_rng(seed: int, epsilon: float, mechanism: str, record_id: str):
    return rng_from_key(
        int(seed),
        float(epsilon),
        mechanism,
        record_id,
        schema=f"{BENCHMARK_SCHEMA}-rng",
    )


def run_models(
    rn: RoadNetwork,
    points,
    times,
    record_id: str,
    *,
    training_trajectories=(),
    epsilon: float,
    transprotect_epsilon_per_km: float,
    transprotect_k: int,
    transprotect_target_count: int,
    transprotect_alpha: float,
    transprotect_probability_smoothing: float,
    transprotect_probability_backoff_weight: float,
    k: int,
    seed: int,
) -> list[tuple[ProtectedRun, MethodCard, RuntimeEvidence]]:
    """Return each protected run, method card, and runtime evidence."""

    real = contract_trajectory(points, times)
    transprotect_epsilon_per_m = float(transprotect_epsilon_per_km) / 1_000.0
    model_factories = (
        lambda: TransProtectAdaptation.from_road_network(
            rn,
            training_trajectories=training_trajectories,
            candidate_k=transprotect_k,
            target_count=transprotect_target_count,
            alpha=transprotect_alpha,
            epsilon=transprotect_epsilon_per_m,
            probability_smoothing=transprotect_probability_smoothing,
            probability_backoff_weight=transprotect_probability_backoff_weight,
            rng=model_rng(
                seed,
                transprotect_epsilon_per_m,
                TransProtectAdaptation.name,
                record_id,
            ),
        ),
        lambda: AnotherMeAdaptation(
            rn,
            # AnotherMe has no epsilon parameter. Its random stream must stay
            # fixed when only the thesis privacy budget is swept.
            rng=model_rng(seed, 0.0, AnotherMeAdaptation.name, record_id),
        ),
        lambda: SemanticCorrelationComparator(
            rn,
            k=k,
            # The semantic scheme is configured by K, not thesis epsilon.
            rng=model_rng(
                seed,
                float(k),
                SemanticCorrelationComparator.name,
                record_id,
            ),
        ),
        lambda: GeoIAnchoredDummyTrajectories(
            epsilon,
            rn,
            k=k,
            rng=model_rng(
                seed,
                epsilon,
                GeoIAnchoredDummyTrajectories.name,
                record_id,
            ),
        ),
    )

    completed = []
    for build_model in model_factories:
        setup_started = time.perf_counter()
        mechanism = build_model()
        setup_runtime_ms = (time.perf_counter() - setup_started) * 1_000.0
        inference_started = time.perf_counter()
        # Every method owns its adapter. Keeping truth separation next to
        # mechanism-specific internals avoids re-encoding a private real index
        # or REM anchor in this generic runner.
        protected = mechanism.protect_run(real)
        inference_runtime_ms = (time.perf_counter() - inference_started) * 1_000.0
        paper_metrics: dict[str, object] = {}
        if isinstance(mechanism, TransProtectAdaptation):
            paper_metrics = {
                "expected_travel_cost_loss_m": round_value(
                    np.mean(mechanism.last_output_utility_losses)
                ),
                "forced_real_membership_events": int(
                    sum(mechanism.last_forced_real_membership)
                ),
                "vehitrack_eie_m": None,
                "vehitrack_eie_status": (
                    "not_available_without_paper_equivalent_VehiTrack_attack"
                ),
            }
        elif isinstance(mechanism, AnotherMeAdaptation):
            trace = mechanism.last_trace
            paper_metrics = {
                "raw_virtual_samples": len(trace.virtual_trajectory) if trace else None,
                "transport_mode": trace.transport_mode.value if trace else None,
                "raw_three_second_grid_preserved": bool(
                    trace
                    and all(
                        current.timestamp_s - previous.timestamp_s == 3.0
                        for previous, current in zip(
                            trace.virtual_trajectory,
                            trace.virtual_trajectory[1:],
                        )
                    )
                ),
                "benchmark_alignment_status": (
                    "adapted_to_real_event_grid_not_paper_temporal_parity"
                ),
            }
        elif isinstance(mechanism, SemanticCorrelationComparator):
            paper_metrics = {
                "paper_asr_percent": None,
                "paper_asr_status": (
                    "not_available_without_calibrated_LSP_posterior"
                ),
                "paper_der": None,
                "paper_der_status": (
                    "not_available_without_paper_effectiveness_labels"
                ),
            }
        completed.append(
            (
                protected,
                mechanism.method_card,
                RuntimeEvidence(
                    setup_runtime_ms,
                    inference_runtime_ms,
                    paper_metrics,
                ),
            )
        )
    return completed
