"""
Benchmark: baseline internship-2 mechanism vs. pure planar Laplace vs. the
proposed road-network exponential mechanisms, on real GeoLife trajectories
(Beijing) over the OSM road graph.

Usage (from repo root):
    python -m experiments.run_benchmark [--quick]

Outputs:
    outputs/benchmark_results.json  — full per-(mechanism, epsilon) table
    printed summary table
"""
import argparse
import json
import os
import time

import numpy as np

from core.road_network import RoadNetwork
from core.mechanisms import (
    PlanarLaplace,
    BaselineThesis,
    RoadExponential,
    TemporalRoadExponential,
    StayMemoizedREM,
    PrivateReuseSMREM,
)
from evaluation import metrics
from evaluation.attacks import (
    BayesianPointAttack,
    HMMTrackingAttack,
    precompute_lognorm,
)
from experiments.rng_util import semantic_rng
from experiments.provenance import provenance, assert_graph_matches_manifest
from data.geolife import load_trajectories

GRAPH_PKL = os.path.join("data", "raw", "beijing_graph.pkl")
QOS_RADIUS = 200.0  # meters — same as the internship-2 experiments
SEED = 42


def build_mechanisms(eps, rn, rng):
    return [
        PlanarLaplace(eps, rng=rng),
        BaselineThesis(eps, QOS_RADIUS, rn, rng=rng),
        RoadExponential(eps, rn, rng=rng),
        TemporalRoadExponential(eps, rn, rng=rng),
        StayMemoizedREM(eps, rn, rng=rng),
        # Matched budget (verifier R2-003/R3-012): epsilon_step_cap=ε splits into
        # ε/2 release + ε/2 test, so a resample step's worst-case per-step cost
        # is ε — the same as REM. Keyword-only API prevents a silent 2ε cap.
        PrivateReuseSMREM(rn, epsilon_step_cap=eps, rng=rng),
    ]


# ε is per meter: the exponential mechanisms' expected displacement is ~2/ε·2=...
# for planar Laplace E[r] = 2/ε (Gamma(2,1/ε)); ε=0.02 → E[r]=100m, ε=0.01 → 200m.
EPSILONS = [0.01, 0.02, 0.05]

# Attacker emission per mechanism, as a multiplier on the column ε, and whether
# the exact input-dependent road normaliser logZ applies. Planar Laplace and the
# capped baseline have a (near-)constant normaliser, so they must NOT be given the
# road logZ (verifier R2-004). PR-SM-REM's release uses ε/2 (matched budget), and
# REM scores exp(-ε/2·d), so its emission multiplier is 0.5·0.5 = 0.25.
EMISSION_SCALE = {
    "planar_laplace": 1.0,
    "baseline_thesis": 1.0,
    "road_exponential": 0.5,
    "temporal_road_exponential": 0.5,
    "stay_memoized_rem": 0.5,
    "pr_sm_rem": 0.25,
}
USE_LOGNORM = {
    "planar_laplace": False,
    "baseline_thesis": False,
    "road_exponential": True,
    "temporal_road_exponential": True,
    "stay_memoized_rem": True,
    "pr_sm_rem": True,
}


def run(n_trajectories=20, epsilons=EPSILONS, quick=False):
    print("Loading road network...", flush=True)
    rn = RoadNetwork.from_pickle(GRAPH_PKL)
    # Fail closed if the graph is not the pinned artifact (verifier R3-008).
    assert_graph_matches_manifest(rn)
    print(f"  {len(rn)} candidate vertices (graph hash verified)")

    print("Loading GeoLife trajectories...", flush=True)
    # 20s sampling keeps consecutive points strongly correlated — the regime
    # where temporal-correlation attacks (and T-REM's consistency weighting)
    # actually matter.
    trajs = load_trajectories(n_trajectories=n_trajectories, interval_s=20)
    print(f"  {len(trajs)} trajectories, "
          f"{sum(len(t['points']) for t in trajs)} points total")
    # Stable IDs of the exact records used, for provenance (verifier R3-003).
    # One .plt can split into several segments, so append a stable index.
    selected_ids = [f"{t['user']}/{t['file']}#{i}" for i, t in enumerate(trajs)]

    knn = metrics.KnnPoiUtility(rn)

    results = []
    raw_rows = []  # tidy per-(mechanism, ε, trajectory) rows (verifier R3-010)
    for eps in epsilons:
        # Exact input-dependent log-normaliser of the REM emission, precomputed
        # once per (ε, scale) and shared by both attacks (verifier V-004).
        lognorm = {
            s: precompute_lognorm(rn, eps, scale=s)
            for s in sorted(set(EMISSION_SCALE.values()))
        }
        zero_norm = np.zeros(len(rn))  # constant normaliser (planar/baseline)
        # Mechanisms are built with a throwaway RNG then given an independent,
        # order-INDEPENDENT stream keyed by the stable semantic identity
        # (root seed, exact ε, mechanism name) — not list position (R3-002).
        rng = np.random.default_rng(SEED)
        for mech in build_mechanisms(eps, rn, rng):
            mech.rng = semantic_rng(SEED, eps, mech.name)
            t0 = time.time()
            scale = EMISSION_SCALE[mech.name]
            ln = lognorm[scale] if USE_LOGNORM[mech.name] else zero_norm
            point_attack = BayesianPointAttack(
                rn, eps, emission_scale=scale, lognorm=ln
            )
            hmm_attack = HMMTrackingAttack(
                rn, eps, emission_scale=scale, lognorm=ln
            )
            per_traj = []
            for ti, traj in enumerate(trajs):
                real, times = traj["points"], traj["times"]
                mech.reset()
                released = [
                    mech.perturb(lat, lon, t=t) for (lat, lon), t in zip(real, times)
                ]
                m = {
                    "mean_disp": metrics.mean_displacement(real, released, rn.proj),
                    "max_disp": metrics.max_displacement(real, released, rn.proj),
                    "qos": metrics.qos_satisfaction(
                        real, released, rn.proj, QOS_RADIUS
                    ),
                    "hausdorff": metrics.hausdorff(real, released, rn.proj),
                    "dtw": metrics.dtw(real, released, rn.proj),
                    "on_road": metrics.on_road_rate(released, rn),
                    "speed_viol": metrics.speed_violation_rate(
                        released, times, rn.proj
                    ),
                    "knn_recall": knn.recall(real, released),
                    "bayes_err": point_attack.error(real, released),
                    "hmm_err": hmm_attack.error(real, released, times),
                }
                per_traj.append(m)
                raw_rows.append({
                    "mechanism": mech.name, "epsilon": eps,
                    "record_id": f"{traj['user']}/{traj['file']}#{ti}",
                    "n_points": len(real), **{k: round(v, 4) for k, v in m.items()},
                })
            row = {"mechanism": mech.name, "epsilon": eps}
            row.update(metrics.summarize(per_traj))
            row["hmm_coverage"] = round(
                hmm_attack.true_covered / max(1, hmm_attack.true_total), 3
            )
            row["runtime_s"] = round(time.time() - t0, 1)
            results.append(row)
            print(
                f"eps={eps:<5} {mech.name:<26} "
                f"disp={row['mean_disp']:6.1f}m qos={row['qos']:.2f} "
                f"road={row['on_road']:.2f} spd_viol={row['speed_viol']:.2f} "
                f"bayes={row['bayes_err']:6.1f}m hmm={row['hmm_err']:6.1f}m "
                f"cov={row['hmm_coverage']:.2f} ({row['runtime_s']}s)",
                flush=True,
            )

    prov = provenance(
        rn, epsilons, root_seeds=[SEED], quick=quick,
        extra={
            "qos_radius": QOS_RADIUS,
            "n_trajectories": len(trajs),
            "n_users": len(set(t["user"] for t in trajs)),
            "selected_record_ids": selected_ids,
            "emission_scale": EMISSION_SCALE,
            "use_lognorm": USE_LOGNORM,
            "pr_sm_rem": {"epsilon_release": "eps/2", "epsilon_test": "eps/2",
                          "epsilon_step_cap": "eps", "theta": 200.0},
            "note": "MLE/HMM attackers are REM-emission proxies (exact for REM only), "
                    "so attacker columns are UPPER BOUNDS for non-REM mechanisms and "
                    "must not be used to rank privacy (R3-007). Guarantees are "
                    "ideal-kernel; sampler is finite-precision (see docs/reviews). "
                    "Single root seed — treat as exploratory, not multi-seed CI (R3-010).",
        })
    os.makedirs("outputs", exist_ok=True)
    out = os.path.join("outputs", "benchmark_results.json")
    with open(out, "w") as f:
        json.dump({"provenance": prov, "rows": results, "raw_rows": raw_rows},
                  f, indent=2)
    print(f"\nSaved {out} ({len(raw_rows)} raw rows)")
    return results


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true", help="5 trajectories, 1 epsilon")
    args = ap.parse_args()
    if args.quick:
        run(n_trajectories=5, epsilons=[0.02], quick=True)
    else:
        run()
