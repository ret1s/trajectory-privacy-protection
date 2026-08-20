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
)
from evaluation import metrics
from evaluation.attacks import BayesianPointAttack, HMMTrackingAttack
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
    ]


# ε is per meter: the exponential mechanisms' expected displacement is ~2/ε·2=...
# for planar Laplace E[r] = 2/ε (Gamma(2,1/ε)); ε=0.02 → E[r]=100m, ε=0.01 → 200m.
EPSILONS = [0.01, 0.02, 0.05]

EMISSION_SCALE = {
    "planar_laplace": 1.0,
    "baseline_thesis": 1.0,
    "road_exponential": 0.5,
    "temporal_road_exponential": 0.5,
}


def run(n_trajectories=20, epsilons=EPSILONS):
    print("Loading road network...", flush=True)
    rn = RoadNetwork.from_pickle(GRAPH_PKL)
    print(f"  {len(rn)} candidate vertices")

    print("Loading GeoLife trajectories...", flush=True)
    # 20s sampling keeps consecutive points strongly correlated — the regime
    # where temporal-correlation attacks (and T-REM's consistency weighting)
    # actually matter.
    trajs = load_trajectories(n_trajectories=n_trajectories, interval_s=20)
    print(f"  {len(trajs)} trajectories, "
          f"{sum(len(t['points']) for t in trajs)} points total")

    knn = metrics.KnnPoiUtility(rn)

    results = []
    for eps in epsilons:
        rng = np.random.default_rng(SEED)
        for mech in build_mechanisms(eps, rn, rng):
            t0 = time.time()
            point_attack = BayesianPointAttack(
                rn, eps, emission_scale=EMISSION_SCALE[mech.name]
            )
            hmm_attack = HMMTrackingAttack(
                rn, eps, emission_scale=EMISSION_SCALE[mech.name]
            )
            per_traj = []
            for traj in trajs:
                real, times = traj["points"], traj["times"]
                mech.reset()
                released = [
                    mech.perturb(lat, lon, t=t) for (lat, lon), t in zip(real, times)
                ]
                per_traj.append(
                    {
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
                )
            row = {"mechanism": mech.name, "epsilon": eps}
            row.update(metrics.summarize(per_traj))
            row["runtime_s"] = round(time.time() - t0, 1)
            results.append(row)
            print(
                f"eps={eps:<5} {mech.name:<26} "
                f"disp={row['mean_disp']:6.1f}m qos={row['qos']:.2f} "
                f"road={row['on_road']:.2f} spd_viol={row['speed_viol']:.2f} "
                f"bayes={row['bayes_err']:6.1f}m hmm={row['hmm_err']:6.1f}m "
                f"({row['runtime_s']}s)",
                flush=True,
            )

    os.makedirs("outputs", exist_ok=True)
    out = os.path.join("outputs", "benchmark_results.json")
    with open(out, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved {out}")
    return results


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true", help="5 trajectories, 1 epsilon")
    args = ap.parse_args()
    if args.quick:
        run(n_trajectories=5, epsilons=[0.02])
    else:
        run()
