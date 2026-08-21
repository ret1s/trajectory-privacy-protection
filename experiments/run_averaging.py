"""
Averaging / home-inference experiment (scenario S4).

Simulates a user reporting repeatedly from a static "home" point and measures
how the adversary's averaged-estimate error changes with the number of reports
n, for each mechanism. This is the documented real-world harm (Strava home-zone
recovery; data-broker home fingerprinting) that noise-per-report mechanisms fail
against and that StayMemoizedREM is designed to close.

Usage (from repo root):
    python -m experiments.run_averaging

Outputs outputs/averaging_results.json and prints the error-vs-n table.
"""
import json
import os

import numpy as np

from core.road_network import RoadNetwork
from core.mechanisms import (
    PlanarLaplace,
    BaselineThesis,
    RoadExponential,
    TemporalRoadExponential,
    StayMemoizedREM,
)
from evaluation.attacks import AveragingAttack
from data.geolife import load_trajectories

GRAPH_PKL = os.path.join("data", "raw", "beijing_graph.pkl")
QOS_RADIUS = 200.0
EPS = 0.02
N_REPORTS = 100
SEED = 42


def build(eps, rn, rng):
    return [
        PlanarLaplace(eps, rng=rng),
        BaselineThesis(eps, QOS_RADIUS, rn, rng=rng),
        RoadExponential(eps, rn, rng=rng),
        TemporalRoadExponential(eps, rn, rng=rng),
        StayMemoizedREM(eps, rn, rng=rng),
    ]


def run():
    rn = RoadNetwork.from_pickle(GRAPH_PKL)
    # Use the first point of the first GeoLife trajectory as the "home" point.
    trajs = load_trajectories(n_trajectories=1, interval_s=20)
    home_lat, home_lon = trajs[0]["points"][0]
    print(f"Home point: {home_lat:.6f}, {home_lon:.6f}")
    print(f"Averaging attack, {N_REPORTS} reports, eps={EPS}\n")

    attack = AveragingAttack(rn)
    ks = [1, 2, 5, 10, 20, 50, 100]
    header = "mechanism".ljust(26) + "".join(f"n={k}".rjust(9) for k in ks)
    print(header)
    print("-" * len(header))

    results = []
    rng = np.random.default_rng(SEED)
    for mech in build(EPS, rn, rng):
        curve = attack.run(mech, home_lat, home_lon, N_REPORTS)
        row = {"mechanism": mech.name, "epsilon": EPS, "curve": curve}
        if hasattr(mech, "distinct_releases"):
            row["distinct_releases"] = mech.distinct_releases
        results.append(row)
        line = mech.name.ljust(26) + "".join(
            f"{curve[k]:8.1f}m" for k in ks if k in curve
        )
        print(line)

    os.makedirs("outputs", exist_ok=True)
    out = os.path.join("outputs", "averaging_results.json")
    with open(out, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved {out}")
    print(
        "\nRead: error is distance (m) from the adversary's averaged estimate to\n"
        "the true home. Lower = adversary wins. A mechanism is robust if the\n"
        "error does NOT shrink as n grows."
    )
    return results


if __name__ == "__main__":
    run()
