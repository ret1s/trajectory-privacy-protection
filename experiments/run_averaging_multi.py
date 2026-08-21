"""
Multi-home averaging / home-inference study (scenario S4), rigorous version.

Turns the single-home illustration into statistical evidence, following the
methodology of the location-privacy literature (Shokri S&P 2011 expected error;
Dhondt et al. CCS 2022 success-rate + bootstrap; RAPPOR CCS 2014 memoization;
Eclipse IEEE TMC 2020 long-term-observation threat):

  * population of REAL stay-point "homes" from GeoLife (Li 2008 / Zheng 2009),
    capped per user;
  * per home × seed, n reports with Gaussian GPS jitter (σ), through each
    mechanism;
  * TWO adversary estimators (verifier V-004/V-005): the naive sample MEAN and
    the consistent mechanism-aware MLE  x̂ = argmax_x[−a·Σd(x,z_i) − n·logZ(x)];
  * metrics: expected inference error vs n, and success probability P[err ≤ r];
  * bootstrap 95% CI over homes, cluster-resampled by user.

Usage (from repo root):
    python -m experiments.run_averaging_multi [--quick]
"""
import argparse
import json
import os

import numpy as np

from core.road_network import RoadNetwork
from core.mechanisms import (
    PlanarLaplace,
    RoadExponential,
    TemporalRoadExponential,
    StayMemoizedREM,
    PrivateReuseSMREM,
)
from evaluation.attacks import AveragingAttack, precompute_lognorm
from data.geolife import load_stay_points

GRAPH_PKL = os.path.join("data", "raw", "beijing_graph.pkl")
EPS = 0.02
N_REPORTS = 100
JITTER_M = 10.0            # consumer-GPS urban jitter (PLOS ONE 2019: 7–13 m)
KS = (1, 2, 5, 10, 20, 50, 100)
RADII = (50.0, 100.0, 200.0)
SCALE = {"planar_laplace": 1.0, "road_exponential": 0.5,
         "temporal_road_exponential": 0.5, "stay_memoized_rem": 0.5,
         "pr_sm_rem": 0.5}


def mech_factories(eps, rn):
    return {
        "planar_laplace": lambda: PlanarLaplace(eps),
        "road_exponential": lambda: RoadExponential(eps, rn),
        "temporal_road_exponential": lambda: TemporalRoadExponential(eps, rn),
        "stay_memoized_rem": lambda: StayMemoizedREM(eps, rn),
        "pr_sm_rem": lambda: PrivateReuseSMREM(eps, rn),
    }


def bootstrap_ci(values_by_user, reducer=np.median, n_boot=1000, seed=0):
    """Percentile 95% CI of `reducer` over homes, resampling USERS with
    replacement (cluster bootstrap) so multiple homes of one user don't inflate
    confidence."""
    rng = np.random.default_rng(seed)
    users = list(values_by_user.keys())
    point = reducer(np.concatenate([values_by_user[u] for u in users]))
    boots = []
    for _ in range(n_boot):
        pick = rng.choice(users, size=len(users), replace=True)
        vals = np.concatenate([values_by_user[u] for u in pick])
        boots.append(reducer(vals))
    lo, hi = np.percentile(boots, [2.5, 97.5])
    return float(point), float(lo), float(hi)


def run(n_homes=40, seeds=8, eps=EPS):
    rn = RoadNetwork.from_pickle(GRAPH_PKL)
    homes = load_stay_points(n_homes=n_homes)
    print(f"{len(homes)} stay-point homes across "
          f"{len(set(h['user'] for h in homes))} users; "
          f"{seeds} seeds; jitter σ={JITTER_M}m; eps={eps}\n")

    lognorm = {s: precompute_lognorm(rn, eps, scale=s) for s in set(SCALE.values())}
    factories = mech_factories(eps, rn)

    # raw[mech][estimator][k] -> {user: [errors]}
    raw = {m: {"mean": {k: {} for k in KS}, "mle": {k: {} for k in KS}}
           for m in factories}

    for hi, h in enumerate(homes):
        user, (hlat, hlon) = h["user"], h["home"]
        for m, factory in factories.items():
            attack = AveragingAttack(rn, eps, emission_scale=SCALE[m],
                                     lognorm=lognorm[SCALE[m]])
            for s in range(seeds):
                mech = factory()
                mech.rng = np.random.default_rng(1000 * hi + s)
                res = attack.run(mech, hlat, hlon, N_REPORTS, jitter_m=JITTER_M,
                                 ks=KS, rng=np.random.default_rng(7_000 * hi + s))
                for est in ("mean", "mle"):
                    for k in KS:
                        raw[m][est][k].setdefault(user, []).append(res[est][k])
        if (hi + 1) % 10 == 0:
            print(f"  ...{hi+1}/{len(homes)} homes", flush=True)

    # Aggregate: median error vs n (both estimators) with CI over homes.
    results = {}
    print(f"\n{'mechanism':<26}{'est':<5}" + "".join(f"n={k}".rjust(8) for k in KS))
    print("-" * (31 + 8 * len(KS)))
    for m in factories:
        results[m] = {}
        for est in ("mean", "mle"):
            row = {}
            for k in KS:
                vbu = {u: np.array(v) for u, v in raw[m][est][k].items()}
                pt, lo, hi_ = bootstrap_ci(vbu)
                row[k] = {"median": round(pt, 1), "ci": [round(lo, 1), round(hi_, 1)]}
            results[m][est] = row
            print(f"{m:<26}{est:<5}" + "".join(f"{row[k]['median']:7.1f}" for k in KS))

    # Success probability P[err <= r] at n=100 (MLE adversary).
    print(f"\nSuccess prob P[MLE err ≤ r] at n={N_REPORTS} (higher = adversary wins):")
    print(f"{'mechanism':<26}" + "".join(f"r={int(r)}m".rjust(10) for r in RADII))
    for m in factories:
        allerr = np.concatenate(
            [np.array(v) for v in raw[m]["mle"][N_REPORTS].values()]
        )
        succ = {r: float((allerr <= r).mean()) for r in RADII}
        results[m]["success_at_n100"] = {int(r): round(succ[r], 3) for r in RADII}
        print(f"{m:<26}" + "".join(f"{succ[r]*100:8.0f}%" for r in RADII))

    os.makedirs("outputs", exist_ok=True)
    out = os.path.join("outputs", "averaging_multi_results.json")
    with open(out, "w") as f:
        json.dump({"config": {"n_homes": len(homes), "seeds": seeds, "eps": eps,
                              "jitter_m": JITTER_M, "n_reports": N_REPORTS},
                   "results": results}, f, indent=2)
    print(f"\nSaved {out}")
    return results


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    args = ap.parse_args()
    if args.quick:
        run(n_homes=6, seeds=3)
    else:
        run()
