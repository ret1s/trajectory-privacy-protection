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
    a REM-emission-form MLE proxy  x̂ = argmax_x[−a·Σd(x,z_i) − n·logZ(x)]
    (exact for REM; a proxy — not the optimal attacker — for the others);
  * metrics: expected inference error vs n, and success probability P[err ≤ r];
  * bootstrap 95% CI over homes, cluster-resampled by user.

Usage (from repo root):
    python -m experiments.run_averaging_multi [--quick]
"""
import argparse
import json
import os

import numpy as np

from experiments.artifact_paths import AVERAGING_RESULTS_PATH
from core.road_network import RoadNetwork
from core.mechanisms import (
    PlanarLaplace,
    RoadExponential,
    TemporalRoadExponential,
    StayMemoizedREM,
    PrivateReuseSMREM,
)
from evaluation.attacks import AveragingAttack, precompute_lognorm
from experiments.rng_util import rng_from_key
from experiments.provenance import provenance, assert_graph_matches_manifest, begin_run
from data.geolife import load_stay_points

GRAPH_PKL = os.path.join("data", "raw", "beijing_graph.pkl")
SEED = 42
BOOT_SEED = 0
N_BOOT = 1000
EPS = 0.02
N_REPORTS = 100
JITTER_M = 10.0            # consumer-GPS urban jitter (PLOS ONE 2019: 7–13 m)
KS = (1, 2, 5, 10, 20, 50, 100)
RADII = (50.0, 100.0, 200.0)
SCALE = {"planar_laplace": 1.0, "road_exponential": 0.5,
         "temporal_road_exponential": 0.5, "stay_memoized_rem": 0.5,
         "pr_sm_rem": 0.25}  # release ε/2, REM exp(-ε/2·d) → 0.5·0.5 (R2-003)
USE_LOGNORM = {"planar_laplace": False, "road_exponential": True,
               "temporal_road_exponential": True, "stay_memoized_rem": True,
               "pr_sm_rem": True}


def mech_factories(eps, rn):
    return {
        "planar_laplace": lambda: PlanarLaplace(eps),
        "road_exponential": lambda: RoadExponential(eps, rn),
        "temporal_road_exponential": lambda: TemporalRoadExponential(eps, rn),
        "stay_memoized_rem": lambda: StayMemoizedREM(eps, rn),
        # matched budget: epsilon_step_cap=ε → ε/2 release + ε/2 test (R2-003/R3-012)
        "pr_sm_rem": lambda: PrivateReuseSMREM(rn, epsilon_step_cap=eps),
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


def run(n_homes=40, seeds=8, eps=EPS, quick=False):
    run_ctx = begin_run()  # capture source/env/start BEFORE computation (R4-008)
    rn = RoadNetwork.from_pickle(GRAPH_PKL)
    assert_graph_matches_manifest(rn)  # fail closed on wrong graph (R3-008)
    homes = load_stay_points(n_homes=n_homes)
    # Unique, source-backed home ID (verifier R4-001): user/file/segment/stay.
    home_ids = [h["uid"] for h in homes]
    assert len(home_ids) == len(set(home_ids)), "home_ids must be unique (R4-001)"
    print(f"{len(homes)} distinct stay-points across "
          f"{len(set(h['user'] for h in homes))} users; "
          f"{seeds} seeds; jitter σ={JITTER_M}m; eps={eps}\n")

    lognorm = {s: precompute_lognorm(rn, eps, scale=s) for s in set(SCALE.values())}
    zero_norm = np.zeros(len(rn))  # constant normaliser (planar Laplace, R2-004)
    factories = mech_factories(eps, rn)

    # raw[mech][estimator][k] -> {user: [errors]}
    raw = {m: {"mean": {k: {} for k in KS}, "mle": {k: {} for k in KS}}
           for m in factories}
    raw_rows = []  # tidy per-(mechanism, home, seed) rows (verifier R3-010)

    for hi, h in enumerate(homes):
        user, (hlat, hlon) = h["user"], h["home"]
        hid = home_ids[hi]
        for m, factory in factories.items():
            attack = AveragingAttack(
                rn, eps, emission_scale=SCALE[m],
                lognorm=lognorm[SCALE[m]] if USE_LOGNORM[m] else zero_norm)
            for s in range(seeds):
                mech = factory()
                # Mechanism RNG keys ON the mechanism (R3-002); the EXOGENOUS GPS
                # jitter RNG does NOT — every mechanism sees the SAME input-noise
                # realization for a given (home, seed), so the comparison is
                # PAIRED (verifier R4-002).
                mech.rng = rng_from_key("avg-mech", SEED, eps, m, hid, s)
                jitter_rng = rng_from_key("avg-jitter", SEED, eps, hid, s)
                res = attack.run(mech, hlat, hlon, N_REPORTS, jitter_m=JITTER_M,
                                 ks=KS, rng=jitter_rng)
                for est in ("mean", "mle"):
                    for k in KS:
                        raw[m][est][k].setdefault(user, []).append(res[est][k])
                raw_rows.append({
                    "mechanism": m, "home_id": hid, "user": user, "seed": s,
                    "mean": {int(k): round(float(res["mean"][k]), 4) for k in KS},
                    "mle": {int(k): round(float(res["mle"][k]), 4) for k in KS},
                })
        if (hi + 1) % 10 == 0:
            print(f"  ...{hi+1}/{len(homes)} homes", flush=True)

    # Composite raw key must be a genuine primary key (verifier R4-001).
    comp = [(r["mechanism"], r["home_id"], r["seed"]) for r in raw_rows]
    assert len(comp) == len(set(comp)), "raw (mechanism, home_id, seed) must be unique"

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

    config = provenance(
        rn, [eps], root_seeds=[SEED], quick=quick, begin=run_ctx,
        extra={
            "estimand": "distinct significant location (stay-point centre); "
                        "near-identical same-user stays deduped within dedup_m",
            "n_locations": len(homes),
            "n_users": len(set(h["user"] for h in homes)),
            "dedup_m": 25.0,
            "selected_home_ids": home_ids,
            "paired_jitter": "GPS jitter RNG keyed by (root,eps,home,seed) only — "
                             "same input noise across mechanisms (R4-002)",
            "seeds": seeds, "eps": eps, "jitter_m": JITTER_M,
            "n_reports": N_REPORTS, "ks": list(KS), "radii": list(RADII),
            "scale": SCALE, "bootstrap": {"n_boot": N_BOOT, "seed": BOOT_SEED,
                                          "cluster_unit": "user"},
            "pr_sm_rem": {"epsilon_release": "eps/2", "epsilon_test": "eps/2",
                          "epsilon_step_cap": "eps", "theta": 200.0},
            "estimator_note": "MLE is a REM-emission-form PROXY (exact for REM only); "
                              "jitter not in likelihood; NOT the optimal attacker "
                              "(R2-004/R3-007) — do not rank privacy across non-REM "
                              "mechanisms from these numbers. Locations are stay-points, "
                              "not ground-truth homes (R2-008). Success probs pooled "
                              "without user-cluster CI (R3-010, future work).",
        })
    AVERAGING_RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    out = AVERAGING_RESULTS_PATH
    with out.open("w", encoding="utf-8") as f:
        json.dump({"config": config, "results": results, "raw_rows": raw_rows},
                  f, indent=2)
    print(f"\nSaved {out} ({len(raw_rows)} raw rows)")
    return results


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    args = ap.parse_args()
    if args.quick:
        run(n_homes=6, seeds=3, quick=True)
    else:
        run()
