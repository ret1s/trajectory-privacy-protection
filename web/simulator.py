"""
LBS privacy simulator — replays real GeoLife trajectories through the privacy
mechanisms and shows, per timestep, what each party sees:

  * USER view      — the true position (green)
  * LBS view       — the released/obfuscated position the service receives (red)
  * ATTACKER view  — a live HMM-tracking adversary's best estimate from the
                     released stream only (black), with its error

plus a concrete LBS use case running on top: a k-nearest-POI query over a fixed
set of SYNTHETIC POIs (random road vertices, not real amenity=pharmacy nodes;
verifier V-011) issued at the released position, scored against the
ground-truth answer — making the privacy/QoS trade-off tangible.

Run from the repo root:
    python -m web.simulator          # http://localhost:5003

Real-life use cases this models (see README section "Use cases"):
  1. Ride-hailing / food delivery: the app needs your rough position for
     dispatch & ETA but should not learn your exact home address.
  2. Health: querying nearby pharmacies/clinics without revealing the exact
     (sensitive) location you are at.
  3. Fitness/social apps: sharing runs without exposing home start points
     to stalkers.
  4. Traffic crowdsensing: contributing speed probes without exposing your
     full commute.
"""
import os

from flask import Flask, render_template, request, jsonify
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
from evaluation.attacks import HMMTrackingAttack, precompute_lognorm
from evaluation import metrics
from data.geolife import load_trajectories

GRAPH_PKL = os.path.join("data", "raw", "beijing_graph.pkl")
QOS_RADIUS = 200.0

app = Flask(__name__)

print("Loading road network + GeoLife trajectories (once)...")
RN = RoadNetwork.from_pickle(GRAPH_PKL)
TRAJS = load_trajectories(n_trajectories=12, interval_s=20)
KNN = metrics.KnnPoiUtility(RN, n_pois=300, k=3, k_prime=3)
print(f"Ready: {len(RN)} road vertices, {len(TRAJS)} trajectories")

MECHS = {
    "planar_laplace": lambda eps: PlanarLaplace(eps),
    "baseline_thesis": lambda eps: BaselineThesis(eps, QOS_RADIUS, RN),
    "road_exponential": lambda eps: RoadExponential(eps, RN),
    "temporal_road_exponential": lambda eps: TemporalRoadExponential(eps, RN),
    "stay_memoized_rem": lambda eps: StayMemoizedREM(eps, RN),
    "pr_sm_rem": lambda eps: PrivateReuseSMREM(RN, epsilon_step_cap=eps),
}
EMISSION_SCALE = {
    "planar_laplace": 1.0,
    "baseline_thesis": 1.0,
    "road_exponential": 0.5,
    "temporal_road_exponential": 0.5,
    "stay_memoized_rem": 0.5,
    "pr_sm_rem": 0.25,
}
# Which mechanisms get the exact input-dependent road normaliser logZ (same as
# the offline benchmark) — so the simulator evaluates the IDENTICAL attacker,
# not a weaker one (verifier R4-007 online/offline parity).
USE_LOGNORM = {
    "planar_laplace": False, "baseline_thesis": False,
    "road_exponential": True, "temporal_road_exponential": True,
    "stay_memoized_rem": True, "pr_sm_rem": True,
}


def _poi_latlon(idx):
    x, y = KNN.poi_xy[idx]
    lat, lon = RN.proj.to_latlon(x, y)
    return [float(lat), float(lon)]


@app.route("/")
def index():
    return render_template(
        "simulator.html",
        trajectories=[
            {"i": i, "user": t["user"], "n": len(t["points"])}
            for i, t in enumerate(TRAJS)
        ],
    )


@app.route("/simulate")
def simulate():
    traj_i = int(request.args.get("traj", 0)) % len(TRAJS)
    mech_name = request.args.get("mech", "temporal_road_exponential")
    eps = float(request.args.get("eps", 0.02))
    seed = int(request.args.get("seed", 0))

    traj = TRAJS[traj_i]
    real, times = traj["points"], traj["times"]

    rng = np.random.default_rng(seed)
    mech = MECHS[mech_name](eps)
    mech.rng = rng
    mech.reset()
    released = [mech.perturb(lat, lon, t=t) for (lat, lon), t in zip(real, times)]

    scale = EMISSION_SCALE[mech_name]
    ln = precompute_lognorm(RN, eps, scale=scale) if USE_LOGNORM[mech_name] else None
    attacker = HMMTrackingAttack(RN, eps, emission_scale=scale, lognorm=ln)
    estimates = attacker.online_estimates(released, times)

    steps = []
    for i, ((rlat, rlon), (zlat, zlon), (alat, alon)) in enumerate(
        zip(real, released, estimates)
    ):
        rx, ry = RN.point_xy(rlat, rlon)
        zx, zy = RN.point_xy(zlat, zlon)
        ax, ay = RN.point_xy(alat, alon)
        poi_true = sorted(KNN._knn(rlat, rlon, KNN.k))
        poi_reported = sorted(KNN._knn(zlat, zlon, KNN.k))
        steps.append(
            {
                "t": times[i].strftime("%H:%M:%S"),
                "real": [rlat, rlon],
                "released": [zlat, zlon],
                "attacker": [alat, alon],
                "disp": round(float(np.hypot(rx - zx, ry - zy)), 1),
                "attacker_err": round(float(np.hypot(rx - ax, ry - ay)), 1),
                "poi_true": [_poi_latlon(p) for p in poi_true],
                "poi_reported": [_poi_latlon(p) for p in poi_reported],
                "poi_hits": len(set(poi_true) & set(poi_reported)),
            }
        )

    summary = {
        "mean_disp": round(metrics.mean_displacement(real, released, RN.proj), 1),
        "qos": round(metrics.qos_satisfaction(real, released, RN.proj, QOS_RADIUS), 2),
        "on_road": round(metrics.on_road_rate(released, RN), 2),
        "mean_attacker_err": round(
            float(np.mean([s["attacker_err"] for s in steps])), 1
        ),
        "knn_recall": round(
            float(np.mean([s["poi_hits"] / KNN.k for s in steps])), 2
        ),
        # ε·T is a worst-case composition CEILING, valid per-release only for
        # REM/T-REM. SM-REM's revisit channel is an ∞-ratio leak (not covered by
        # ε·T) and the legacy baseline has no valid per-release ε (verifier R4-010).
        "epsilon_ceiling_eps_times_T": round(eps * len(real), 3),
        "epsilon_note": {
            "road_exponential": "ε·T là trần composition hợp lệ per-release",
            "temporal_road_exponential": "ε·T là trần composition hợp lệ per-release",
            "stay_memoized_rem": "ε·T KHÔNG bao pattern thăm-lại (ratio ∞); chỉ static same-cell",
            "pr_sm_rem": "per-step (ε_test+ε_release); w-event manager chưa cài",
            "planar_laplace": "ε per-release hợp lệ (không on-road)",
            "baseline_thesis": "ε danh nghĩa KHÔNG phải guarantee hợp lệ (surrogate)",
        }.get(mech_name, ""),
    }
    return jsonify({"steps": steps, "summary": summary, "qos_radius": QOS_RADIUS})


if __name__ == "__main__":
    app.run(debug=False, port=5003)
