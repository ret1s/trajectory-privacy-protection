"""Common task metrics and stronger full-transcript empirical adversaries.

These are operational definitions of this benchmark, not renamed ASR/DER or
an assertion of a universal LBS evaluation standard.
"""
import numpy as np

from evaluation.scenario_metrics import public_features, continuity_estimates, road_filter_estimates


def full_path_estimates(public, rn, travel, population, latent_count=24):
    """Offline Viterbi-style path decoder, not a calibrated posterior.

    Uses the same road-time/centroid observation assumptions as the existing
    causal decoder, but backtracks the full visible sequence. No truth input.
    Explicitly restart segments if no transition survives quantization slack.
    """
    states, scores, parents = [], [], []
    previous_t = None
    for event in public["events"]:
        center = public_features({"events": [event]}, rn, "S1")[0]
        if public["output_kind"] == "real_plus_dummies":
            ids = np.array(sorted({rn.nearest(c["lat"], c["lon"])[0] for c in event["candidates"]}))
            emission = np.zeros(len(ids))
        else:
            ids = np.atleast_1d(rn.tree.query(center, k=min(latent_count, len(rn)))[1]).astype(int)
            emission = -np.linalg.norm(rn.xy[ids] - center, axis=1) / 200.
        current = np.log(population[ids]) + emission
        parent = np.full(len(ids), -1, dtype=int)
        if states:
            dt = event["timestamp_s"] - previous_t
            current = np.full(len(ids), -np.inf)
            for i, vertex in enumerate(states[-1]):
                reached = travel.reachable(vertex, dt + 10.)
                for j, target in enumerate(ids):
                    tt = reached.get(rn.node_ids[target])
                    if tt is not None:
                        value = scores[-1][i] - tt / max(dt, 1.) + emission[j]
                        if value > current[j]:
                            current[j], parent[j] = value, i
            if not np.isfinite(current).any():
                current = np.log(population[ids]) + emission
        current -= current.max()
        states.append(ids); scores.append(current); parents.append(parent)
        previous_t = event["timestamp_s"]
    path = []
    selected = int(np.argmax(scores[-1]))
    for t in range(len(states) - 1, -1, -1):
        path.append(rn.xy[states[t][selected]])
        previous = parents[t][selected]
        if t:
            selected = int(previous) if previous >= 0 else int(np.argmax(scores[t - 1]))
    return np.array(path[::-1])


def full_features(public, rn):
    centers = public_features(public, rn, "S1")
    return np.c_[centers, np.tile(centers.mean(axis=0), (len(centers), 1))]


def attack_candidates(public, rn, scenario, travel, population, learned=None):
    centers = public_features(public, rn, "S1")
    candidates = {"centroid": centers,
                  "prior": np.tile(np.average(rn.xy, axis=0, weights=population), (len(centers), 1)),
                  "continuity": continuity_estimates(public, rn),
                  "road_filter": road_filter_estimates(public, rn, travel, population)[0]}
    if scenario != "S1":
        candidates["full_path"] = full_path_estimates(public, rn, travel, population)
        candidates["full_mean"] = np.tile(centers.mean(axis=0), (len(centers), 1))
    if scenario == "S2":
        candidates["running_mean"] = public_features(public, rn, "S2")
        if public["output_kind"] == "real_plus_dummies":
            candidates["stationary_intersection"] = centers.copy()
            sets = [{rn.nearest(c["lat"], c["lon"])[0] for c in e["candidates"]} for e in public["events"]]
            intersection = set.intersection(*sets)
            if intersection:
                chosen = max(sorted(intersection), key=lambda i: population[i])
                candidates["stationary_intersection"] = np.tile(rn.xy[chosen], (len(centers), 1))
    if learned is not None:
        candidates["shadow_full_knn"] = learned.predict(full_features(public, rn))
    return candidates


def target_errors(prediction, record, rn):
    if record["scenario"] in {"S9", "S10"}:
        index = 0 if record["scenario"] == "S9" else -1
        truth = np.array([rn.point_xy(*record["hidden_target"])])
        prediction = np.asarray(prediction)[[index]]
    else:
        truth = np.array([rn.point_xy(*p) for p in record["points"]])
    return np.linalg.norm(np.asarray(prediction) - truth, axis=1)


def utility_metrics(service, public, truth):
    """All available categories; recall after client union/dedup/local ranking.

    Extra road distance is conditional on complete result cardinality, paired
    with completion rate. Never give an empty result a zero distance penalty.
    """
    details, extra, completed = [], [], []
    for category in service.categories:
        result = service.evaluate(public, truth, category)
        for point, row in zip(truth, result["poi_rows"]):
            if row["reference"]:
                complete = len(row["returned"]) == len(row["reference"])
                completed.append(complete)
                if complete:
                    distances = service.distances(point)
                    gap = np.mean([distances[p] for p in row["returned"]]) - np.mean([distances[p] for p in row["reference"]])
                    extra.append(max(0., float(gap)))
                else:
                    gap = None
                row = {**row, "complete": complete, "extra_distance_m": gap}
            details.append({"category": category, **row})
    valid = [r["recall"] for r in details if r["recall"] is not None]
    return {"poi_recall_at_5": float(np.mean(valid)) if valid else None,
            "poi_complete_rate": float(np.mean(completed)) if completed else None,
            "poi_extra_distance_m": float(np.mean(extra)) if extra else None,
            "poi_extra_distance_n": len(extra), "poi_evaluable_n": len(valid),
            "poi_rows": details}
