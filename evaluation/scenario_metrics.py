"""End-to-end POI task and explicit, modest empirical attacker baselines."""
import gzip
import json
from pathlib import Path
import xml.etree.ElementTree as ET

import networkx as nx
import numpy as np
from scipy.spatial import cKDTree


def read_osm_pois(path, bbox):
    """Node amenities only; retain public OSM IDs, not invented businesses."""
    opener = gzip.open if str(path).endswith(".gz") else open
    west, south, east, north = bbox
    result = []
    with opener(path, "rb") as stream:
        iterator = ET.iterparse(stream, events=("start", "end"))
        _, root = next(iterator)
        count = 0
        for event, element in iterator:
            if event != "end" or element.tag not in {"node", "way", "relation"}:
                continue
            if element.tag == "node":
                lat, lon = float(element.attrib["lat"]), float(element.attrib["lon"])
                if west <= lon <= east and south <= lat <= north:
                    tags = {tag.attrib["k"]: tag.attrib["v"] for tag in element.findall("tag")}
                    if tags.get("amenity") in {"restaurant", "cafe", "pharmacy", "fuel", "hospital", "clinic"}:
                        result.append({"id": "osm/node/" + element.attrib["id"],
                                       "lat": lat, "lon": lon, "category": tags["amenity"]})
            element.clear()
            count += 1
            if count % 1000 == 0:
                root.clear()
    return sorted(result, key=lambda p: p["id"])


class PoiService:
    """Directed road distance, nearest-vertex access approximation, stable ties."""
    def __init__(self, rn, pois, k=5, max_access_m=250):
        if k < 1:
            raise ValueError("k must be positive")
        self.rn, self.k = rn, int(k)
        self.pois = []
        self.excluded = []
        self._distances = {}
        for poi in pois:
            vertex, offset = rn.nearest(poi["lat"], poi["lon"])
            item = {**poi, "vertex": int(vertex), "access_offset_m": float(offset)}
            (self.pois if offset <= max_access_m else self.excluded).append(item)
        self.categories = sorted({p["category"] for p in self.pois})
        self.by_id = {p["id"]: p for p in self.pois}

    def distances(self, point):
        vertex, _ = self.rn.nearest(*point)
        if vertex not in self._distances:
            node = self.rn.node_ids[vertex]
            lengths = nx.single_source_dijkstra_path_length(self.rn.graph, node, weight="length")
            self._distances[vertex] = {
                p["id"]: float(lengths[self.rn.node_ids[p["vertex"]]])
                for p in self.pois if self.rn.node_ids[p["vertex"]] in lengths
            }
        return self._distances[vertex]

    def query(self, point, category):
        distances = self.distances(point)
        return sorted((pid for pid in distances if self.by_id[pid]["category"] == category),
                      key=lambda pid: (distances[pid], pid))[:self.k]

    def evaluate(self, public, real_points, category):
        if len(public["events"]) != len(real_points):
            raise ValueError("This task evaluator requires aligned real/public events")
        rows = []
        for event, truth in zip(public["events"], real_points):
            reference = self.query(truth, category)
            union = set()
            replies = []
            for candidate in event["candidates"]:
                reply = self.query((candidate["lat"], candidate["lon"]), category)
                replies.append(reply)
                union.update(reply)
            distances = self.distances(truth)
            returned = sorted((p for p in union if p in distances),
                              key=lambda p: (distances[p], p))[:self.k]
            rows.append({
                "reference": reference, "returned": returned,
                "recall": len(set(reference) & set(returned)) / len(reference) if reference else None,
                "response_json_bytes": len(json.dumps(replies, separators=(",", ":")).encode()),
            })
        valid = [r["recall"] for r in rows if r["recall"] is not None]
        return {
            "poi_recall_at_k": float(np.mean(valid)) if valid else None,
            "poi_evaluable_events": len(valid), "poi_empty_reference_events": len(rows) - len(valid),
            "poi_rows": rows,
        }


def public_features(public, rn, scenario):
    """No truth access. S2 assumes the released interval is known stationary."""
    centers = np.array([
        np.mean([rn.point_xy(c["lat"], c["lon"]) for c in e["candidates"]], axis=0)
        for e in public["events"]
    ])
    if scenario == "S2":
        return np.cumsum(centers, axis=0) / np.arange(1, len(centers) + 1)[:, None]
    if scenario == "S3":
        return np.array([np.r_[center, centers[max(0, i - 1)]]
                         for i, center in enumerate(centers)])
    return centers


class ShadowKnnAttack:
    """Method-specific k-NN decoder trained on disjoint shadow users.

    A reproducible empirical adversary, NOT a Bayes-optimal/posterior or a
    reproduction of VehiTrack. Absolute coordinates encode a learnt prior.
    """
    def __init__(self, features, locations, neighbors=5):
        self.features = np.asarray(features, dtype=float)
        self.locations = np.asarray(locations, dtype=float)
        if len(self.features) == 0 or len(self.features) != len(self.locations):
            raise ValueError("nonempty aligned shadow data required")
        self.tree = cKDTree(self.features)
        self.neighbors = min(neighbors, len(self.features))

    def predict(self, features):
        _, ids = self.tree.query(features, k=self.neighbors)
        ids = np.asarray(ids).reshape(len(features), self.neighbors)
        return self.locations[ids].mean(axis=1)


def continuity_estimates(public, rn):
    """Causal minimum-motion path filter over public coordinates, no truth.

    Maintains costs for all candidates; emits the current minimum-cost endpoint.
    Not a calibrated posterior, road-constrained attack, or offline traceback.
    """
    previous, costs, previous_time = None, None, None
    estimates = []
    for event in public["events"]:
        points = np.array([rn.point_xy(c["lat"], c["lon"]) for c in event["candidates"]])
        timestamp = event["timestamp_s"]
        if previous is None:
            costs = np.zeros(len(points))
        else:
            dt = max(float(timestamp - previous_time), 1.)
            transitions = np.linalg.norm(points[:, None, :] - previous[None, :, :], axis=2) / dt
            costs = np.min(costs[None, :] + transitions ** 2, axis=1)
            costs -= costs.min()
        estimates.append(points[int(np.argmin(costs))])
        previous, previous_time = points, timestamp
    return np.asarray(estimates)


def attack_scores(estimates_xy, truth_xy):
    estimates, truth = np.asarray(estimates_xy), np.asarray(truth_xy)
    if estimates.shape != truth.shape or truth.ndim != 2 or truth.shape[1] != 2 or not len(truth):
        raise ValueError("nonempty aligned (n,2) coordinates required")
    errors = np.linalg.norm(estimates - truth, axis=1)
    return {"location_mae_m": float(errors.mean()),
            "location_hit_100m": float((errors <= 100).mean()),
            "per_event_error_m": errors.tolist()}


class RoadTravelTimes:
    """Public directed, speed-limited graph distances, cached by time budget."""
    def __init__(self, rn, v_max=25.):
        self.rn, self.v_max, self.cache = rn, float(v_max), {}
        self.indices = {node: i for i, node in enumerate(rn.node_ids)}

    def _weight(self, u, v, data):
        edges = data.values() if self.rn.graph.is_multigraph() else [data]
        fallback = np.linalg.norm(self.rn.xy[self.indices[u]] - self.rn.xy[self.indices[v]])
        return min(float(edge.get("length", fallback)) /
                   min(self.v_max, max(1e-9, float(edge.get("speed", self.v_max))))
                   for edge in edges)

    def reachable(self, vertex, seconds):
        key = int(vertex), float(seconds)
        if key not in self.cache:
            self.cache[key] = nx.single_source_dijkstra_path_length(
                self.rn.graph, self.rn.node_ids[int(vertex)], cutoff=max(0., seconds),
                weight=self._weight)
        return self.cache[key]


def road_filter_estimates(public, rn, travel, population, *, sigma_m=200., latent_count=24):
    """Causal road-time path decoder; all inputs except public are public context.

    True-containing sets use their candidates as states. Replacement and
    dummy-only emissions use nearby latent road states around the public
    centroid with a distance likelihood. The observation kernel is a stated
    heuristic, not a fitted mechanism likelihood or a calibrated posterior.
    A 10-second grace accommodates nearest-junction quantization. If no path
    survives, restart from the current observation and count the restart.
    """
    previous_ids, previous_scores, previous_t = None, None, None
    estimates, restarts = [], 0
    for event in public["events"]:
        center = np.mean([rn.point_xy(c["lat"], c["lon"]) for c in event["candidates"]], axis=0)
        if public.get("output_kind") == "real_plus_dummies":
            ids = np.array(sorted({int(rn.nearest(c["lat"], c["lon"])[0]) for c in event["candidates"]}))
            emission = np.zeros(len(ids))
        else:
            _, ids = rn.tree.query(center, k=min(latent_count, len(rn)))
            ids = np.atleast_1d(ids).astype(int)
            emission = -np.linalg.norm(rn.xy[ids] - center, axis=1) / sigma_m
        initial = np.log(np.asarray(population)[ids]) + emission
        scores = initial
        if previous_ids is not None:
            dt = float(event["timestamp_s"] - previous_t)
            scores = np.full(len(ids), -np.inf)
            for prev, cost in zip(previous_ids, previous_scores):
                reachable = travel.reachable(int(prev), dt + 10.)
                for j, index in enumerate(ids):
                    tt = reachable.get(rn.node_ids[index])
                    if tt is not None:
                        scores[j] = max(scores[j], cost - tt / max(dt, 1.) + emission[j])
            if not np.isfinite(scores).any():
                scores = initial
                restarts += 1
        scores -= scores.max()
        estimates.append(rn.xy[int(ids[int(np.argmax(scores))])])
        previous_ids, previous_scores, previous_t = ids, scores, event["timestamp_s"]
    return np.asarray(estimates), restarts


def stable_track_validity(public, rn, travel):
    """Only stable declared tracks have a point-to-point feasibility fraction."""
    events = public["events"]
    if len(events) < 2:
        return None
    previous, passed, total = {}, 0, 0
    for event in events:
        current = {}
        for candidate in event["candidates"]:
            key = candidate["candidate_id"]
            vertex = int(rn.nearest(candidate["lat"], candidate["lon"])[0])
            if key in previous:
                old_vertex, old_time = previous[key]
                total += 1
                passed += rn.node_ids[vertex] in travel.reachable(old_vertex, event["timestamp_s"] - old_time)
            current[key] = vertex, event["timestamp_s"]
        previous = current
    return passed / total if total else None
