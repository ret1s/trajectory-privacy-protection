"""Independent arithmetic, lineage and deterministic-replay audit of pilot JSON.

Run: python -m experiments.verify_report_demo [--compare another/results.json]
No benchmark source is rewritten and no empirical success claim is inferred.
"""
import argparse
from collections import Counter, defaultdict
import hashlib
import json
import math
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
DEFAULT = ROOT / "artifacts/benchmarks/report_demo/results.json"


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def near(a, b):
    assert math.isclose(a, b, rel_tol=1e-9, abs_tol=1e-6), (a, b)


def verify(path=DEFAULT, *, source_check=True):
    path = Path(path)
    assert sha(path) == path.with_suffix(".sha256").read_text().strip(), "checksum"
    data = json.loads(path.read_text())
    assert data["schema"] == "report-demo-v1"
    if source_check:
        for name, expected in data["source_sha256"].items():
            assert sha(ROOT / name) == expected, f"stale source: {name}"
    manifests = {m["seed"]: m for m in data["manifests"]}
    samples = {(s["seed"], s["record_id"]): s for s in data["dataset_records"]}
    assert len(samples) == len(data["dataset_records"])
    for m in manifests.values():
        split = list(m["split"].values())
        assert all(set(a).isdisjoint(b) for i, a in enumerate(split) for b in split[i+1:])
        assert m["provenance"]["scenario"] == "controlled_beijing_passenger_stops_v1"
        assert set(m["provenance"]["commands"]) == {"sumo", "netconvert", "randomTrips.py"}
    for s in samples.values():
        assert s["vehicle_id"] in manifests[s["seed"]]["split"]["test"]
        t = s["times"]
        assert len(t) == len(s["points"]) == len(s["fcd"]) <= 12
        assert all(b > a for a, b in zip(t, t[1:]))
        near(s["checks"]["span_s"], t[-1] - t[0])
        if s["scenario"] == "S1":
            assert len(t) == 1
        elif s["scenario"] == "S2":
            assert t[-1] - t[0] >= 120
            assert all(f["speed_m_s"] <= .05 for f in s["fcd"])
            assert s["checks"]["max_radius_from_first_m"] <= 5
        else:
            assert s["scenario"] == "S3" and t[-1] - t[0] >= 60
            assert all(f["speed_m_s"] > .05 for f in s["fcd"])
        assert s["points"] == [[f["lat"], f["lon"]] for f in s["fcd"]]
    groups, keys, ids = defaultdict(list), set(), set()
    event_count = 0
    for r in data["rows"]:
        key = (r["seed"], r["record_id"], r["method"], r["k"])
        assert key not in keys and r["id"] not in ids
        keys.add(key); ids.add(r["id"])
        groups[r["scenario"], r["method"], r["k"]].append(r)
        assert (r["seed"], r["record_id"]) in samples
        if r["status"] != "ok":
            assert r["status"] in {"failed", "not_applicable"} and r["error"]
            continue
        sample = samples[r["seed"], r["record_id"]]
        assert r["truth"] == sample["points"]
        public = r["public"]
        forbidden = {"truth", "real_candidate_ids", "real_trajectory", "seed", "anchors", "truth_input"}
        def check_keys(obj):
            if isinstance(obj, dict):
                assert not forbidden.intersection(obj)
                for value in obj.values():
                    check_keys(value)
            elif isinstance(obj, list):
                for value in obj:
                    check_keys(value)
        check_keys(public)
        assert [e["timestamp_s"] for e in public["events"]] == sample["times"]
        assert all(e["candidates"] for e in public["events"])
        lat0 = manifests[r["seed"]]["projection_lat0"]
        metres = 6371000 * math.pi / 180
        xy = np.array([[lon * metres * math.cos(math.radians(lat0)), lat * metres]
                       for lat, lon in r["truth"]])
        errors = np.linalg.norm(np.array(r["attack_xy"]) - xy, axis=1)
        near(float(errors.mean()), r["metrics"]["location_mae_m"])
        near(float((errors <= 100).mean()), r["metrics"]["location_hit_100m"])
        assert np.allclose(errors, r["metrics"]["per_event_error_m"], atol=1e-6, rtol=1e-9)
        recalls = []
        for u in r["utility"]["poi_rows"]:
            assert len(u["reference"]) == len(set(u["reference"]))
            assert len(u["returned"]) == len(set(u["returned"])) <= 5
            if u["reference"]:
                recall = len(set(u["reference"]) & set(u["returned"])) / len(u["reference"])
                near(recall, u["recall"])
                recalls.append(recall)
            else:
                assert u["recall"] is None
        if recalls:
            near(float(np.mean(recalls)), r["metrics"]["poi_recall_at_k"])
        else:
            assert r["metrics"]["poi_recall_at_k"] is None
        near(np.mean([len(e["candidates"]) for e in public["events"]]),
             r["metrics"]["coordinates_per_request"])
        n = len(public["events"])
        payload_bytes = len(json.dumps({"events": public["events"], "query_category": r["query_category"]},
                                      separators=(",", ":")).encode())
        near((payload_bytes + sum(u["response_json_bytes"] for u in r["utility"]["poi_rows"])) / n,
             r["metrics"]["payload_json_bytes_per_request"])
        if r["method"] == "geo_i_anchored_dummy_road" and n > 1:
            near(r["metrics"]["directed_track_validity"], 1.)
        event_count += n
    for s in data["summary"]:
        group = groups[s["scenario"], s["method"], s["k"]]
        good = [r for r in group if r["status"] == "ok"]
        assert s["attempted"] == len(group)
        assert s["completed"] == len(good)
        assert s["failures"] == sum(r["status"] == "failed" for r in group)
        assert s["not_applicable"] == sum(r["status"] == "not_applicable" for r in group)
        for field in ("location_mae_m", "location_hit_100m", "poi_recall_at_k", "directed_track_validity"):
            values = [r["metrics"][field] for r in good if r["metrics"].get(field) is not None]
            assert s[field + "_n"] == len(values)
            if values:
                near(float(np.mean(values)), s[field])
            else:
                assert s[field] is None
        delivered = [r["metrics"].get("poi_recall_at_k", 0) if r["status"] == "ok" else 0
                     for r in group if r["status"] != "not_applicable" and r["poi_reference_evaluable"]]
        delivered = [x for x in delivered if x is not None]
        if delivered:
            near(float(np.mean(delivered)), s["poi_delivered_recall_at_k"])
        else:
            assert s["poi_delivered_recall_at_k"] is None
    causal = [c for c in data["causal_checks"] if c["prefix_invariance"] is not None]
    assert all(c["prefix_invariance"] for c in causal if c["method"] != "anotherme_adaptation")
    return {"artifact_sha256": sha(path), "source_files_checked": len(data["source_sha256"]) if source_check else 0,
            "records": len(samples), "rows": len(keys), "evaluated_events": event_count,
            "status": dict(Counter(r["status"] for r in data["rows"])), "summary_rows": len(data["summary"]),
            "prefix_checks": len(causal), "prefix_pass": sum(c["prefix_invariance"] for c in causal)}


def deterministic_view(data):
    # Wall-clock time and XML provenance timestamps are not deterministic.
    rows = []
    for original in data["rows"]:
        r = {**original, "metrics": {k: v for k, v in original["metrics"].items()
                                     if k not in {"amortized_generation_ms", "model_initialization_ms"}}}
        rows.append(r)
    return {"rows": rows, "samples": data["dataset_records"], "roads": data["roads"],
            "causal": data["causal_checks"],
            "selections": [m.get("attacker_selection") for m in data["manifests"]]}


def verify_raw(path):
    """Trace every saved truth point to FCD; recompute POI result IDs from roads.

    XML headers may change on a deterministic rerun. Compare parsed FCD records
    rather than asserting unchanged bytes for overwritten simulator outputs.
    """
    from dataclasses import asdict
    import networkx as nx
    from data.sumo_demo import parse_fcd, load_sumo_road_network, _existing_default_osm
    from evaluation.scenario_metrics import read_osm_pois
    data = json.loads(Path(path).read_text())
    matched, queries = 0, 0
    for m in data["manifests"]:
        root = ROOT / "cache/report_demo" / f"seed_{m['seed']}"
        traces = parse_fcd(root / "beijing_smoke.fcd.xml")
        for sample in data["dataset_records"]:
            if sample["seed"] != m["seed"]:
                continue
            lookup = {p.timestamp_s: asdict(p) for p in traces[sample["vehicle_id"]]}
            for fcd in sample["fcd"]:
                assert lookup[fcd["timestamp_s"]] == fcd
                matched += 1
        source = _existing_default_osm()
        assert sha(source) == m["provenance"]["sha256"]["osm"]
        rn = load_sumo_road_network(root / "beijing_smoke.net.xml")
        pois = []
        for poi in read_osm_pois(source, m["config"]["bbox"]):
            vertex, offset = rn.nearest(poi["lat"], poi["lon"])
            if offset <= 250:
                pois.append({**poi, "node": rn.node_ids[vertex]})
        assert len(pois) == m["pois"]["accepted"]
        cached = {}
        def ordering(point, category):
            node = rn.node_ids[rn.nearest(*point)[0]]
            key = node, category
            if key not in cached:
                distances = nx.single_source_dijkstra_path_length(rn.graph, node, weight="length")
                cached[key] = [p["id"] for p in sorted(
                    [p for p in pois if p["category"] == category and p["node"] in distances],
                    key=lambda p: (distances[p["node"]], p["id"]))]
            return cached[key]
        for row in data["rows"]:
            if row["seed"] != m["seed"] or row["status"] != "ok":
                continue
            category = row["query_category"]
            for event, truth, utility in zip(row["public"]["events"], row["truth"], row["utility"]["poi_rows"]):
                reference_order = ordering(truth, category)
                union = set()
                for c in event["candidates"]:
                    union.update(ordering((c["lat"], c["lon"]), category)[:5])
                assert utility["reference"] == reference_order[:5]
                assert utility["returned"] == [pid for pid in reference_order if pid in union][:5]
                queries += 1
    return {"raw_fcd_points_matched": matched, "poi_queries_recomputed_from_osm": queries}


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--path", type=Path, default=DEFAULT)
    p.add_argument("--compare", type=Path)
    p.add_argument("--raw", action="store_true")
    args = p.parse_args()
    result = verify(args.path)
    if args.raw:
        result.update(verify_raw(args.path))
    if args.compare:
        assert deterministic_view(json.loads(args.path.read_text())) == deterministic_view(json.loads(args.compare.read_text()))
        result["deterministic_rerun"] = "identical excluding timing and provenance timestamps"
    print(json.dumps(result, indent=2))
