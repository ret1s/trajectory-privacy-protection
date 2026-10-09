"""Independent fixed-Q dynamic-workload arithmetic and public-source audit.

Public graph/routing primitives are shared; provider hash, epoch cache, available
reference/candidate intersections, denominators, aggregation and wire costs are
reconstructed below without importing the workload runner or its scorer.
This certifies an inspected synthetic sensitivity study, not fresh confirmation,
provider unpredictability, a deployed service, or a privacy theorem.
"""
import argparse
from collections import defaultdict, OrderedDict
from datetime import datetime, timezone
import gzip
import hashlib
import json
import math
from pathlib import Path
import statistics

import numpy as np
from scipy.sparse.csgraph import dijkstra

from benchmark.public_poi_context import PublicPoiContext
from data.lane_states import build_lane_states
from evaluation.lane_travel import LanePoiService, matrix

ROOT = Path(__file__).resolve().parents[1]
THIS = "experiments/verify_dynamic_provider_status_20261006_v2.py"
TEST = "tests/test_dynamic_provider_status_verifier_v2.py"
OUT = ROOT / "artifacts/benchmarks/dynamic_provider_status_20261006_v1"
PUBLIC = "artifacts/benchmarks/research_loop/resources.json"
PURPOSES = ("nearest_distance", "fastest_travel", "within_radius", "minimum_detour")
ARMS = ("geoi_l20_current", "geoi_l20_epoch60", "geoi_l30_current", "geoi_l30_epoch60",
        "raw_l20_current", "raw_l30_current", "static_catalogue_stale_safe",
        "current_catalogue_bulk_oracle", "static_catalogue_stale_unsafe")
SEED = 2026100623
EPOCH = 60
FIXTURE_COUNT = 10


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    path = Path(path)
    raw = path.read_bytes()
    return json.loads(gzip.decompress(raw) if path.suffix == ".gz" else raw)


def canonical(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def size(value):
    return len(json.dumps(value, separators=(",", ":"), ensure_ascii=False, allow_nan=False).encode())


def save(path, value):
    with Path(path).open("x") as stream:
        stream.write(json.dumps(value, indent=2, allow_nan=False) + "\n")


def public_bit(poi_id, epoch):
    assert isinstance(epoch, int) and not isinstance(epoch, bool) and epoch >= 0
    encoded = json.dumps(["synthetic-availability-v1", SEED, str(poi_id), epoch], separators=(",", ":")).encode()
    return int.from_bytes(hashlib.sha256(encoded).digest()[:8], "big") < (1 << 63)


def same(left, right, path="root"):
    if isinstance(left, dict):
        assert isinstance(right, dict) and set(left) == set(right), path
        for key in left:
            same(left[key], right[key], path + "/" + str(key))
    elif isinstance(left, (list, tuple)):
        if path.endswith("/rows"):
            left, right = canonical_rows(left), canonical_rows(right)
        assert isinstance(right, (list, tuple)) and len(left) == len(right), path
        for index, (a, b) in enumerate(zip(left, right)):
            same(a, b, path + "/" + str(index))
    elif isinstance(left, float):
        assert isinstance(right, (int, float)) and math.isfinite(left) and math.isfinite(right), path
        assert math.isclose(left, right, rel_tol=1e-12, abs_tol=1e-12), (path, left, right)
    else:
        assert left == right, (path, left, right)


def source_contract(out):
    """Only pinned metadata/public files; does not open dynamic scores."""
    out = Path(out)
    p = read(out / "protocol.json")
    assert sha(out / "protocol.json") == (out / "protocol.sha256").read_text().strip()
    assert p["schema"] == "frozen-Q-dynamic-provider-status-v1"
    assert p["workload_seed"] == SEED and p["availability_probability"] == .5
    assert p["status_epoch_seconds"] == EPOCH and p["depths"] == [20, 30]
    assert tuple(p["arms"]) == ARMS and p["draws"] == [1, 2, 3] and p["split"] == "test"
    assert p["public_departures_s"] == [1500. * slot for slot in range(8)]
    assert len(p["families"]) == len(set(p["families"])) == 24
    assert p["included_blocks"] == [f"{f}--draw{d}.json.gz" for f in p["families"] for d in (1, 2, 3)]
    assert len(p["included_blocks"]) == 72
    for name, pin in p["source_sha256"].items():
        assert sha(ROOT / name) == sha(out / "source_snapshot" / name) == pin, name
    for folder, field in [(ROOT / p["source_output"], "source_files_sha256"), (ROOT / p["base_Q_output"], "base_files_sha256")]:
        for name, pin in p[field].items():
            assert sha(folder / name) == pin, name
    depth = ROOT / p["source_output"]
    cert = read(depth / "validation.json")
    assert cert["status"] == "pass"
    assert cert["protocol_sha256"] == sha(depth / "protocol.json")
    assert cert["readout_sha256"] == sha(depth / "readout.json")
    assert cert["paired_readout_sha256"] == sha(depth / "paired_readout.json")
    assert cert["verifier_sha256"] == sha(ROOT / "experiments/verify_qplanner_response_depth_generalization_20261006.py")
    assert cert["no_new_Q_GPS_anchors_or_ledger_reads"] is True
    assert sha(ROOT / p["dataset_path"]) == p["dataset_sha256"]
    assert sha(ROOT / PUBLIC) == p["public_resource_sha256"]
    assert sha(out / "public_reply60.npz") == p["reply_cache_sha256"]
    return p


def verification_sources(p):
    return sorted(set(p["source_sha256"]) | {THIS, TEST, "experiments/verify_dynamic_provider_status_20261006.py", "tests/test_dynamic_provider_status_verifier.py"})


def canonical_rows(rows):
    """Only serialization order changes; event/arm inventory must be unique."""
    keys = [(row["slot"], row["t"], row["arm"]) for row in rows]
    assert len(set(keys)) == len(keys), "Duplicate public event/arm row"
    return [row for _, row in sorted(zip(keys, rows), key=lambda pair: pair[0])]


def declare(out):
    out = Path(out)
    p = source_contract(out)
    original = read(out / "verification_protocol.json")
    failure = read(out / "validation_attempt_v1_failure.json")
    assert failure["status"] == "failed" and failure["failed_key"].endswith("/rows/2/arm")
    assert failure["verification_protocol_sha256"] == sha(out / "verification_protocol.json")
    assert failure["readout_sha256"] == sha(out / "readout.json")
    assert failure["verifier_sha256"] == sha(ROOT / "experiments/verify_dynamic_provider_status_20261006.py")
    assert sha(out / "verification_protocol.json") == (out / "verification_protocol.sha256").read_text().strip()
    for name, pin in original["source_sha256"].items():
        assert sha(ROOT / name) == sha(out / "verification_source_snapshot" / name) == pin
    pins = {name: sha(ROOT / name) for name in verification_sources(p)}
    inputs = {name: sha(out / name) for name in ("readout.json", "resources.json", "status_world.json.gz", "verification_protocol.json", "validation_attempt_v1_failure.json")}
    value = dict(schema="dynamic-status-independent-verification-review-protocol-v2",
                 created_utc=datetime.now(timezone.utc).isoformat(), workload_protocol_sha256=sha(out / "protocol.json"),
                 source_sha256=pins, synthetic_fixture_count=FIXTURE_COUNT, completed_inputs_sha256=inputs,
                 independent_hash_epoch_cache_reference_denominators_aggregation_cost=True,
                 shared_public_graph_and_lane_service=True, scoring_started_before_declaration=True,
                 correction="Serialization-only canonical unique(slot,t,arm) rows; original arithmetic unchanged, failed checker/source/declaration retained",
                 scope="Explicit post-score checker correction for secondary inspected synthetic workload; not a new preregistration, defense tuning, deployed provider or privacy theorem")
    save(out / "verification_review_protocol_v2.json", value)
    (out / "verification_review_protocol_v2.sha256").write_text(sha(out / "verification_review_protocol_v2.json") + "\n")
    for name in pins:
        target = out / "verification_review_source_snapshot_v2" / name
        target.parent.mkdir(parents=True, exist_ok=True)
        with target.open("xb") as stream:
            stream.write((ROOT / name).read_bytes())
    return value


def verification_contract(out, p):
    out = Path(out)
    v = read(out / "verification_review_protocol_v2.json")
    assert sha(out / "verification_review_protocol_v2.json") == (out / "verification_review_protocol_v2.sha256").read_text().strip()
    assert v["schema"] == "dynamic-status-independent-verification-review-protocol-v2"
    assert v["workload_protocol_sha256"] == sha(out / "protocol.json")
    assert v["synthetic_fixture_count"] == FIXTURE_COUNT and v["scoring_started_before_declaration"] is True
    assert set(v["source_sha256"]) == set(verification_sources(p))
    for name, pin in v["source_sha256"].items():
        assert sha(ROOT / name) == sha(out / "verification_review_source_snapshot_v2" / name) == pin, name
    for name, pin in v["completed_inputs_sha256"].items():
        assert sha(out / name) == pin, name
    assert v["source_sha256"][THIS] == sha(__file__)
    return v

def score_ordered(cases, truth, static, known, available, unsafe=None):
    """Use plain sets/list filters; no production scorer/array mask reduction."""
    truth, static, known, available = map(set, (truth, static, known, available))
    candidates = static & known & available if unsafe is None else set(unsafe)
    assert unsafe is not None or not (known & available) - truth
    result = {}
    for purpose, entries in cases.items():
        recall, completion = [], []
        counts = dict(reference_poi_total=0, overlap_total=0, empty_reference_categories=0,
                      retrieval_miss_reference_pois=0, current_status_unknown_reference_pois=0,
                      known_available_reference_pois=0, returned_items=0, returned_unavailable_items=0,
                      returned_current_status_unknown_items=0, reference_exists_zero_answer_categories=0,
                      current_known_unavailable_candidates=len(static & known - available),
                      current_status_unknown_candidates=len(static - known))
        for ordered in entries:
            reference = [int(i) for i in ordered if int(i) in truth][:5]
            answer = [int(i) for i in ordered if int(i) in candidates][:5]
            counts["returned_items"] += len(answer)
            counts["returned_unavailable_items"] += len(set(answer) - truth)
            counts["returned_current_status_unknown_items"] += len(set(answer) - known)
            if not reference:
                counts["empty_reference_categories"] += 1
                continue
            overlap = len(set(reference) & set(answer))
            recall.append(overlap / len(reference))
            completion.append(min(len(set(answer) & truth), len(reference)) / len(reference))
            counts["reference_poi_total"] += len(reference)
            counts["overlap_total"] += overlap
            counts["retrieval_miss_reference_pois"] += len(set(reference) - static)
            counts["current_status_unknown_reference_pois"] += len(set(reference) & static - known)
            counts["known_available_reference_pois"] += len(set(reference) & static & known & available)
            counts["reference_exists_zero_answer_categories"] += not answer
        if unsafe is None:
            assert counts["returned_unavailable_items"] == counts["returned_current_status_unknown_items"] == 0
            assert counts["reference_poi_total"] == sum(counts[k] for k in ("retrieval_miss_reference_pois", "current_status_unknown_reference_pois", "known_available_reference_pois"))
        result[purpose] = dict(recall5=statistics.mean(recall) if recall else None,
                               completion=statistics.mean(completion) if completion else None,
                               reference_category_count=len(recall), all_category_count=len(entries), **counts)
    return result


class ReceivedHistory:
    def __init__(self):
        self.static, self.bits = set(), {}
        self.epoch, self.last = None, None

    def observe(self, clock, received):
        assert math.isfinite(clock) and clock >= 0 and (self.last is None or clock > self.last)
        epoch = int(clock // EPOCH)
        old = self.bits if epoch == self.epoch else {}
        updates = {}
        for index, bit in received:
            assert isinstance(index, int) and not isinstance(index, bool) and index >= 0 and type(bit) is bool
            assert index not in old or old[index] is bit
            assert index not in updates or updates[index] is bit
            updates[index] = bit
        self.bits = dict(old, **{})
        self.bits.update(updates)
        self.static.update(updates)
        self.epoch, self.last = epoch, clock
        return (set(updates), set(updates), {i for i, bit in updates.items() if bit}), (set(self.static), set(self.bits), {i for i, bit in self.bits.items() if bit})


class OrderedRoadOracle:
    """Explicit directed shortest-distance/time/radius/detour formula oracle."""
    def __init__(self, context):
        self.context = context
        self.distance, self.time = matrix(context.rn), matrix(context.rn, time=True)
        self.vertices = np.asarray([p["vertex"] for p in context.pois], int)
        self.cost_cache, self.orders = OrderedDict(), {}

    def costs(self, state, mode):
        key = int(state), mode
        if key not in self.cost_cache:
            graph = self.time if mode == "time" else self.distance
            if mode == "reverse":
                graph = graph.transpose().tocsr()
            self.cost_cache[key] = dijkstra(graph, directed=True, indices=int(state))
            if len(self.cost_cache) > 64:
                self.cost_cache.popitem(last=False)
        self.cost_cache.move_to_end(key)
        return self.cost_cache[key]

    def ordered(self, state, destination):
        key = int(state), int(destination)
        if key not in self.orders:
            distance = self.costs(state, "distance")
            travel = self.costs(state, "time")
            reverse = self.costs(destination, "reverse")
            base = distance[self.vertices]
            detour = base + reverse[self.vertices] - distance[destination]
            if not math.isfinite(distance[destination]):
                detour[:] = np.inf
            else:
                detour[np.isfinite(detour)] = np.maximum(0., detour[np.isfinite(detour)])
            radius = base.copy()
            radius[radius > 1000.] = np.inf
            result = {}
            for purpose, scores in zip(PURPOSES, (base, travel[self.vertices], radius, detour)):
                result[purpose] = []
                for category in self.context.categories:
                    eligible = [i for i, poi in enumerate(self.context.pois) if poi["category"] == category and np.isfinite(scores[i])]
                    result[purpose].append(sorted(eligible, key=lambda i: (float(scores[i]), self.context.pois[i]["id"])))
            self.orders[key] = result
        return self.orders[key]


def wire(positions, clock, categories, L):
    epoch = int(clock // EPOCH)
    original = [dict(timestamp_s=clock, lat=float(lat), lon=float(lon), categories=list(categories), L=L) for lat, lon in positions]
    extended = [dict(row, status_epoch=epoch, include_availability=True) for row in original]
    return len(positions), sum(size(row) for row in original), sum(size(row) for row in extended) - sum(size(row) for row in original)


def replies_cost(ids, context, epoch, live):
    ids = list(ids)
    records = [{k: context.pois[i][k] for k in ("id", "category", "lat", "lon")} for i in ids]
    status = [dict(id=context.pois[i]["id"], available=i in live) for i in ids]
    return size({"results": records}), size({"epoch": epoch, "status": status})


def expected_block(bundle, depth, context, oracle, world):
    truth, streams = bundle["evaluator_only"], bundle["public"]["streams"]
    all_ids = set(range(len(context.pois)))
    snapshots = {int(e): {i for i, bit in enumerate(bits) if bit} for e, bits in world["snapshots"].items()}
    stale, seen = snapshots[0], set()
    histories = {L: ReceivedHistory() for L in (20, 30)}
    rows, max_cache = [], {"20": 0, "30": 0}
    costs = {arm: dict(requests=0, static_request_bytes=0, status_upload_bytes=0, static_reply_bytes=0, status_download_bytes=0) for arm in ARMS}
    old_wires = {(r["method"], r["slot"], r["t"]): r for r in depth["wire"]}
    inventory = []
    for slot, session in enumerate(streams["legacy_l10"]):
        raw = streams["raw"][slot]["events"]
        assert [e["timestamp_s"] for e in raw] == [e["timestamp_s"] for e in session["events"]]
        destination = context.rn.nearest(raw[-1]["candidates"][0]["lat"], raw[-1]["candidates"][0]["lon"])[0]
        for event, gps in zip(session["events"], raw):
            t = old_wires["service_l20", slot, event["timestamp_s"]]["t"]
            clock = 1500. * slot + t
            epoch, live = int(clock // EPOCH), snapshots[int(clock // EPOCH)]
            inventory.append([slot, event["timestamp_s"], event["event_id"], clock])
            point = gps["candidates"][0]
            state = context.rn.nearest(point["lat"], point["lon"])[0]
            cases, configs = oracle.ordered(state, destination), {}
            positions = [(q["lat"], q["lon"]) for q in event["candidates"]]
            assert len(positions) == 5
            for L in (20, 30):
                reply_lists = []
                for lat, lon in positions:
                    indices = context.query_indices(context.rn.nearest(lat, lon)[0])[:, :L].ravel()
                    reply_lists.append([int(i) for i in indices if i >= 0])
                old = old_wires[f"service_l{L}", slot, t]
                assert reply_lists == old["reply_poi_ids_by_Q"]
                current, cached = histories[L].observe(clock, [(i, i in live) for reply in reply_lists for i in reply])
                max_cache[str(L)] = max(max_cache[str(L)], len(cached[0]))
                for policy, masks in [("current", current), ("epoch60", cached)]:
                    arm = f"geoi_l{L}_{policy}"
                    configs[arm] = (*masks, None)
                    requests, request_bytes, added = wire(positions, clock, context.categories, L)
                    assert request_bytes == old["request_bytes"]
                    static_bytes = sum(replies_cost(ids, context, epoch, live)[0] for ids in reply_lists)
                    assert static_bytes == old["reply_bytes"]
                    values = (requests, request_bytes, added, static_bytes, sum(replies_cost(ids, context, epoch, live)[1] for ids in reply_lists))
                    for key, value in zip(costs[arm], values):
                        costs[arm][key] += value
                indices = context.query_indices(state)[:, :L].ravel()
                ids = [int(i) for i in indices if i >= 0]
                configs[f"raw_l{L}_current"] = (set(ids), set(ids), set(ids) & live, None)
                count, rb, added = wire([(point["lat"], point["lon"])], clock, context.categories, L)
                sb, statusb = replies_cost(ids, context, epoch, live)
                for key, value in zip(costs[f"raw_l{L}_current"], (count, rb, added, sb, statusb)):
                    costs[f"raw_l{L}_current"][key] += value
            configs["static_catalogue_stale_safe"] = (all_ids, all_ids if epoch == 0 else set(), stale if epoch == 0 else set(), None)
            configs["current_catalogue_bulk_oracle"] = (all_ids, all_ids, live, None)
            configs["static_catalogue_stale_unsafe"] = (all_ids, all_ids if epoch == 0 else set(), stale, stale)
            if not seen:
                sb, statusb = replies_cost(range(len(context.pois)), context, 0, stale)
                for arm in ARMS[-3:]:
                    costs[arm]["requests"] += 1
                    costs[arm]["static_request_bytes"] += size({"schema": "full_public_catalogue_v1"})
                    costs[arm]["static_reply_bytes"] += sb
                    if arm != "current_catalogue_bulk_oracle":
                        costs[arm]["status_download_bytes"] += statusb
            if epoch not in seen:
                costs["current_catalogue_bulk_oracle"]["requests"] += 1
                costs["current_catalogue_bulk_oracle"]["status_upload_bytes"] += size({"schema": "full_catalogue_status_v1", "epoch": epoch})
                costs["current_catalogue_bulk_oracle"]["status_download_bytes"] += replies_cost(range(len(context.pois)), context, epoch, live)[1]
                seen.add(epoch)
            for arm in ARMS:
                static, known, available, unsafe = configs[arm]
                scores = score_ordered(cases, live, static, known, available, unsafe)
                rows.append(dict(family_id=truth["family_id"], draw=truth["draw"], split="test", arm=arm, slot=slot,
                                 t=t, absolute_t=clock, epoch=epoch, operationally_valid=arm != "static_catalogue_stale_unsafe", purposes=scores))
    return dict(schema="dynamic-status-frozen-Q-block-v1", family_id=truth["family_id"], draw=truth["draw"], split="test",
                rows=rows, cost=costs, static_cache_max_records=max_cache, frozen_controls=depth["frozen_controls"],
                private_reads_or_Q_regeneration=False), inventory


def aggregate(blocks):
    groups, costs = defaultdict(list), {arm: defaultdict(int) for arm in ARMS}
    local, summary = [], {}
    for block in blocks:
        for row in block["rows"]:
            phases = ["all"] + (["cold"] if row["slot"] == 0 else []) + (["temporal_tail_400_600"] if row["t"] >= 400 else [])
            for phase in phases:
                for purpose in PURPOSES:
                    groups[row["arm"], phase, purpose, row["family_id"], row["draw"]].append(row["purposes"][purpose])
        for arm, row in block["cost"].items():
            for key, value in row.items():
                costs[arm][key] += value
    for arm in ARMS:
        summary[arm] = {}
        for phase in ("all", "cold", "temporal_tail_400_600"):
            result = {}
            for purpose in PURPOSES:
                family, draws, den = defaultdict(list), defaultdict(list), defaultdict(int)
                for (a, ph, pu, f, draw), values in sorted(groups.items()):
                    if (a, ph, pu) != (arm, phase, purpose):
                        continue
                    valid = [v["recall5"] for v in values if v["recall5"] is not None]
                    mean = statistics.mean(valid) if valid else None
                    family[f].append(mean)
                    draws[draw].append(mean)
                    local.append(dict(arm=arm, phase=phase, purpose=purpose, family_id=f, draw=draw,
                                      recall5=mean, total_events=len(values), defined_events=len(valid)))
                    den["total_events"] += len(values)
                    den["defined_events"] += len(valid)
                    for row in values:
                        for key, value in row.items():
                            if key not in ("recall5", "completion"):
                                den[key] += value
                means = {f: statistics.mean([v for v in values if v is not None]) if any(v is not None for v in values) else None for f, values in family.items()}
                defined = [v for v in means.values() if v is not None]
                result[purpose] = dict(family_mean=statistics.mean(defined) if defined else None, family_values=means,
                                       within_draw_family_mean={str(d): statistics.mean([v for v in values if v is not None]) if any(v is not None for v in values) else None for d, values in draws.items()}, explicit_denominators=dict(den))
            macro = {}
            for f in result[PURPOSES[0]]["family_values"]:
                values = [result[p]["family_values"][f] for p in PURPOSES if result[p]["family_values"][f] is not None]
                if values:
                    macro[f] = statistics.mean(values)
            per_draw = {}
            for d in (1, 2, 3):
                values = []
                for f in macro:
                    defined = [r["recall5"] for r in local if (r["arm"], r["phase"], r["family_id"], r["draw"]) == (arm, phase, f, d) and r["recall5"] is not None]
                    if defined:
                        values.append(statistics.mean(defined))
                per_draw[str(d)] = statistics.mean(values) if values else None
            result["equal_purpose_macro"] = dict(family_mean=statistics.mean(macro.values()) if macro else None,
                                                  family_values=macro, minimum_family_mean=min(macro.values()) if macro else None,
                                                  within_draw_family_mean=per_draw)
            summary[arm][phase] = result
    return dict(summary=summary, family_draw_purpose_values=local, cost={arm: dict(row) for arm, row in costs.items()},
                max_static_cache_records={str(L): max(block["static_cache_max_records"][str(L)] for block in blocks) for L in (20, 30)})


def verify(out):
    out = Path(out)
    p = source_contract(out)
    verification_contract(out, p)
    world = read(out / "status_world.json.gz")
    assert canonical(world) == p["status_world_sha256"]
    ids = [poi["id"] for poi in p["public_catalogue"]]
    assert world["poi_ids"] == ids and world["seed"] == SEED and world["epoch_seconds"] == EPOCH
    for epoch, bits in world["snapshots"].items():
        assert bits == [public_bit(pid, int(epoch)) for pid in ids]
    dataset = read(ROOT / p["dataset_path"])
    network = ROOT / dataset["network"]["compressed_path"]
    assert sha(network) == dataset["network"]["compressed_sha256"]
    assert hashlib.sha256(gzip.decompress(network.read_bytes())).hexdigest() == dataset["network"]["native_sha256"]
    rn = build_lane_states(network, spacing_m=40.)
    pois = [{k: v for k, v in row.items() if k not in ("vertex", "access_offset_m")} for row in read(ROOT / PUBLIC)["pois_used"]]
    context = PublicPoiContext(LanePoiService(rn, pois, k=60), out / "public_reply60.npz")
    assert context.sha256 == p["public_reply60_sha256"] and list(context.pois) == p["public_catalogue"]
    oracle = OrderedRoadOracle(context)
    actual_readout = read(out / "readout.json")
    assert actual_readout["protocol_sha256"] == sha(out / "protocol.json")
    assert actual_readout["status_world_file_sha256"] == sha(out / "status_world.json.gz")
    assert set(actual_readout["block_files_sha256"]) == set(p["included_blocks"])
    blocks, rows_checked, events_checked, epochs = [], 0, 0, {0}
    for index, name in enumerate(p["included_blocks"], 1):
        base, depth = ROOT / p["base_Q_output"] / "families" / name, ROOT / p["source_output"] / "families" / name
        assert sha(base) == p["base_family_files_sha256"][name] and sha(depth) == p["source_family_files_sha256"][name]
        b, d = read(base), read(depth)
        assert canonical(b["public"]["streams"]["legacy_l10"]) == d["frozen_controls"]["Q_stream_sha256"]
        assert d["frozen_controls"] == p["frozen_controls"][name]
        expected, inventory = expected_block(b, d, context, oracle, world)
        assert inventory == p["public_event_inventory"][name]
        epochs.update(int(row[-1] // EPOCH) for row in inventory)
        expected.update(source_bundle_sha256=sha(base), source_depth_reply_bundle_sha256=sha(depth))
        target = out / "blocks" / name
        assert sha(target) == actual_readout["block_files_sha256"][name]
        same(expected, read(target), name)
        blocks.append(expected)
        rows_checked += len(expected["rows"])
        events_checked += len(inventory)
        print(f"Independent dynamic audit {index}/72 blocks; {events_checked} fixed public events", flush=True)
    assert epochs == set(map(int, world["snapshots"]))
    expected = aggregate(blocks)
    for key, value in expected.items():
        same(value, actual_readout[key], key)
    assert actual_readout["protected_Q_clocks_ledger_unchanged"] is True
    assert actual_readout["no_private_key_GPS_queries_or_sampler_calls"] is True
    assert actual_readout["source_already_inspected_secondary_only"] is True
    result = dict(schema="dynamic-status-independent-verification-v2", status="pass",
                  protocol_sha256=sha(out / "protocol.json"), readout_sha256=sha(out / "readout.json"),
                  verification_protocol_sha256=sha(out / "verification_protocol.json"),
                  verification_review_protocol_v2_sha256=sha(out / "verification_review_protocol_v2.json"),
                  post_score_checker_row_order_correction=True, verifier_sha256=sha(__file__),
                  blocks=72, test_family_clusters=24, nested_draws=3, public_events=events_checked,
                  arm_event_rows=rows_checked, provider_epochs=len(epochs), public_pois=len(ids),
                  public_status_hashes_rebuilt=True, all_fixed_Q_static_topL_replies_preserved=True,
                  independent_directed_reference_order_and_availability_recomputed=True,
                  safe_epoch_invalidation_and_unknown_partitions_recomputed=True,
                  all_family_draw_purpose_denominators_and_costs_recomputed=True,
                  unchanged_ledger_certificate_inherited_from_pinned_fresh_base=True,
                  private_key_or_sampler_required=False, fresh_confirmation=False, privacy_theorem_certified=False,
                  scope="Inspected same-map synthetic status sensitivity; full evaluator/provider hash world is withheld by the experimental client API only; static+status bytes are simulated JSON, no deployment/network/energy measurement.")
    source_contract(out)
    verification_contract(out, p)
    if (out / "validation.json").exists():
        same(result, read(out / "validation.json"))
    else:
        save(out / "validation.json", result)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=OUT)
    parser.add_argument("--declare", action="store_true")
    args = parser.parse_args()
    print(json.dumps(declare(args.output) if args.declare else verify(args.output), indent=2))
