"""Independent causal local-GPS sensitivity audit on immutable protected Q.

The public road graph and projection are explicit shared geometry primitives.
Sensor perturbations, causal estimates, answer/reference intersections and
denominators are reconstructed here without importing the workload scorer.
This is a same-map diagnostic, not a fresh confirmation or Geo-I certificate.
"""
import argparse
from collections import defaultdict
from datetime import datetime, timezone
import gzip
import hashlib
import json
import math
from pathlib import Path
import statistics

import numpy as np

from benchmark.public_poi_context import PublicPoiContext
from data.lane_states import build_lane_states, catalogue_summary
from evaluation.lane_travel import LanePoiService
from experiments.verify_dynamic_provider_status_20261006_v2 import OrderedRoadOracle

ROOT = Path(__file__).resolve().parents[1]
THIS = "experiments/verify_local_gps_robustness_20261007.py"
TEST = "tests/test_local_gps_robustness_verifier.py"
OUT = ROOT / "artifacts/benchmarks/local_gps_robustness_20261007_v1"
PURPOSES = ("nearest_distance", "fastest_travel", "within_radius", "minimum_detour")
SEED = 2026100701
FIX_INTERVAL = 60
SPEED_CAP = 8.
PUBLIC = "artifacts/benchmarks/research_loop/resources.json"
VARIANTS = {"exact_event_oracle": dict(mode="exact_event_oracle", sigma_m=0.)}
for _mode in ("hold_last60", "velocity_twofix60"):
    for _sigma in (0, 5, 15):
        VARIANTS[f"{_mode}_s{_sigma}"] = dict(mode=_mode, sigma_m=float(_sigma))
ARMS = tuple(f"{variant}--L{depth}" for variant in VARIANTS for depth in (20, 30))
FIXTURE_COUNT = 10


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    path = Path(path)
    raw = path.read_bytes()
    return json.loads(gzip.decompress(raw) if path.suffix == ".gz" else raw)


def canonical(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def save(path, value):
    with Path(path).open("x") as stream:
        stream.write(json.dumps(value, indent=2, allow_nan=False) + "\n")


def same(expected, actual, path="root"):
    if isinstance(expected, dict):
        assert isinstance(actual, dict) and set(expected) == set(actual), path
        for key in expected:
            same(expected[key], actual[key], path + "/" + str(key))
    elif isinstance(expected, (list, tuple)):
        assert isinstance(actual, (list, tuple)) and len(expected) == len(actual), path
        for index, (left, right) in enumerate(zip(expected, actual)):
            same(left, right, path + "/" + str(index))
    elif isinstance(expected, float):
        assert isinstance(actual, (int, float)) and math.isfinite(expected) and math.isfinite(actual), path
        assert math.isclose(expected, actual, rel_tol=1e-12, abs_tol=1e-9), (path, expected, actual)
    else:
        assert expected == actual, (path, expected, actual)


def standardized_offset(family_id, draw, slot, fix_t):
    """Public synthetic Box-Muller sensor tape; no private sampler state."""
    assert isinstance(draw, int) and not isinstance(draw, bool) and draw in (1, 2, 3)
    assert isinstance(slot, int) and not isinstance(slot, bool) and 0 <= slot < 8
    assert math.isfinite(fix_t) and 0 <= fix_t <= 600 and fix_t % FIX_INTERVAL == 0
    domain = ["synthetic-local-gps-gaussian-v1", SEED, str(family_id), draw, slot, float(fix_t)]
    digest = hashlib.sha256(json.dumps(domain, separators=(",", ":")).encode()).digest()
    uniforms = []
    for start in (0, 8):
        word = int.from_bytes(digest[start:start + 8], "big")
        value = ((word >> 11) + .5) / (1 << 53)
        uniforms.append(min(math.nextafter(1., 0.), max(math.nextafter(0., 1.), value)))
    radius = math.sqrt(-2. * math.log(uniforms[0]))
    angle = 2. * math.pi * uniforms[1]
    return radius * math.cos(angle), radius * math.sin(angle)


def causal_estimate(fixes, clock, mode):
    """Use last one/two already available noisy fixes; future coordinates unread."""
    assert math.isfinite(clock) and clock >= 0
    assert mode in ("hold_last60", "velocity_twofix60")
    past = []
    last = None
    for fix in fixes:
        timestamp = fix[0]
        assert math.isfinite(timestamp) and timestamp >= 0 and (last is None or timestamp > last)
        last = timestamp
        if timestamp <= clock:
            xy = tuple(float(value) for value in fix[1])
            assert len(xy) == 2 and all(math.isfinite(value) for value in xy)
            past.append((timestamp, xy))
    assert past, "At least one available local fix required"
    t1, position = past[-1]
    velocity = (0., 0.)
    if mode == "velocity_twofix60" and len(past) >= 2:
        t0, previous = past[-2]
        velocity = tuple((position[i] - previous[i]) / (t1 - t0) for i in range(2))
        norm = math.hypot(*velocity)
        if norm > SPEED_CAP:
            velocity = tuple(value * SPEED_CAP / norm for value in velocity)
    age = clock - t1
    return tuple(position[i] + velocity[i] * age for i in range(2)), velocity, age, t1


def causal_fix_history(raw_by_t, family_id, draw, slot, event_t, sigma_m, projection):
    """Only declared0/60/... sensor fixes are read; current events are not fixes."""
    assert sigma_m in (0., 5., 15.) and 0 <= event_t <= 600
    history = []
    for fix_t in range(0, int(event_t // FIX_INTERVAL) * FIX_INTERVAL + 1, FIX_INTERVAL):
        point = raw_by_t[fix_t]
        xy = projection(point["lat"], point["lon"])
        offset = standardized_offset(family_id, draw, slot, float(fix_t))
        history.append((float(fix_t), tuple(float(xy[i]) + sigma_m * offset[i] for i in range(2))))
    return history


def rank_score(reference_orders, estimated_orders, received, invalid=False):
    """True-current top5 denominator; local-estimated ranking forms the answer."""
    received = set(map(int, received))
    result = {}
    assert invalid or set(reference_orders) == set(estimated_orders)
    for purpose in reference_orders:
        references = reference_orders[purpose]
        estimates = [[] for _ in references] if invalid else estimated_orders[purpose]
        assert len(references) == len(estimates)
        recall, completion = [], []
        counts = dict(reference_category_count=0, all_category_count=len(references),
                      overlap_total=0, reference_poi_total=0, returned_items=0,
                      returned_outside_true_domain=0, empty_reference_categories=0,
                      zero_answer_reference_categories=0, estimated_empty_categories=0,
                      invalid_estimate_reference_categories=0, answered_empty_reference_categories=0)
        for ordered, approximate in zip(references, estimates):
            reference = list(map(int, ordered[:5]))
            answer = [int(i) for i in approximate if int(i) in received][:5]
            assert len(reference) == len(set(reference)) and len(answer) == len(set(answer))
            true_eligible = set(map(int, ordered))
            counts["returned_items"] += len(answer)
            counts["returned_outside_true_domain"] += len(set(answer) - true_eligible)
            counts["estimated_empty_categories"] += not approximate
            if not reference:
                counts["empty_reference_categories"] += 1
                counts["answered_empty_reference_categories"] += bool(answer)
                continue
            overlap = len(set(reference) & set(answer))
            recall.append(overlap / len(reference))
            completion.append(min(len(set(answer) & true_eligible), len(reference)) / len(reference))
            counts["reference_category_count"] += 1
            counts["reference_poi_total"] += len(reference)
            counts["overlap_total"] += overlap
            counts["zero_answer_reference_categories"] += not answer
            counts["invalid_estimate_reference_categories"] += invalid
        result[purpose] = dict(recall5=statistics.mean(recall) if recall else None,
                               completion=statistics.mean(completion) if completion else None, **counts)
    return result


def estimate_state(fixes, clock, mode):
    """Rebuild causal counters and unclipped-speed diagnostics independently."""
    xy, _, _, last_t = causal_estimate(fixes, clock, mode)
    past = [fix for fix in fixes if fix[0] <= clock]
    previous = past[-2] if len(past) > 1 else None
    raw_speed = 0.
    if mode == "velocity_twofix60" and previous is not None:
        raw_speed = math.hypot(*[(past[-1][1][i] - previous[1][i]) / (last_t - previous[0]) for i in range(2)])
    return dict(estimated_xy=list(xy), last_fix_t=last_t, previous_fix_t=None if previous is None else previous[0],
                local_fixes_observed=len(past), stored_fix_count=min(len(past), 2 if mode == "velocity_twofix60" else 1),
                raw_velocity_m_s=raw_speed, velocity_clipped=raw_speed > SPEED_CAP,
                first_fix_fallback=mode == "velocity_twofix60" and previous is None)


def source_contract(output):
    """Metadata and immutable source hashes only; no diagnostic score opened."""
    output = Path(output)
    p = read(output / "protocol.json")
    assert sha(output / "protocol.json") == (output / "protocol.sha256").read_text().strip()
    assert p["schema"] == "fixed-Q-local-gps-robustness-v1"
    assert p["noise_seed"] == SEED and p["sensor_interval_s"] == 60. and p["speed_ceiling_m_s"] == SPEED_CAP
    assert p["variants"] == VARIANTS and p["arms"] == list(ARMS) and p["depths"] == [20, 30]
    assert p["purposes"] == list(PURPOSES) and p["local_fix_clocks"] == list(map(float, range(0, 601, 60)))
    assert p["reference_k"] == 5 and p["radius_m"] == 1000. and p["split"] == "test" and p["draws"] == [1, 2, 3]
    assert len(p["families"]) == len(set(p["families"])) == 24
    names = [f"{family}--draw{draw}.json.gz" for family in p["families"] for draw in (1, 2, 3)]
    assert p["included_blocks"] == names and len(names) == 72
    assert set(p["base_family_files_sha256"]) == set(p["source_family_files_sha256"]) == set(names)
    for name, pin in p["source_sha256"].items():
        assert sha(ROOT / name) == sha(output / "source_snapshot" / name) == pin, name
    for folder, field in ((ROOT / p["source_output"], "source_files_sha256"), (ROOT / p["base_Q_output"], "base_files_sha256")):
        for name, pin in p[field].items():
            assert sha(folder / name) == pin, name
    base, depth = ROOT / p["base_Q_output"], ROOT / p["source_output"]
    base_protocol = read(base / "protocol.json")
    required_privacy_sources = {"core/mechanisms.py", "core/session_budget.py", "benchmark/anchor_belief.py",
                                "benchmark/engines/filtered_cover.py", "benchmark/engines/matched_filter.py", "benchmark/engines/paced_guard.py"}
    assert set(p["frozen_privacy_source_sha256"]) == required_privacy_sources
    for name, pin in p["frozen_privacy_source_sha256"].items():
        assert base_protocol["source_sha256"][name] == pin, name
    # Authenticate the whole original Q-generation closure as well as the six
    # explicitly highlighted primitive entries; this is read-only provenance.
    for name, pin in base_protocol["source_sha256"].items():
        assert sha(ROOT / name) == sha(base / "source_snapshot" / name) == pin, name
    certificate = read(depth / "validation.json")
    assert certificate["status"] == "pass" and certificate["protocol_sha256"] == sha(depth / "protocol.json")
    assert certificate["readout_sha256"] == sha(depth / "readout.json")
    assert certificate["paired_readout_sha256"] == sha(depth / "paired_readout.json")
    assert certificate["no_new_Q_GPS_anchors_or_ledger_reads"] is True
    assert certificate["verifier_sha256"] == sha(ROOT / "experiments/verify_qplanner_response_depth_generalization_20261006.py")
    assert sha(ROOT / p["dataset_path"]) == p["dataset_sha256"]
    assert sha(ROOT / PUBLIC) == p["public_resource_sha256"]
    assert sha(output / "public_reply60.npz") == p["reply_cache_sha256"]
    assert sha(output / "local_sensor_tape.json.gz") == p["local_sensor_tape_sha256"]
    return p


def verification_sources(p):
    original = read(ROOT / p["base_Q_output"] / "protocol.json")["source_sha256"]
    return sorted(set(p["source_sha256"]) | set(original) | {
        THIS, TEST, "experiments/verify_dynamic_provider_status_20261006_v2.py"})


def declare(output):
    output = Path(output)
    p = source_contract(output)
    assert not (output / "replay_started.json").exists() and not (output / "readout.json").exists()
    assert not (output / "blocks").exists(), "Checker must be frozen before diagnostic scoring"
    value = dict(schema="local-gps-independent-verification-protocol-v1", created_utc=datetime.now(timezone.utc).isoformat(),
                 workload_protocol_sha256=sha(output / "protocol.json"),
                 source_sha256={name: sha(ROOT / name) for name in verification_sources(p)}, synthetic_fixture_count=FIXTURE_COUNT,
                 scoring_started_before_declaration=False, sensor_reference="Separate declared11 virtual local fixes per session; actual-event GPS reference only",
                 ranking="Independent reused directed road oracle + newly reconstructed causal sensor/answer arithmetic; public graph/context primitives shared",
                 scope="Inspected same-map synthetic local-GPS sensitivity, no privacy proof, real sensor calibration, device energy, or new confirmation")
    save(output / "verification_protocol.json", value)
    (output / "verification_protocol.sha256").write_text(sha(output / "verification_protocol.json") + "\n")
    for name in value["source_sha256"]:
        target = output / "verification_source_snapshot" / name
        target.parent.mkdir(parents=True, exist_ok=True)
        with target.open("xb") as stream:
            stream.write((ROOT / name).read_bytes())
    return value


def verification_contract(output, p):
    output = Path(output)
    v = read(output / "verification_protocol.json")
    assert sha(output / "verification_protocol.json") == (output / "verification_protocol.sha256").read_text().strip()
    assert v["schema"] == "local-gps-independent-verification-protocol-v1"
    assert v["workload_protocol_sha256"] == sha(output / "protocol.json") and v["synthetic_fixture_count"] == FIXTURE_COUNT
    assert v["scoring_started_before_declaration"] is False
    assert set(v["source_sha256"]) == set(verification_sources(p))
    for name, pin in v["source_sha256"].items():
        assert sha(ROOT / name) == sha(output / "verification_source_snapshot" / name) == pin, name
    assert v["source_sha256"][THIS] == sha(__file__)
    return v


def expected_block(bundle, depth, context, oracle, tape):
    truth, streams = bundle["evaluator_only"], bundle["public"]["streams"]
    assert len(streams["legacy_l10"]) == len(streams["raw"]) == len(tape) == 8
    old_scores = {(row["method"], row["slot"], row["t"]): row for row in depth["utility"] if row["cache"] == "current"}
    old_wires = {(row["method"], row["slot"], row["t"]): row for row in depth["wire"]}
    rows, estimates, inventory = [], [], []
    for slot, session in enumerate(streams["legacy_l10"]):
        raw = streams["raw"][slot]["events"]
        assert [row["timestamp_s"] for row in raw] == [row["timestamp_s"] for row in session["events"]]
        raw_by_t = {row["timestamp_s"]: row["candidates"][0] for row in raw}
        assert len(raw_by_t) == len(raw) and all(len(row["candidates"]) == 1 for row in raw)
        assert raw[0]["timestamp_s"] == 0 and raw[-1]["timestamp_s"] == 600
        assert [fix["t"] for fix in tape[slot]] == list(map(float, range(0, 601, 60)))
        for fix in tape[slot]:
            same(fix, dict(t=fix["t"], lat=raw_by_t[fix["t"]]["lat"], lon=raw_by_t[fix["t"]]["lon"],
                           normal=list(standardized_offset(truth["family_id"], truth["draw"], slot, fix["t"]))), "sensor_tape")
        final = raw_by_t[600]
        destination = context.rn.nearest(final["lat"], final["lon"])[0]
        for event in session["events"]:
            t = event["timestamp_s"]
            inventory.append([slot, t, event["event_id"]])
            point = raw_by_t[t]
            true_xy = context.rn.point_xy(point["lat"], point["lon"])
            true_state = context.rn.nearest(point["lat"], point["lon"])[0]
            references = oracle.ordered(true_state, destination)
            positions = [(q["lat"], q["lon"]) for q in event["candidates"]]
            assert len(positions) == 5
            pools = {}
            for L in (20, 30):
                lists = [[int(i) for i in context.query_indices(context.rn.nearest(*position)[0])[:, :L].ravel() if i >= 0] for position in positions]
                assert lists == old_wires[f"service_l{L}", slot, t]["reply_poi_ids_by_Q"]
                pools[L] = {i for reply in lists for i in reply}
            assert pools[20] <= pools[30]
            for variant, config in VARIANTS.items():
                if config["mode"] == "exact_event_oracle":
                    state = dict(estimated_xy=list(true_xy), last_fix_t=t, previous_fix_t=None, local_fixes_observed=None,
                                 stored_fix_count=1, raw_velocity_m_s=0., velocity_clipped=False, first_fix_fallback=False)
                else:
                    fixes = causal_fix_history(raw_by_t, truth["family_id"], truth["draw"], slot, t, config["sigma_m"], context.rn.point_xy)
                    state = estimate_state(fixes, t, config["mode"])
                xy = state["estimated_xy"]
                assert all(math.isfinite(v) for v in xy)
                snap_distance, index = context.rn.tree.query(xy)
                index = int(index)
                predicted = oracle.ordered(index, destination)
                before = math.hypot(*(xy[i] - true_xy[i] for i in range(2)))
                after = math.hypot(*(float(context.rn.xy[index][i]) - true_xy[i] for i in range(2)))
                estimates.append(dict(slot=slot, t=t, event_id=event["event_id"], variant=variant, **state,
                                      estimated_state=index, snap_distance_m=float(snap_distance), error_before_snap_m=before,
                                      error_after_snap_m=after, invalid_reason=None))
                for L in (20, 30):
                    scores = rank_score(references, predicted, pools[L])
                    if variant == "exact_event_oracle":
                        for purpose in PURPOSES:
                            for key in ("recall5", "completion", "reference_category_count", "all_category_count", "overlap_total", "reference_poi_total"):
                                same(scores[purpose][key], old_scores[f"service_l{L}", slot, t]["purposes"][purpose][key], "original_exact_control")
                    rows.append(dict(family_id=truth["family_id"], draw=truth["draw"], split="test", slot=slot, t=t,
                                     event_id=event["event_id"], variant=variant, depth=L, arm=f"{variant}--L{L}", purposes=scores))
    assert set(old_wires) == {(f"service_l{L}", slot, t) for slot, t, _ in inventory for L in (20, 30)}
    return dict(schema="local-gps-robustness-block-v1", family_id=truth["family_id"], draw=truth["draw"], split="test",
                estimates=estimates, rows=rows, wire=depth["wire"], frozen_controls=depth["frozen_controls"],
                local_sensor_fix_count_per_session=[11] * 8, private_GPS_or_Q_regeneration=False,
                source_exact_current_scores_recovered=True), inventory


def percentile(values, quantile):
    values = sorted(values)
    if not values:
        return None
    rank = (len(values) - 1) * quantile
    lower, upper = math.floor(rank), math.ceil(rank)
    return values[lower] + (values[upper] - values[lower]) * (rank - lower)


def aggregate(blocks):
    """Own family/draw/category arithmetic and explicit event-pooled errors."""
    cells, costs, errors = defaultdict(list), defaultdict(lambda: defaultdict(int)), defaultdict(list)
    for block in blocks:
        for row in block["rows"]:
            phases = ["all"] + (["cold"] if row["slot"] == 0 else []) + (["temporal_tail_400_600"] if row["t"] >= 400 else [])
            for phase in phases:
                for purpose in PURPOSES:
                    cells[row["arm"], phase, purpose, row["family_id"], row["draw"]].append(row["purposes"][purpose])
        for wire in block["wire"]:
            for variant in VARIANTS:
                arm = f"{variant}--L{wire['method'].removeprefix('service_l')}"
                for field in ("requests", "request_bytes", "reply_bytes"):
                    costs[arm][field] += wire[field]
        for row in block["estimates"]:
            errors[row["variant"]].append(row)
    summary, local = {}, []
    for arm in ARMS:
        summary[arm] = {}
        for phase in ("all", "cold", "temporal_tail_400_600"):
            result = {}
            for purpose in PURPOSES:
                family_draw, by_draw, denominators = defaultdict(list), defaultdict(list), defaultdict(int)
                for (a, ph, pu, family, draw), values in sorted(cells.items()):
                    if (a, ph, pu) != (arm, phase, purpose):
                        continue
                    defined = [v["recall5"] for v in values if v["recall5"] is not None]
                    value = statistics.mean(defined) if defined else None
                    family_draw[family].append(value)
                    by_draw[draw].append(value)
                    local.append(dict(arm=arm, phase=phase, purpose=purpose, family_id=family, draw=draw,
                                      recall5=value, total_events=len(values), defined_events=len(defined)))
                    denominators["total_events"] += len(values)
                    denominators["defined_events"] += len(defined)
                    for event in values:
                        for key, count in event.items():
                            if key not in ("recall5", "completion"):
                                denominators[key] += count
                fvalues = {family: statistics.mean([v for v in vals if v is not None]) if any(v is not None for v in vals) else None for family, vals in family_draw.items()}
                defined = [v for v in fvalues.values() if v is not None]
                result[purpose] = dict(family_mean=statistics.mean(defined) if defined else None, family_values=fvalues,
                                       within_draw_family_mean={str(d): statistics.mean([v for v in vals if v is not None]) if any(v is not None for v in vals) else None for d, vals in by_draw.items()},
                                       explicit_denominators=dict(denominators))
            for name, purposes in (("equal_purpose_macro", PURPOSES), ("three_purpose_macro", PURPOSES[:3])):
                fvalues = {}
                for family in result[PURPOSES[0]]["family_values"]:
                    defined = [result[p]["family_values"][family] for p in purposes if result[p]["family_values"][family] is not None]
                    if defined:
                        fvalues[family] = statistics.mean(defined)
                draw_values = {}
                for draw in (1, 2, 3):
                    families = []
                    for family in fvalues:
                        defined = [row["recall5"] for row in local if (row["arm"], row["phase"], row["family_id"], row["draw"]) == (arm, phase, family, draw)
                                   and row["purpose"] in purposes and row["recall5"] is not None]
                        assert defined, "Every published family/draw macro requires a defined purpose"
                        families.append(statistics.mean(defined))
                    draw_values[str(draw)] = statistics.mean(families) if families else None
                result[name] = dict(family_mean=statistics.mean(fvalues.values()) if fvalues else None, family_values=fvalues,
                                    minimum_family_mean=min(fvalues.values()) if fvalues else None, within_draw_family_mean=draw_values)
            summary[arm][phase] = result
    positions = {}
    for variant, values in errors.items():
        fields = {}
        for key in ("error_before_snap_m", "error_after_snap_m", "snap_distance_m"):
            defined = [row[key] for row in values if row[key] is not None]
            fields[key] = dict(mean=statistics.mean(defined) if defined else None, p50=percentile(defined, .5),
                               p95=percentile(defined, .95), max=max(defined) if defined else None, defined_events=len(defined))
        positions[variant] = dict(events=len(values), invalid_events=sum(row["invalid_reason"] is not None for row in values),
                                 velocity_clipped_events=sum(row["velocity_clipped"] for row in values),
                                 first_fix_fallback_events=sum(row["first_fix_fallback"] for row in values), **fields)
    return dict(summary=summary, family_draw_purpose_values=local, cost={arm: dict(value) for arm, value in costs.items()}, position_error=positions)


def verify(output):
    output = Path(output)
    p = source_contract(output)
    verification_contract(output, p)
    started = read(output / "replay_started.json")
    assert started["protocol_sha256"] == sha(output / "protocol.json")
    assert started["verification_protocol_sha256"] == sha(output / "verification_protocol.json")
    assert started["local_sensor_tape_sha256"] == p["local_sensor_tape_sha256"]
    assert started["stage"] == "before_public_context_and_scoring"
    assert started["started_utc"] >= read(output / "verification_protocol.json")["created_utc"]
    data = read(ROOT / p["dataset_path"])
    network = ROOT / data["network"]["compressed_path"]
    assert sha(network) == data["network"]["compressed_sha256"]
    assert hashlib.sha256(gzip.decompress(network.read_bytes())).hexdigest() == data["network"]["native_sha256"]
    rn = build_lane_states(network, spacing_m=40.)
    pois = [{key: value for key, value in row.items() if key not in ("vertex", "access_offset_m")} for row in read(ROOT / PUBLIC)["pois_used"]]
    context = PublicPoiContext(LanePoiService(rn, pois, k=60), output / "public_reply60.npz")
    assert context.sha256 == p["public_reply60_sha256"]
    oracle = OrderedRoadOracle(context)
    tape = read(output / "local_sensor_tape.json.gz")
    assert tape["schema"] == "public-clock-synthetic-local-gps-tape-v1" and tape["seed"] == SEED
    assert set(tape["tapes"]) == set(p["included_blocks"])
    actual = read(output / "readout.json")
    assert actual["schema"] == "local-gps-robustness-readout-v1" and actual["protocol_sha256"] == sha(output / "protocol.json")
    assert actual["local_sensor_tape_sha256"] == p["local_sensor_tape_sha256"]
    assert actual["verification_protocol_sha256"] == sha(output / "verification_protocol.json")
    assert actual["replay_started_sha256"] == sha(output / "replay_started.json")
    assert actual["baseline_recovery_sha256"] == sha(output / "baseline_recovery.json")
    assert set(actual["block_files_sha256"]) == set(p["included_blocks"])
    assert {path.name for path in (output / "blocks").iterdir()} == set(p["included_blocks"])
    blocks, events = [], 0
    for index, name in enumerate(p["included_blocks"], 1):
        base, depth = ROOT / p["base_Q_output"] / "families" / name, ROOT / p["source_output"] / "families" / name
        assert sha(base) == p["base_family_files_sha256"][name] and sha(depth) == p["source_family_files_sha256"][name]
        bundle, protected = read(base), read(depth)
        assert canonical(bundle["public"]["streams"]["legacy_l10"]) == protected["frozen_controls"]["Q_stream_sha256"]
        assert protected["frozen_controls"] == p["frozen_controls"][name]
        expected, inventory = expected_block(bundle, protected, context, oracle, tape["tapes"][name])
        assert inventory == p["public_event_inventory"][name]
        expected.update(source_bundle_sha256=sha(base), source_depth_bundle_sha256=sha(depth))
        assert sha(output / "blocks" / name) == actual["block_files_sha256"][name]
        same(expected, read(output / "blocks" / name), name)
        blocks.append(expected)
        events += len(inventory)
        print(f"Independent local-GPS audit {index}/72 blocks; {events} unchanged public events", flush=True)
    expected = aggregate(blocks)
    for key, value in expected.items():
        same(value, actual[key], key)
    same(dict(catalogue=catalogue_summary(rn), public_reply60_sha256=context.sha256, pois=len(context.pois)), read(output / "resources.json"), "public_resources")
    same(dict(schema="local-gps-exact-baseline-recovery-v1", protocol_sha256=sha(output / "protocol.json"), blocks=72,
              public_events=events, depths=[20, 30],
              metrics=["recall5", "completion", "reference_category_count", "all_category_count", "overlap_total", "reference_poi_total"],
              completed_before_sensor_arm_scoring=True, all_exact_current_scores_recovered=True),
         read(output / "baseline_recovery.json"), "exact_baseline_recovery_receipt")
    assert actual["total_public_events"] == events and actual["virtual_local_sensor_fixes"] == 72 * 8 * 11
    for key in ("fixed_Q_clocks_anchors_ledger_wire", "no_private_key_or_protection_sampler_calls", "source_already_inspected_secondary_only",
                "local_clock_is_not_total_device_GNSS_reads", "all_exact_current_scores_recovered"):
        assert actual[key] is True, key
    result = dict(schema="local-gps-independent-verification-v1", status="pass", protocol_sha256=sha(output / "protocol.json"),
                  readout_sha256=sha(output / "readout.json"), verification_protocol_sha256=sha(output / "verification_protocol.json"),
                  verifier_sha256=sha(__file__), blocks=72, test_family_clusters=24, nested_draws=3,
                  public_events=events, local_virtual_fixes=72 * 8 * 11, local_estimate_rows=events * 7, utility_arm_rows=events * 14,
                  causal_public60s_sensor_noise_and_estimates_rebuilt=True, actual_current_reference_denominators_rebuilt=True,
                  exact_event_oracle_recovers_source_current_metrics=True, fixed_Q_replies_clocks_anchors_ledger_wire_inherited_and_checked=True,
                  family_draw_purpose_arithmetic_and_cost_recomputed=True, no_private_sampler_or_keys_required=True,
                  local_snapping_is_not_a_privacy_mechanism=True, destination_is_known_local_oracle=True,
                  three_purpose_macro_excludes_destination=True, fresh_confirmation=False, privacy_theorem_certified=False,
                  scope="Inspected same-map controlled per-axis Gaussian local-GPS sensitivity; separate11 virtual local fixes/session, exact event reference/destination oracle, event-pooled positional quantiles; not calibrated real GPS/total GNSS reads/energy or privacy evidence")
    source_contract(output)
    verification_contract(output, p)
    if (output / "validation.json").exists():
        same(result, read(output / "validation.json"))
    else:
        save(output / "validation.json", result)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=OUT)
    parser.add_argument("--declare", action="store_true")
    args = parser.parse_args()
    print(json.dumps(declare(args.output) if args.declare else verify(args.output), indent=2))
