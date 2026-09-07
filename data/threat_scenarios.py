"""Versioned scenario eligibility from actual SUMO FCD; no GeoLife fallback."""
from dataclasses import asdict
import numpy as np

TAXONOMY_VERSION = "urban-threats-2026-09-07-v2"
SCENARIOS = {
    "S1": ("Định vị một lần gửi", "current_location"),
    "S2": ("Suy luận điểm dừng", "stop_location"),
    "S3": ("Tái dựng đường đã đi", "past_trajectory"),
    "S4": ("Liên kết người hoặc thiết bị", "identity"),
    "S5": ("Dự đoán bước tiếp theo", "next_edge"),
    "S6": ("Dự đoán đích đến", "future_destination"),
    "S7": ("Suy luận nhu cầu truy vấn", "query_intent"),
    "S8": ("Suy luận qua đồng hành", "co_location"),
    "S9": ("Suy luận điểm xuất phát", "hidden_origin"),
    "S10": ("Suy luận điểm kết thúc", "hidden_endpoint"),
}


def catalogue():
    return [{"id": key, "name": value[0], "target": value[1],
             "status": "runnable" if key in {"S1", "S2", "S3"} else "specified_not_evaluated"}
            for key, value in SCENARIOS.items()]


def records_from_fcd(traces, rn, *, interval_s=20, max_events=12, dwell_s=120):
    """Eligibility uses simulator truth; attacks never receive this object.

    S2 explicitly grants the attacker knowledge that its observation interval
    is stationary. The location remains secret. S1 gives one observation only.
    """
    if interval_s <= 0 or dwell_s < 60 or max_events < 2 or (max_events - 1) * interval_s < dwell_s:
        raise ValueError("Sampling window must preserve the required dwell interval")
    records, rejected = [], []
    for vehicle_id, trace in sorted(traces.items()):
        # This release consumes 1 Hz SUMO FCD; missing/time-reversed records
        # must not silently create a long continuous stop or moving interval.
        if any(not 0 < b.timestamp_s - a.timestamp_s <= 1.01 for a, b in zip(trace, trace[1:])):
            rejected.append({"vehicle_id": vehicle_id, "reason": "non_contiguous_1hz_fcd"})
            continue
        stopped = []
        longest = []
        for sample in trace:
            if sample.speed_m_s is not None and sample.speed_m_s <= 0.05:
                stopped.append(sample)
                if len(stopped) > len(longest):
                    longest = stopped[:]
            else:
                stopped = []
        if not longest or longest[-1].timestamp_s - longest[0].timestamp_s < dwell_s:
            rejected.append({"vehicle_id": vehicle_id, "reason": "no_dwell_120s"})
            continue
        stop_xy = np.array([rn.point_xy(p.lat, p.lon) for p in longest])
        if np.linalg.norm(stop_xy - stop_xy[0], axis=1).max() > 5:
            rejected.append({"vehicle_id": vehicle_id, "reason": "stop_drift_exceeds_5m"})
            continue
        # Select the longest contiguous moving interval of sufficient span.
        moving, segments = [], []
        for sample in trace:
            if sample.speed_m_s is not None and sample.speed_m_s > 0.05:
                moving.append(sample)
            elif moving:
                segments.append(moving)
                moving = []
        if moving:
            segments.append(moving)
        moving = max(segments, key=len, default=[])
        if len(moving) < 2 or moving[-1].timestamp_s - moving[0].timestamp_s < 60:
            rejected.append({"vehicle_id": vehicle_id, "reason": "no_moving_interval_60s"})
            continue
        for scenario, source in (("S1", moving[:1]), ("S2", longest), ("S3", moving)):
            selected = []
            for point in source:
                if not selected or point.timestamp_s - selected[-1].timestamp_s >= interval_s:
                    selected.append(point)
                if len(selected) >= max_events:
                    break
            xy = np.array([rn.point_xy(p.lat, p.lon) for p in selected])
            records.append({
                "record_id": f"{vehicle_id}/{scenario}", "scenario": scenario,
                "vehicle_id": vehicle_id, "taxonomy_version": TAXONOMY_VERSION,
                "points": [[p.lat, p.lon] for p in selected],
                "times": [p.timestamp_s for p in selected],
                "fcd": [asdict(p) for p in selected],
                "checks": {
                    "events": len(selected),
                    "strictly_increasing_time": all(b.timestamp_s > a.timestamp_s
                                                    for a, b in zip(selected, selected[1:])),
                    "span_s": selected[-1].timestamp_s - selected[0].timestamp_s,
                    "max_radius_from_first_m": float(np.linalg.norm(xy - xy[0], axis=1).max()),
                    "max_edge_offset_m": max(rn.dist_to_edge(p.lat, p.lon) for p in selected),
                },
            })
    return records, rejected
