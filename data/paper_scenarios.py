"""Current runnable cases, including controlled hidden trajectory endpoints.

S9/S10 target the first/last recorded FCD point of completed simulated trips,
not an inferred home or an unobserved real-life destination.
"""
from dataclasses import asdict
import xml.etree.ElementTree as ET

from data.threat_scenarios import records_from_fcd, catalogue


def records_for_study(traces, rn, vehicle_routes_path):
    records, rejected = records_from_fcd(traces, rn)
    users = {r["vehicle_id"] for r in records}
    completed = {v.attrib["id"] for v in ET.parse(vehicle_routes_path).getroot().findall("vehicle")
                 if float(v.attrib.get("arrival", -1)) >= 0}
    for user in sorted(users):
        trace = traces[user]
        if user not in completed or trace[-1].timestamp_s - trace[0].timestamp_s < 300:
            rejected.append({"vehicle_id": user, "reason": "endpoint_needs_completed_trip_300s"})
            continue
        for scenario in ("S9", "S10"):
            target = trace[0] if scenario == "S9" else trace[-1]
            start = trace[0].timestamp_s + 60 if scenario == "S9" else trace[-1].timestamp_s - 280
            stop = start + 220
            candidates = [p for p in trace if start <= p.timestamp_s <= stop]
            selected = []
            for p in candidates:
                if not selected or p.timestamp_s - selected[-1].timestamp_s >= 20:
                    selected.append(p)
                if len(selected) == 12:
                    break
            if len(selected) != 12:
                rejected.append({"vehicle_id": user, "scenario": scenario, "reason": "endpoint_window_incomplete"})
                continue
            records.append({"record_id": f"{user}/{scenario}", "scenario": scenario, "vehicle_id": user,
                            "points": [[p.lat, p.lon] for p in selected], "times": [p.timestamp_s for p in selected],
                            "fcd": [asdict(p) for p in selected], "hidden_target": [target.lat, target.lon],
                            "hidden_target_fcd": asdict(target), "mask_seconds": 60,
                            "target_definition": "first_or_last_FCD_of_completed_simulated_trip",
                            "checks": {"events": 12, "completed_trip": True, "mask_s": 60}})
    return records, rejected


def study_catalogue():
    return [{**item, "status": "runnable" if item["id"] in {"S1", "S2", "S3", "S9", "S10"}
             else "specified_not_evaluated"} for item in catalogue()]
