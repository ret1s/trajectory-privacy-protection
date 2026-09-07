"""Purpose-built route families, executed by real SUMO (no trajectory editing)."""
from collections import defaultdict
import random
import xml.etree.ElementTree as ET


def successors(edge):
    return sorted((e for e, connections in edge.getOutgoing().items()
                   if not e.getID().startswith(':') and e.allows("passenger") and any(c.getFromLane().allows("passenger") and
                                                     c.getToLane().allows("passenger") for c in connections)),
                  key=lambda e: e.getID())


def predecessors(edge):
    return sorted((e for e in edge.getIncoming() if not e.getID().startswith(':') and edge in successors(e)), key=lambda e: e.getID())


def plan(network, seed):
    """Choose a connected route, a loop, merging origins and diverging endings.

    Randomness is used before simulation, independently of any defender. A
    public map may be shared across splits; each related family stays together.
    """
    rng = random.Random(seed)
    edges = sorted((e for e in network.getEdges(withInternal=False) if e.allows("passenger") and
                    e.getLength() >= 25), key=lambda e: e.getID())
    for _ in range(1500):
        origin, end = rng.sample(edges, 2)
        path, length = network.getShortestPath(origin, end, vClass="passenger")
        if not path or not 2500 <= length <= 5000 or len(path) < 12:
            continue
        # A real directed loop returning to a lane long enough to park on.
        stop_i = next((i for i in range(2, len(path)//2) if path[i].getLength() >= 40), None)
        if stop_i is None:
            continue
        back, _ = network.getShortestPath(path[-1], path[stop_i], vClass="passenger")
        if not back or len(back) < 3:
            continue
        fork_i = next((i for i in range(len(path)//3, len(path)-3)
                       if any(e != path[i+1] for e in successors(path[i]))), None)
        if fork_i is None:
            continue
        other = next(e for e in successors(path[fork_i]) if e != path[fork_i+1])
        alternate = list(path[:fork_i+1]) + [other]
        # Finish on an alternative edge. No reversal/teleport is inserted.
        for _ in range(4):
            choices = [e for e in successors(alternate[-1]) if e not in alternate]
            if not choices:
                break
            alternate.append(rng.choice(choices))
        merge_i = next((i for i in range(2, len(path)//2)
                        if any(e != path[i-1] and e not in path for e in predecessors(path[i]))), None)
        if merge_i is None:
            continue
        alternate_origin = next(e for e in predecessors(path[merge_i]) if e != path[merge_i-1] and e not in path)
        merged = [alternate_origin] + list(path[merge_i:])
        loop = list(path) + list(back[1:]) + list(path[stop_i+1:])
        access_i = next((i for i in range(len(path)-2, len(path)//2, -1) if len(predecessors(path[i])) == 1), None)
        if access_i is None:
            continue
        families = {"base": list(path), "alternative": alternate, "merge": merged, "return": loop,
                    "single_access": list(path[:access_i+1])}
        specs = []
        # SUMO IDs identify sessions, NOT people, physical cars, or devices.
        roles = [
            ("base", "base", 0, "p0", "d0", "car0", 0),
            ("companion", "base", 3, "p1", "d1", "car1", 0),
            ("partial", "alternative", 6, "p2", "d2", "car2", 0),
            ("incidental", "alternative", 9, "p3", "d3", "car3", 0),
            ("merged", "merge", 12, "p4", "d4", "car4", 0),
            ("short_stop", "base", 30, "p5", "d5", "car5", 30),
            ("long_stop", "base", 60, "p6", "d6", "car6", 180),
            ("return_stop", "return", 90, "p7", "d7", "car7", 45),
            ("access_endpoint", "single_access", 120, "p9", "d9", "car9", 0),
            ("repeat", "base", 2500, "p0", "d0", "car0", 0),
            ("new_device", "base", 5000, "p0", "d8", "car8", 0),
            ("shared_device", "base", 7500, "p8", "d0", "car0", 0),
        ]
        for role, family, depart, person, device, car, duration in roles:
            route = families[family]
            lane = next(l for l in path[stop_i].getLanes() if l.allows("passenger"))
            stops = [] if not duration else [{"lane": lane.getID(), "endPos": round(lane.getLength() * .7, 2),
                                              "duration": duration}] * (2 if role == "return_stop" else 1)
            if role == "return_stop":
                # Force the two visits onto different loop occurrences, not two
                # consecutive stops at the same location without moving.
                turn_lane = next(l for l in path[-1].getLanes() if l.allows("passenger"))
                stops.insert(1, {"lane": turn_lane.getID(), "endPos": round(turn_lane.getLength()*.7, 2), "duration": 1})
            specs.append({"session_id": f"u{seed}_{len(specs):02d}", "role": role, "family": family,
                          "person_id": f"{seed}/{person}", "device_id": f"{seed}/{device}",
                          "physical_vehicle_id": f"{seed}/{car}", "depart_s": depart,
                          "route_edges": [e.getID() for e in route], "stops": stops})
        return {"seed": seed, "family_id": f"family-{seed}", "fork_edge": path[fork_i].getID(),
                "merge_edge": path[merge_i].getID(), "sessions": specs}
    raise ValueError("No connected route family meeting declared design found; do not fabricate traces")


def route_xml(design):
    root = ET.Element("routes")
    ET.SubElement(root, "vType", id="urban", vClass="passenger", maxSpeed="8", speedFactor="1", sigma="0.3")
    for s in sorted(design["sessions"], key=lambda s: s["depart_s"]):
        v = ET.SubElement(root, "vehicle", id=s["session_id"], type="urban", depart=str(s["depart_s"]),
                          departLane="best")
        ET.SubElement(v, "route", edges=" ".join(s["route_edges"]))
        for stop in s["stops"]:
            ET.SubElement(v, "stop", **{k: str(v) for k, v in stop.items()}, parking="true")
    ET.indent(root)
    return ET.tostring(root, encoding="unicode", xml_declaration=True)


def parse_fcd(path, network):
    traces = defaultdict(list)
    for _, step in ET.iterparse(path, events=("end",)):
        if step.tag != "timestep":
            continue
        t = float(step.attrib["time"])
        for v in step.findall("vehicle"):
            a = v.attrib
            lane = network.getLane(a["lane"])
            # Preserve lane-relative progress, including junction-internal lanes.
            traces[a["id"]].append({"time_s": t, "lon": float(a["x"]), "lat": float(a["y"]),
                "speed_m_s": float(a["speed"]), "lane_id": a["lane"], "edge_id": lane.getEdge().getID(),
                "lane_pos_m": float(a["pos"]), "angle_deg": float(a["angle"])})
        step.clear()
    return dict(traces)
