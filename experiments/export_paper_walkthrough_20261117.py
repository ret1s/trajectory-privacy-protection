"""Render a multi-step illustration from a fixed, previously retained TEST trip.

Reads existing truth, protected anchors, private ledgers, Q frames and reply masks.
No defense sampling, attacker fitting, belief reconstruction or benchmark mutation.
"""
from fractions import Fraction
from io import BytesIO
from pathlib import Path
import argparse
import gzip
import hashlib
import json
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
OUT = ROOT / "artifacts/figures/paper_walkthrough_20261117_v1"
FAMILY_ID = "cache-046"
SESSION_ID = FAMILY_ID + "_day7"
PRIMARY = ROOT / f"artifacts/benchmarks/paper_comparison_20261028_v1/families/{FAMILY_ID}.json.gz"
UPSTREAM = ROOT / f"artifacts/benchmarks/main_depth_20261014_v1/families/{FAMILY_ID}.json.gz"
NETWORK = ROOT / "artifacts/datasets/future_controlled_20261005_v1/public_native.net.xml.gz"
INDICES = (0, 1, 2, 3, 6, 9)


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_gzip(path):
    return json.loads(gzip.decompress(path.read_bytes()))


def sample_data():
    import numpy as np
    from core.research_release import load_release
    from data.lane_states import build_lane_states
    load_release()
    trip = read_gzip(PRIMARY)["trips"][0]
    upstream = read_gzip(UPSTREAM)["trips"][0]
    assert trip["native_session_id"] == upstream["native_session_id"] == SESSION_ID
    assert trip["truth"] == upstream["truth"], "Use the same retained paired trajectory"
    # Reconstruct only the public coordinate projection, never a protection mechanism.
    with tempfile.TemporaryDirectory(prefix="paper-public-projection-") as directory:
        net = Path(directory) / "network.net.xml"
        net.write_bytes(gzip.decompress(NETWORK.read_bytes()))
        rn = build_lane_states(net, spacing_m=40.)
    origin = np.asarray(trip["truth"][0]["xy"])
    certificates = upstream["Q_certificates"]["cache_target_g1"]
    records = []
    for i in range(10):  # Fixed 0--180 s prefix, not selected by privacy/utility score.
        point, event, ledger = trip["truth"][i], trip["events"]["geoi_bq"][i], upstream["GeoI_ledger"][i]
        certificate = certificates[i]
        assert point["t"] == event["timestamp_s"] == certificate["t"] == 20 * i
        assert len(event["coordinates"]) == 5
        assert event["coordinates"] == upstream["events"]["cache_target_g1"][i]["coordinates"]
        assert ledger["spent_units"] <= 23
        block = int(certificate["block"])
        assert block == i // 3 + 1
        reserved = Fraction(block, block + 1)
        assert abs(float(reserved) - certificate["reserved_prefix_epsilon_Q"]) < 1e-14
        anchor = upstream["GeoI_anchors"][i]
        anchor_xy = np.asarray(rn.proj.to_xy(*anchor))
        q = np.asarray(event["coordinates"])
        q_xy = np.column_stack(rn.proj.to_xy(q[:, 0], q[:, 1]))
        records.append(dict(
            index=i, t_s=point["t"], gps_xy_m=point["xy"], z_latlon=anchor,
            z_xy_m=anchor_xy.tolist(), q_xy_m=q_xy.tolist(),
            gps_east_m=round(float(point["xy"][0] - origin[0]), 6),
            z_east_m=round(float(anchor_xy[0] - origin[0]), 6),
            q1_east_m=round(float(q_xy[0, 0] - origin[0]), 6),
            q_latlon=event["coordinates"], branch=ledger["branch"],
            private_read=ledger["private_read"], geo_i_cost_units=ledger["cost_units"],
            geo_i_spent_units=ledger["spent_units"],
            geo_i_spent_per_m=ledger["spent_units"] / 100,
            belief_operation="update" if ledger["private_read"] else "predict",
            segment=block, frame=i % 3 + 1, q_reserved_upper_exact=str(reserved),
            cache_unique=int(trip["masks"]["geoi_bq"][i]["session"], 16).bit_count()))
    assert [r["geo_i_spent_units"] for r in (records[i] for i in INDICES)] == [1, 1, 1, 3, 5, 7]
    assert records[9]["branch"] == "fresh" and records[9]["z_latlon"] != records[6]["z_latlon"]
    assert records[0]["q_latlon"] == records[1]["q_latlon"] == records[2]["q_latlon"]
    points = np.concatenate([np.asarray([r["gps_xy_m"], r["z_xy_m"]] + r["q_xy_m"])
                             for r in records]) - origin
    lo, hi = points.min(axis=0) - 300, points.max(axis=0) + 300
    local = np.asarray([[r["gps_xy_m"], r["z_xy_m"]] for r in records]).reshape(-1, 2) - origin
    zoom_lo, zoom_hi = local.min(axis=0) - [110, 90], local.max(axis=0) + [110, 90]
    zoom_center_y = (zoom_lo[1] + zoom_hi[1]) / 2
    zoom_half_height = max(200, (zoom_hi[1] - zoom_lo[1]) / 2)
    zoom_lo[1], zoom_hi[1] = zoom_center_y - zoom_half_height, zoom_center_y + zoom_half_height
    # Exact passenger-lane geometry from the same public SUMO snapshot; no tiles,
    # synthetic street grid, inferred private path or background-map service.
    from data.sumo_demo import _load_sumolib
    with tempfile.TemporaryDirectory(prefix="paper-public-roads-") as directory:
        path = Path(directory) / "network.net.xml"
        path.write_bytes(gzip.decompress(NETWORK.read_bytes()))
        network = _load_sumolib().net.readNet(str(path), withInternal=False)
        roads = []
        for edge in sorted(network.getEdges(), key=lambda e: e.getID()):
            lanes = [lane for lane in edge.getLanes() if lane.allows("passenger")]
            if not lanes:
                continue
            lonlat = np.asarray([network.convertXY2LonLat(*p) for p in lanes[0].getShape()])
            xy = np.column_stack(rn.proj.to_xy(lonlat[:, 1], lonlat[:, 0])) - origin
            if np.all(xy.max(axis=0) >= lo) and np.all(xy.min(axis=0) <= hi):
                roads.append(dict(edge_id=edge.getID(), xy_m=xy.round(3).tolist()))
    assert roads
    paths = [PRIMARY, UPSTREAM, NETWORK, ROOT / "data/lane_states.py", ROOT / "core/road_network.py",
             ROOT / "benchmark/engines/filtered_cover.py", ROOT / "benchmark/anchor_belief.py"]
    return dict(
        schema="paper-walkthrough-data-v1", native_session_id=trip["native_session_id"],
        selection="First retained trip of cache-046; 0--180 s multi-turn geometry selected for illustration at the user's request, without attack/utility-score selection",
        shown_indices=list(INDICES), K=5, L_per_category=60, categories=6,
        geo_i_unit_per_m=.01, geo_i_cap_per_m=.23, gamma_Q=1,
        public_projection_lat0=rn.proj.lat0, origin_xy_m=origin.tolist(), records=records,
        map=dict(source="Public benchmark SUMO network with OpenStreetMap provenance",
                 geometry="First passenger lane per external edge; display coordinates rounded to 1 mm",
                 bounds_m=[lo.tolist(), hi.tolist()], zoom_bounds_m=[zoom_lo.tolist(), zoom_hi.tolist()],
                 roads=roads),
        scope="Retained two-dimensional locations on the actual public road map; belief operations, not reconstructed posterior probabilities",
        sources_sha256={str(p.relative_to(ROOT)): sha(p) for p in paths})


def render(data):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle
    from matplotlib.collections import LineCollection
    from matplotlib.lines import Line2D
    import matplotlib.patheffects as pe
    import numpy as np
    plt.rcParams.update({
        "font.family": "DejaVu Serif", "font.size": 7.5, "pdf.fonttype": 42,
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.linewidth": .65, "xtick.major.width": .65, "ytick.major.width": .65,
    })
    fig = plt.figure(figsize=(7.0, 4.0))
    rows = data["records"]
    origin = np.asarray(data["origin_xy_m"])
    gps = np.asarray([r["gps_xy_m"] for r in rows]) - origin
    z = np.asarray([r["z_xy_m"] for r in rows]) - origin
    q = np.asarray([r["q_xy_m"] for r in rows]) - origin
    roads = [r["xy_m"] for r in data["map"]["roads"]]
    lo, hi = np.asarray(data["map"]["bounds_m"])
    detail_lo, detail_hi = np.asarray(data["map"]["zoom_bounds_m"])
    colors = {0: "#0072b2", 3: "#009e73", 6: "#d55e00", 9: "#8b5aa5"}

    def background(target, limits, width=.35):
        target.set_facecolor("#f8f9f7")
        target.add_collection(LineCollection(roads, colors="#c8cecc", linewidths=width, zorder=0))
        target.set(xlim=(limits[0][0], limits[1][0]), ylim=(limits[0][1], limits[1][1]),
                   aspect="equal", xticks=[], yticks=[])
        for spine in target.spines.values():
            spine.set_visible(True)
            spine.set_color("#bcc3c0")

    def outlined(target, x, y, value, **kw):
        return target.text(x, y, value, path_effects=[pe.withStroke(linewidth=2, foreground="white")], **kw)

    def scale(target, x, y, length, label):
        target.plot([x, x+length], [y, y], color="#222222", lw=1.1)
        height = length*.035
        target.vlines([x, x+length], y-height, y+height, color="#222222", lw=.7)
        outlined(target, x+length/2, y+length*.10, label, ha="center", fontsize=7)

    # Separate overview and large detail: no inset hides the published Q.
    overview = fig.add_axes([.035, .41, .34, .51])
    background(overview, (lo, hi))
    overview.set_title("(a) Sent Q: full query area", fontsize=8, pad=7)
    for k in range(5):
        overview.scatter(q[:, k, 0], q[:, k, 1], s=np.linspace(7, 14, len(rows)),
                         color="#087f7a", edgecolors="white", linewidths=.25, zorder=3)
        overview.scatter(*q[0, k], s=22, facecolors="white", edgecolors="#087f7a", lw=.8, zorder=4)
        overview.scatter(*q[-1, k], s=21, color="#087f7a", marker="^", zorder=5)
        outlined(overview, q[0, k, 0]+85, q[0, k, 1]+80, f"Q{k+1}",
                 fontsize=7, color="#075e5b", zorder=6)
    overview.add_patch(Rectangle(detail_lo, *(detail_hi-detail_lo), fill=False,
                                edgecolor="#b66422", lw=1.0, zorder=6))
    outlined(overview, *(detail_lo+(detail_hi-detail_lo)/2), "B",
             ha="center", va="center", color="#9b5018", fontweight="bold", fontsize=8, zorder=7)
    scale(overview, lo[0]+150, lo[1]+.58*(hi[1]-lo[1]), 1000, "1 km")
    overview.annotate("N", xy=(.08, .80), xytext=(.08, .67), xycoords="axes fraction",
                      fontsize=7, ha="center", arrowprops=dict(arrowstyle="-|>", lw=.7, color="#444444"))
    q_key = [Line2D([], [], marker="o", color="#087f7a", mfc="white", ls="", ms=3.5, label="0 s"),
             Line2D([], [], marker="o", color="#087f7a", ls="", ms=2.5, label="20 s ticks"),
             Line2D([], [], marker="^", color="#087f7a", ls="", ms=3.5, label="180 s")]
    overview.legend(handles=q_key, loc="upper center", bbox_to_anchor=(.5, -.045),
                    fontsize=6.5, frameon=False, ncol=3, handletextpad=.15, columnspacing=.65)

    detail = fig.add_axes([.465, .42, .50, .49])
    background(detail, (detail_lo, detail_hi), width=.65)
    for spine in detail.spines.values():
        spine.set_color("#b66422")
    detail.set_title("(b) GPS and Z: detail B (local only)", fontsize=8, pad=7)
    detail.scatter(gps[:, 0], gps[:, 1], s=11, color="#333333", zorder=3)
    # Dashed arrows show same-time reference displacement, not a travelled path.
    gps_offsets = {0: (13, -7), 3: (-14, 9), 6: (-13, -10), 9: (15, -9)}
    z_offsets = {0: (-14, -9), 3: (-14, 10), 6: (-14, 10), 9: (14, 10)}
    for i, color in colors.items():
        detail.annotate("", xy=z[i], xytext=gps[i],
                        arrowprops=dict(arrowstyle="->", color=color, lw=.85, ls=(0, (3, 2)), alpha=.85),
                        zorder=2)
        detail.scatter(*gps[i], s=24, color=color, edgecolors="white", linewidths=.4, zorder=4)
        detail.scatter(*z[i], s=38, marker="D", facecolors="white",
                       edgecolors=color, linewidths=1.0, zorder=5)
        for point, label, offset in [(gps[i], str(rows[i]["t_s"])+" s", gps_offsets[i]),
                                     (z[i], "Z"+str(rows[i]["t_s"]), z_offsets[i])]:
            detail.annotate(label, xy=point, xytext=offset, textcoords="offset points",
                            fontsize=7.1, ha="center", va="center", color=color,
                            arrowprops=dict(arrowstyle="-", color=color, lw=.45),
                            bbox=dict(facecolor="white", edgecolor="none", pad=.3), zorder=6)
    scale(detail, detail_hi[0]-350, detail_lo[1]+35, 200, "200 m")
    detail_key = [Line2D([], [], marker="o", color="#333333", ls="", ms=3.5, label="GPS"),
                  Line2D([], [], marker="D", color="#333333", mfc="white", ls="", ms=4, label="Z"),
                  Line2D([], [], color="#777777", lw=.9, ls="--", label="same-time pair")]
    detail.legend(handles=detail_key, loc="upper center", bbox_to_anchor=(.5, -.045),
                  fontsize=7, frameon=False, ncol=3, handlelength=1.5,
                  handletextpad=.4, columnspacing=1.2)

    panel = fig.add_axes([.045, .085, .92, .215])
    panel.set(xlim=(0, 1), ylim=(0, 1))
    panel.axis("off")
    fig.text(.045, .321, "(c) Selected ticks: protected reads, committed K5 frames and cache", fontsize=8)
    labels = ["Time (s)", "GPS / Z", "Belief op.", r"Geo-I spent (m$^{-1}$)",
              "Q reserved (unitless)", "Sent K5 frame", "POIs cached"]
    yy = [.95, .80, .65, .50, .35, .20, .05]
    xx = np.linspace(.31, .95, 6)
    chosen = [rows[i] for i in data["shown_indices"]]
    panel.add_patch(Rectangle((0, .275), 1, .60, color="#f2f2f2", zorder=0))
    panel.add_patch(Rectangle((0, .13), 1, .145, color="#edf5f4", zorder=0))
    for label, y in zip(labels, yy):
        panel.text(.008, y, label, va="center", fontsize=7,
                   fontweight="bold" if label in ("Time (s)", "Sent K5 frame") else "normal")
    for x, row in zip(xx, chosen):
        branch = ("init" if row["index"] == 0 else
                  {"fresh": "refresh", "reuse": "reuse", "public_clock_skip": "skip"}[row["branch"]])
        values = [str(row["t_s"]), branch, row["belief_operation"],
                  f'{row["geo_i_spent_per_m"]:.2f}',
                  f'{float(Fraction(row["q_reserved_upper_exact"])):.3f}']
        for j, (y, value) in enumerate(zip(yy[:5], values)):
            panel.text(x, y, value, ha="center", va="center", fontsize=7,
                       color=colors.get(row["index"], "#333333") if j == 0 else "#222222",
                       fontweight="bold" if j == 0 else "normal")
        panel.text(x, yy[5], r"$A_{%d}^{(%d)}$" % (row["segment"], row["frame"]),
                   ha="center", va="center", fontsize=8, color="#075e5b")
        panel.text(x, yy[6], str(row["cache_unique"]), ha="center", va="center", fontsize=7.1)
    panel.plot([.24, 1], [.875, .875], color="#555555", lw=.6)
    fig.text(.045, .045,
             r"$A_j^{(f)}$: frame $f$ of segment $j$. Maps use different scales; north is up. Roads: benchmark SUMO/OSM.",
             fontsize=6.7, color="#444444")
    fig.text(.045, .009,
             r"Cached POIs $\rightarrow$ local filter/sort (GPS + request) $\rightarrow$ full sorted answer.  Only K5 + cover is uploaded.",
             fontsize=6.9, color="#333333")
    buffer = BytesIO()
    fig.savefig(buffer, format="pdf", metadata={
        "CreationDate": None, "ModDate": None, "Creator": "Frozen GeoI-BQ sample walkthrough"})
    plt.close(fig)
    return buffer.getvalue()

def export(check=False):
    data = sample_data()
    # The pinned network already stores all road geometry. Bind the rendered
    # subset by digest/count rather than duplicating thousands of polylines.
    roads = data["map"]["roads"]
    recorded_map = {k: v for k, v in data["map"].items() if k != "roads"}
    recorded_map.update(road_edge_count=len(roads), road_geometry_sha256=hashlib.sha256(
        json.dumps(roads, separators=(",", ":")).encode()).hexdigest())
    encoded = json.dumps(dict(data, map=recorded_map), indent=2) + "\n"
    pdf = render(data)
    record = dict(schema="paper-walkthrough-receipt-v1", date="2026-11-17",
                  data_sha256=hashlib.sha256(encoded.encode()).hexdigest(),
                  figure_sha256=hashlib.sha256(pdf).hexdigest(),
                  sources_sha256=data["sources_sha256"],
                  exporter_sha256=sha(Path(__file__)),
                  benchmark_changed=False, defense_or_attackers_rerun=False)
    contents = {"sample.json": encoded.encode(), "figure_walkthrough.pdf": pdf,
                "receipt.json": (json.dumps(record, indent=2) + "\n").encode()}
    for name, content in contents.items():
        target = OUT / name
        if check:
            assert target.read_bytes() == content, target
        else:
            OUT.mkdir(parents=True, exist_ok=True)
            target.write_bytes(content)
    return data


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--check", action="store_true")
    args = p.parse_args()
    export(check=args.check)
    print("PASS: fixed retained sample and deterministic figure; no new model outputs")
