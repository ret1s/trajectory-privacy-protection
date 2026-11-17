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
PRIMARY = ROOT / "artifacts/benchmarks/paper_comparison_20261028_v1/families/cache-029.json.gz"
UPSTREAM = ROOT / "artifacts/benchmarks/main_depth_20261014_v1/families/cache-029.json.gz"
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
    assert trip["native_session_id"] == upstream["native_session_id"] == "cache-029_day7"
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
    assert [r["geo_i_spent_units"] for r in (records[i] for i in INDICES)] == [1, 1, 1, 3, 5, 6]
    assert records[9]["z_latlon"] == records[6]["z_latlon"]
    assert records[0]["q_latlon"] == records[1]["q_latlon"] == records[2]["q_latlon"]
    paths = [PRIMARY, UPSTREAM, NETWORK, ROOT / "data/lane_states.py", ROOT / "core/road_network.py",
             ROOT / "benchmark/engines/filtered_cover.py", ROOT / "benchmark/anchor_belief.py"]
    return dict(
        schema="paper-walkthrough-data-v1", native_session_id=trip["native_session_id"],
        selection="First trip in fixed first TEST family cache-029; prefix fixed at 0--180 s",
        shown_indices=list(INDICES), K=5, L_per_category=60, categories=6,
        geo_i_unit_per_m=.01, geo_i_cap_per_m=.23, gamma_Q=1,
        public_projection_lat0=rn.proj.lat0, origin_xy_m=origin.tolist(), records=records,
        scope="Retained sample; east coordinate only; belief operations, not reconstructed posterior probabilities",
        sources_sha256={str(p.relative_to(ROOT)): sha(p) for p in paths})


def render(data):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle
    import numpy as np
    plt.rcParams.update({
        "font.family": "DejaVu Serif", "font.size": 7.5, "pdf.fonttype": 42,
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.linewidth": .65, "xtick.major.width": .65, "ytick.major.width": .65,
    })
    fig = plt.figure(figsize=(7.0, 2.85))
    ax = fig.add_axes([.082, .20, .277, .68])
    rows = data["records"]
    times = [r["t_s"] for r in rows]
    ax.plot(times, [r["gps_east_m"] for r in rows], "-o", color="#222222",
            ms=2.8, lw=1.15, label="GPS")
    ax.step(times, [r["z_east_m"] for r in rows], where="post", color="#777777",
            lw=1.05, label="Z (local)")
    reads = [r for r in rows if r["private_read"]]
    ax.scatter([r["t_s"] for r in reads], [r["z_east_m"] for r in reads],
               color="#777777", marker="D", s=14, zorder=4)
    ax.plot(times, [r["q1_east_m"] for r in rows], "--s", color="#087f7a",
            ms=2.4, lw=1.05, label="Q1 of K5")
    ax.set(xlim=(-5, 185), ylim=(-230, 2780), xticks=[0, 60, 120, 180],
           yticks=[0, 1000, 2000], xlabel="Time (s)", ylabel="East from start (m)")
    ax.set_title("(a) Recorded coordinate traces", fontsize=8, pad=7)
    ax.grid(alpha=.15, lw=.5)
    ax.legend(loc="center left", bbox_to_anchor=(.01, .63), fontsize=6.9,
              frameon=False, handlelength=1.6, borderaxespad=0, labelspacing=.35)
    ax.annotate("same Z", xy=(180, rows[9]["z_east_m"]),
                xytext=(126, 1150), fontsize=6.9,
                arrowprops=dict(arrowstyle="->", lw=.6, color="#555555"))

    panel = fig.add_axes([.405, .20, .575, .68])
    panel.set(xlim=(0, 1), ylim=(0, 1))
    panel.axis("off")
    panel.set_title("(b) Selected ticks: local state and published frames", fontsize=8, pad=7)
    labels = ["Time (s)", "GPS / Z", "Belief op.", "Geo-I spent", "Q reserved",
              "Sent K5 frame", "POIs cached"]
    yy = [.96, .81, .68, .55, .42, .25, .08]
    xx = np.linspace(.30, .95, 6)
    chosen = [rows[i] for i in data["shown_indices"]]
    panel.add_patch(Rectangle((0, .355), 1, .52, color="#f2f2f2", zorder=0))
    panel.add_patch(Rectangle((0, .155), 1, .18, color="#edf5f4", zorder=0))
    for label, y in zip(labels, yy):
        panel.text(.008, y, label, va="center", fontsize=7.1,
                   fontweight="bold" if label in ("Time (s)", "Sent K5 frame") else "normal")
    for x, row in zip(xx, chosen):
        first = row["index"] == 0
        branch = ("init" if first else
                  {"fresh": "refresh", "reuse": "reuse", "public_clock_skip": "skip"}[row["branch"]])
        values = [str(row["t_s"]), branch, row["belief_operation"],
                  f'{row["geo_i_spent_per_m"]:.2f}',
                  f'{float(Fraction(row["q_reserved_upper_exact"])):.3f}']
        for y, value in zip(yy[:5], values):
            panel.text(x, y, value, ha="center", va="center", fontsize=7.0)
        panel.text(x, yy[5]+.018, r"$A_{%d}^{(%d)}$" % (row["segment"], row["frame"]),
                   ha="center", va="center", fontsize=8.0, color="#075e5b")
        panel.scatter(x + np.linspace(-.026, .026, 5), [.19]*5,
                      s=4, color="#087f7a", clip_on=False)
        panel.text(x, yy[6], str(row["cache_unique"]), ha="center", va="center", fontsize=7.2)
    panel.plot([.235, 1], [.895, .895], color="#555555", lw=.6)
    panel.plot([0, 1], [.345, .345], color="#777777", lw=.65, ls="--")
    # Only these three frames share a single initially selected segment.
    panel.plot([xx[0]-.035, xx[0]-.035, xx[2]+.035, xx[2]+.035],
               [.13, .115, .115, .13], color="#087f7a", lw=.65)
    fig.text(.405, .135, r"$A_j^{(f)}$: frame $f$ of segment $j$; every frame contains 5 Q.",
             fontsize=6.9)
    fig.text(.405, .087, r"Geo-I spent: m$^{-1}$ (cap 0.23); Q reserved: unitless (cap 1).",
             fontsize=6.9)
    fig.text(.082, .025,
             r"Cached POIs $\rightarrow$ local filter/sort (GPS + request) $\rightarrow$ full sorted answer.  Only K5 + cover is uploaded.",
             fontsize=6.9, color="#333333")
    buffer = BytesIO()
    fig.savefig(buffer, format="pdf", metadata={
        "CreationDate": None, "ModDate": None, "Creator": "Frozen GeoI-BQ sample walkthrough"})
    plt.close(fig)
    return buffer.getvalue()


def export(check=False):
    data = sample_data()
    encoded = json.dumps(data, indent=2) + "\n"
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
