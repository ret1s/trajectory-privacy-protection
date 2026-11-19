"""Translate figure labels only; preserve the retained English sample and map geometry."""
from pathlib import Path
from fractions import Fraction
from io import BytesIO
import json
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from experiments.export_paper_walkthrough_20261117 import sample_data

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
    overview.set_title("(a) Q đã gửi: toàn vùng truy vấn", fontsize=8, pad=7)
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
             Line2D([], [], marker="o", color="#087f7a", ls="", ms=2.5, label="mỗi 20 s"),
             Line2D([], [], marker="^", color="#087f7a", ls="", ms=3.5, label="180 s")]
    overview.legend(handles=q_key, loc="upper center", bbox_to_anchor=(.5, -.045),
                    fontsize=6.5, frameon=False, ncol=3, handletextpad=.15, columnspacing=.65)

    detail = fig.add_axes([.465, .42, .50, .49])
    background(detail, (detail_lo, detail_hi), width=.65)
    for spine in detail.spines.values():
        spine.set_color("#b66422")
    detail.set_title("(b) GPS và Z: vùng B (chỉ ở local)", fontsize=8, pad=7)
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
                  Line2D([], [], color="#777777", lw=.9, ls="--", label="cùng thời điểm")]
    detail.legend(handles=detail_key, loc="upper center", bbox_to_anchor=(.5, -.045),
                  fontsize=7, frameon=False, ncol=3, handlelength=1.5,
                  handletextpad=.4, columnspacing=1.2)

    panel = fig.add_axes([.045, .085, .92, .215])
    panel.set(xlim=(0, 1), ylim=(0, 1))
    panel.axis("off")
    fig.text(.045, .321, "(c) Các mốc: đọc GPS có bảo vệ, khung K5 đã cam kết và cache", fontsize=8)
    labels = ["Thời gian (s)", "GPS / Z", "Xử lý belief", r"Geo-I đã dùng (m$^{-1}$)",
              "Q dự phòng (không đơn vị)", "Khung K5 đã gửi", "Số POI trong cache"]
    yy = [.95, .80, .65, .50, .35, .20, .05]
    xx = np.linspace(.31, .95, 6)
    chosen = [rows[i] for i in data["shown_indices"]]
    panel.add_patch(Rectangle((0, .275), 1, .60, color="#f2f2f2", zorder=0))
    panel.add_patch(Rectangle((0, .13), 1, .145, color="#edf5f4", zorder=0))
    for label, y in zip(labels, yy):
        panel.text(.008, y, label, va="center", fontsize=7,
                   fontweight="bold" if label in ("Thời gian (s)", "Khung K5 đã gửi") else "normal")
    for x, row in zip(xx, chosen):
        branch = ("khởi tạo" if row["index"] == 0 else
                  {"fresh": "tạo mới", "reuse": "giữ lại", "public_clock_skip": "bỏ qua"}[row["branch"]])
        values = [str(row["t_s"]), branch, {"update": "cập nhật", "predict": "dự đoán"}[row["belief_operation"]],
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
             r"$A_j^{(f)}$: khung $f$ của đoạn $j$. Hai bản đồ khác tỷ lệ; hướng bắc ở trên. Đường: SUMO/OSM của benchmark.",
             fontsize=6.7, color="#444444")
    fig.text(.045, .009,
             r"POI trong cache $\rightarrow$ lọc/sắp xếp local (GPS + request) $\rightarrow$ toàn danh sách. Chỉ gửi K5 + truy vấn chung.",
             fontsize=6.9, color="#333333")
    buffer = BytesIO()
    fig.savefig(buffer, format="pdf", metadata={
        "CreationDate": None, "ModDate": None, "Creator": "Vietnamese labels on frozen GeoI-BQ sample"})
    plt.close(fig)
    return buffer.getvalue()


def export():
    data = sample_data()  # Read retained output and the pinned public road map only.
    retained = json.loads((ROOT / "artifacts/figures/paper_walkthrough_20261117_v1/sample.json").read_text())
    assert {k: v for k, v in data.items() if k != "map"} == {k: v for k, v in retained.items() if k != "map"}
    import hashlib
    roads = data["map"]["roads"]
    assert hashlib.sha256(json.dumps(roads, separators=(",", ":")).encode()).hexdigest() == retained["map"]["road_geometry_sha256"]
    target = ROOT / "paper_hk261_vi/figure_walkthrough.pdf"
    target.write_bytes(render(data))
    print("PASS: Vietnamese labels; retained sample, outputs and map geometry unchanged")


if __name__ == "__main__":
    export()
