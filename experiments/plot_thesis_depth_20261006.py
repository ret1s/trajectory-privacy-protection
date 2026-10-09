"""Print-size thesis figure from certified fixed-Q L20/L30 evidence.

No scores, samples, confidence intervals, selection or private keys are created.
Run ``python -m experiments.plot_thesis_depth_20261006`` from the repository.
"""
import hashlib
import json
from pathlib import Path
import statistics

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "artifacts/benchmarks/qplanner_response_depth_generalization_20261006_v1"
OUT = ROOT / "artifacts/reports/thesis_figures_20261006"


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    paired = json.loads((BASE / "paired_readout.json").read_text())
    config = json.loads((BASE / "recommended_configuration.json").read_text())
    cert = json.loads((BASE / "validation.json").read_text())
    assert cert["status"] == "pass" and paired["primary_criterion_result"]["passes"]
    assert cert["paired_readout_sha256"] == sha(BASE / "paired_readout.json")
    for path, digest in config["provenance_sha256"].items():
        assert sha(BASE / path) == digest, path
    purposes = [
        ("nearest_distance", "Gần nhất"),
        ("fastest_travel", "Nhanh nhất"),
        ("within_radius", "Trong bán kính"),
        ("minimum_detour", "Ít vòng đường nhất"),
        ("equal_purpose_macro", "Trung bình đều\n(chính)"),
    ]
    rows = []
    for purpose, label in purposes:
        means = [100 * statistics.mean(paired["conditional_family_cells"][
            f"service_l{L}--test--current--all"][purpose]["family_values"].values()) for L in (20, 30)]
        contrast = paired["contrasts"][f"service_l30--minus--service_l20--test--current--all--{purpose}"]
        gain = 100 * contrast["mean_difference"]
        assert abs(means[1] - means[0] - gain) < 1e-10
        assert contrast["independent_family_clusters"] == 24
        rows.append(dict(purpose=purpose, label=label, L20=means[0], L30=means[1],
            gain_pp=gain, interval95_pp=[100*x for x in contrast["percentile95_family_bootstrap"]],
            reference_coverage=contrast.get("left_reference_coverage"),
            interval_scope="primary predeclared" if purpose=="equal_purpose_macro" else "secondary, multiplicity unadjusted"))
    cost = config["actual_TEST_cost"]
    assert cost["L20"]["requests"] == cost["L30"]["requests"]
    growth = 100 * (cost["L30"]["reply_bytes"] / cost["L20"]["reply_bytes"] - 1)
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10,
        "axes.titlesize": 10.5, "axes.labelsize": 10, "pdf.fonttype": 42,
        "axes.spines.top": False, "axes.spines.right": False})
    fig, axes = plt.subplots(2, 1, figsize=(6.3, 6.1))
    fig.subplots_adjust(left=.255, right=.96, bottom=.255, top=.86, hspace=.64)
    y = np.arange(len(rows))[::-1]
    blue, orange = "#174A7E", "#C36B26"
    for i, row in enumerate(rows):
        axes[0].plot([row["L20"], row["L30"]], [y[i], y[i]], color="#a8a8a8", lw=1.2)
    axes[0].scatter([r["L20"] for r in rows], y, facecolors="white", edgecolors=blue,
        s=36, linewidths=1.3, label="L20", zorder=3)
    axes[0].scatter([r["L30"] for r in rows], y, marker="s", color=orange,
        s=30, label="L30", zorder=4)
    axes[0].set_xlim(75, 100)
    axes[0].set_xticks([75, 80, 85, 90, 95, 100])
    axes[0].set_xlabel("Recall@5 (%) — trục phóng từ 75 đến 100")
    axes[0].set_title("(a) Chất lượng từ phản hồi hiện tại", loc="left", pad=12)
    axes[0].legend(loc="lower right", bbox_to_anchor=(1.01, 1.02), ncol=2,
        frameon=False, borderaxespad=0, handletextpad=.4, columnspacing=1.2)
    for i, row in enumerate(rows):
        lo, hi = row["interval95_pp"]
        axes[1].errorbar(row["gain_pp"], y[i], xerr=[[row["gain_pp"]-lo],[hi-row["gain_pp"]]],
            fmt="D" if i==4 else "o", color=blue, capsize=3, markersize=5, lw=1.3)
        axes[1].text(hi+.15, y[i], f"+{row['gain_pp']:.2f}", va="center", fontsize=9.5)
    axes[1].axvline(0, color="#5c5c5c", lw=.9)
    axes[1].set_xlim(-.05, 7.2)
    axes[1].set_xticks([0, 1, 2, 3, 4, 5, 6, 7])
    axes[1].set_xlabel("Gain L30 − L20 (điểm phần trăm)")
    axes[1].set_title("(b) Chênh lệch và CI95% ghép theo nhóm tuyến", loc="left", pad=12)
    for ax in axes:
        ax.set_yticks(y, [r["label"] for r in rows])
        ax.set_ylim(-.55, 4.55)
        ax.grid(axis="x", color="#dedede", lw=.5)
        ax.set_axisbelow(True)
        ax.tick_params(axis="y", length=0)
    fig.text(.02, .975, "24 nhóm tuyến mới × 3 lần lấy mẫu/nhóm; tổng hợp cùng bản đồ",
        ha="left", va="top", fontsize=10)
    fig.text(.02, .12, f"Reply JSON: {cost['L20']['reply_bytes']/1e6:.2f} → "
        f"{cost['L30']['reply_bytes']/1e6:.2f} MB (+{growth:.2f}%).", fontsize=10)
    fig.text(.02, .087, f"{cost['L20']['requests']:,} requests; Q, GPS bảo vệ và ngân sách giữ nguyên.", fontsize=10)
    fig.text(.02, .05, "Recall khi có đáp án tham chiếu; CI theo từng mục đích là phân tích phụ.", fontsize=8.8)
    fig.text(.02, .025, "MB = 10⁶ byte; chưa đo HTTP/TLS, độ trễ hoặc năng lượng thiết bị.", fontsize=8.8)
    OUT.mkdir(parents=True, exist_ok=True)
    pdf, png = OUT / "depth_utility_cost.pdf", OUT / "depth_utility_cost.png"
    fig.savefig(pdf, metadata={"CreationDate": None, "ModDate": None})
    fig.savefig(png, dpi=200)
    plt.close(fig)
    evidence = dict(schema="thesis-depth-figure-v1", rows=rows, cost=cost,
        source_sha256={str((BASE/p).relative_to(ROOT)): sha(BASE/p) for p in
            ("paired_readout.json", "recommended_configuration.json", "validation.json")},
        builder_sha256=sha(__file__), matplotlib_version=matplotlib.__version__,
        output_sha256={p.name:sha(p) for p in (pdf,png)},
        scope="Formatting certified retained statistics only; no rescoring or new uncertainty.")
    (OUT/"source.json").write_text(json.dumps(evidence, indent=2, ensure_ascii=False)+"\n")
    print("Created print-size thesis figure from certified statistics:",pdf.relative_to(ROOT))


if __name__ == "__main__":
    main()
