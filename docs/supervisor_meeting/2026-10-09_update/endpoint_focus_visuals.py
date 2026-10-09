"""Render historical endpoint evidence without running a defense or attacker.

The input is the source-pinned endpoint focus audit. The caller supplies the
existing scientific SVG helpers and owns the slide title, footer and export.
This module does not reinterpret the historical delay arm as current L30.
"""

import math

from focus_visuals import table


def _number(value, digits=0):
    value = float(value)
    if not math.isfinite(value):
        raise ValueError("Finite saved evidence required")
    return f"{value:,.{digits}f}".replace(",", " ").replace(".", ",")


def _time(value):
    value = float(value)
    if not math.isfinite(value) or value < 0:
        raise ValueError("Finite nonnegative saved time required")
    return _number(value, 0 if value.is_integer() else 1)


def historical_delay_timeline(audit, *, text, line, ink, teal, blue, gray,
                              orange, **helpers):
    """Show seven saved transitions of the u701_00 explanatory delay run.

    Q(t) denotes the protected query set generated at source time t. The
    publication column uses the actual release clock, not the source clock.
    Both clocks are evaluator-side explanatory annotations of a saved run.
    """
    saved = audit["historical_walkthrough"]
    sample = saved["samples"]["boundary"]
    close = float(saved["close_s"])
    policy = sample["policy"]
    if (saved["session_id"] != "u701_00" or close != 384
            or policy != {"head_s": 60.0, "delay_s": 60.0}):
        raise ValueError("This layout requires the declared historical case")

    by_time = {float(row["t"]): row for row in sample["trigger_rows"]}
    times = (0, 60, 80, 120, 360, 380, 384)
    if not all(t in by_time for t in times):
        raise ValueError("Historical trigger rows are missing")
    if sample["head_skipped_source_times_s"] != [0.0, 20.0, 40.0]:
        raise ValueError("Head suppression differs from the saved example")
    cancelled = sample["cancelled_source_times_s"]
    if cancelled != [340.0, 360.0, 380.0, 384.0]:
        raise ValueError("Close cancellation differs from the saved example")

    rows = []
    for t in times:
        row = by_time[t]
        label = "0, 20, 40" if t == 0 else _time(t)
        if row["head_skipped"]:
            gps = "Bỏ đầu, chưa gọi Geo-I"
            queue = "Chưa sinh Q"
        else:
            if row["private_read"]:
                if row["branch"] == "fresh":
                    gps = "Đọc GPS, tạo Z mới"
                elif row["branch"] == "reuse":
                    gps = "Đọc GPS, giữ Z cũ"
                else:
                    raise ValueError("Unexpected private-read branch")
            else:
                gps = "Không đọc GPS mới"
            queue = "Giữ " + ("thêm " if t != 60 else "") + f"Q({_time(t)})"
            if t == close:
                queue = f"Thêm Q({_time(t)}), đóng phiên"
        released = row["released_source_times"]
        if len(released) > 1:
            raise ValueError("Saved example requires a single release per tick")
        if released:
            published = f"Q({_time(released[0])}), lúc {_time(t)} s"
        else:
            published = "Chưa gửi" if t != close else "Hủy Q chưa gửi"
        rows.append((label, gps, queue, published))

    body = text(56, 150,
                "u701_00: bỏ đầu 60 s, giữ Q 60 s, đóng phiên tại 384 s",
                27, color=blue)
    rendered, _ = table(
        ["t (s)", "GPS cho Geo-I", "Hàng đợi tại thiết bị", "Gửi lên máy chủ"],
        rows, [115, 315, 340, 398], text=text, line=line,
        ink=ink, gray=gray, teal=teal, y=206, row_h=47,
        size=25, header_size=25,
        colors=[gray, ink, gray, ink, ink, gray, orange],
    )
    body += rendered
    body += text(56, 609,
                 "Khi đóng: hủy Q(340), Q(360), Q(380), Q(384). Không gửi bù phần cuối.",
                 25, color=orange, weight=700)
    body += text(56, 643,
                 "Q(60) được tạo tại 60 s, nhưng bản tin công bố mang thời điểm 120 s.",
                 24, color=gray)
    return body


def historical_delay_benchmark(audit, *, text, line, ink, teal, blue, gray,
                               orange, **helpers):
    """Render every scored arm of the four-family round-two readout.

    MAE and Hit are readouts of separately selected finite-bank attackers.
    Delay L10 and Endpoint20 L20 differ in both reply depth and privacy budget.
    JSON byte cost uses the saved per-input, family-weighted readout.
    """
    study = audit["historical_matched_study"]
    if (study["families"] != 4 or study["executions_per_method"] != 16
            or study["input_events_per_method"] != 170):
        raise ValueError("This layout requires the declared round-two cohort")
    records = {row["method"]: row for row in study["rows"]}
    methods = (
        ("raw", "GPS thật", blue),
        ("scale100_L10", "GeoI-Slack L10", ink),
        ("scale100_L20", "GeoI-Slack L20", ink),
        ("delay60_L10", "Delay 60 s, L10", orange),
        ("scale025_L20", "Endpoint20, L20", teal),
    )
    if set(records) != {name for name, _, _ in methods}:
        raise ValueError("All five historical arms must be retained")
    rows, colors = [], []
    for method, label, color in methods:
        record = records[method]
        for scenario in ("S9", "S10"):
            score = record[scenario]
            expected_hit = 1.0 if method == "raw" else 0.0
            if (score["families"] != 4 or score["observations"] != 16
                    or score["hit100"] != expected_hit):
                raise ValueError("Historical denominator or Hit100 changed")
        if record["input_events"] != 170:
            raise ValueError("The full input denominator must remain 170")
        rows.append((
            label,
            _number(record["S9"]["mae_m"], 1),
            _number(record["S10"]["mae_m"], 1),
            _number(100 * record["recall"], 2) + "%",
            f'{record["released_events"]}/{record["input_events"]}',
            _time(record["mean_publication_delay_s"]) + " s",
            _number(record["bytes_per_input"]),
        ))
        colors.append(color)

    body = text(56, 150,
                "4 nhóm tuyến, 16 lần chạy mỗi phương pháp, 170 mốc đầu vào",
                27, color=blue)
    rendered, _ = table(
        ["Phương pháp", "S9 MAE\n(m)", "S10 MAE\n(m)", "Recall@5\n(%)",
         "Gửi /\nđầu vào", "Delay\n(s)", "JSON byte\n/mốc"],
        rows, [280, 165, 165, 150, 155, 100, 153],
        text=text, line=line, ink=ink, gray=gray, teal=teal,
        y=206, row_h=57, size=25, header_size=24, colors=colors,
    )
    body += rendered
    body += text(56, 605,
                 "Hit100 ở S9/S10: GPS thật 100%, các bản Geo-I 0% trước bank hữu hạn đã chọn.",
                 25, color=gray)
    body += text(56, 642,
                 "Delay: L10, C = 0,23/m. Endpoint20: L20, C = 0,0575/m. Chi phí khác nhau.",
                 24, color=orange)
    return body
