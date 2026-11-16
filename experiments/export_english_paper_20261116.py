"""Export English manuscript tables from authenticated, frozen research evidence.

No simulation, defense sampling, training, bootstrap rerun or benchmark mutation.
"""
from pathlib import Path
import argparse
import gzip
import hashlib
import json
import re
import sys

ROOT = Path(__file__).resolve().parents[1]
PAPER = ROOT / "paper"
AUTHORS = ("Hanh Ngo-Minh", "Phan Trong Nhan")
sys.path.insert(0, str(ROOT))
METHODS = ("raw", "geo_i_Z_k1", "transprotect_markov", "semantic_poi", "fake_queries", "geoi_bq")
LABELS = ("Raw GPS", "REM-Z (K1)", "TransProtect*", "Semantic*", "Fake queries*", r"\textbf{GeoI-BQ v1}")
PURPOSES = ("macro", "nearest_distance", "fastest_travel", "within_radius", "minimum_detour")
PLABELS = ("Macro", "Nearest", "Fastest", "Radius", "Detour")
SCENARIOS = ("S1", "S2", "S3", "S9", "S10")
B1 = ROOT / "artifacts/benchmarks/paper_comparison_20261028_v1"
B2 = ROOT / "artifacts/benchmarks/remaining_scenarios_20261109_v1"
B3 = ROOT / "artifacts/benchmarks/s8_fusion_20261109_v2"


def read(path):
    return json.loads(path.read_text())


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def n(value, digits=3):
    return "N/A" if value is None else f"{float(value):.{digits}f}"


def pct(value, digits=3):
    return n(100 * value, digits)


def ci(values, scale=100, digits=3):
    return "[" + ", ".join(n(v * scale, digits) for v in values) + "]"


def row(values):
    return " & ".join(map(str, values)) + r" \\" + "\n"


def esc(value):
    return str(value).replace("_", r"\_\allowbreak{}").replace("%", r"\%").replace("&", r"\&")


def table(caption, label, spec, headers, rows, note="", long=False):
    head = row([r"\textbf{" + h + "}" for h in headers])
    body = "".join(row(r) for r in rows)
    if long:
        return (r"\begingroup\footnotesize" + "\n" + r"\begin{longtable}{@{}" + spec + "@{}}\n"
                + r"\caption{" + caption + r"}\label{tab:" + label + r"}\\" + "\n"
                + r"\toprule" + head + r"\midrule\endfirsthead" + "\n"
                + r"\multicolumn{" + str(len(headers)) + r"}{l}{\textit{Continued from previous page}}\\"
                + "\n" + r"\toprule" + head + r"\midrule\endhead" + "\n"
                + r"\midrule\multicolumn{" + str(len(headers))
                + r"}{r}{\textit{Continued on next page}}\\\endfoot" + "\n"
                + r"\bottomrule\endlastfoot" + "\n" + body
                + r"\end{longtable}" + "\n" + note + "\n" + r"\endgroup" + "\n")
    return (r"\begin{table*}[t]\centering\footnotesize" + "\n"
            + r"\caption{" + caption + r"}\label{tab:" + label + "}\n"
            + r"\begin{tabularx}{\linewidth}{@{}" + spec + r"@{}}\toprule" + "\n"
            + head + r"\midrule" + "\n" + body + r"\bottomrule\end{tabularx}" + "\n"
            + (r"\par\smallskip\raggedright " + note + "\n" if note else "")
            + r"\end{table*}" + "\n")


def generate():
    from core.research_release import load_release
    spec = load_release()
    provenance = read(ROOT / "thesis/generated/sources.json")
    # Authenticate all inherited sources, rather than trusting printed thesis tables.
    for name, digest in provenance["sources_sha256"].items():
        assert sha(ROOT / name) == digest, name
    r, extra, fusion = (read(p / "results.json") for p in (B1, B2, B3))
    audit = read(B1 / "validation.json")
    assert audit["status"] == "pass" and audit["results_sha256"] == sha(B1 / "results.json")
    assert extra["full_coverage"] and fusion["full_coverage"]
    assert not extra["failures"] and not fusion["failures"]
    assert not r["proposed_model_changed"] and not r["proposed_randomness_regenerated"]
    a = {(t["method"], t["scenario"]): t for t in r["attackers"]}
    u = {(t["method"], t["cache"]): t for t in r["utility"]}
    comp = {t["control"]: t["metrics"] for t in r["comparisons"]}
    tasks = {t["task"]: t for t in extra["tasks"]}
    out = {}
    out["table_privacy.tex"] = table(
        r"Localization results: Hit100 (\%, lower) / MAE (m, higher). Same time windows.",
        "privacy", "P{2.7cm}YYYYY", ("Method",) + SCENARIOS,
        [[label] + [pct(a[m, s]["Hit100"]) + " / " + n(a[m, s]["MAE_m"], 1)
                    for s in SCENARIOS] for m, label in zip(METHODS, LABELS)],
        r"* Audited adaptations, not official author implementations. Each loss has its own selected attacker. Full CIs appear in the supplement.")
    out["table_utility.tex"] = table(
        r"Session-cache Recall@5 (\%, higher). The application returns a full sorted list.",
        "utility", "P{2.7cm}YYYYY", ("Method",) + PLABELS,
        [[label] + [pct(u[m, "session"]["metrics"][p]["mean"]) for p in PURPOSES]
         for m, label in zip(METHODS, LABELS)])
    out["table_cost.tex"] = table(
        "Mean wire cost per 600-second session; KB denotes 1,000 bytes.",
        "cost", "P{2.7cm}P{2.2cm}P{2.2cm}P{2.2cm}P{2.2cm}", ("Method", "Packets", "Coordinates", "Request KB", "Reply KB"),
        [[label, n(r["cost"][m]["packets"], 2), n(r["cost"][m]["requests"], 2),
          n(r["cost"][m]["request_bytes"] / 1000), n(r["cost"][m]["reply_bytes"] / 1000)]
         for m, label in zip(METHODS, LABELS)],
        r"Request/reply serialization is the replay contract; reply bytes encode POI indices, not full production records, TLS or network energy.",
        long=True)
    out["table_service.tex"] = table(
        r"Session Recall@5 (\%, higher) by purpose and acquisition cost per 600 s.",
        "service", "P{2.7cm}YYYYYYY",
        ("Method",) + PLABELS + ("Coordinates", "Reply KB"),
        [[label] + [pct(u[m, "session"]["metrics"][p]["mean"]) for p in PURPOSES]
         + [n(r["cost"][m]["requests"], 2), n(r["cost"][m]["reply_bytes"] / 1000)]
         for m, label in zip(METHODS, LABELS)],
        r"KB = 1,000 bytes of reply-index JSON; full records, transport overhead, latency and energy are excluded.")
    method_rows = [
        ["Raw GPS", "One real coordinate", "Unprotected control"],
        ["REM-Z (K1)", "One REM coordinate", r"$u=0.01\,\mathrm m^{-1}$; frozen references"],
        ["TransProtect*", "One replacement coordinate", r"Markov predictor; $K=10$, 8 TRAIN targets, $\alpha_{\rm TP}=10000$, $\varepsilon=0.005\,\mathrm m^{-1}$"],
        ["Semantic*", r"True point + 4 dummies", "OSM/empirical predictor; candidate pool 128; replaces AMap/LSTM"],
        ["Fake queries*", "K5 sets + inserted fake events", r"Road/history predictor; up to 2 fake events, waits 5--15 s, 24 candidates, similarity 0.75"],
        [r"\textbf{GeoI-BQ v1}", "K5 committed frames", r"$\Gamma_Q=1$, $C_s=0.23\,\mathrm m^{-1}$; protected read interval $\ge60$ s"],
    ]
    out["table_methods.tex"] = table(
        "Controls, adaptations and the frozen proposed method.", "methods", "P{2.7cm}P{3.6cm}P{7.3cm}",
        ("Method", "Public output contract", "Configuration / fidelity"), method_rows,
        r"The methods have different guarantees, output cardinalities and costs. We do not equate their epsilon values.",
        long=True)
    # English coverage descriptions are manually audited translations of the pinned literature matrix.
    lit_rows = [
        [r"TransProtect (2024)\cite{transprotect}", "Synthetic coordinate", "D: S1, S3; P: S2", "EIE / VehiTrack; travel-cost distortion", "Markov adaptation"],
        [r"Semantic (2026)\cite{semantic}", "Real + dummy set", "D: S1; P: S2, S3", "ASR; DER; generation time", "OSM/empirical adaptation"],
        [r"Fake queries (2026)\cite{fake}", "Mixed and fake-only sets", "D: S1, S3; P: S2", "ASR; indistinguishable paths; runtime/RSS", "Road/history adaptation"],
        [r"EV querying (2024)\cite{ev}", "Noisy/dummy queries", "P: S1--S3", r"Approx.\ Geo-I; IBU/Wasserstein; cost of privacy", "Survey only; Edge contract differs"],
        [r"CPCROK (2025)\cite{cpcrok}", "Pseudonyms / beacons", "P: S4", "Relinking; fake routes; communication", "Different VANET contract"],
        [r"DP-FETC (2025)\cite{dpfetc}", "Offline trajectory data", "P: S3, S8--S10", "DP; mutual information; JSD", "Different release contract"],
        [r"PTPPM (2025)\cite{ptppm}", "Road perturbation", "U: S1, S3", "Geo-I; distortion / QoS", "Preprint; no audited comparator"],
        [r"SoK (2024)\cite{sok}", "Synthesis survey", "Not a comparator", "Privacy units; reconstruction; utility", "Context and evaluation guidance"],
    ]
    out["table_related.tex"] = table(
        "Recent related work: contracts, task coverage and evidence status.", "related",
        "P{2.5cm}P{2.05cm}P{2.15cm}YP{2.35cm}",
        ("Work", "Output", "Coverage", "Privacy / utility metrics", "This study"), lit_rows,
        r"D: directly evaluated task under source assumptions; P: partial overlap; U: insufficient audited evidence for a stronger mapping. Unlisted scenarios have no direct coverage established here. D does not imply universal protection.")
    summary_rows = []
    for m, label in zip(METHODS, LABELS):
        s4 = tasks["S4--" + m]["metrics"]["balanced_accuracy"]
        s6 = tasks["S6--post_turn--" + m]["metrics"]["accuracy"]
        s8 = next(t for t in fusion["tasks"] if t["task"] == "S8--protected_partner--" + m)["metrics"]["Hit100"]
        summary_rows.append([label, pct(s4["mean"], 2), pct(s6["mean"], 2), pct(s8["mean"], 2)])
    out["table_scenario_summary.tex"] = table(
        r"Extended diagnostics (\%); lower inference is preferable, subject to task validity.",
        "scenario-summary", "P{2.7cm}YYY",
        ("Method", "S4 balanced acc.", "S6 post-turn acc.", "S8 protected-partner Hit100"), summary_rows,
        r"S4 is route-family association, not verified person/vehicle identity. S6 has two candidates (50\% chance). S8 is the post-V1 fusion diagnostic on the inspected cohort.")
    out["table_privacy_full.tex"] = table(
        r"All primary localization results and family-bootstrap 95\% CIs.",
        "privacy-full", "P{2.5cm}P{.6cm}P{1.25cm}P{2.7cm}P{1.6cm}P{3.1cm}",
        ("Method", "S", r"Hit100 \%", "95\\% CI", "MAE m", "95\\% CI"),
        [[label, s, pct(a[m, s]["Hit100"]), ci(a[m, s]["Hit100_CI95"]),
          n(a[m, s]["MAE_m"], 1), ci(a[m, s]["MAE_m_CI95"], 1, 1)]
         for m, label in zip(METHODS, LABELS) for s in SCENARIOS], long=True)
    out["table_attackers.tex"] = table(
        "Loss-specific attackers selected before the primary TEST readout.", "attackers",
        "P{2.5cm}P{.6cm}P{4.5cm}P{4.5cm}",
        ("Method", "S", "Hit100 attacker", "MAE attacker"),
        [[label, s, esc(a[m, s]["selected_attacker"]["hit100"]), esc(a[m, s]["selected_attacker"]["mae"])]
         for m, label in zip(METHODS, LABELS) for s in SCENARIOS], long=True)
    out["table_utility_full.tex"] = table(
        r"Session Recall and paired differences: GeoI-BQ minus each control (pp).",
        "utility-full", "P{2.5cm}P{1.4cm}P{1.2cm}P{2.6cm}P{1.6cm}P{2.7cm}",
        ("Control", "Purpose", r"Recall \%", "Absolute 95\\% CI", r"$\Delta$ pp", "Paired 95\\% CI"),
        [[label, plabel, pct(u[m, "session"]["metrics"][p]["mean"]),
          ci(u[m, "session"]["metrics"][p]["absolute95"]),
          pct(comp[m][p]["delta"]), ci(comp[m][p]["paired95"])]
         for m, label in zip(METHODS[:-1], LABELS[:-1]) for p, plabel in zip(PURPOSES, PLABELS)],
        long=True)
    out["table_bq_utility.tex"] = table(
        r"GeoI-BQ session Recall@5 and absolute 95\% CIs.", "bq-utility", "P{3cm}P{3cm}P{5cm}",
        ("Purpose", r"Recall \%", "95\\% CI"),
        [[plabel, pct(u["geoi_bq", "session"]["metrics"][p]["mean"]),
          ci(u["geoi_bq", "session"]["metrics"][p]["absolute95"])] for p, plabel in zip(PURPOSES, PLABELS)], long=True)
    out["table_cache.tex"] = table(
        r"Macro Recall@5 (\%) by cache scope on the same public outputs.", "cache",
        "P{3cm}P{2.5cm}P{2.5cm}P{2.5cm}", ("Method", "Current packet", "60 s epoch", "Session"),
        [[label] + [pct(u[m, c]["metrics"]["macro"]["mean"]) for c in ("current", "known", "session")]
         for m, label in zip(METHODS, LABELS)], long=True)
    out["table_native.tex"] = table(
        "Source-contract metrics: computed values and unavailable entries.", "native",
        "P{2.5cm}P{1.8cm}P{2cm}P{1.4cm}P{1.4cm}P{1.8cm}",
        ("Method", r"$\Delta c$ m", "Eligible draws", "ASR", "DER", "Path count"),
        [[label, n(r["native_metrics"][m]["Eq13"]["mean"], 2),
          r["native_metrics"][m]["Eq13"]["eligible"], "N/A", "N/A", "N/A"]
         for m, label in zip(METHODS, LABELS)], long=True)
    out["table_coverage.tex"] = table(
        "Primary execution coverage; no failures are treated as privacy success.", "coverage",
        "P{3cm}P{2.5cm}P{2.5cm}P{2.5cm}", ("Method", "Attempted", "Completed", "Failed"),
        [[label] + [r["coverage"][m][k] for k in ("attempted", "completed", "failed")]
         for m, label in zip(METHODS, LABELS)], long=True)
    # Extended tasks retain every method and every phase/view, including negative controls.
    out["table_s4.tex"] = table(
        r"S4 route-family association; balanced accuracy and grouped AUC, 95\% CIs.",
        "s4", "P{2.5cm}P{4.5cm}P{4.5cm}", ("Method", "Balanced accuracy \\%", "Grouped AUC"),
        [[label, pct(tasks["S4--" + m]["metrics"]["balanced_accuracy"]["mean"], 2) + "; " +
          ci(tasks["S4--" + m]["metrics"]["balanced_accuracy"]["ci95"], digits=2),
          n(tasks["S4--" + m]["metrics"]["group_mean_roc_auc"]["mean"]) + "; " +
          ci(tasks["S4--" + m]["metrics"]["group_mean_roc_auc"]["ci95"], 1)]
         for m, label in zip(METHODS, LABELS)], long=True)
    for scenario, phases in (("S5", ("before_fork", "turn_window")),
                             ("S6", ("before_fork", "turn_window", "post_turn"))):
        rows = []
        for phase in phases:
            for m, label in zip(METHODS, LABELS):
                t = tasks[scenario + "--" + phase + "--" + m]
                rows.append([label, esc(phase),
                             pct(t["metrics"]["accuracy"]["mean"], 2) + "; " +
                             ci(t["metrics"]["accuracy"]["ci95"], digits=2),
                             n(t["metrics"]["brier"]["mean"]),
                             n(t["metrics"]["log_loss"]["mean"])])
        out["table_" + scenario.lower() + ".tex"] = table(
            scenario + r" two-candidate inference: accuracy and 95\% CI, Brier score and log loss.",
            scenario.lower(), "P{2.5cm}P{2cm}P{4cm}P{1.2cm}P{1.4cm}",
            ("Method", "Phase", "Accuracy \\%; 95\\% CI", "Brier", "Log loss"), rows, long=True)
    out["table_s7.tex"] = table(
        "S7 fixed-wire interface audit under the common cover wrapper.", "s7",
        "P{2.5cm}P{2cm}P{2.2cm}P{2.2cm}P{2.2cm}",
        ("Method", "Equal wire \\%", "Purpose vuln.", "Category vuln.", "Joint vuln."),
        [[label, pct(tasks["S7--" + m]["equal_wire_rate"]),
          n(tasks["S7--" + m]["metrics"]["purpose_vulnerability"]["mean"]),
          n(tasks["S7--" + m]["metrics"]["category_vulnerability"]["mean"]),
          n(tasks["S7--" + m]["metrics"]["joint_vulnerability"]["mean"])]
         for m, label in zip(METHODS, LABELS)], long=True)
    f = {t["task"]: t for t in fusion["tasks"]}
    rows, selected = [], []
    for view in ("target_only", "protected_partner", "raw_partner", "unrelated_partner"):
        for m, label in zip(METHODS, LABELS):
            t = f["S8--" + view + "--" + m]
            metrics = t["metrics"]
            rows.append([label, esc(view), pct(metrics["Hit100"]["mean"], 2) + "; " +
                         ci(metrics["Hit100"]["ci95"], digits=2),
                         n(metrics["MAE_m"]["mean"], 1) + "; " + ci(metrics["MAE_m"]["ci95"], 1, 1)])
            selected.append([label, esc(view), esc(t["selected"]["hit100"]), esc(t["selected"]["mae"])])
    out["table_s8.tex"] = table(
        r"S8 V2 fusion diagnostic; Hit100 and MAE with block-bootstrap 95\% CIs.",
        "s8", "P{2.5cm}P{2.6cm}P{3.7cm}P{4.3cm}",
        ("Method", "View", r"Hit100 \%; CI", "MAE m; CI"), rows, long=True)
    out["table_s8_attackers.tex"] = table(
        "Loss-specific selected S8 V2 attackers.", "s8-attackers",
        "P{2.5cm}P{2.6cm}P{3.8cm}P{3.8cm}",
        ("Method", "View", "Hit100 attacker", "MAE attacker"), selected, long=True)
    # Fixed first TEST identifier, first trip; no score-based sample selection.
    trip = json.loads(gzip.decompress((B1 / "families/cache-029.json.gz").read_bytes()))["trips"][0]
    old_folder = ROOT / "artifacts/benchmarks/main_depth_20261014_v1"
    old_trip = json.loads(gzip.decompress((old_folder / "families/cache-029.json.gz").read_bytes()))["trips"][0]
    rows = []
    pair = lambda a: "(" + ", ".join(n(v, 4) for v in a) + ")"
    x0 = trip["truth"][0]["xy"]
    for i in (0, 1, 2, 3, 6, 9):
        pt, ledger = trip["truth"][i], old_trip["GeoI_ledger"][i]
        branch = {"fresh": "Refresh", "reuse": "Reuse", "public_clock_skip": "Skip"}[ledger["branch"]]
        rows.append([pt["t"], "(" + ", ".join(n(pt["xy"][k] - x0[k], 0) for k in (0, 1)) + ")",
                     branch + " / " + str(ledger["spent_units"]), pair(old_trip["GeoI_anchors"][i]),
                     pair(trip["events"]["geoi_bq"][i]["coordinates"][0]),
                     int(trip["masks"]["geoi_bq"][i]["session"], 16).bit_count()])
    out["table_sample.tex"] = table(
        "Fixed multi-step example: cache-029, first trip; only Q1 of each K5 frame is displayed.",
        "sample", "P{.6cm}P{1.5cm}P{1.8cm}P{3cm}P{3cm}P{1.1cm}",
        ("s", r"$\Delta$GPS m", "Branch / units", "Z (lat, lon)", "Q1 (lat, lon)", "Cache"),
        rows, long=True)
    old = read(old_folder / "results.json")
    ou = {(t["method"], t["cache"]): t for t in old["utility"]}
    ablations = (("baseline", "Service cover"), ("finite_static_g1", "Initial only"),
                 ("cache_explore_g1", "Local explore"), ("cache_target_g0", r"Public base $\Gamma_Q=0$"),
                 ("cache_target_g1", "GeoI-BQ v1"))
    out["table_ablation.tex"] = table(
        r"Preserved internal diagnostics (14 October): session Recall@5 (\%). Not the paper comparison.",
        "ablation", "P{3cm}P{1.6cm}P{1.6cm}P{1.6cm}P{1.6cm}P{1.6cm}",
        ("Variant",) + PLABELS,
        [[label] + [pct(ou[m, "session"]["metrics"][p]["mean"]) for p in PURPOSES]
         for m, label in ablations], long=True)
    bib = (ROOT / "thesis/generated/references.tex").read_text()
    for a0, b0 in ((" và cộng sự", " et al."), (" và ", " and "),
                   ("Nguồn tác giả", "Author version"), ("Bản tác giả:", "Author version:"),
                   ("Nguồn hướng dẫn trích dẫn:", "Citation guidance:"),
                   ("Truy cập 09/11/2026; nguồn địa lý được lưu theo snapshot trong repository.",
                    "Accessed 9 November 2026; geographic snapshots are versioned in the research repository.")):
        bib = bib.replace(a0, b0)
    bib += (r"\bibitem{iwpe26} L. Santos, M. Cunha and J. Vilela. Location Privacy Protection through Road "
            r"Network Adaptability and User Mobility Prediction. IWPE 2026 program abstract, 6 July 2026. "
            r"\url{https://www.iwpe.info/program.html}. Program abstract only; no full-paper reproduction claimed." + "\n")
    # Numeric references follow first citation order in the actual expanded article.
    entries = {re.search(r"\\bibitem\{([^}]+)\}", line)[1]: line
               for line in bib.splitlines() if line.strip()}
    def expand_for_citations(name):
        if not name.endswith(".tex"):
            name += ".tex"
        text = out[name] if name in out else (PAPER / name).read_text()
        return re.sub(r"\\input\{([^}]+)\}", lambda m: expand_for_citations(m[1]), text)
    citation_order = []
    for name in ("introduction", "related_work", "method", "analysis", "evaluation", "conclusion"):
        for group in re.findall(r"\\cite(?:\[[^]]*\])?\{([^}]+)\}", expand_for_citations(name)):
            for key in group.split(","):
                if key not in citation_order:
                    citation_order.append(key)
    assert set(citation_order) == set(entries), (citation_order, entries.keys())
    bib = "\n".join(entries[k] for k in citation_order) + "\n"
    out["references.tex"] = bib
    sources = dict(provenance["sources_sha256"])
    for folder in (B1, B2, B3, ROOT / "artifacts/analysis/thesis_arguments_20261116_v1"):
        for path in sorted(folder.glob("*.json")):
            sources[str(path.relative_to(ROOT))] = sha(path)
    sources["thesis/generated/sources.json"] = sha(ROOT / "thesis/generated/sources.json")
    figures = {"figure_tradeoff.pdf": ROOT / "thesis/figures/privacy_utility_cost.pdf",
               "figure_bound_profile.pdf": ROOT / "thesis/figures/privacy_bound_profile_20261116.pdf"}
    for name, path in figures.items():
        assert sha(path) == provenance["figures_sha256"][path.name], path
        sources[str(path.relative_to(ROOT))] = sha(path)
    manifest = dict(schema="english-paper-evidence-v1", date="2026-11-16",
                    method="GeoI-BQ v1", authors=list(AUTHORS),
                    author_information="Order confirmed by user; affiliations and corresponding-author details pending",
                    frozen_release_pins_verified=len(spec["files_sha256"]),
                    main_privacy_cells=len(a), extended_tasks=len(extra["tasks"]), fusion_tasks=len(fusion["tasks"]),
                    models_fitted=False, defense_outputs_regenerated=False,
                    new_empirical_runs=False, scope="Controlled comparison on the previously inspected synthetic cohort",
                    generated_tables=len(out) - 1, generated_sha256={
                        name: hashlib.sha256(content.encode()).hexdigest() for name, content in out.items()},
                    sources_sha256=sources, figure_sha256={name: sha(path) for name, path in figures.items()},
                    additional_literature=dict(key="iwpe26", url="https://www.iwpe.info/program.html",
                                              type="official program abstract", accessed="2026-11-16"))
    out["evidence_manifest.json"] = json.dumps(manifest, indent=2) + "\n"
    return out, figures


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--check", action="store_true")
    args = p.parse_args()
    out, figures = generate()
    for name, text in out.items():
        target = PAPER / name
        if args.check:
            assert target.read_text() == text, target
        else:
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(text)
    for name, source in figures.items():
        target = PAPER / name
        if args.check:
            assert target.read_bytes() == source.read_bytes(), target
        else:
            target.write_bytes(source.read_bytes())
    print(f"PASS: {len(out)-2} English tables, 18 references, 108 frozen pins; no new experiments")


if __name__ == "__main__":
    main()
