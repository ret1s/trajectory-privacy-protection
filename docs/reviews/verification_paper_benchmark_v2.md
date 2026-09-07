# BR-Dummy / thesis research cycle v2 — verification and next improvements

Date: 2026-09-07. Parent release: `1ece299bf5f41e76a8bcdee3e5b6dd35c3d7ba79`.
This file is an evidence snapshot for subsequent research/code agents. It does
not certify publication readiness, reproduced SOTA, or ten-scenario protection.

## Outcome

The thesis now includes an executable budgeted dummy-generation candidate,
justified metric consolidation, a stronger evaluation protocol, measured endpoint
tasks and explicit negative results. One canonical cover remains. The apparent
second title page was the default `report` chapter opening; chapter headings are
now compact. Duplicate continued-table entries in the list of tables and a nearly
empty trailing chapter page were also removed.

Canonical PDF: `artifacts/reports/graduation_thesis.pdf`, 49 pages.
SHA-256: `af8d85941fd66c63f9556c4b54eeb47815ee3ed8a3405e785c9c79a4b3719945`.

Result: `artifacts/benchmarks/paper_benchmark/results.json`.
SHA-256: `2df043bfb73c3b2f630046381daf5b4bc9f6941fb5be925b0967d6de98f27538`.
The old v1 artifact/source verification still passes unchanged.

## What changed scientifically

1. `benchmark/engines/budgeted.py`: BR-Dummy combines the existing REM and
   PrivateReuseSMREM primitives with a public H-event ledger, separate randomness
   for anchors/dummies, largest public strongly connected component and full
   reachable-set sampling. It does not intersect reachability with a small
   secret-centered radius. After H it ignores new private coordinates and only
   postprocesses the final protected anchor. A reset starts another budget.
2. `evaluation/research_protocol.py`: full-window path backtracking, full-window
   shadow features, and stationary candidate intersection. MAE and Hit100 select
   different adversaries when appropriate. Empty intersections fall back to the
   centroid, retaining the same validation denominator.
3. `data/paper_scenarios.py`: S9/S10 derive from completed SUMO trips, hide 60 s at
   the respective boundary and expose a 220 s/12-event window. Labels are the
   first/last recorded FCD point, not actual homes or user identities.
4. `experiments/run_paper_benchmark.py`: separate model, shadow, attack-selection,
   defense-selection and test users. Seeds 81–83 are new relative to v1's 71–73.
   Six BR parameter options are selected on defense validation, not test.
5. `thesis/metrics_consolidation.tex`: source-by-source rationale for keeping
   inference error, task utility and cost distinct. AnotherMe's exact original
   metric definitions remain unverified; no invented AUC or paper accuracy.
6. The read-only Flask replay now uses v2, exposes five cases and BR variants, and
   shows hidden endpoint truth only in evaluator mode.

## Data and denominators

- SUMO + pinned local OSM only; no GeoLife trajectory, schedule or parameter.
- Demand: 32 vehicles per seed. Qualified: 31, 31, 32. One rejected vehicle in
  each of 81/82, none in 83. Demand was increased from 24 only after eligibility
  failure (23 qualified), before any protection/test metrics; amendment recorded
  in `thesis/notes/paper_cycle_v2_protocol.md`.
- Per seed: test 4, shadow 6, attack validation 4, defense validation 4, model
  training 13/13/14. Five-way disjointness checked.
- Test: 12 distinct `(seed, vehicle)` trips; 60 correlated scenario windows;
  K=3/5; 12 reported configurations. These are not 60 independent people.
- 1,440 attempted rows: 1,343 completed; 96 N/A; one failure.
- Failure: AnotherMe, seed 81, `smoke_10/S3`, K=5: no distinct reachable virtual
  endpoint. It remains a failure, not silently dropped or replaced.
- N/A: AnotherMe S1/S2/S9/S10 under this complete-route adaptation contract.
- 11,316 released events; 67,896 category-query evaluations; 120 summary rows.
- Means weight each trip equally within scenario/method/K. SD is not a CI;
  vehicles within the same simulation can interact, and only three seeds exist.

## Verification executed

```bash
venv/bin/python -m pytest -o addopts='' -q tests
venv/bin/python -m experiments.verify_paper_benchmark --raw \
  --compare tmp/paper_benchmark_replay/results.json
venv/bin/python -m experiments.verify_report_demo
venv/bin/python -m experiments.export_paper_benchmark
venv/bin/python -m experiments.qa_report_demo_browser --port 5051
```

- **148 tests passed.** New tests cover stream/batch equality, prefix invariance,
  worst-case accounting, ignoring NaN private coordinates after exhaustion,
  increasing times, dead-end exclusion, intersection fallback, endpoint eligibility,
  conditional POI travel loss, selection constraints and public/private separation.
- **37 scientific source hashes verified.** Source mismatch fails closed.
- **506 FCD points and 24 hidden endpoint labels** matched independently to raw
  SUMO FCD and completed vehicle-route output.
- **67,896 POI results independently recomputed** from OSM, directed distances,
  union/dedup and client ranking, including response bytes and extra-distance loss.
- **10,704 BR directed transitions independently recomputed** on a separately
  constructed weighted directed graph; no trust in the stored 100% field.
- All 264 causal prefix tests passed. Six AnotherMe tests intentionally differ
  because the offline algorithm receives the whole segment (270 total checks).
- Full scientific replay identical: public coordinates, errors, service results,
  selections, labels, causal checks and summaries. Excluded fields are clocks,
  runtime/provenance XML headers and source-hash metadata; outcome fields remain.
- Flask/Chromium: initial public-only view, evaluator opt-in, 120 summary rows,
  BR playback, endpoint display, N/A disabled playback, DLS, hiding evaluator and
  390 px layout. Zero page errors and zero external requests. The in-app browser
  was unavailable; standalone local Chromium was used for this test.
- PDF: XeLaTeX/latexmk; no undefined references, missing glyphs or overfull boxes.
  Rendered all pages for layout inspection and rechecked changed algorithm/result/
  conclusion pages at readable resolution. Only bibliography underfull boxes remain.
  Reviewed PDF copied to the canonical artifact, not an unreviewed editor output.
- `git diff --check` passed.

## Findings that must not disappear in a later rewrite

**The joint utility target FAILED in all six seed/K searches.** No private-reuse
grid option achieved minimum scenario Recall@5 >= .90 on defense validation.
`br_selected` is the predefined fallback maximizing minimum-case recall; it is
not a feasible optimum or a proven best algorithm.

At S2/K3, fixed private reuse has Hit100 16.7% versus 33.3% for fresh BR; Recall is
94.6% versus 96.1%. This is a finite-sample signal, not a significance test or a
proof that reuse alone caused the difference. Private reuse halves release epsilon
to pay for the test, so fresh-anchor noise distributions also differ.

At S3/K3, fresh BR improves Recall from the old road variant's 81.0% to 86.5%, with
100% graph-node validity in both. Multiple components changed; do not attribute
all gain to SCC filtering or reachable-set selection alone. The selected private
variant's Recall is only 81.7%; it is not uniformly better than fixed variants.

The unprotected S3/K3 baseline has only 48.3% validity under nearest-node time
checks. SUMO is not thereby invalid: nearest-junction projection discards along-edge
progress. This representation mismatch can create dummy lag and confound the
privacy/utility tradeoff. Fix it before claiming realistic road-level dominance.

Endpoint results include difficulty caused by masking itself. Even the unprotected
baseline sees only the retained window, not the hidden endpoint. Zero hits among
12 endpoints is not zero attack probability in a population. Our endpoint decoder
is not a reproduction of the Dhondt et al. metadata-based EPZ attack.

## Literature review and resulting criticism

- [Predictive Geo-I, PETS 2014](https://petsymposium.org/2014/papers/Chatzikokolakis.pdf)
  already studies private prediction tests and budget management. Cite the inherited
  primitive; do not sell reuse or budgeting alone as new. BR's fixed worst-case
  ledger is conservative and does not refund based on a secret branch.
- [Shokri et al., IEEE S&P 2011](https://www.ieee-security.org/TC/SP2011/PAPERS/2011/paper016.pdf)
  supports explicit adversary/error evaluation. Our point-estimate MAE is not an
  exact posterior expected inference error; entropy is not a substitute.
- [TransProtect](https://arxiv.org/abs/2409.09495) motivates inference and
  destination-related travel utility. Our POI extra-distance metric is not its
  original equation (13); both must retain their operational definitions.
- [Semantic correlation](https://link.springer.com/article/10.1007/s44443-026-00899-w)
  motivates examining temporal/semantic consistency. ASR requires a success rule;
  generating K points is not posterior success. DER descriptions need reconciliation.
  This is a journal article, not a verified “Q1 conference” claim.
- [AnotherMe official code](https://github.com/fang-zhiyou/AnotherMe) supports the
  VTGA and detector-family audit, but does not establish every full-paper metric
  or recreate its training/evaluation split. Full-source access is a remaining gap.
- [Dhondt et al., ACM CCS 2022](https://doi.org/10.1145/3548606.3560616)
  attacks endpoint privacy using metadata, roads and zone-entry information.
  Our time-mask/window task omits parts of that observation, so coverage must stay
  labelled controlled endpoint inference, not that complete real-world attack.

## Prioritized next cycle and acceptance criteria

1. **Representation before tuning:** keep longitudinal position on SUMO edges or
   lanes, honor legal turns, and verify raw truth is feasible under the same model
   used for dummy validation. Do not “repair” truth to make a metric pass.
2. **Isolate components:** no-reuse anchors with the same release epsilon as private
   reuse; controlled-offset/SCC/reachability ablations; independent random trials.
   Compare at matched utility and payload as well as matched declared privacy caps.
3. **Stronger threat-matched attacks:** a calibrated mechanism likelihood or stronger
   learned decoder; full retained trajectories and EPZ-relevant metadata for endpoints;
   report prior-only and masked-unprotected references under the same observations.
4. **Utility feasibility:** sweep budget, offsets, prediction threshold and public
   horizons on development/validation only. Retain negative feasibility records.
   Seeds 81–83 become development if used to choose the next design; confirm on fresh
   seeds and another city. Do not repeatedly tune to this held-out table.
5. **Long sessions and deployment:** rolling-window/privacy-filter design, bounded
   caches, cold/warm costs, device measurements, network overhead and actual query
   mix. Current p95 is a mean of per-trip p95s, not a pooled/device/network p95.
6. **Publication claims:** acquire missing original comparator definitions/assets,
   reproduce trained components and quantify uncertainty by simulation/person units.
   State novelty only after an explicit closest-work comparison. Add S4–S8 only
   with their own labels, observations and target-specific attacks, not inferred
   from location MAE or the number of scenario definitions.

Ideal bound: BR with B=.24 and H=12 composes to at most B under D-infinity on
represented inputs. At radius 100 m, exp(24) is a loose bound. Do not turn this
into 1/K success, GPS-continuous Geo-I, endpoint/home privacy or machine-arithmetic
certification. Real-arithmetic assumptions and public schedule/context matter.
