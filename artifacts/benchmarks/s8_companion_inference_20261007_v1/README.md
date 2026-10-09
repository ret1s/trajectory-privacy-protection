# Historical frozen-tape S8 companion-inference diagnostic

Protocol, source snapshots and independent checker were declared before fitting or scoring. Status: **scored; independent verification PASS**. Existing datasets, protected Q and original evidence are unchanged.

## Actual data and the narrow task

The repository already contains synthetic simultaneous SUMO pairs: `urban_scenarios_v2` has 18 S8 records/6 families, `urban_fresh_v2` 33/12, `research_loop_expanded_v1` 34/12, and `endpoint_holdout_expanded_v1` 89/32. They distinguish partial companionship, full-route companionship and incidental proximity. These are planned relationship labels plus measured FCD proximity, not real social relationships. Original SUMO network cache files are absent. The new 60-family native Q cohort contains sequential history/query trips, no actual S8 pair; pairing those trips as if simultaneous would be invalid.

This diagnostic reuses the frozen historical `identity_future_20261005` GeoI-Slack+Raw tapes on the **actual expanded** base/partial/full-route pairs. Only 22 realized S8.A/B records have both required protected tapes; there is no protected incidental-role tape, so none is fabricated. The split remains 6/3/3 route families, with 12/5/5 pair records and 100/44/41 pair-events. Those represent **50/27/24 unique target events** and 6/3/3 target sessions, respectively: one base trajectory can occur in both A/B. Cases are averaged inside each family and all related windows stay together. There is one retained protected draw, not independent repetitions. Deduplicated counts are descriptive and do not replace the frozen equal-case/family weighting.

The target is current true GPS at each saved target emission, every60s. First-visible offsets preserve the common pair clock. The associated partner account and those offsets are an **explicit observer-information grant**. A relationship tag does not grant exact current co-location: the observer never receives `proximity_pairs`, actual distance, route, identity labels, future points or separation time as features. Partner prefixes stop at the same public cut; a partner observation3s after the cut is not pulled back by nearest-time alignment. Missing partner data remain explicit.

| View | Attacker observations |
|---|---|
| Target-only | Causal protected target prefix |
| Joint protected partner | Above plus causal protected associated-partner prefix |
| Joint public Raw partner | Above plus explicitly granted saved Raw partner prefix at the SAME60s clock |
| Joint unrelated protected partner | Replace partner by next-family same-role protected tape within the SAME split, fixed before scoring |
| Raw target positive control | Saved causal one-coordinate Raw target prefix |

The unrelated control is a synthetic negative pairing, not an observed social relation. Raw auxiliary access is declared as an observer limit/control; no additional evaluator GPS is silently supplied. This measures localization sensitivity, not relationship classification or complete S8.

## Frozen mechanism and inferential limits

The producer is historical GeoI-Slack on a reconstructed public map, K5, nominal B=.24, H12, effective cap≤.23/session, private reads60s, theta200m and slack.03. It is **not** current GeoI-Epoch8-H12 or the L30 workload. No new GPS, sampler, protection call or network reconstruction is needed. Historical synthetic streams were reproducible from session IDs; the finite observer bank does not exploit seed inversion. These tapes cannot establish protection against an observer with the private-generator state or publication-grade unpredictable randomness.

The bank retains ExtraTrees96/leaf2/depth16, standardized weighted kNN1/5/15, train prior and public last/velocity geometry. Joint banks may ignore the partner. Fits use train only; one attacker per view minimizes equal-case/equal-family MAE on selection. `attacker_selection.json` is written before test prediction/scoring. Every finite-bank test candidate is retained descriptively; no winner is selected using test. Absolute MAE/Hit100 and the change from target-only are reported, with family values. Three test families and correlated windows do not justify a tick-level confidence interval or population claim.

Geo-I bounds how nearby secrets alter the mechanism's output law; it does not guarantee a large absolute error when side information already localizes the person. For ideal independent protected reports of a shared co-located position, costs compose; hiding one person's output cannot remove information independently supplied by an unprotected companion. This is a conditional inference/theory distinction, not a finding from this diagnostic. See [Andrés et al., sections3.2–3.3](https://arxiv.org/html/1212.1984v3), and [Olteanu et al., sections2 and6](https://www.mhumbert.com/publications/tmc16.pdf). Their co-location attack motivates the joint-view contrast; this runner is not a faithful reproduction of their Bayesian-network attack.

## Actual bounded result

| Observer view | Selection-chosen attacker | Family-macro MAE (m) | Family-macro Hit100 |
|---|---|---:|---:|
| Protected target only | Latest public Q-set centroid | 664.27 | 3.70% |
| Target + protected associated partner | Latest public Q-set centroid | 664.27 | 3.70% |
| Target + explicitly public Raw partner | Causal partner-prefix mean-velocity extrapolation | 594.33 | 1.85% |
| Target + unrelated protected partner | Latest public Q-set centroid | 664.27 | 3.70% |
| Raw target positive control | Latest single-coordinate public Raw point | 0.00 | 100.00% |

The joint protected bank chose to ignore the partner. This is **no observed incremental localization gain for this selected finite bank**, not evidence of independence, complete S8 protection or group privacy. Learned attacks generalize poorly across the three test route families; a stronger joint mobility/REM decoder or more training families may change the result. Historical public/session-derived sampler streams remain a material limitation: no observer that reconstructs those streams was tested, so the table is a statistical-attacker diagnostic only.

Explicit Raw auxiliary access reduces mean error by 69.94m, but Hit100 falls; the improvement is not uniform across metrics/families. For the selected SAME attacker, exploratory case-family means are partial-route 716.43m (3 families/24 pair-events) and full-route 370.91m (2 families/17 pair-events), versus target-only 664.27m and 669.44m respectively. Those subgroup counts differ and are not pooled as independent samples. Do not replace the selected Raw-partner velocity model by the test-best Raw-partner last point (570.43m): that candidate remains descriptive in the full saved bank.

Joint views lack an observed partner prefix at five initial pair-events; they keep those events and use the explicit missing-data branch. No future partner sample fills the gap. [pre_scoring_execution.json](pre_scoring_execution.json) retains scores-absent/selection-absent flags and deduplicated event counts; [attacker_selection.json](attacker_selection.json) was written before [predictions.json](predictions.json) and [readout.json](readout.json).

[validation.json](validation.json) independently reproduces the source clock/tape ancestry, causal feature vectors, train-only fits, selection-only choices, 205 selected prediction rows, and equal-case/family denominators. Protocol SHA256 is `338477cac3898e480d462c486a77ac5f8dfb4bc3ce3fbdcd632d994bd8782c78`; readout SHA256 is `295f8a850754ce9120f07e5123dd2303b908c30f62d4f1436a798d239f3a76c9`. No original protected output, GPS, label, source or protocol was edited after the first score.

## Reproduce

```sh
python -m pytest -q tests/test_s8_companion_inference.py
python -m experiments.s8_companion_inference_20261007 contract
python -m experiments.s8_companion_inference_20261007 run
python -m experiments.verify_s8_companion_inference_20261007
```

Declaration/source paths and selection/results are write-once. The separate verifier does not import the S8 production runner/scorer: it independently rebuilds causal prefixes, projected features, clock/pair ancestry, family-case arithmetic, train-only fits, selection and saved predictions. Both use the pinned historical regression library. The verifier declaration is [verification_protocol.json](verification_protocol.json); the task design/source closure is [protocol.json](protocol.json), with readable counts in [protocol_preview.json](protocol_preview.json).
