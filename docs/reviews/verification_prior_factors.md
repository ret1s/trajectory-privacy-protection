# Verification: prior factors and loss-aware shadow inference

Date: 2026-09-08. Parent: `50d15250603070e9f2f861a40eba2d5df52b5fea`.

## Assessment: share with caveats, not submission-ready

This cycle separates the explanation of a prior result, implements four new
protection variants and expands the attack bank. It does NOT establish a new
independent winner, a stronger Geo-I theorem or faithful SOTA superiority.

The locked rule still selects **uniform U/U**, with `utility_feasible=false`.
Fixed-mixture M/M is a promising exploratory candidate, not a post-hoc replacement.
The main negative finding is the limited auxiliary support of all shadow banks:
360 observations correspond to only 36 distinct XY labels from two families.

## Protocol, inputs and trust boundary

- Declaration: `thesis/notes/prior_factor_protocol.md`, written before new
  generation/attack scores. No post-result changes to the declaration or frozen
  generation/attack sources.
- SQLite release `urban-scenarios-v3`, semantic SHA:
  `213886fc2722bbecf5e55f61e8978d2c2842bb6018b7f65f5021a261bd79358d`.
- DB and update log unchanged; no new dataset release is justified by changing
  protection algorithms. No GeoLife inputs.
- Training families 101/102; validation 103/104; reused development 201--204.
  The historical immutable `confirmation` split is NOT fresh confirmation here.
- Nine S1--S3 cases, K=5, at most 12 allowed events, three paired RNG draws,
  B=.24/m, theta=200 m. No scenario exclusion based on scores.
- Full trajectories of new variants use their own dummy histories. Paired
  private anchors, composition ledger and directed free-flow lane constraints
  are unchanged. No raw-GPS truncation, future access or corridor was added.
- Original BR-lane, L/L and U/U generation fields remain exact. New artifacts
  extend old attack errors, while old files are never overwritten.

## Method change and interpretation

`benchmark/prior_factors.py` provides a factorized anchor model: one model owns
the initial prior/emission/context, another owns prediction and long-gap reset.
Both must share grid, public map/context and emission normalizers.

L = learned prior; U = uniform mass per occupied public cell; C = learned lane
density divided by cell state count and normalized; M = .5*C + .5*U. Four new
full-run combinations are L/U, U/L, C/C and M/M.

Changing initial prior also changes the fixed tie-break center and initial
dummy configuration. Its later effects propagate through reachable sets.
Changing motion includes both local transition weighting and long-gap reset.
Do not describe this as initial-prior-only causality or an independently
estimated traffic model. Balancing C does not remove all discretization effects;
the anchor output catalogue/kernel remains unchanged.

## Results: reused development only

| Configuration | Recall | Worst-case mean | Cafe/restaurant mean | Selected Hit100 | Expanded Hit envelope |
|---|---:|---:|---:|---:|---:|
| BR-lane | 87.48% | 84.01% | 65.48% | 8.46% | 16.81% |
| L/L | 84.12% | 78.89% | 55.36% | 0.00% | 5.06% |
| U/U | 92.35% | 86.01% | 78.00% | 2.78% | 14.30% |
| L/U | 84.06% | 78.89% | 55.17% | 0.00% | 5.20% |
| U/L | 92.34% | 86.49% | 78.28% | 2.78% | 14.14% |
| C/C | 85.14% | 79.72% | 57.83% | 1.00% | 6.42% |
| M/M | 93.04% | 88.61% | 80.15% | 2.08% | 8.72% |

- Initialization L->U: +8.22 pp under learned motion; +8.29 pp under uniform
  motion. Motion L->U: -0.058 pp under learned initialization, +0.006 pp under
  uniform initialization. Conditional contrasts, not universal effect estimates.
- C/C gains only +1.02 pp versus L/L; density balancing alone does not recover
  U/U's utility. No claim that lane density has no impact under other grids.
- M/M gains +0.70 pp over U/U on average. Per-family differences are +2.78,
  +0.34, 0.00 and -0.33 pp: NOT four positive independent replications.
- M/M S3.A/S3.C = 88.61%/89.55%, still below the 90% requirement. Its validation
  minimum =88.09%, versus selected U/U =88.13%; selection remains unchanged.
- All methods: 4,017 evaluable category-events and 69 empty references kept N/A.
  Every new cover has five distinct coordinates throughout these runs; this is
  not a k-anonymity guarantee. Six service categories are counterfactual tasks,
  not a measured production query distribution.
- M/M's selected MAE is 1,082.9 m versus U/U 1,283.6 m. Lower attack Hit alone
  must not be framed as dominance: smaller error favors the attacker.

## Expanded attacker: implementation and caveats

For neighbor counts 15 and 45, add a minimum-Euclidean-risk action among neighbor
coordinates plus mean, and maximum 100 m disk mass among those actions plus
circle centers through feasible pairs. Add mean45 as a control. The mean is
not generally optimal for either objective; this follows standard decision
theory, not a novel attack theorem.

All predictions use only training data and public-prefix features. Case labels
are used to select/report attacks in the evaluator, not supplied to the new
predictors. Existing full-window attacks remain in the bank. Attacker guesses
can be off-road and outside the dummy set. The disk-construction tolerance
1e-7 m is not applied to the strict raw-GPS Hit<=100 benchmark threshold.

The empirical neighbor posterior is not calibrated and is not the mechanism's
actual likelihood. MAE is optimized on a finite action set, not the entire
plane. No proof of global optimal attack is made.

On development, expanded-bank maximum Hit is unchanged for all seven models.
C/C's selected Hit changes from .93% to 1.00%; other selected Hit macro values
stay the same. L/L's selected MAE *worsens* from 1,349.5 m to 1,467.2 m after
validation selects a different attack. This is selection transfer failure, not
stronger defense: underlying outputs are identical. An expanded bank is not
empirical evidence that its new members are stronger on new routes.

## Literature review and publication critique

See `docs/research/prior_factor_literature_review.md` for primary links and
explicit access limits. Read PoPETs 2017 on remapping/prior mismatch/mean versus
Euclidean loss, and PoPETs 2015 Privacy Games on objective-aware optimal attacks.
Screened the 2025 PIVE critique and August 2026 LAHEC paper; the former's proof
and the latter's full PDF were not accessible. Their abstracts are not treated
as reproduced guarantees or comparison results.

High-priority publication gaps:

1. **Auxiliary-data support and likelihood.** Generate more independent SUMO
   shadow routes and diverse allowed query windows. Diagnose coverage/distance
   to training labels before a fresh confirmation. Do not merely add RNG draws
   or tune neighbor counts on the development set until they look favorable.
2. **Formal privacy.** Ideal exp(24) bound at 100 m is unchanged and loose.
   Keep absolute inference risk separate; audit floating-point and adjacency
   assumptions before advertising a formal implementable guarantee.
3. **Utility/generalization.** Two difficult cases still miss the target; use
   held-out maps/longer sessions/budget curves after locking the next candidate.
4. **Novelty and comparators.** Prior mixing, POI context and loss-aware Bayes
   decisions already have precedents. Faithful comparators, output/trust budget
   alignment and a precise contribution are still required. No new SOTA or
   S4--S10 protection results were produced in this cycle.
5. **Statistics/reproducibility.** Independent families, not steps/RNG replicas,
   are the sampling units. Four reused families cannot support a fresh-test
   significance claim. Some pinned raw/cache inputs remain external to Git;
   an arbitrary OSM download is not a reproducible substitute.

## Verification and artifact QA

- Unit/regression suite: **250 passed in 20.57 s** on the final rerun
  (the first full run also passed all 250 tests).
- Independent geometric decision enumeration matches 200 fixed-seed synthetic
  supports (sizes 1/2/15/45); max numeric difference 1.14e-13, zero mismatches.
- Independent full verifier: **PASS** on all 1,512 rows / 9,765 events, including
  1,134 scored validation/development rows and 648 exactly reused control rows.
- Recomputed 43,470 category queries, 14,490 loss-aware decisions and 5,580
  covered-POI objective evaluations. Checked 41,265 directed transitions.
- Replayed 288 new-method runs (replicate 1, every phase/record/configuration),
  with 288 prefix comparisons; 192 prefixes are strictly shorter than the full
  run. Other replicates have score/constraint checks, not full generation replay.
- Checked 432 first-event comparisons: changing only motion leaves the first
  output unchanged. Reconstructed all seven shadow banks and their labels.
- Independent export audit recomputed development aggregation from 756 rows,
  per-category scores, validation-selected attacks for both old/new banks,
  paired family differences, training-support counts and both exact LaTeX tables.
  All frozen generation sources and artifact hashes still match their manifests.
- SQLite file SHA remains
  `fb28dc526fe7825ac025ff076470d3da3ada32cbb06c15791b28490423081499`;
  the dataset and its update history are unchanged.

### Reproduction and handoff

```bash
venv/bin/python -m pytest --override-ini addopts='' -q
venv/bin/python -m experiments.verify_prior_factors
venv/bin/python -m experiments.export_prior_factors
```

Generation commands and the pinned-input prerequisite are in
`artifacts/benchmarks/prior_factors/README.md`. Do not regenerate over frozen
evidence or rename reused development as a fresh test split. Exporting only
recreates readout/tables; it does not rerun the scientific stages.

Independent receipt: `artifacts/benchmarks/prior_factors/verification.json`, SHA
`0e3d921847033bbe062f7d6e3b0240f0c60bdba26bb169e2b09f4fa69cd459d0`.

| Stage | SHA-256 |
|---|---|
| Training | `d3ec90d22ccd0c2aafcc365355864c354e2f17eb6443929c985f95d0eb2f6fe7` |
| Validation | `5a76f89c651bd9387bc29eb301faae71dd3c0ee839ab26e779c2eb46d1f4b68c` |
| Selection | `f41f47fd483bfc08e04b31fe3f9a8319aeee61feadcba068fa1ea6969fa2e6ba` |
| Development | `55a11dd9b7d3d0fd860985e90ea27d29be6ff32295cfef1d0fb8068df11b6e4b` |

Thesis reading path: Section 5.11 (method factors), Section 6.11 / Tables
6.20--6.21 (results and attack limitations), Section 7.2 (publication gates).
The abstract distinguishes the historical confirmation from its later reuse.
Only fully read primary papers support detailed methodological claims; the
2025/2026 screening and access limits remain in the linked literature review.

### PDF release QA

- Canonical release: `artifacts/reports/graduation_thesis.pdf`, **88 A4 pages**,
  441,228 bytes; SHA-256
  `ca01a5fda9fd199cca699004a4b9c84e6bafc657b41edda8a9cfda8bfa7f23cc`.
- Built using XeLaTeX/latexmk. No overfull boxes, undefined references, LaTeX
  warnings, empty pages, out-of-page word boxes or duplicated title pages.
- Rendered every page, inspected all contact sheets and enlarged the new
  mathematical definitions / both comparison tables. Tightened only the TOC's
  paragraph spacing to avoid a fourth contents page containing just three lines.
- Final wording explicitly excludes coincident point pairs in the disk-center
  construction and describes long-gap prediction as gradual prior mixing.
  Re-rendering showed only physical page 81 changed after the last wording edit;
  that page was re-inspected at 120 dpi. All other rendered pages were identical.
- The release PDF is byte-identical to the checked `build/thesis/main.pdf`.
  Generated tables are exact-value comparisons, not a cross-contract SOTA
  leaderboard; no chart or claim of statistical significance was added.
