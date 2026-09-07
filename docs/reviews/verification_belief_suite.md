# Verification: belief-weighted BR-lane + SUMO scenario suite v2

Date: 2026-09-07. Starting main commit:
`8b530de6c944b590be3aaaaf3379181029409563`.

## Verdict and handoff

**Verified research extension; not a promoted default, all-scenario protection
claim, or publication-ready SOTA result.** The new dataset fills the two
previously empty subcases and doubles the number of related simulation families.
The proposed method now uses an approximate distribution over locations derived
from protected-anchor history to score expected POI coverage. A separate
validation/confirmation run exposes both improvements and counterexamples.

For K5, the validation-selected `belief24` reduces confirmation macro Hit100
23.15% →18.70% and raises selected-attack MAE480.6→509.7m, with macro Recall
91.73%→91.44%. It does **not** meet the all-case utility constraint. For K3,
the weakest-case Recall worsens and the exploratory privacy envelope worsens.
Do not describe this as a universally better model or a successful final result.

The proposed model remains unconstrained regarding neural/non-neural design.
This cycle is an internal mechanism ablation; existing clean-room comparator
limitations and web-app defaults are unchanged. All data generation and fitting
here are SUMO+OSM only. No GeoLife trajectories or fitting enter this cycle.

## Files and lineage

- `data/scenario_suite_v2/` and `experiments/build_scenario_suite_v2.py` extend
  frozen v1 generation without editing historical sources.
- `benchmark/anchor_belief.py`: public grid, emission likelihood and approximate
  temporal filter. `benchmark/engines/belief_lane.py`: expected POI coverage.
- `experiments/run_belief_suite.py`: separate validation/confirmation stages.
- `experiments/verify_scenario_suite_v2.py`, `scenario_v2_checks.py` and
  `verify_belief_suite.py`: independently written arithmetic/gate/native audits.
  The isolated base-case checker derives from v1 but permits verified short
  unsampled terminal edges; the frozen historical verifier is not modified.
- `experiments/render_belief_suite.py`: exact-lookup tables and macro summaries.
- Canonical thesis: updated §3.8 dataset specification, new §5.8 method and
  §6.8 experiment, abstract and conclusion. No duplicate title page/draft report.
- The canonical scenario table now cites v2. Its regression test was updated to
  v2, while historical v1 checks and evidence remain unchanged.

Frozen artifact checksums:

| Artifact | SHA256 |
|---|---|
| v2 dataset | `e5ee6dad69074f870f2ef4cd2c1ada1a33d3fd71b846c8ef771a3b7b72856ef8` |
| Validation | `009d5c0a73ca43d439bd938e374a507f39660d25f3a94e5943e0449a5dc6a5c3` |
| Selection, before confirmation | `a5d80de1aa1299ecce57c5a49a4f744409a1b5f603c6a17f6850e6efe3d854b6` |
| Confirmation | `0319caba8029aa59f7069ccf58500ab355bb180cda6ec9595eda7b5bdf8382cc` |
| Generated thesis tables | `5c2a72a3f5d48aa4c0a89f0354700f2ff9e4b0bfed7d1cee9a526df947128a19` |
| Public belief model | `d2bdb241c5dde09a3add02cc6124349ecdb995407994e2376e207addcab1b20e` |

The 32 experiment source hashes are in each stage JSON; verifier source hashes
are separately retained in `verification.json`. All 19 scientific sources and
the result hash of the preceding contextual study remain unchanged. Earlier
v1 dataset, paper benchmark and lane comparison checksums were also rechecked.
Their numbers must not be pooled with this different prior/case/family sample.

## Dataset construction and QA findings

Six seeds:101/102 training,103/104 validation,105/106 within-city confirmation.
Twenty-two sessions per family, simulated in nine separate day runs: the
fourteen-session base day plus eight one-trip calendar days. Shared person,
device, car, route/day relations stay within the same split and cannot overlap
in time. Synthetic activity and identity labels are not empirical human data.

Public route design precedes protection outcomes: single-successor origin,
alternative endpoint separation, a shortened nearby endpoint, and a route close
to a rare-category OSM POI. The generator may retry public route proposals but
does not search seeds using attacker/defender scores. Rare means <=5% category
share among418 retained POIs and <=150m realized FCD proximity, not a claim of
sensitivity or visit rarity. The selected training/validation/confirmation
routes do not necessarily visit every rare category.

S6.C uses day1–6 history with five routine and one rare destination; day7/8 are
routine/rare query trips. A record contains seven sessions: **target_slot=6**.
The current trip is limited to a prefix preceding its endpoint. Raw historical
positions require their own protection before attacker access; no cross-day
privacy ledger is provided by the dataset. One query trip lacks a sampled fork
cut, leaving11 records instead of12. Overnight activity and return-home legs
are outside these deliberately synthetic observation windows.

Three pre-protection issues were found rather than suppressed:

1. Initial public endpoint search included internal SUMO edges. SUMO rejected
   the route before results existed. Endpoint candidates now exclude internal
   edges; actual native internal connections remain in motion verification.
2. The first full bundle had two13–15m FCD displacements at a one-second parking
   waypoint in family102. Native `stops.xml` confirms parking905–906s on the same
   lane while longitudinal position barely changes. Only route-enforcing
   one-second stops were changed to `parking=false`; real stationary visits
   retain their intended declaration. All six families were rerun with the
   same seeds. No coordinates were edited or interpolated.
3. Independent labels caught two S5.C pairs where a short immediate successor
   is absent from1Hz FCD. The next *observed* edge was not the next *adjacent*
   edge. Reject these records for S5.C, retaining legitimate views for other
   tasks. A subsequent full rebuild reproduced all94,334 FCD samples exactly;
   only the corrected scenario records changed. Three trips also cross tiny
   final edges after their last sample: require native route completion,
   <=8.1m remaining external distance and <=1s until arrival. Do not invent a
   final GPS point or relax arbitrary continuity failures.

Failed preflight bundles remain recoverable in ignored local cache directories
`cache/scenario_suite_v2_preflight_bundle/` and
`cache/scenario_suite_v2_preflight2_bundle/`. They are **not** published evidence.
The final native run files are under `cache/scenario_suite_v2_final/`.

Final counts:132/132 completed sessions,94,334 raw samples,207 records,30/30
subcases in the union. Per-family coverage26/28/27/29/26/30. S5.C has only one
record, in confirmation: it cannot support trained/validated attack comparison.
S6.A/S6.B/S10.B have three records each. This is still a small challenge suite,
not a complete benchmark population. The30-subcase denominator is retained.

## Method, literature and limits of the proof

The public120m grid has2,076 latent states; each is an actual lane state near
its cell center. The prior sums training occupancy mass in each cell. This
prior includes synthetic repeated-day trips, not representative query counts.

The ideal REM coordinate likelihood normalizes over all102,123 original lane
states, not just the latent grid, and sums duplicate-coordinate multiplicities.
After the first release:

`P(anchor | x, previous) = (1-q(x))*fresh(anchor|x) + q(x)*I(anchor=previous)`.

`q` is the Laplace-CDF noisy-reuse probability. An unchanged anchor can come from
reuse **or a fresh sample** at the same coordinate. The filter never sees a
private reuse-decision flag. After H12 it predicts without treating the retained
anchor as a new observation. The ideal spent bound is at most.23/m for12
observations under configured cap B=.24/m.

For dt<=120s, approximate motion is.6 stay+.4 occupancy-weighted isotropic
diffusion. Longer gaps explicitly mix toward the training prior with
`persistence=.6**(dt/20)`. This is not a learned/calibrated directed transition.
Output feasibility is separately enforced on the unchanged directed lane graph.
POI weights average reference answers over the belief and reward newly covered
answers, not repeated duplicates. The geometry remains anchored to the protected
point except in the explicitly named `belief24_center` ablation.

Primary source rechecked: Chatzikokolakis, ElSalamouny and Palamidessi,
*Efficient Utility Improvement for Location Privacy*, PoPETs2017(4),308–328,
[final proceedings PDF](https://petsymposium.org/popets/2017/popets-2017-0051.pdf),
DOI10.1515/popets-2017-0051. The earlier draft PDF has different pagination/DOI;
use the final proceedings reference. It motivates remapping with a prior/loss,
but our coarse filter and constrained sequential dummy choice are **not its
optimal Bayesian remapping**, nor proof of novelty.

SUMO primary documentation checked:
[vehicle/route definitions](https://sumo.dlr.de/docs/Definition_of_Vehicles,_Vehicle_Types,_and_Routes.html),
[FCD output](https://sumo.dlr.de/docs/Simulation/Output/FCDOutput.html) and
[stop output](https://sumo.dlr.de/docs/Simulation/Output/StopOutput.html).
These define simulation fields; rarity thresholds and activity patterns are
our experimental choices, not standards prescribed by SUMO or a paper.

For fixed public prior/context and observation schedule, the new generator only
postprocesses protected anchors and retains the existing **ideal** composition
bound. It does not establish executable pure Geo-I under floating-point sampling,
protect query timing/content, or solve long-horizon budgeting. B=.24/m still
gives the very loose exp(24) upper bound at100m under the stated D-infinity
adjacency. Low observed attack accuracy is not a formal privacy proof.

## Experimental protocol and results

Recorded before scores in `thesis/notes/belief_evaluation_protocol.md`. Five
variants, K3/5, nine cases, two families per phase, one paired RNG replicate.
The first12 allowed events are retained. All S2.C views contain9 samples from
the first visit and3 from the second, with the true long public time gap intact.
The attacker task is location inference, not inference of a home/work semantic
label. No new S4–S10 attack was run in this experiment.

Each method/K/case has separate validation-selected MAE/Hit attackers. Group
averages first average events within a record, then the two families; macro
scores weight all nine cases equally. Events and overlapping record views are
not independent repetitions. Confirmation preserves every variant, including
nonselected ablations, but never reselects based on their scores.

Selection: require every validation case mean Recall>=.90, then minimize macro
worst-attacker Hit100. Otherwise maximize minimum-case Recall then minimize Hit,
with12-digit decision precision. Both K choose`belief24`, **infeasible**:

| K | Baseline minimum validation Recall | Selected fallback minimum | Feasible? |
|---|---:|---:|---|
| 3 | 74.05% | 77.92% | No |
| 5 | 71.67% | 79.35% | No |

Confirmation, baseline →selected fallback:

| K | Macro Recall | Worst-case Recall | Selected Hit100 | Exploratory Hit envelope | Selected MAE |
|---|---|---|---|---|---|
| 3 | 91.24→91.34% | 83.33→77.78% | 12.69→12.22% | 26.11→27.22% | 384.0→430.9m |
| 5 | 91.73→91.44% | 85.33→83.61% | 23.15→18.70% | 33.57→28.15% | 480.6→509.7m |

Important counter-evidence:

- S3.B/K3 Recall83.33→77.78%; do not hide this behind macro Recall.
- S3.C/K5 selected Hit0→10%; do not claim every attack weakens.
- S2.B/K5 selected Hit100→50%, Recall100→99.44%; encouraging but only two families.
- Mean-center variant lowers selected Hit to zero, but Recall is67.98%/70.52%.
  This is an unacceptable utility tradeoff under the preset criterion.
- Extra-distance diagnostic, equal record means on these balanced cases:
  K3 baseline81.21m→belief24 86.84m; K5 75.90m→78.59m. Mean-center gives
  560.25m/459.29m. These supplemental diagnostics do not alter selection.
- All complete-query rates are100%: returning five POIs need not return the
  correct five. Small category catalogues partly saturate this indicator.
  S1.C targeted rare-category Recall is100% for both baseline and selected
  method in both confirmation families/K; it is not an observed improvement.
- Selected dummy streams have as few as2 distinct coordinates at K3 and4 at K5.
  Duplicate lane-coordinate states do not establish K-anonymity.

Exploratory private-filter diagnostics (not attacker scores): at S1.C the belief
mean is about1.91–2.80km from truth, versus anchors0.14–0.68km away across the
four family/K runs. This supports investigating prior/multimodality issues; it
does not isolate their causal contribution. Never expose these private means
as if the LSP had received them.

## Executed verification

- Full raw dataset audit:54 SUMO runs,94,334 exact FCD records,7,052 external
  route transitions; max1s displacement9.203m;12 generation source hashes.
- All360 result rows replay exactly, including private anchors, belief summaries,
  public events and motion states. All360 prefix comparisons pass.
- 72 paired-anchor groups,2,370 released events,14,220 POI queries,8,040 directed
  motion transitions,41,016 native connections checked.
- Ten POI origins additionally checked against independent NetworkX directed
  distances; every run's complete POI response recomputed during replay.
- Each per-attack error, event/category Recall, group summary and validation-only
  selection recomputed. Selection/source checksums required by confirmation.
- 197 regression tests pass, including exact full-support/duplicate-coordinate
  emission arithmetic, fresh-vs-reuse mixture, stale-cache rejection, finite
  temporal filtering, POI expectation, prefix and private-GPS canaries, budget
  exhaustion and adversarial v2-label mutations.
- Generated tables compare exactly against their renderer; all previous frozen
  scientific sources and result checksums remain intact.

Native motion checks certify free-flow reachability only; not a traffic-light,
acceleration, congestion or lane-change-aware SUMO replay of each dummy.
Timings: cold graph2.41s, training prior3.35s, belief model1.70s; warm belief
load0.18s. Selected generation median3.03/3.22ms, p9535.64/62.54ms for K3/5,
excluding model initialization, context setup and network I/O. CPU development
measurements are not phone latency promises.

## Priorities for the next agent/cycle

1. **Freeze this cycle.** Families105/106 are now inspected. New tuning requires
   new declared validation/confirmation families; do not overwrite these files.
2. **Fix scarce scenario geometry before attacks.** Ensure both alternatives
   have an observable immediate successor at1Hz, and a shared observed fork cut;
   target balanced S5.C/S6/S10 support in every split. Do not lower gates merely
   to reach a sample count. Consider native edge-entry events when evaluating
   immediate-next-edge targets, explicitly changing the record contract.
3. **Calibrate the belief on validation data.** Compare occupancy/uniform or
   regularized public priors, appropriate cell sizes, lane/direction-aware
   transitions from protected history, and multimodal rather than mean-center
   goals. Use posterior coverage/calibration diagnostics, not just training fit.
   Any learned predictor is allowed if it respects data/privacy boundaries.
4. **Improve path-level POI tradeoffs.** The current greedy step may enter a
   direction that later loses useful POIs. Test bounded multi-step planning over
   protected/public state; do not read raw future GPS or true destination.
5. **Strengthen adversaries and utility stress.** Add mechanism-aware likelihood
   inference learned/fitted on separate simulations, attack ablations, repeated
   seeds, and family-level uncertainty. Keep selected-vs-envelope distinction.
   Use category-level Recall and directed extra distance; test larger/more
   heterogeneous POI catalogues instead of relying on saturated completeness.
6. **Extend coverage deliberately.** S4 needs person/device linkage separately;
   S5/S6 require future-route/destination attackers and historical-release
   budgets; S7 content policy; S8 companion side information; S9/S10 boundary
   attacks on full allowed windows. Data existence is not protection evidence.
7. **Then revisit comparator fairness and scale.** Same-data/output-contract
   comparisons, full neural reproduction gaps, multiple budgets/horizons/maps,
   traffic realism, and on-device profiling remain prerequisites for paper claims.

## Reproduction and presentation

```bash
venv/bin/python -m experiments.verify_scenario_suite_v2 --raw
venv/bin/python -m experiments.verify_belief_suite --replay
venv/bin/python -m experiments.render_belief_suite --check
venv/bin/python -m pytest -q
```

Fresh build/run commands are in the two artifact READMEs. Raw caches/OSM/SUMO
dependencies must be available; they are intentionally not committed. Full
result/dataset JSON is evaluator-private synthetic evidence, not an attacker
export. The existing web app is unchanged and does not serve the new bundles.

Presentation contract: canonical LaTeX/PDF with exact per-case lookup tables,
all five aggregate variants and visible negative evidence. Tables are chosen
for precise lookup, not an underpowered ranking or significance chart. The
reusable CLIs are the reproducible companion; no duplicate notebook/HTML report.
Data-quality and validation workflows determined the gate repairs, split/metric
caveats and replay checks; the PDF workflow governs final render inspection.

Final PDF QA:71 A4 pages; one title page. All pages rendered and inspected in
overview, with detail inspection of dataset tables/definitions (PDF26–29),
method equations (50–51), new experiment tables/text (65–67) and conclusion(68).
The new equation overflow and near-empty spill pages were corrected. Final
LaTeX log has no overfull boxes, undefined references/citations, duplicate labels
or missing characters; some inherited bibliography underfull-spacing warnings
remain without clipping. Tables6.12–6.15 are generated directly from checked
results; text and percentages were cross-checked against those artifacts.

Published `artifacts/reports/graduation_thesis.pdf` SHA256:
`3dbe3f6963c05b9534e8f78afc9b15b22a418a679bd72b2f245a33fb1ca28862`.
The file is byte-identical to the inspected final build. PDF pages25–30 cover
§3.8,50–51 cover§5.8, and64–67 cover§6.8 (printed page numbers differ).
