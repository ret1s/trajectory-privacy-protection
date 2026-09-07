# Verification: controlled S1–S10 data and proposed-method framing

Date: 2026-09-07. Baseline commit: `8dc7f855223ef1b59902b7090027594cd3d5a4f3`.

## Assessment

**Share with caveats as a reproducible development dataset and thesis update.**
Do not present this release as a ten-scenario protection benchmark or a SOTA
improvement. It adds data, labels, gates and reproducible inspection, not new
attack scores, model training or a proven all-scenario algorithm.

The user's clarification is binding: **non-deep-learning methods belong in the
expanded comparator set; the proposed method is NOT constrained to be non-DL.**
Chapter 5 now explicitly identifies the proposed stateful, private-anchor dummy
generator. BR-Dummy is its current experimental implementation; REM is its
anchor primitive. The current code has no neural component, but this is an
implementation fact, not a research constraint or contribution. Do not relabel
an untrained model as trained, or claim a new mechanism was implemented in this
dataset cycle. Chapter 4 retains DLS and the broader non-DL literature as
comparators. Existing TransProtect/semantic/AnotherMe implementations remain
adaptations, not verified paper-equivalent reproductions.

## What changed

- New isolated `data/scenario_suite/` package: declared 30 cases, legal route
  families, full 1 Hz FCD including lane progress and angle, post-run eligibility,
  person/device/physical-vehicle/session assignments, query intents and proximity.
- `experiments/build_scenario_suite.py`: rebuilds an OSM passenger network, runs
  actual SUMO at seeds 91/92/93, stores commands, source hashes and actual arrivals.
- `experiments/verify_scenario_suite.py`: independently recomputes gates and
  audits raw SUMO output, without calling the record builder.
- `experiments/inspect_scenario_suite.py`: bounded local-input examples, evaluator
  metadata only with an explicit flag; no fake “protected” transcript.
- `tests/test_scenario_suite.py`: allowlists, future-coordinate isolation, shared
  S8 clock, query-cover invariance, raw integration, table/data agreement and
  eight deliberate-corruption checks.
- Thesis §3.5/§3.8: all ten scenarios are research targets, precise recipes and
  conditions, missing cases, independence and module-to-evaluation mapping.
- Thesis Chapter 5: proposed method, inherited foundations, implemented scope,
  open role of learning, and distinction from non-DL comparators.

## Verified data

Dataset: `artifacts/datasets/urban_scenarios_v1/dataset.json`.
SHA-256: `826bf236588448e5e59b5e4e0304ddef3e3cf85751df8dddf901b72ae88aabb9`.

| Quantity | Verified result |
|---|---:|
| Related route families | 3 |
| Completed / planned sessions | 36 / 36 |
| Raw FCD samples | 21,818 |
| Scenario records (NOT independent trips) | 96 |
| Scenario groups with generated data | 10 / 10 |
| Declared subcases with at least one generated record | 28 / 30 |
| Full directed route transitions checked | 2,071 |
| Route-edge occurrences between consecutive FCD samples | 292 |
| Maximum one-second geographic displacement | 8.818 m |
| Raw FCD points compared field-for-field to XML | 21,818 |
| Scientific input/source files pinned | 6 |

Record counts S1–S10 respectively: **6, 9, 9, 9, 9, 4, 27, 7, 7, 9**.
Development family record counts: seed91=32, seed92=33, seed93=31.
These are reused views of 36 sessions, in only three related families; do not
use 96 or 21,818 as the independent sample size for confidence intervals.

S1.C rare POI context and S6.C routine/rare destinations across days are not
generated. S6.B, S8.A and S9.A are present in only one family. The catalogue
keeps absent cases with a zero count and reasons. All new case-level
`attack_evaluated` and `protection_evaluated` fields remain false.

## Checks and issues corrected during this cycle

1. **Do not count SUMO internal junction edges as independent access roads.**
   Loading internal lanes is necessary for FCD fidelity but causes incoming-edge
   lists to include via edges. Road-degree gates now exclude these edges while
   retaining them in motion data. Otherwise a true single-access edge could be
   incorrectly counted as having two entrances.
2. **Do not require exact equality between sampled edge lists and full routes.**
   At 1 Hz, short edges can be traversed between samples. FCD must preserve
   monotone route order; the complete legal sequence is separately compared to
   SUMO vehroute XML. No interpolation or manual coordinate repair was used.
3. **Repeated destination is not identical last FCD position.** The final
   fractional second before arrival is not sampled. Repeated-endpoint cases
   verify shared planned edge and compatible recorded endpoints, rather than
   forcing raw endpoints to coincide.
4. **A return stop must involve actual movement.** An intermediate stop on the
   far part of the loop forces two visits to the same lane onto different route
   occurrences. Two consecutive stop entries at the same position would not
   suffice. Both stationary intervals and the moving gap are checked against
   FCD; scheduled stops are also matched to actual SUMO stop output.
5. **S8 must retain cross-user timing.** The pair uses a common relative epoch.
   Independently resetting each user to time zero would invent simultaneity.
6. **S7 controls need more than one class.** Each reference policy has three
   balanced synthetic intent templates. Sequence templates share the first query
   but diverge later. These are diagnostic examples, not realistic human behavior
   or a tested privacy result.
7. **Separate identities and resources.** Same-person/new-device and
   different-person/shared-device labels differ. No person, device or physical
   car is assigned to overlapping sessions; related entities do not cross the
   development splits.
8. **Enforce observation boundaries.** Device input excludes route/identity
   metadata and unobserved coordinates. S5 futures lie after the cutoff; S6
   endpoints are unseen; S9/S10 hide true boundaries and do not expose raw trip
   duration. The bundle itself stays evaluator-only.

## Reproduction and validation results

```bash
venv/bin/python -m experiments.build_scenario_suite --output tmp/scenario_suite_release_replay --workdir cache/scenario_suite_release_replay
venv/bin/python -m experiments.verify_scenario_suite --raw --compare tmp/scenario_suite_release_replay/dataset.json
venv/bin/python -m pytest -o addopts='' -q tests
venv/bin/python -m experiments.verify_paper_benchmark --raw
```

- Independent SUMO rebuild: **scientific payload identical** (all observations,
  records, gates, labels, source versions, designs and arrivals). XML timestamps,
  raw provenance hashes and output paths are not scientific equality criteria.
  For fresh clones without original XML, audit the rebuilt bundle with
  `--raw --reference artifacts/datasets/urban_scenarios_v1/dataset.json`.
  This checks the rebuild's raw sources and exact scientific equivalence to the
  checksummed reference; it does not pretend to recreate original XML timestamps.
- Full suite: **163 passed**. Fresh clones need to regenerate ignored raw SUMO
  caches for integration tests; those tests explicitly skip without the network.
- Existing paper-v2 artifact remains unchanged:
  `2df043bfb73c3b2f630046381daf5b4bc9f6941fb5be925b0967d6de98f27538`.
  Reaudit passed: 37 source files, 60 samples, 1,440 rows, 11,316 events,
  67,896 category queries, 506 raw FCD points, 24 hidden targets and 10,704
  independently recomputed BR transitions. Its pre-existing 96 N/A rows and
  one AnotherMe failure remain reported; they were not removed or “fixed” by
  changing data. Its 264/270 prefix passes likewise retain the six offline
  AnotherMe failures. No old score was recomputed into the new dataset.
- PDF: XeLaTeX/latexmk build, all-page contact review and full-size review of
  dataset/method pages. No overfull boxes, undefined references or missing glyph
  warnings. One cover only. Canonical PDF release is updated with this commit.
  **54 pages**, SHA-256:
  `6dd6f7210e1ea9d54ca4bf838cbdf6ba558f040b0624ff52e279f4be950568e9`.
  Proposed method starts at printed page 34 (PDF page 41); dataset specification
  at printed page 17 (PDF page 24); non-DL comparators at §4.3.

## Literature cross-check and interpretation

- [SUMO FCD fields](https://sumo.dlr.de/docs/Simulation/Output/FCDOutput.html):
  keep geographic coordinates, lane-relative progress and heading; FCD is
  simulator movement, not human identity or query intent.
- [SUMO vehicle/route specification](https://sumo.dlr.de/docs/Definition_of_Vehicles,_Vehicle_Types,_and_Routes.html):
  a vehicle, route and type are distinct objects. LBS person/device assignments
  are an explicit additional model, not attributes inferred from a vehicle ID.
- [de Montjoye et al., 2013](https://www.nature.com/articles/srep01376): motivates
  mobility linkage risk, not the validity or realism of our synthetic people.
- [Ziebart et al., AAAI 2008](https://www.cs.cmu.edu/~bziebart/publications/maximum-entropy-inverse-reinforcement-learning.html):
  route prediction background; S5's chosen next-edge target is our declared
  operationalization, not a reproduced evaluation from this paper.
- [Dhondt et al., CCS 2022](https://people.cs.kuleuven.be/~stijn.volckaert/papers/2022_CCS_Fitness_Tracking.pdf):
  endpoint-zone inference uses side information including distance metadata.
  Our time masks and route cutoffs are narrower controlled tasks, not equivalent
  EPZ attacks and not privacy guarantees for homes.

The data-quality/validation workflow influenced the release by retaining zero
coverage, rejecting non-realized gates, separating reused records from units of
independence, and keeping data readiness separate from defense effectiveness.

## Priorities for the next agent

1. **Respect the corrected method framing.** Non-DL is a comparator axis, not a
   restriction on the main method. Any learned component needs an explicit
   training split, causal input contract, latency measurements and renewed
   privacy analysis if it accesses private history. Do not make novelty claims
   from an existing Geo-I primitive plus an untested neural component.
2. **Expand independent families before attack fitting.** The current three
   groups are development fixtures, not a final test cohort. Balance every case
   over train/attack-selection/defense-selection/test families; reserve a fresh
   test set after the generation protocol stabilizes. Never improve a headline
   by dropping failed/unrepresented subcases.
3. **Fill missing semantics and activity.** Pin OSM POI categories/access points
   and declare rarity before S1.C; generate several days and held-out personal
   routines before S6.C. Traffic stops need hard-negative labels and observed
   conditions, rather than treating all zero speed as a sensitive visit.
4. **Implement threat-specific attackers, then protection modules.** S4 pairwise
   linkage is distinct from closed-set identification; S5 predicts the next edge,
   S6 the eventual destination; S7 query intent needs content-aware attacks;
   S8 requires declared knowledge of the companion. Keep plaintext and perfect
   companion-information limit controls.
5. **Resolve motion representation in the mechanism.** New truth includes lane
   progress, but BR-Dummy still uses graph vertices. Do not claim the previous
   node-projection validity mismatch was fixed in the algorithm by adding fields
   to a dataset. Introduce a public, fixed edge/lane-state support and re-derive
   the metric privacy statement before claiming continuity improvements.
6. **Integrate at equal utility/cost.** Query-cover, pseudonym and endpoint
   policies change observable contracts. Count missed service, bandwidth and
   latency, as well as extra private-data accesses. Compare against native paper
   metrics and credible non-DL baselines; do not silently substitute a Markov
   adaptation for the paper's trained deep architecture.
7. **Web integration is still pending.** Existing dashboard shows paper-v2
   scores, not this suite's S4–S8 protection results. A future scenario browser
   should separate local truth, attacker-visible observations and labels, and
   explicitly display unavailable evaluations instead of empty charts implying
   completed benchmarks.
