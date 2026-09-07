# Contextual BR-lane: a development ablation

This experiment adds two actual proposed-method components: directed-road
potential and marginal public POI coverage. **Neither validation-selected
configuration passes the all-five-case Recall >= 0.90 requirement.** Utility
improves locally but privacy can worsen, so these remain opt-in research
variants, not a newly promoted default or a final SOTA result.

No neural/non-neural constraint is imposed on future proposed models. This is
an internal ablation, not an additional comparator-paper reproduction.

## Contents and access boundary

- `results.json` / `results.sha256`: frozen public releases, per-query results,
  strengthened attacker errors, validation-only selections, exploratory
  summaries, source hashes and **private synthetic evaluator anchors/states**.
- `results_tables.tex`: two generated tables included in the canonical thesis.

**Do not serve the JSON directly to an attacker or public web endpoint.** This
cycle does not wire the new variants into the web app. Existing demo defaults
and old releases are unchanged. Only each row's `public` field is the simulated
LSP-visible transcript; neither raw truth nor evaluator anchors belong there.

## Fixed protocol

SUMO/OSM only, frozen `urban_scenarios_v1`. Family91 provides a public spatial
prior, family92 selects, family93 reports development results. No fresh final
holdout is claimed. Cases S1.A/S2.B/S3.A/S3.B/S3.C: single release, repeated stop,
moving prefix, low-branching corridor and sparse 60-second observations.

Ten related input records, 69 events counted within records (overlap possible),
K=3/5, two RNG replicates and four weight pairs => 160 rows. Route weight is
0 or 0.5; coverage weight is 0 or 6. All four variants share the exact same
protected anchors within a record/K/replicate. Zero/zero reproduces the old
BR-lane public events. Two RNG repetitions are not two independent users.

This experiment adds attacks that exploit individual stable dummy tracks and
their full-window means. These apply equally to the four variants. Do not
compare new Hit/MAE numerically with the earlier, weaker-attack lane tables.
The directed POI service is unchanged from that lane experiment, and differs
from the older junction study. No all-ten-scenario protection claim follows.

## Reproduce and verify

Run from the repository root with the existing venv and pinned scenario-suite
SUMO network / OSM caches. Use a new result destination to preserve this release.
The public POI cache is built locally from the entire public map, never fetched
according to the current private location. A stale cache is rejected; when its
source changes, move the old cache aside or use a new cache version explicitly.

```sh
venv/bin/python -m experiments.run_contextual_lane --output tmp/contextual_rebuild
venv/bin/python -m experiments.verify_contextual_lane --results tmp/contextual_rebuild/results.json --replay
venv/bin/python -m experiments.render_contextual_lane --directory tmp/contextual_rebuild
venv/bin/python -m experiments.render_contextual_lane --directory tmp/contextual_rebuild --check
```

Verify the committed release without regenerating or overwriting it:

```sh
venv/bin/python -m experiments.verify_contextual_lane --replay
venv/bin/python -m experiments.render_contextual_lane --check
venv/bin/python -m pytest -o addopts= -q tests
```

Replay excludes hardware-dependent timings, but requires exact public events,
evaluator state/anchor streams and utility responses. It also rebuilds the public
POI index without its cache and uses independent NetworkX motion checks. The
index query checks sample public states and all emitted states, not the entire
state catalogue. A cache/service membership disagreement is explicitly counted;
the current checked release has zero.

Read `thesis/notes/contextual_lane_protocol.md` for the fixed protocol and
`docs/reviews/verification_contextual_lane.md` for counter-evidence, limitations
and the next improvement plan. Family93 remains development data even though
the stored split field is named `development_test`.
