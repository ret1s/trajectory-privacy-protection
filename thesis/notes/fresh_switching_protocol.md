# Fresh-family evaluation protocol — 2026-09-10

Frozen design before generating or scoring new families. This is a report-week
experiment, not a declaration of conference readiness or a new privacy theorem.

## Dataset

- SUMO 1.27.1 + the existing SHA-pinned urban OSM passenger network only; no GeoLife.
- Fixed seeds 301–306: validation; 307–312: confirmation. Generate all 22 sessions
  over nine synthetic days per family using the established 30-subcase demand plan.
- All stops are on-road (`parking=false`), including one-second waypoints, to avoid
  off-lane parking coordinates. Preserve failed sessions and scenario rejections;
  do not replace seeds according to protection results. Halt if physical checks fail.
- Independently check native FCD, route completion/connectivity, stops, identities,
  scenario predicates, exact route/window overlap with old and opposite-split data.
  Shared map edges are permitted and reported: this is same-city, new-family transfer.
- Publish a new immutable SQLite shard with a release/update log and a reference to
  the historical database's sealed releases. Do not grow or rewrite the 95 MiB old DB.
- Dataset coverage includes S1–S10; this algorithm experiment evaluates S1–S3 only,
  first 12 observed events, keeping any absent subcase explicit in the readout.

## Proposed extension

Keep the existing private anchor and reachable coverage selector. Replace only its
approximate single-motion belief with a finite two-mode forward filter:

- Initial joint belief: public uniform-cell spatial prior × (0.5 stopped, 0.5 moving).
- Mode stay probability `(1 + exp(-dt/100))/2`, symmetric across modes. This is a
  declared heuristic, not fitted to confirmation data or calibrated traffic behavior.
- For dt <= 120 s, obtain public diffusion D from the existing transition
  T = 0.6 I + 0.4 D. Stopped kernel = 0.95 I + 0.05 D; moving = 0.10 I + 0.90 D.
- Mix modes, propagate spatial belief, multiply by the existing exact private-anchor
  emission and normalize jointly. For dt > 120 s use the old public prior-reset
  predictor in both modes (do not pretend to reconstruct an unobserved trip).
- Sum over modes, compute expected POI coverage, use the existing mean objective,
  greedy initialization + at most three improving one-track exchanges.
- Inputs are protected anchor history, public timestamps/map/POIs and fixed constants;
  no raw velocity, true coordinate, target label, or future event enters this filter.
- This is standard HMM filtering adapted as postprocessing, NOT a novel HMM/IMM claim.

## Comparisons and selection

- B=0.24, K=5, H=12, anchor threshold=200 m; three paired randomness replicates.
- Four variants: geometric, mean_greedy, mean_exchange, switching_exchange.
  These are internal controls/ablations, not additional reproduced SOTA papers.
- Client top-5 POIs, server depth L=5 and L=10, all six existing categories. Identical
  protected transcripts at both L values. Count requests, reply items and serialized
  POI-ID bytes; ID bytes are not total network traffic or local runtime.
- Reuse the exact old training population (core 101/102 plus 64 auxiliary families).
  Reuse pinned control attack fits; generate/fix the new variant's attacks using only
  those same training records. No fitting on either new split.
- Existing direct/retrospective attacks, core kNN/loss-aware attacks, expanded kNN/
  loss-aware attacks and two 128-tree ExtraTrees attacks, with frozen parameters.
- Select attack separately per method/subcase for MAE and Hit@100m on validation.
  Report both this frozen choice and the confirmation descriptive attack envelope.
- Per L, eligibility is minimum subcase mean Recall@5 >= 0.90 on validation;
  among eligible methods choose smallest macro strongest-attacker Hit@100m (name
  breaks ties). If none qualify, report no eligible method, not a relaxed threshold.
- Seal training, validation and selection hashes before scoring confirmation.
- Report all four variants even if the new filter loses. Report per-family changes;
  six confirmation families, not RNG replicates/events, are the independent units.
  No retuning on confirmation. A failed candidate remains useful negative evidence.

## Verification and scope

Recompute metrics from public releases and ground truth using the independent
shortest-path service oracle; refit/check attack arrays; check complete/prefix replay,
common anchors and road reachability. Keep raw rows and hash-pinned source provenance.
The ideal anchor budget survives data-independent postprocessing under the existing
assumptions; finite precision, loose distance-scaled bounds, same-city synthetic
support and the untested extension targets remain explicit limitations.
