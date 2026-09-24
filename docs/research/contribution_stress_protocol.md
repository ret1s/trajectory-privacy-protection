# Contribution stress review — 24 September 2026

This is a retrospective analysis of already inspected artifacts, not a new sealed
confirmation experiment. No defender, threshold or historical artifact is changed.

1. S1–S3: use all 636 fresh-switching confirmation rows, frozen validation-selected
   MAE/Hit100 attackers, K=5, budget=.24. Contrast switching_exchange with geometric,
   mean_greedy and mean_exchange. Average replicas within family/case, then families
   within case, then the nine cases equally. Paired percentile bootstrap (10,000,
   seed 240926) resamples six whole families jointly across methods/cases. Missing
   S3.B family remains missing; reject draws containing no observed S3.B family.
   Report raw method differences and 95% intervals for MAE, Hit100, Recall L5/L10,
   step latency and reply bytes. These are exploratory intervals with six clusters,
   not multiplicity-adjusted or proof of generalization.
2. S9/S10: challenge every saved K=5 method with public-only endpoint estimators:
   candidate mean/median, each candidate track, and linear extrapolation using
   2/3/6 boundary events over 30/60/120 seconds. Include zero-velocity boundary and
   whole-window mean controls. Candidate order is observable in these artifacts;
   track-index attacks apply only to this output contract.
   Choose an attack separately for MAE and Hit100 on the other two simulation seeds,
   evaluate on the held-out seed (four trips). Never select per target or test seed.
   Repeat at fixed extra cuts 0/40/80 seconds; cut variants reuse saved core outputs
   and are NOT executions of BoundaryProtectedStream. Recompute service recall over
   all original queries, assigning zero to dropped queries (no cache).
3. Retain negative effects and all comparators, distinguishing local paper
   adaptations from faithful reproductions. Only historical source already masked
   60 seconds is available; full-trip utility, repeated-site attacks, new defender
   runs, A/B/C coverage and independent endpoint confirmation remain unmeasured.
4. Exact SUMO network and OSM source recovery is a prerequisite for a comparable
   full rerun. Never replace the network silently or reuse exposed confirmation
   families to tune a method and claim independent confirmation.
