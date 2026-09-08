# Pre-attack data-quality amendment, 2026-09-08

No expanded shadow training or attack scoring had begun at this amendment.
The initial fixed design exposed two data problems in native validation:

1. SUMO `parking=true` can report a parked point 6.793 m away from the lane
   center at its reported longitudinal position (family 1048). Change every
   planned stop to `parking=false`, retaining the same network/seeds/routes.
   This is the intended on-lane urban-stop scope, not a relaxed geometry gate.
2. After that correction, 80 families / 175,620 FCD samples passed native
   geometry/completion, but five stationary windows were exact duplicates:
   three within training, one across training/holdout (1036/1069), one within
   holdout. No old-v3 window overlap was found. Different full routes are not
   sufficient for independence of a stationary query window.

The revised PUBLIC-MAP planner additionally reserves a distinct stop lane and
one-metre planned-stop cell for each family. It selects without replacement
before simulation, without reading any target-v3 coordinates or attack scores.
All 80 seeds and 64/16 roles stay fixed; invalid designs are recorded as planning
rejections, not discarded after seeing privacy. The full native verification
and exact-window leakage gates are rerun; no observation is jittered, moved,
interpolated or silently dropped. This is a documented amendment, not a claim
that the initially declared data generator passed unchanged.

Rejected JSON/native outputs and generator snapshots are preserved locally at
`cache/urban_shadow_v1_rejected_parking_*` and
`cache/urban_shadow_v1_rejected_duplicates_*`; they are not accepted registry
releases. The final verifier records their content hashes for traceability.

The attack specification stays unchanged. In particular, `max_features=1` is
the integer parameter: one candidate feature at an ExtraTrees split, not the
fraction `1.0`. This distinction will be explicit in the fitted-model manifest.
