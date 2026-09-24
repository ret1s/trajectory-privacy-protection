# Contribution stress evidence

Reproduce from the repository root (Python + NumPy):

```sh
python -m experiments.contribution_stress
python -m pytest tests/test_contribution_stress.py tests/test_boundary_stream.py tests/test_boundary_switching.py -q
```

Protocol: `docs/research/contribution_stress_protocol.md`.
Inputs are immutable saved artifacts; results contain source, helper, protocol and
runner SHA-256. This is retrospective analysis, not new defender generation.

- 636 fresh-switching rows: paired family bootstrap, 10,000 draws. Six families,
  nine cases; replicas do not increase the number of independent families.
- 264 historical endpoint transcripts: 792 transcript/cut conditions. Eleven
  applicable methods, 12 trips/scenario, three seed folds. AnotherMe's 24 missing
  endpoint rows are explicitly recorded, not scored as zero.
- Public-only estimators: 71 for K=5, 31 for a single coordinate. Test seed errors
  never participate in selecting that fold's attacker. Model selection uses
  eight training trips, making it unstable; these are not attack-optimal scores.
- Fixed six-centroid/30-second extrapolation hits S9 br_fresh 7/12 and br_private
  2/12; the separately cross-seed selected attacks hit 7/12 and 0/12. The latter
  zero cannot be interpreted as no successful attack in the pool.
- S10 masked unprotected baseline also has cross-seed Hit100=0/12: the test lacks
  discrimination. Broader thresholds, road/POI priors and repeated-site attacks
  are still required. No full-session or new boundary-defender claims follow.
- Utility denominator covers the original saved 12-event window, not the hidden
  initial/final 60 seconds. Dropped additional queries score zero without cache.

Report sections 10.4–10.6 distinguish observed advantages, counterexamples,
uncertainty, runtime cost, local paper adaptations and missing new confirmation.
