# Boundary-window diagnostic, frozen before computing the additional results

Date: 2026-09-21. Status: exploratory reanalysis of previously examined artifacts,
not a new held-out test or a newly trained defender.

Source: `artifacts/benchmarks/paper_benchmark/results.json`, paper-v2.
Methods: masked-unprotected and fixed `br_private`; K=5. All successful S9/S10
rows, seeds 81–83, four test trips per seed (12 trips per scenario).

Compare fixed extra cuts of 0, 2 and 4 events (0, 40, 80 seconds at the recorded
20-second cadence). S9 removes leading events; S10 removes trailing events.
Existing masks are 60 seconds. Do not tune the grid or select a winner afterward.
This is deterministic postprocessing of saved public transcripts, not regeneration
of the latest lane/coverage mechanism. S10 trimming is offline; it does not imply
that a live service can foresee the user's destination or erase sent events.

Three fixed, untrained attackers see only the retained public coordinates and
cadence/mask policy: boundary centroid; full-window mean centroid; linear
extrapolation using the first/last three centroid events, over the known 60-second
mask plus the declared extra cut. No endpoint coordinate or trip label enters
any prediction function. Report all three and the descriptive aggregate envelope;
do not select the best prediction for each individual target.

Errors: same local equirectangular projection as paper-v2, using the saved
per-seed projection latitude. One target per trip, equal trip weight. Report MAE,
nearest-rank median/p90, Hit50/100/200 and group count. No significance claim.

Utility: replay saved independent query responses. Recompute Recall from POI IDs.
Report (a) retained-only Recall and (b) service Recall across all original 12
query events, giving newly dropped queries zero service in the no-cache model.
This denominator still excludes the original 60-second mask: it is a conditional
window utility, not full-trip utility. No counterfactual claim about cache quality.
No additional networking, service responses, training, or on-device timing.

Audit: input SHA-256, derivation code SHA-256, per-row predictions/errors and
utility numerators/denominators. Verify source metrics from saved errors/POI IDs;
check no mutation, correct boundary cuts, no ground-truth dependency and exact
preservation of the no-extra-cut transcript. Keep original artifacts immutable.
