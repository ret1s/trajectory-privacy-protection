# Contextual BR-lane: development protocol (2026-09-07)

Recorded before executing the new mechanism comparison. Existing dataset and
lane benchmark are frozen. This is an algorithm-development cycle, not a final
paper test or a neural/non-neural design restriction.

Question: can a directed-road potential and a marginal POI-coverage score improve
moving-case utility while preserving the private-input boundary and ledger?

Fixed 2x2 ablation: route weight 0 or 0.5; POI-coverage weight 0 or 6. The old
Euclidean BR-lane is the zero/zero control. B=.24/m, H=12, theta=200 m, offset=80 m,
temperature=60 m, K=3/5, two random replicates. Both additions are postprocessing
of protected anchors and public map/POIs. Same RNG key per paired record/K/rep;
separate anchor and dummy streams. No private truth in either score.

Use the frozen family91 public prior; family92 development selection; family93
development reporting. Cases S1.A/S2.B/S3.A/S3.B/S3.C, first at most12 permitted
events; record all truncation. This extends the moving-case checks to corridors
and sparse queries; still no new independent families or all-ten evaluation.

Build POI signatures over the whole public lane graph, before any protection,
including the service's coordinate-to-query-state convention. No online
position-dependent context download. This assumes the defender knows a static
public POI catalogue, not proprietary/live LSP information.

Retain validation-selected MAE/Hit attackers plus a separately named exploratory
report-family attack envelope. Report privacy and utility even if they worsen.
Never retune on family93 or silently drop a failed row. Select one weight pair
per K, common to all five cases, using family92 only: require every case's mean
Recall >=.90, then lowest macro audit Hit100. If none feasible, select highest
minimum-case Recall, then lowest macro Hit100, and mark infeasible.

Verification correction after the first development run: compare selection
rates rounded to 12 decimal places so a one-ulp arithmetic discrepancy between
equal fractions cannot defeat the next tie-breaker. This fixes numerical
selection, not the model grid or data cases. The first run is retained locally
in `tmp/contextual_lane_initial_20260907/`; the shipped run is regenerated with
this rule. Family93 remains explicitly development data, not a fresh holdout.

Validation oracle: exact zero/zero public-event parity with old BR-lane; same
anchor stream across pairs; independent NetworkX directed transitions; compare
public POI signatures to service queries at fixed public states and emitted
locations; full regeneration and prefix invariance; unchanged source hashes for
the old releases. Timings include score work, with public context preprocessing
and model initialization reported separately. No phone-real-time claim.

Literature motivation: Chatzikokolakis, ElSalamouny and Palamidessi, *Efficient
Utility Improvement for Location Privacy*, PoPETs 2017(4), DOI
10.1515/popets-2017-0051, studies utility-improving remapping of protected outputs.
The current score is a road-constrained heuristic, **not their optimal Bayesian
remap**. A preserved Geo-I bound does not imply preserved empirical inference
error. Both are tested/interpreted separately.
