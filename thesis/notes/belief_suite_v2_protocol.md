# Belief-aware BR-lane and scenario suite v2 (2026-09-07)

Baseline: 8b530de. This plan is recorded before new simulation/protection results.
Keep v1 datasets and prior experiment sources/results unchanged. SUMO+OSM only.

## Dataset improvement, independent of protection outcomes

Six new route families: seeds 101/102 train, 103/104 validation, 105/106 reporting.
All related sessions, synthetic persons/devices/cars, day-history windows and
scenario views stay in one split. New reporting families are a small within-city
confirmation, not a representative population or an unseen city. Do not change
the generator or method after inspecting them without retiring that claim.

Reuse the fixed passenger network and pinned public OSM context. Extend the
existing twelve-session family with a rare-POI route, a nearby-destination route,
and eight independently simulated calendar-day trips. Five of the six historical
days use the routine destination, one uses an alternative; two query days cover
one routine and one rare destination. These frequencies are deliberately
synthetic challenge controls, not empirical human behavior. Overnight activity
and return trips are outside the simulated observation windows.

Rare POI means category share <=5% among the six retained public OSM categories,
and an observed FCD point within 150 m of such a POI. Rarity is catalogue category
frequency, not visit frequency or guaranteed sensitive meaning. Plan such a
route before SUMO; select the closest realized point after SUMO. Record failures.

Improve scarce base cases via public route-design constraints: single-exit
origin; a nearby shortened destination; an alternative continuation sufficiently
separated from the base endpoint. Realized case eligibility is still checked
after SUMO. No trajectory coordinates are fabricated/interpolated or selectively
edited. Do not rerun/drop a family based on defender/attacker scores. Keep all
30 subcases in the denominator, including any ungenerated ones.

Pre-protection data QA correction: the first generation exposed two 13--15 m
native FCD displacements at a one-second off-road parking waypoint in family
102. Make only these route-enforcing one-second stops in-lane (`parking=false`),
not actual scheduled stationary visits. Regenerate every family with the same
seeds, without editing any coordinates. Preserve the failed preflight bundle
locally for diagnosis; no defender results had been computed at this point.
Further gate audit found that two S5.C pairs skip a short immediate successor
between 1 Hz samples. Reject those pairs for the immediate-next-edge task, rather
than relabel the next observed edge as adjacent. Other valid task views of their
unchanged trajectories remain available. Completed trips may also cross short
terminal edges after the final FCD sample; verify native route completion and
the <=1 second unsampled boundary, rather than fabricating an endpoint sample.

## Method hypothesis

The preceding point-anchor POI score can optimize the wrong neighborhood.
Test expected POI coverage under a finite-grid belief using protected-anchor
history and public context only. Model the noisy-reuse observation correctly:
an unchanged anchor can result from reuse OR a fresh draw at the same coordinate.
Compute the REM normalizer over the full original lane-state support, including
coordinate multiplicities. The latent-location grid and temporal transition are
approximations, not a claim of a calibrated true user posterior. No extra GPS
input or future route is allowed in postprocessing.

Keep B=.24/m, H=12, theta=200 m, offset=80 m, temperature=60 m; K=3/5.
Controls: old BR-lane and point-anchor POI coverage. New belief variants use
coverage weights 6 or 24, route weight 0, to isolate the utility-target change.
The new scorer can use the belief's center for its geometric goal only in an
explicit separately named variant; do not silently alter the control.

Screen on old development data if needed; final selection uses only new family
103/104 validation. Retain a fixed grid and all negative results. Selection:
every case mean Recall >=.90, then lowest macro Hit100; if no candidate qualifies,
mark infeasible and report best minimum-case Recall, then Hit (12-digit decision
rounding). One configuration per K shared across evaluated cases. No tuning to
reporting family outcomes. Extra hypotheses need a new declared cycle.

## Evaluation and reporting

Mechanism comparison covers all nine S1/S2/S3 subcases if data exists, at most
12 allowed events per record. Preserve truncation and interval information,
especially S2.C where the second visit may fall outside that prefix: do not
count a truncated one-visit observation as an evaluated return attack.
At least one replay seed per record/K/variant; increase replication only before
viewing reporting outcomes. Match anchor streams across variants. Reuse the
expanded public-track attacker set and the same directed POI service. Report
validation-selected attackers separately from exploratory envelopes.

The dataset may cover S4-S10 without those attacks having been evaluated by this
method experiment. Keep these coverage notions separate. Expose neither labels,
raw history/future nor evaluator anchors/beliefs to the LSP export. Verify raw
SUMO provenance, split isolation, every new gate, emission arithmetic on a small
exact oracle, prefix invariance, motion and POI responses; retain source hashes.

Deliver canonical LaTeX/PDF plus a detailed verifier and immutable artifacts.
The repository's reusable build/verify CLIs are the inspectable companion (no
duplicate notebook or unrelated HTML report). Tables support exact per-case
lookup; do not imply statistical significance from a tiny family count.

Literature: Chatzikokolakis et al., PoPETs 2017, final DOI
10.1515/popets-2017-0051 (not the earlier draft DOI 0035), remapping/postprocessing;
SUMO official vehicle/route documentation for explicit departures and routes.
