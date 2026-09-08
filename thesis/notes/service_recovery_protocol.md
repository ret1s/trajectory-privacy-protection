# Service-recovery diagnostic and development protocol — 2026-09-08

Parent evidence: commit 28ef61e. The preceding service-cover cycle failed
confirmation utility. All its sources, outputs and DB releases remain frozen.
This cycle first diagnoses that failure, then implements a bounded candidate.
No result from families 201–204 is a new held-out confirmation: they are now
development diagnostics, without rewriting their historical v3 split labels.

## First diagnostic (fixed before computation)

Use every S1–S3 record in v3 families 201–204, K=5 and RNG replicate 1 from the
frozen service-cover artifact. Keep each event's preceding published dummy
states fixed. Compare learned-belief coverage, protected-anchor coverage and
true-location coverage with and without the directed reachable-set constraint.
The latter two interventions change one-step decisions, not complete trajectories.
True-location coverage is evaluator-only and cannot be a protection method.
Greedy true-location coverage is a diagnostic reference, not a proven optimum.

Also compare the learned location belief with a uniform-public-cell belief,
using the identical anchor emission and approximate transition construction.
Report belief mean error and POI recall. Uniform refers to occupied spatial
cells, not lane-state multiplicity. Record per-event values and aggregate
equally over cases/families. These interventions can separate local objective
effects from feasibility at a frozen history; they cannot identify the global
effect of an alternative entire trajectory or a unique cause of failure.

## Development boundary

The candidate must use public context and protected anchors only, preserve the
existing raw-input anchor/ledger, and enforce directed output motion. No access
to hidden future positions, identities, labels, query truth or raw GPS in a
postprocessor. Do not change metric definitions or discard difficult cases.

After diagnosis, write a separate candidate design/ablation declaration before
scoring its full trajectories. Candidate selection is development-only unless
a new confirmation set and locked evaluation protocol are actually completed.
Prefer a small mechanism change with explicit controls over a large parameter
search. Keep the old default if the evidence does not justify promotion.

## Deliverables and validation

Save diagnostic records, method outputs, independent verification, unit tests,
literature critique, and canonical LaTeX/PDF. Use executable experiment modules
for reproducibility, not an additional notebook or HTML reporting surface.
The chosen primary artifact is the existing thesis. No new dashboard/chart is
needed for a small exact comparison table. Distinguish numerical tests from
proof, empirical Hit from a calibrated optimal adversary, and data provenance
from realistic population coverage. Literature review is targeted, not systematic.
