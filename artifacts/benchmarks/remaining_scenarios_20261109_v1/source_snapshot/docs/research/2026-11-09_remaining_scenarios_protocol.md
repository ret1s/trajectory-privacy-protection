# S4–S8: frozen-output supplement, 09/11/2026

Keep GeoI-BQ v1, its 108 release pins and every protected output unchanged.
Compare all six arms of `paper_comparison_20261028_v1`; do not regenerate GPS,
Q, REM anchors or comparator draws. This is a controlled extension on the
previously inspected 48-family synthetic cohort, not untouched confirmation.
Freeze this protocol, evaluator, tests and feature code before scoring.

## Shared evaluation

Use the retained 20 TRAIN / 8 SELECTION / 20 TEST split, two distinct native
sessions per family, one train/selection draw and three test draws. Fit attackers
on TRAIN; choose per arm/task on SELECTION using group mean−SE accuracy (or
mean+SE MAE); use log loss then lexical ties for classification. Archive the
selection before TEST predictions. Keep all bank predictions and failures.
Bootstrap 10,000 times at the independent group level, public seed20261109.
Do not interpret a value below chance or a zero-hit bootstrap as a guarantee.

Features receive only public coordinates/timestamps and, for S5/S6, the two
public candidate geometries. No family/person/vehicle ID, private chosen edge,
routine label, native future point or evaluator slot enters feature APIs.
All fake packets up to the cutoff remain visible; nothing after the cutoff is
read by an online feature. Native truth/route IDs are labels only.

## S4 — session association proxy, not measured person/vehicle identity

The current cohort has route-family membership, not independent human and
physical-vehicle identity labels. Score whether day7/day8 sessions belong to
the same synthetic route family. Never rename this result person re-ID or
vehicle re-ID. Report those two targets N/A with the missing-label reason.

Within each split, pair successive sorted families into disjoint two-family
blocks. For each draw, retain two positive within-family day7→day8 pairs and
two crossed negative pairs. Do not count different draws of the same native
journey as independent positive sessions. All four pairs in a block stay
together in bootstrap: 10 TRAIN / 4 SELECTION / 10 TEST blocks. TEST has120
pairs, balanced labels. Negatives are a declared association workload, not
an exhaustive real-user candidate population.

Bank: symmetric translation-invariant prefix/pair features, ExtraTrees96,
logistic, kNN1/5, start-distance scores at2/10/25/50/100/200/500/1000m and
uniform. Select thresholds .1/.25/.5/.75/.9 on SELECTION. Report balanced
accuracy, ROC-AUC, macro-F1, log loss and CI; include chance50%.

## S5 — future external-edge choice from a public prefix

Use public `shared_fork_t` and `turn_visible_t` from counterfactual native
calibration, fixed before private route choices. Observe retained packets
with timestamp≤cut, not an extra GPS at the cutoff. The selected next external
edge must first occur strictly after the cutoff in native truth. Both public
edge alternatives/turn shapes are allowed auxiliary information. Score exact
candidate-edge accuracy, balanced accuracy, Brier and log loss. This is a
two-choice fork task, not unrestricted next-route prediction.

The 20s emission grid can miss a short turn window. Retain every family; report
the last genuinely observed raw-packet phase and the fraction already showing
the turn. Raw at chance before the fork is intrinsic ambiguity, not a defense
contribution. Never cut at a secret label-dependent point or remove poor cases.

## S6 — future destination from public prefixes

Use the same pre-fork/turn cutoffs plus `ceil(turn_visible_t/20)*20` for a
post-turn observation phase. Native destination arrival must remain in the
future. Both candidate destinations are public; the true destination/choice
stays evaluator-only. No six-trip protected-history claim: current sealed
outputs cover only query days7/8. Fit geometric/candidate ExtraTrees96 and
kNN1/5, with uniform control; report candidate accuracy, Hit100/MAE destination,
Brier, log loss and family CI. S5/S6 labels are balanced across the two days.

## S7 — conditional payload identifiability audit

For each retained stream, hold GPS/Q/clock/provider realization fixed and vary
four private purposes × six categories. Exercise the actual `SessionCoverClient`
request interface with a deterministic empty-response probe; it is a contract
test, not a new POI utility trial. Archive hashes of actual generated requests
and of the frozen reply-mask signatures. Replies under a fixed provider law
cannot add direct payload information when requests coincide.

Report equal-wire rate and empirical conditional Bayes vulnerability: for
balanced counterfactuals, purpose1/4, category1/6 and joint1/24 when wire is equal.
An explicit-private-fields control must recover all labels. These are exact
conditional audit quantities, not trained natural-intent accuracy. Apply the
same wrapper to all six methods; this is a shared interface property, not a
unique advantage over paper originals. Route-correlated intent, account,
activation timing and human query distributions remain unmeasured.

## S8 — aligned-session correlated-channel diagnostic

Use distinct day7/day8 tapes from the same family/draw, align their already
relative public clocks, and cut at `20*floor(shared_fork_t/20)`. Native GPS must
match throughout the measured prefix. This is a synthetic perfectly correlated
shared-prefix disclosure model; the sessions did not run simultaneously in
SUMO. Do not call it real companionship or companion detection.

Target: GPS at that cutoff. Four views: target-only; target + protected aligned
partner; target + explicitly granted raw partner; target + protected unrelated
partner (the other family in the disjoint block). The association is granted,
not learned; raw partner GPS is explicitly public in that control. Count all
observed packets/coordinates, including fake events and both channels.

Bank: public prefix geometry/velocity plus ExtraTrees64/kNN1/5/prior regressors;
joint views include target-only estimators so the attacker can ignore auxiliary
information. Fit/select each view separately. Report Hit100/MAE, paired changes
from target-only and CI bootstrapped by the same two-family blocks (needed for
unrelated partners). Raw-partner control should recover the target because of
the declared exact shared-prefix correlation. Single-session Q bounds do not
automatically certify the joint observation or raw external side information.

## Completion and verification

Keep all declared arms/records and failures. Verify inputs/source hashes,
group-disjoint splits, future-edge/destination eligibility, strict prefix
causality, no evaluator fields in features, complete fake-packet accounting,
actual common-wrapper request equality, raw controls and independent metric/CI
recomputation from saved predictions. No proposed-method winning gate.
Store new evidence in `artifacts/benchmarks/remaining_scenarios_20261109_v1/`;
old artifacts and report remain intact. Provide a separate dated report addendum
with scope labels for every measured and unmeasured target.
