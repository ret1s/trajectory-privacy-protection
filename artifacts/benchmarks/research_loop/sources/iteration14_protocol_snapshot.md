# Algorithm improvement loop — started 24 September 2026

## Acceptance criteria (fixed before new method scores)

A thesis claim is ready only if all of the following hold:

1. Explicit online input/output and information boundary, a correct budget
   argument, and separate endpoint/metadata assumptions. No claim of invention
   for Geo-I, predictive reuse, greedy maximum coverage, or truncation itself.
2. S1/S2/S3/S9/S10 each has applicable A/B/C evidence. Whole-session service loss,
   suppressed requests, actual delay, preprocessing/storage/communication and
   generation time are measured; no denominator changes to make utility pass.
3. At matched privacy budget, K and service contract, a candidate has a meaningful
   supported gain over at least one strong matched baseline, with ablation that
   attributes the gain. Report all tradeoffs. Preserve the existing 90% per-case
   Recall gate, with failure explicitly marked; changing a service requirement
   requires a separately justified protocol, not retrospective threshold tuning.
4. Endpoint attacks must work on an unprotected control, include map/POI and
   repeated-site inference, and be selected outside test. Report multiple hit
   radii plus MAE, rather than declaring success from an insensitive Hit100=0.
5. New final confirmation uses unseen groups/sites. Existing fresh-switching
   confirmation is exposed and cannot be recycled as independent confirmation.
6. Comparison distinguishes faithful reproductions, transparent local adaptations,
   and inapplicable papers. A literature table is not experimental superiority.

These are operational research criteria, not a promise of degree acceptance or
universal privacy. A failed criterion leaves the corresponding claim open.

## Source recovery

- Downloaded BBBike Beijing source dated 20 September 2026.
- gzip SHA256: c7de4d7d52467820912d73f5edb34b0b21f0174d80fa4e03160c67154f8f7a4b
- XML SHA256: 1a1e98a335da912b6c2551af13164290afe2bab37619fa5343b7b3eab2264e10
- Source differs from historical source. New network has 102,404 lane states,
  110,945 arcs; catalogue SHA256 cc0ba4ada19cc0abf3469e0cae33c5d97a756300c43931f51b14cdc112f6f573.
- Use `cache/research_loop_20260924`, not historical cache paths. First development
  runs may replay old development GPS against the new public map, explicitly
  labeled as such. They are not a replay of the old SUMO benchmark.

## Iteration 1: exact reduction of service-equivalent candidates

Hypothesis: many reachable lane states return identical POI lists. Evaluating all
of them in every greedy/exchange step wastes work without changing service.
For each track, group feasible states by their identical ordered category/POI
signature and retain the minimum public `(movement, center_distance, state_id)`.
These signatures and costs depend only on public context and protected history.

For fixed groups, every removed state has identical marginal gain to its retained
representative for every partial selection. The representative wins the original
tie rule. Thus greedy choices and bounded exchanges are identical by induction.
Because the chosen states are identical, future reachable groups and transcript
are identical as well under a coupled anchor random stream. This preserves the
existing privacy/utility distribution; it claims only reduced computation.

Use ordered signatures to preserve floating-point summation order. Grouping only
by unordered sets would need separate handling of numerical ties. Four property/
integration tests compare exhaustive objectives, selected states, transcripts,
anchors and ledger. City latency must be measured before claiming a speedup.

## Research anchors

- Predictive private tests and budget management precede this thesis:
  https://arxiv.org/abs/1311.4008
- Optimal utility under Geo-I also precedes this thesis:
  https://arxiv.org/abs/1402.5029
- S-TT prior art directly treats sensitive endpoint truncation:
  https://link.springer.com/chapter/10.1007/978-3-031-35374-1_3

The proposed contribution must be the precise new combination/algorithmic
improvement and evidence under the stated service contract, not a renamed
instance of these established principles.

## Iteration 2: static-cache control (measured)

On 27 old development records replayed against the recovered map, 1,740 category
queries matched the reference exactly without remote location requests. The
public POI JSON is 57,585 bytes, precomputed index 12,698,096 bytes, and SUMO XML
20,074,934 bytes. Lookup mean 0.010 ms on this machine. These are separate size
components, not a measured compressed network payload. The map/context is already
needed by the original selector. This is a material service-model limitation:
static public POI retrieval alone does not justify sending perturbed positions.

Do not silently replace static POIs with a live-service task. A question about the
intended server-only information is pending. Until clarified, retain this control
and do not claim a privacy/utility advantage over it.

## Iteration 3: whole-session boundary development (fixed before scores)

New SUMO families 501/502 (train), 601/602 (validation), all 22 sessions/family.
These are development only, no final confirmation. Initial whole-session stress
uses base/repeat/access_endpoint roles. It is S9/S10 mechanism development, NOT
all A/B/C coverage. Query every 20 seconds from session start; no window-based
budget reset. Compare raw GPS, core H=12, core H=64, H64+head60, H64+tail60 and
H64+both60 at per-session budget .24, K=5. Two repeated sessions therefore compose
to .48; report this rather than claim a .24 person-level guarantee.

Delayed replies are evaluated at the user's CURRENT delivery-time position, with
missing deliveries scoring zero over all scheduled queries. No cache credit.
Record exact delay and generated/cancelled requests. Endpoint attacker uses only
public transcript (whole released session), centroid/track/extrapolation bank,
plus public-road nearest-state variants. Select on training families only; report
MAE and Hits 50/100/200/500 separately. Include raw GPS as positive attack control.
Strong learned, road-transition and repeated-site attacks still remain necessary.

## Iteration 4: prospective predictive budget filter

Implemented a separate fixed-cap filter; see `predictive_filter_argument.md`.
72 prior full-session rows remain immutable. New 48 runs used 13–18 private reads
instead of 12, but validation Recall stayed 81.86% and endpoint results matched.
Do not promote this as a measured utility improvement. The diagnostic revealed
1–4 distinct query sets across 33–41 validation events, despite new anchors.

## Iteration 5: service-equivalent directed progress (before scores)

Hypothesis: minimum-movement ties trap the myopic coverage selector in a service
region. For each step, first obtain the usual reachable coverage solution. Then
replace a selected state ONLY with a reachable state returning the identical
ordered POI signature. Choose the replacement that minimizes directed travel time
towards a global coverage plan computed from the protected belief. Match tracks
to plan goals using a public minimum-cost assignment. No private GPS, speed, true
endpoint or future trace enters this planning stage.

This preserves the complete selected POI union at that step conditional on the
same history, and preserves the privacy bound by postprocessing. It does not
preserve the next-step feasible sets, transcript distribution, or empirical attack
resistance. Measure a 2x2: fixed/filter ledger × stay/progress selection, plus a
progress/filter/boundary60 combination and raw control on the same full sessions.
Reject any claim of global optimality or trajectory-wide utility dominance.

Iteration 5 outcome: whole-session validation Recall rose from 81.86% to 83.79%
with progress alone and 84.58% with filter+progress. Filter alone stayed 81.86%,
so the observed benefit is an interaction with mobility of the query set. The
S10 selected-attacker MAE fell from 1,952 m to 1,648 m (a privacy disadvantage),
while Hits at 50/100/200/500 m stayed zero in this small finite attack test. Keep
this tradeoff; do not call it a Pareto improvement. Boundary60 still has only
83.60% deliveries, imposing a no-cache utility ceiling below the 90% requirement.

## Iteration 6: reply-depth control (before scores)

Re-evaluate the same iteration-5 transcripts with service reply depth L=5 and
L=10 while client reference/answer stays top-5. This does not change the defender
or privacy observations. Apply both depths to EVERY control, not only the proposed
method. Charge actual serialized POI-ID reply bytes, explicitly excluding HTTP
and full POI payload. Report missing deliveries as zero and keep all scheduled
current-position queries in the denominator. Deeper retrieval is a service-cost
tradeoff, not by itself an algorithmic contribution.

Iteration 6 outcome: filter+progress at L10 reaches mean Recall 90.57%, but worst
validation session is 87.71%; this tail diagnostic does not itself decide the 90% per-case gate, which is still untested for all cases. At matched L10, fixed
baseline mean is 89.18% and worst 85.53%. ID reply cost is about 4.51 kB/query,
versus 2.74 kB at L5. Do not pass the per-case requirement using the overall mean alone.

## Iteration 7: small bounded cover slack (before scores)

Test only slack 0.01 and 0.03, alongside zero-slack progress and fixed core. Permit
a directed move towards a protected-belief goal if the *total* step objective
remains at least the zero-slack value minus slack. This may escape a service-cell
boundary which equal signatures cannot cross. The bound is on an approximate
belief objective, not true Recall. Every proposed replacement remains reachable;
no raw position/future label enters the decision. Compare privacy as well as
utility; additional motion may expose more trajectory information. Apply L5/L10
to all variants, not selectively. This remains development on exposed validation
families; new final confirmation has not been used or claimed.

## Iteration 8: tighten the privacy-budget match (before scores)

The original B=.24/H12 controller actually has a tighter worst-case bound .23
(one initial release + 11 test/release pairs). Earlier filtered candidates respect
nominal .24 but can spend .24. They match the advertised ceiling, not that tighter
bound. Retain those results with this limitation; do not attribute all gain to
planning under exactly equal tight privacy bounds.

Add a separate matched-filter variant capped at 23 units with the SAME .01 unit
cost. It now has the same .23 worst-case guarantee as fixed H12. Re-evaluate zero
slack and .01/.03 slack, whole-session utility and endpoint attacks at L5/L10.
This check is required before selecting a candidate. Historical implementations
and artifacts remain unchanged.

## Iteration 9: include the latest switching-belief core (before scores)

The full-session ablations so far used the simpler one-mode belief to isolate
planning and ledger effects. The existing method also has a stopped/moving
switching belief. Include its unchanged behavior (using the exact quotient
acceleration) and the matched-cap filter/progress integration. Compare both to
the prior candidate and fixed simple core on the SAME full sessions, seeds,
K5, tight bound .23 and L5/L10. Do not claim an advance over the latest method
from a comparison only to an older ablation. Two integration tests passed.

## Iteration 10: exact A/B/C views of a persistent defended session (before scores)

Use every record for S1/S2/S3/S9/S10 in the new development dataset. For each
physical session, fix an evaluation request clock: the union of a 20-second
background clock and the exact sample timestamps required by those records.
Generate one continuous mechanism state from the real session start, with no
record/window reset. Reuse those outputs for all views of that session. This
clock is specified by the benchmark using its selected samples, so the privacy
claim is explicitly CONDITIONAL ON THAT FIXED REQUEST CLOCK; it does not protect
the times at which the benchmark chose to query or membership of a scenario.
It is a different request workload from iterations 3–9; do not mix their Recall.
The online defender receives only the current time/GPS, never the full schedule,
case, target, future or route. No schedule decision is credited to the defender.

Publish only each record's allowed events, rebased to its first allowed time,
with fresh local event numbering. Prior protected states remain internal. Measure
Recall L5/L10 on both the case queries and ALL session queries. A pair shares two
session budgets, .46 in total; overlapping record views are not independent trips.
Raw, the existing switching core, matched filter/progress and its switching
variant get identical clocks and paired random seeds; use two RNG repetitions.
Select geometric/road-projection attacks on families 501/502, separately for MAE
and each hit radius, then evaluate 601/602. For repeated-site C, also fuse estimates
across both disclosed trips. This finite attack test does not satisfy the stronger
shadow-learning/directed-road endpoint requirement; retain it as an open criterion.

## Iteration 11: learned inversion of the first public query set (before scores)

S9 full-session testing must include a mechanism-aware attacker rather than only
centroids/extrapolation. Generate 2,000 synthetic single-query shadows by drawing
true positions from a public 120 m cell-balanced road prior and running the exact
first-step mechanism (epsilon_release=.01, K5). This first step is shared by the
simple/switching and fixed/filtered candidates. It trains on public simulations,
not validation endpoint labels. Fixed kNN (1/5/15), loss-aware neighbor actions
(15/45), and ExtraTrees direct/residual estimators use only the five public
coordinates. Select the estimator on development families 501/502; score the same
full-session validation families 601/602. Include the previous attack bank when
selecting so increasing attack capability cannot weaken its training objective.
No claim of exact Bayesian inversion, optimal attacker, S10 protection or new
independent defender test is implied. Persist shadow inputs/outputs and provenance.

## Iteration 12: public-clock early privacy allocation (before scores)

Prior turn classified as progress: 11 completed experiments revealed both an
S9 learned-inverse counterexample and incomplete case utility. The unresolved
server-only service definition prevents final service claims, but testing an
origin privacy response is independent useful work.

Compare the matched filter/progress control, an origin-first guard and an
origin-60-second guard, plus raw. Early release/test epsilon is one quarter
(.0025 instead of .01). The primitive remains full-support ideal REM with a
private noisy reuse test, NOT planar Laplace noise. Set an integer-quarter-unit
cap of 92 (.23 total); reserve the worst next-step cost before private reads.
Do not spend leftover units without reservation. This may leave a tighter
realized bound than .23; report the ledger. The change responds to first-query
S9 inversion and is not a new DP primitive.

Keep private anchor history across the public phase change, and switch the
belief emission normalizer to match actual release/test parameters. Shared
public state grid/prior/POI objective and causal directed progress remain fixed.
Use the same 12 whole-session sources and paired RNG as iteration 8 at L5/L10;
these are exposed development families, never final confirmation. Train a NEW
2,000-sample public-road shadow inverse on the guarded first step; do not report
privacy against a learner trained only on the old epsilon. Keep full-session
geometric attacks too: a later query may reveal the origin even if the first
query is harder to invert. Four phase/budget/emission/causality tests passed.

## Iteration 13: spread private reads over public time (before scores)

The origin-guard experiment lowered full-session utility. Independently, late
rare-POI queries often arrive after all privacy reads are spent. Test one fixed
60-second minimum private-read interval; every scheduled public query is still
answered by the protected-belief planner, with no extra raw-GPS decision between
reads. Compare unpaced filter/progress, paced filter/progress, and paced guarded
first release, plus the existing switching core and raw. Do not try many spacings
and select on validation. A zero-interval test reproduces the unpaced candidate.

Use ALL exact A/B/C persistent-session clocks of iteration 10, same paired RNG
and two repetitions. The clock, reference eligibility, K5 and L5/L10 are fixed.
A new shadow first-query inverse is already trained for guarded initialization;
ordinary pacing leaves the first query unchanged and therefore inherits the
known S9 first-query weakness. This is a budget/planning ablation, not evidence
that throttling private observations is a novel primitive.

The extended-transcript proof is unchanged: between-read decisions are public
postprocessing; at read times reserve/charge the actual primitive epsilon under
the same .23 ideal-kernel cap. Elapsed time is not used to reset or replenish
the privacy budget. Long sessions can still exhaust it.

## Iteration 14: broader SUMO shadow routes for sequential attackers (before scores)

The existing first-query shadow identifies an S9 weakness but cannot validate
S3 or infer S10 from a long public sequence. Generate 80 auxiliary route families
(1201–1280), each with a cruise trip and a returning/stopping trip, on the recovered
network. Plan using only the public network; do NOT consult core benchmark GPS,
labels, families or scores. Reuse the existing public-map route generator with
its distinct departure-cell and stop-site constraints. First 64 auxiliary
families train attackers; last 16 select attacker configurations. These are
attacker background data, not independent confirmation of defender quality.
Check native SUMO completion; do not silently discard failed routes.

Defender variants to be simulated are the unpaced matched filter/progress,
guarded-first-release, paced unguarded and paced guarded models. Query every
20 seconds from each session start; preserve whole-session mechanism state.
Train fixed kNN and direct/residual ExtraTrees on current public coordinates
plus protected-output history. For origin/destination inference use only
whitelisted public sequence summaries, with both full and masked windows.
Retain family separation, raw positive controls and calibration measurements
on auxiliary holdout before evaluating exposed core development records.
No classifier may assume that one of the K dummy-only points is the real user.
