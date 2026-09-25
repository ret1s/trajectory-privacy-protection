# Algorithm improvement loop — started 24 September 2026

## Iteration 28: basic live-availability service, frozen before scoring

The user delegated selection of a basic service. Choose nearest currently
available POIs on the existing public directed road map. This is a new synthetic
service workload, not a repair or relabeling of historical static-service results.
The server owns current availability; a point request returns the top L=10
available POIs per category, and the client locally ranks top five. Keep original
coordinates, full-session clocks, budgets, GPS datasets and all 15 A/B/C cases.

Freeze 60-second availability epochs, nominal p=.8, sensitivity p={.5,.8,.95},
and three evaluator-only world seeds in `iteration28_protocol.json` BEFORE
scoring. Worlds are shared across methods and independent of target movements;
no seed selection. Do not force an available POI in rare categories; retain empty
reference denominators. This is synthetic availability, not real business data.

Replay both expanded paced parents and public fixed K5/K12. Compare current
responses with a causal client union cache cleared on each availability epoch.
The cache changes neither request coordinates nor timing; it can only improve
current-epoch union Recall, and fixed queries should gain nothing from it.
This conventional caching ablation is not a novel privacy primitive. Responses
are from a point API: the planner is not given the full current status vector.

Retain static local catalogue, stale epoch-zero full-status cache, raw current
query, and a live full-status bitmap download as controls. The bitmap is an
alternative bulk API, only 53 payload bytes per refresh for 419 known POIs; do
not assert that a remote point-query method beats it or that bulk data are costly
by assumption. Report actual payload sizes and separately identify protocol
headers, refresh frequency, and unknown network latency. Existing coordinate
attacks transfer only under the stated independent-status, unchanged-transcript
assumptions. Availability worlds and deterministic service replays are not new
stochastic protection runs, SUMO journeys, or independent confirmation.

## Iteration 27: public service supplement at an explicit query cost

The previous goal turn made progress: iterations25/26 rejected new candidates
with all-case screens and a matched first-query privacy counterexample. Avoid
another optimizer sweep on the same S1.C labels. Instead quantify the cost of
retaining all five existing adaptive queries and appending a public fixed cover.
Freeze supplement counts B={1,2,3}, giving total K={6,7,8}, for BOTH existing
expanded response-paced/slack methods. Plans use only the original public prior,
map and service; no fitting to failure locations or scenario labels.

Unlike iteration23, the five adaptive outputs and their future state are
unchanged. Fixed tracks are explicitly public and identifiable. Deleting them
recovers the original transcript exactly; appending the public constants recovers
the hybrid. Thus coordinate information is equivalent conditional on the public
region/catalogue/clock. Existing attacks transfer by this exact inverse; no claim
that extra dummies improve privacy. The service union can only grow, at an explicit
additional query/reply cost. These elementary composition properties are not a
new privacy primitive, and do not cover timing or response-side metadata.

Evaluate every expanded S1/S2/S3/S9/S10 A/B/C record using the saved full-session
transcripts. This deterministic lifting is not a new stochastic defender run,
SUMO trip or confirmation dataset. Implement a causal online wrapper and test
equivalence with lifted outputs. Compare GPS-independent fixed-query controls
at the SAME total K6/7/8, keeping the existing public K12 and static-cache controls.
Report all B values, not only the cheapest successful one after viewing scores.
Keep the 90% case gate and denominator unchanged; report original learned attacks
as exactly inherited, including S9.C counterexamples and S10.B raw weakness.
This is a conditional protocol cost audit while the server-only workload question
remains unanswered; it does not establish service necessity or thesis readiness.

## Iteration 26: standard planar-Laplace anchor ablation

While iteration25 runs, prepare an independent primitive comparison. The current
anchor uses REM logits -epsilon*d/2 on the full road set with an input-dependent
normalizer. Compare standard full-plane planar Laplace density proportional to
exp(-epsilon*d), with radius Gamma(2,1/epsilon), under the same ideal epsilon
bound and the existing private reuse test. Use the fixed public XY projection;
do not use a latitude-dependent degrees conversion, clip noise, snap private
inputs, or truncate support around the true location.

Keep the original response-aware belief grid/transition, K5, L10, public pacing,
23 budget units and the two fixed slack settings 0/.03. Change the belief emission
to the matching continuous mixture: a reuse atom at the previous anchor and a
fresh continuous density elsewhere. The finite-grid belief remains approximate.
Compare all 15 core cases and identical persistent query clocks; anchors and
realized branch ledgers need not match REM. Do not transplant REM-trained attacks
or equate a shared upper privacy bound with identical empirical privacy.

This is a primitive ablation based on CCS2013/PETS2014, not a faithful reproduction
of their full service protocol or budget policies, and not a new noise mechanism.
Tests cover the radial sampler contract, atom/density distinction, numerical
likelihood bounds, prefix causality, physical outputs and no reads after stopping.
Full execution uses floating-point random generators, so ideal-kernel privacy
does not certify exact pure DP of the executable. This comparison is development;
runtime recorded under overlapping jobs cannot establish comparative speed.

Core kernel screen is complete: planar variants pass 14/15 and 13/15, still
fail S1.C; slack shows a geometric S10.A hit50 counterexample (1/2 RNG rows,
one eligible family). Do not promote either. Add a first-query privacy challenge
as a bounded continuation: reuse the exact 2,000 public training locations and
fixed learner bank from iteration16, regenerate outputs with the planar kernel,
and select on first queries from the 16 existing auxiliary-selection families.
Score all core first-query origins, both RNGs, with raw truth as positive control.
This is an explicitly full-session first-query probe, NOT S9.A/B/C masked-window
evidence. No attacker tuning on the core labels and no reuse of REM-trained fit.

## Iteration 25: saturate service per plausible location

Previous goal turn made progress: fixed-backbone ablation completed and a new
auxiliary-selected repeated-site attack produced a S9.C counterexample. The
backbone's S1.C gain costs S1.A/B utility, so do not sweep the fixed split until
one favorable case wins. Test the objective instead, retaining all five adaptive
tracks, original response-aware belief, directed motion, L10 and .23 cap.

Let R_i(A) be the client's macro-category top-five Recall predicted for public
latent state i from the union of replies at selected query states A. Replace
E_b[R_i(A)] by E_b[min(R_i(A), .9)]. Saturation occurs separately per plausible
location BEFORE expectation, unlike the existing per-category saturation of an
already averaged objective. The cap .9 comes from the fixed service threshold,
not a search on case scores. It is still an expectation under an approximate
belief: no guarantee of per-user Recall, robust worst-case loss, or privacy gain.

The weighted capped-coverage objective remains monotone submodular; this is an
application of established optimization, not a new greedy theorem. Service
equivalence and directed progress remain valid when a move preserves the entire
reply set. Implementation must compute true union coverage (no duplicate POI
counting), use all latent mass, and verify marginal gains against exhaustive
small examples. A disabled-cap boundary must reproduce the parent exactly.

Freeze cap=.9 with slack 0/.03. Profile the exact sparse implementation before
running all 15 core cases and paired persistent clocks; report preprocessing
separately from step time. Do not prune small posterior masses or relax graph
constraints to improve timings. These are development results, followed by
matched attacks and expanded testing only if the evidence supports promotion.

Engineering profile: the direct sparse implementation takes roughly .7 s/step
on the 12-event auxiliary prefix. Its 2,073 reference-utility rows contain only
458 distinct rows. Before case scoring, implement exact grouping of equal rows,
sum their current belief mass and apply the same cap. No mass is discarded;
this is a real-arithmetic identity, not a different objective. Check numerical
agreement and paired prefix outputs; disclose possible floating-point tie
changes rather than assuming whole-transcript bitwise equivalence. Preserve
the initial profile and its pre-hook engine source under `sources/`.

## Iteration 24: prior-corrected repeated-site density attack

While the frozen iteration23 screen runs, strengthen S9.C/S10.C evaluation of
the existing expanded response-paced/slack shortlist and raw controls. Existing
joint-mean regressors can discard multimodal evidence. Fit nearest-neighbor KDE
scores from each mechanism's existing auxiliary-only endpoint training arrays;
use 64 feature neighbors, spatial bandwidths {100,250,500} metres, the fixed
2,073 public grid points, and .05 mixing with the training-target spatial prior.
This is an empirical density score, not an exact/calibrated posterior.

For two permitted same-site views, compare arithmetic density pooling with
the normalized product divided by the prior (avoid counting the prior twice).
Conditional independence is an approximation; both views may remain correlated.
Select attacker/decision separately per metric on the same 16 auxiliary-selection
families, including the old attack bank. Keep old-only selections as controls;
then score all 12 expanded S9.C and S10.C records, both defender RNGs. No tuning
on expanded labels or pooling unrelated sites. Public-grid MAP, weighted mean,
finite-iteration geometric median and grid hit-mass decisions are available.
Report all regressions and prior-only performance. Raw controls must be present.
This adds repeated-site inference, not a full directed-road/semantic attacker,
and does not score the changed iteration20/21/23 mechanisms without matched fit.

## Iteration 23: public backbone plus adaptive residual coverage

The previous goal turn made progress: iterations 20–22 implemented/validated
auxiliary mobility, lookahead, and a GPS-independent fixed-query cost frontier.
No thesis-readiness claim. K5 public coverage does better on expanded S1.C than
the K5 adaptive shortlist but worse on the other 14 cases. Test whether a small
permanent public coverage backbone prevents adaptation from abandoning useful
service support when its approximate belief is wrong.

Keep TOTAL K=5, L10 replies, top-five reference, the same .23 cap, fixed read
pacing, original response-aware belief and road constraints. Fit two stationary
public query locations from the public prior only; allocate three moving queries
to residual POI coverage not already supplied by the fixed two. Greedy/exchange
and directed service-equivalent progress operate on the residual objective;
the two stationary tracks are explicit public knowledge, not secret decoys.
The selection objective is still approximate expected utility. No claim that
two is optimal; freeze this split before case scores, with no split-size sweep.

Run two ablations, slack 0 and .03, on ALL 15 core case views with paired anchors
and persistent whole-session clocks. Replay raw/response-paced/slack controls
exactly. A zero-backbone boundary must exactly reproduce the parent. The public
backbone adds no private read; adaptive transcripts still require matched learned
attackers (which may discard the known static tracks). If the utility screen
does not justify promotion, retain the negative result and do not cherry-pick
S1.C or claim privacy from misleading the geometric centroid attacker.

## Iteration 22: fixed public remote-query control

After the iteration 21 pilot still fails S1.C, audit the value of personalization.
An exploratory K5 fixed public cover has ALREADY been inspected on expanded
S1.C (81.94%); do not call this untouched preregistration or confirmation. Freeze
the same public-only selector now and evaluate ALL 173 expanded target records
and all 102 source-session clocks. Fit from the public grid prior, top-5 target
weights, top-10 replies and the same largest road SCC, never from target traces.
Three bounded exchange passes, identical tie rule to the quotient selector.
All sessions reuse the same stationary query coordinates; zero private reads.

Report K in {1,3,5,8,12}. K5 is the matched-cost control. Other K values are a
bandwidth frontier, not an algorithmic improvement at equal cost. Give per-case
and whole-session utility, queries and returned POI slots; retain empty-reference
eligibility. No RNG repetitions needed for this deterministic protocol. Its
coordinate transcript is independent of GPS conditional on the public request
clock/region; this says nothing about identifying a person from timestamps or
session boundaries. Static cache remains the stronger no-query control under
the present static-public service. Do not invent a live server workload.

## Iteration 21: two-slice reachable service planning (before pilot scores)

Iteration 20 improves auxiliary transition NLL (2.5297 → 2.1961), and core
S1.C without slack (74.17% → 77.50%), but still fails that gate. Do not interpret
forecast quality as service success. Test an explicit current/future service
objective under the SAME directed graph, K5/L10, .23 cap and auxiliary-selected
mobility model. Forecast H=120 s from protected belief only; never use true
future positions, destination, scenario label or future query times.

For each track, form a finite pool of feasible pairs (current output, forecast
output) along shortest directed paths to the current and forecast global cover
goals. Include the legacy current choice and stay continuations. Greedy/exchange
selects one pair per track to maximize the equally weighted union coverage at
the two slices. Only the current coordinates are sent; replan at the next query.
Accept only if the current protected-belief objective is at least the legacy
choice minus fixed slack δ, otherwise retain the legacy output. This floor is
NOT a bound on true-user Recall. The finite path pool is not all feasible paths.

Pilot all four core S1.C source sessions, both RNG repetitions and their exact
whole-session clocks. Three fixed ablations: forecast with δ=0; forecast with
δ=.03; no belief forecast (future weights equal current) with δ=.03. H=0 must
exactly replay the parent and is a test-only boundary control. No tuning H or δ
after this pilot. This is mechanism diagnosis, not full A/B/C or learned privacy
confirmation. Follow with all-case testing if evidence justifies continuation.

Predictive privacy control already exists: Molina et al. IFAC 2023 and Control
Engineering Practice 2025. Do not claim invention of MPC or look-ahead privacy.
The implementation tests finite-road multi-query service-union planning as
postprocessing of an accounted private history; novelty priority is unproven.

## Iteration 20: public auxiliary mobility prior (before model selection)

The preceding turn made progress: completed frozen expanded screening, matched
attackers and reserve-read experiments. No final acceptance or blocked audit.
Inspection finds all 12 expanded S1.C targets inside the permitted largest SCC;
simply dropping that constraint is not justified. Existing prediction uses a
coarse isotropic, high-stay transition. Test a learned continuous-time Markov
generator on the same 2,073 public latent cells. No output road constraints,
query cadence, privacy primitive, epsilon, cap or POI metric may change.

Fit state dwell exposures and off-diagonal transition counts ONLY on the 64
`auxiliary_train` families, from their native 1-second FCD. Smooth each row with
the existing public diffusion generator `(P_20-I)/20`, using pseudo-exposure
τ0 in the predeclared set {20,100,500} seconds. Select τ0 by mean per-family
20-second transition negative log likelihood on the 16 `auxiliary_selection`
families. Compare that likelihood with the original P_20; record all choices.
The public occupancy prior and exact anchor emission remain unchanged. This is
an approximate mobility prior, not a posterior-calibration or privacy theorem.

If the selected public model improves auxiliary likelihood, integrate it with
response-aware paced progress, with and without fixed .03 slack, and evaluate
all core A/B/C views under paired anchors. These are development results.
Changed output requires new matched attackers before making a privacy claim.
Do not select transition parameters using S1.C or other defender case scores.

Markov/HMM mobility and predictive reuse are established prior research, e.g.
[Xiao–Xiong](https://arxiv.org/abs/1410.5919) and
[Chatzikokolakis et al.](https://arxiv.org/abs/1311.4008). The candidate contribution
being tested is the mobility prior's interaction with constrained multi-query
service planning, not invention of a Markov model or temporal-correlation DP.

## Iteration 19: reserve-aware read pacing (fixed before candidate scores)

Expanded screening confirms the rare-POI S1.C deficit in all four candidates.
Inspection of rep-0 slack runs found late queries up to 1,429 seconds but the
last private read as early as 780 seconds. Some other failing queries occur
before exhaustion, so allocation alone cannot be assumed to solve S1.C.

Test one fixed scheduling rule with the same epsilon units, .23 cap, K5/L10,
and unchanged request/delivery clock. After a private read, use interval
`60 * max_units / max(1, remaining_units)` seconds instead of constant 60.
The remaining ledger comes only from the already-accounted protected branch
history. No true velocity, final trip duration, scenario label or future GPS is
consulted. It is allocation over time, not a new Geo-I primitive or an epsilon
increase. Queries between private reads still receive all K outputs.

Two ablations: response-aware progress, and the same with fixed .03 slack.
Replay raw/fixed-pacing/fixed-pacing+slack controls from iteration 17 exactly;
run all core case clocks and both RNG repetitions. Keep both train and exposed
validation labels unchanged. A changed public transcript requires new matched
learners before any privacy improvement claim. First screen utility and report
every case, including regressions. There is no promise that this rule succeeds;
do not tune its strength after reading scores in this iteration.

The core screen finished: reserve pacing has S1.C 73.33%, reserve+slack 78.33%,
versus 74.17% and 82.50% for their fixed-pacing controls. Do not promote either
as an improvement. Core validation S1.C occurs at 530/643 seconds, before the
longest expanded failures, so this does not test the late-budget hypothesis
fully. Follow up with **all 12 expanded S1.C source sessions**, both RNGs and
both unchanged candidates, using the exact full-session query clocks already
stored in iteration 18. Keep every family, not only long or favorable ones.
This 48-execution diagnostic is a continuation of iteration 19, not a new
confirmation or matched learned-privacy result.

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

Iteration 14 implementation clarification before any fitted sequence-attack
scores: select S1 attackers on isolated auxiliary snapshots, separately from
full/late current-location windows. Select A endpoints on matching 60-second
masks, S10.B on prefixes, and C on two linked trips; retain all estimators and
unfused alternatives. The fixed forest uses 128 trees, leaf size 5, depth 18,
and `max_features=1.0` (all feature dimensions, not integer 1 / a single feature).
No search over these settings. Auxiliary clocks are 20 s; exact core-case clocks
have additional benchmark-selected events, so temporal/domain mismatch remains
an explicit limitation. Road projection is not a directed-transition attack.

## Iteration 15: align the selector with the existing reply contract (before scores)

All L10 comparisons already allow each query to return ten POIs/category and
score recovery of the true top five. The current planner still optimizes the
union of top-five responses. Test an explicit separation: keep expected true
reference weights computed from the protected belief and top-five reference,
but optimize the union of actual top-ten replies. This is objective alignment,
not extra bandwidth, not a new DP primitive, and not a claim to invent maximum
coverage. The L10 service contract, K5, epsilon/cap, public clock, true reference,
eligibility and per-case 90% gate remain unchanged. Report L5 too as a mismatch
ablation, never apply a top-ten optimizer's result to an L5-only service claim.

Implement an immutable public model view with the same latent grid/emission/
transition/prior and top-five target weights, but deeper public reply signatures.
Test exact expected-Recall objective on a small fixture and identity when reply
and reference depth match. Test whole-session causality, budget and directed
reachability. Compare unpaced and 60-second paced response-aware planners on all
56 existing A/B/C records with paired seeds. Existing controls may be replayed
from immutable iteration 13, explicitly counted as replay rather than rerun.
These are exposed development families. If utility improves, its changed outputs
need newly trained mechanism-matched attackers before any privacy claim.

Before iteration 15 starts, include a third fixed interaction ablation: paced
response-aware planning with the already tested quarter-epsilon first release.
Keep that guard unchanged; do not tune a new epsilon from validation. This checks
whether objective alignment recovers any of the guarded candidate's service loss.

Iteration 14 follow-up, before new raw scores: the first raw positive control
uses geometric/road-snap attacks, whereas protected endpoints additionally face
learned sequence attacks. Retain that evidence and train the SAME fixed empirical
endpoint learners on raw auxiliary traces too. Duplicate the single public raw
coordinate into five identical feature slots only to reuse the feature shape;
this supplies no extra information. Use the same 64/16 auxiliary family split,
matching masks and pair selection. Store this as a separate raw-control audit,
without altering defender outputs or earlier selected attacks. An intrinsically
ambiguous prefix may still have low raw Hit100; do not call that protection.

## Iteration 16: mechanism-matched attacks for top-L selection (before scores)

Previous continuation made concrete progress (iterations 12–15, not a blocked
turn). The server-only workload question remains unresolved, but testing changed
public outputs and diagnosing S1.C are independent useful work.

Generate two top-L selection variants, unpaced and paced, on the SAME 80 auxiliary
families of iteration 14, using its public 20-second request clock and paired
RNG schema. Preserve 64 attacker-fit / 16 attacker-selection groups. Do not spend
another dataset as defender confirmation. Refit the same fixed public-feature
kNN/ExtraTrees/loss-aware bank; preserve mask/prefix/pair selection, raw controls
and all estimators. Score iteration-15 A/B/C and a separately labeled whole-session
endpoint probe. Those groups are still exposed defender development.

Also rerun the 2,000 public-road first-query simulation at the top-L selector:
first queries coincide for the unpaced/paced versions, so share only that exact
common first-step attack. Preserve the prior first-query test rather than letting
a weaker sequence model replace it. No new defender hyperparameter is selected
in this iteration. Diagnose S1.C using the protected-belief oracle cover and a
true-reference oracle only as evaluator upper-bound controls, never as deployed
methods or evidence of privacy. Keep K, L, targets and budget unchanged.

## Iteration 17: bounded planning loss to leave service plateaus (before scores)

Iteration-16 evaluator diagnosis found that the paced candidate's true-reference
reachable oracle equals its actual 75% mean S1.C Recall on validation. With those
previous public tracks and that final time gap, changing only the last-step
selector cannot reach 90%. Its unconstrained protected-belief goals would return
95% on the same cases, motivating movement earlier rather than extra private GPS.
This is conditional on the existing track history, not a universal impossibility.

Test one fixed combination: response-aware top-L objective + 60-second pacing +
the existing bounded-slack progress rule at slack .03. The slack rule was already
tried without pacing/response alignment; do not present it as a newly invented
primitive. It permits at most .03 loss in the CURRENT protected-belief objective
relative to that step's no-slack solution, to advance on directed paths toward
protected-belief goals. This is not a true-Recall guarantee or whole-run bound.
No real target, oracle solution, future trace or case label may enter the planner.

Use the same all-case clocks, 2 RNG, K5/L10, cap .23 and fixed 90% gate. Replay
immutable raw and paced controls explicitly; run 66 new whole-session executions.
Keep failed cases. If it improves utility, require matched sequence attacks;
its unchanged first query can share ONLY the exact first-step challenge of
iteration 16. This is another development ablation, not final confirmation.

## Iteration 18: expand development before further tuning (before new data/scores)

Do not keep choosing planner thresholds from the same four exposed families.
Generate twelve additional fixed families, seeds 701–712, with the unchanged
22-session SUMO scenario design on the same recovered map. Label ALL of them
`expanded_development`, not final confirmation. Require 22/22 completed trips
per family and retain eligibility/rejections for every A/B/C record. Do not
replace seeds based on protection results. Preserve all old evidence.

The provisional fixed service remains top-five references, K5 and L10; this
geometry/service stress does not resolve the pending server-only information
question. Freeze the initial screening shortlist before looking at these data:
raw; paced original objective; response-aware unpaced; response-aware paced;
response-aware paced with fixed .03 planning slack. Use the same .23 cap and
persistent union-of-20s/case request clocks, 2 paired RNG repetitions. Attack
selection must remain on the existing auxiliary split, never these new labels.
Report all 15 case gates, family-level values and dense-category diagnostics;
missing eligible cases remain missing. A matched learned attacker for the slack
variant is still required before its privacy claims. Any later changes make this
screening development evidence, not an independent final thesis confirmation.

Iteration 18 evaluation completion: also generate the slack candidate on the
same 80 auxiliary families (160 additional defended runs), and refit/select the
unchanged sequence-attacker bank on the same 64/16 split. This fills the missing
matched attacker for the frozen shortlist; no expanded-development labels enter
attacker training or selection. Its first query is identical to response-aware
pacing and must be checked before sharing that separate first-step challenge.

## Iterations 29–30: category-specific public query plans

The continuation request on 24 September authorizes further implementation and
experiments. The working Geo-I K5/L10 configuration remains an explicit control;
new public category plans have a different distinct-coordinate interface and
must not be presented as the same K5 method. Protocols are stored separately in
`iteration29_protocol.json` and `iteration30_protocol.json` before their scores.

Fit a deterministic macro-category greedy cover from public static rankings
only. Freeze a 30 category-coordinate request plan and an unconstrained
maximum-cover diagnostic; both use L10. The latter stops when no additional
static POI can be covered, and must not be called a full-catalogue certificate
if any public targets remain uncovered. Preserve all nine existing service
worlds and all 15 target cases. Measure actual request/response bodies, category
requests and distinct coordinates. Mathematical noninterference concerns the
coordinate channel conditional on region/catalogue/clock/server state; it does
not imply zero attack success or conceal metadata.

After freezing both plans, generate four new families, seeds 901–904, with no
replacement based on results. Require 22/22 completed SUMO journeys per family.
Evaluate all eligible records and preserve null reference cases, three new
availability seeds 24093001–24093003, and probabilities .5/.8/.95. Keep fixed
K5/K12 and bulk controls; public-epoch refresh must be offered equally to fixed
controls. This is a small new-family check on the same city and generator, not
new-city or final thesis confirmation. The previous adaptive defender is not
run on this new dataset, so do not compare it across dataset versions.

Completion: plan30 has 19 distinct coordinates; the wider plan has 67 category
requests and 52 coordinates. Plan30 passes 15/15 nominal cases on both datasets
but only 11/15 in the new-family .95 stress. Wider67 returns 100% on evaluated
cases at a higher byte cost, while eight public POIs remain outside its static
coverage certificate. All outcomes and the complete cost frontier are retained
in `docs/research/category_cover_results.md` and `iteration29_*`/`iteration30_*`.
