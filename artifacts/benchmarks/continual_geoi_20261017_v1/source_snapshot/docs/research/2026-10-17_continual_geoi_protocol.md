# Fixed-unit Geo-I behind bounded Q: October 17 pilot

Status: opt-in experiment. The sealed GeoI-BQ v1 reference and October17 report
remain unchanged. This protocol must be frozen before fresh defense draws.

## Question and mechanisms

Can a fixed per-operation Geo-I epsilon, without an upstream stopping cap,
improve adaptation on an unknown-duration trip while preserving the existing
bounded-Q logical-wire theorem? Keep REM, noisy reuse, belief, directed-road
Q support, K5, per-category L60, three-frame segments,20s publication,60s private
updates, static cache and local purpose sorting unchanged.

Arms: `capped_g1` is the frozen .23/m upstream cap with Gamma1 Q;
`continual_g1` updates with u=.01/m indefinitely behind the SAME Gamma1 Q;
`continual_g0` retains the public-base control. Pair upstream random streams
and Q random streams, never choose keys or data on outcomes. No automatic
promotion and no requirement that every metric improve.

## What is and is not proved

For N public scheduled private-read slots, the ideal continual upstream has
finite-prefix log likelihood ratio at most (2N-1)u D_infinity. A sharper public
bound weights the first slot once and each later slot twice. Realized hidden
branch expenditure is a local diagnostic, NOT a replacement for that public
worst-case theorem. There is no finite constant upstream cap over all time.

The Q proof tolerates arbitrary distributions of hidden upstream scores. At
the same augmented public-output history, positive common base masses and
bounded centered scores give per-segment likelihood ratio <=1+epsilon_j.
Mix over upstream states, compose adaptively, then project hidden sampled
lane/full-segment IDs to coordinate wire. With epsilon_j=Gamma/[j(j+1)],
R_n=product_j(1+epsilon_j) <=exp(Gamma), and for Gamma1 the existing all-time
upper bound is <2.437372. Hence the ideal spatial prefix bound can be combined
as min{(2N-1)u D_infinity, log R_n}; NOT min{.23 D_infinity,log R_n} for the
continual arm. Common clock/provider law, no GPS/Z/branch/account identifiers
on the logical wire, trusted nonrollbackable Q state and the existing ideal
upstream/entropy assumptions remain necessary. Reopening a session does not
renew Gamma; linked sessions compose. No pure floating-point Geo-I certificate.

As j grows, epsilon_j and private adaptation shrink. Updating belief forever
does not prove high utility forever. Existing matched-public-base whole-path
TV/utility and static-cache monotonicity arguments still apply, with their
existing conditions and no zero proxy-error assumption. The old finite-prefix
adversarial oracle already allows perfectly revealing hidden upstream states;
it does not claim an unprotected state can be sent to the provider.

## Frozen evaluation

Use all48 existing native-cohort families (20TRAIN/8SELECTION/20TEST), days7/8,
one fresh draw/trip, no private-score filtering. First600s are REUSED native
GPS, not untouched geometry. Add a1200s synthetic continuation on directed
road paths, with public hashed session seeds, public goals and20s movement.
Generate ALL truth before fresh defense keys. This continuation is NOT native
SUMO, real-user field data, or independent long-trip generalization. Retain
the first10min control separately; capped and continual Gamma1 outputs must
match exactly on this prefix. Detour uses the destination of the30min test;
it is therefore not a numerical replication of the old10min detour benchmark.

Report native0..600, full0..1800, and synthetic720..1800 windows; all four
purposes and current/epoch/session cache. Paired-family percentile95% intervals
use10000 frozen bootstrap draws. Display the existing -2pp utility gate but
do not auto-promote. Static418-POI cache saturation can mask adaptation gains.
Keep query/index byte counts, Geo-I read/spend diagnostics and Q prefix
allowances. No wall-time or battery claim from indexed response sizes.

Attackers fit TRAIN only; loss-specific robust selection on SELECTION precedes
TEST predictions. Retain S1/S2/S3/S9/S10 with the existing ExtraTrees/kNN,
geometry and ordered-endpoint banks, public close1800s. S2 remains a parked
window coordinate proxy even if the synthetic suffix moves: interpret it as
late-window coordinate recovery here, not a semantic sensitive-stop metric.
Keep all control results and finite-bank limitations. One draw is a pilot,
not a stable stochastic replication claim.

## Execution and acceptance

Run `python -m experiments.continual_geoi_study_20261017 run` in the isolated
scientific runtime. The worker freezes source/config, prepares truth, generates
all arms, audits road/Q prefixes, fits attackers, writes results and verifies.
Refuse overwrite/reseed; retain failures under the same protocol ID and use a
new ID for a corrected freeze. Fresh keys and SQLite live outside the repo.

The isolated runtime is declared in `requirements-continual-geoi.txt`.
`python -m experiments.run_continual_geoi_background_20261017 --commit-and-push`
is the one-shot detached worker: after successful verification it commits ONLY
this experiment's artifacts and pushes the current configured upstream. Its
status/log are `/private/tmp/trajectory-continual-geoi-20261017-v1/worker_status.json`
and `run.log`. It does not schedule future runs or send external messages.

Checks before launch: unchanged behavior before old cap;31 read slots through
1800s without a stopping cap; private-read skip ignores GPS; invalid coordinates
and time rejected; no false .23/m cap in continual local-run metadata; existing
adversarial hidden-upstream Q oracle and session-budget prefix tests.

Results live in `artifacts/benchmarks/continual_geoi_20261017_v1/`. A final
validation receipt authenticates frozen sources, dataset and family evidence,
recomputes paired utility/attacker summaries, and audits all directed Q paths
and reservation prefixes. It is a verification run, not independent authorship
of an alternative sampler or an automated symbolic proof.
