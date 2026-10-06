# Fixed-epoch linked-session Geo-I development evidence

Promoted readout: [round3/readout.json](round3/readout.json). This adds accounting
around the unchanged Geo-I REM/noisy-reuse pipeline; it is not a new location
mechanism or a replacement for the original 72-session S4 benchmark.

Before scores, the public policy fixed six slots/day, K5, L20, 60-second reads,
H12 versus H8 at the SAME effective total cap 0.23/m. The old per-session reset
reference composes to 0.345/m across six; its equal-total global control must
produce identical coordinates/states/ledgers. All 144 control sessions matched.

The promoted run retains 720 executions: 72 source sessions, two independent
secret-keyed RNG repetitions, raw plus four public configurations. All linked
sessions/repetitions of one family remain in one split. Previously inspected
701–712 family cohorts make this development evidence, not independent
confirmation. No data or privacy configuration was selected to improve scores.

`round3/public_transcripts.json.gz` contains only public indexes, method labels
and Q/time events. `evaluator_executions.json.gz` separately contains SYNTHETIC
truth/session IDs, branch accounting, service scores and states; it is never an
attacker input. Actual SQLite ledgers, secret master key and raw HMAC material
remain exclusively in private local `/private/tmp` state, outside these files.

`protocol.json` pins source/backbone hashes and the old 72-session artifacts.
`epoch_accounting.json` records allocated/spent composition by research family
and repetition. `verification.json` independently validates saved accounting,
control equality, public/evaluator separation, utility summaries and selected
attack metrics. Run `python -m experiments.verify_session_budget` read-only.

H8 development AUC person/vehicle is 0.526/0.504 with 97.23% Recall@5. Residual
maximum family AUC across the fixed bank remains 0.694/0.681. Only seven test
post-filter ticks exist; train post-filter Recall is 87.51%, so long-trip utility
is not established. H12's lower cap alone did not improve empirical linkage.
Coordinate privacy cannot hide account/IP or known person/vehicle identifiers.
The existing finite-precision sampler remains an ideal-kernel approximation.

Two setup failures are preserved: root protocol/failure records a wrong PUBLIC
prior dimension before private data access; round2 records an overly strict
floating-point equality assertion before attacker scores. Neither triggered
parameter fitting. Round3 used the same declared protection policies.
