# Timing measurement clarification — 2026-09-08

Declared while the three budget jobs are still generating training outputs,
before validation or development scores are available. No metric threshold,
method, private budget, dataset or attacker choice changes.

Generation jobs run concurrently on the same host. Their recorded wall times
are useful for reproducibility diagnostics but are **not clean comparative
latency measurements**. Do not use those times for a Pareto dominance claim.

After all generation/verification jobs have ended, separately profile the fixed
S1.A, S3.A and S3.C validation records of family-103, replica 1, for all four
methods and all three budgets, sequentially in one process. Report mean and
p95 event time, initialization separately, exact record IDs and event counts.
Replay must match stored public outputs before accepting each timing. This is
a 36-run local-host profile, not a mobile-device or production real-time claim.

Primary exploratory frontier uses mean recall, selected Hit100 and reply ID
bytes. A secondary frontier includes the separately measured sequential mean
step time. Neither is an uncertainty-aware statistically significant frontier;
three profiling records are not the same population as the utility readout.
The subset timing is descriptive only and labelled adjacent to it. Non-dominance
is computed **within each B**, never by allowing a weaker formal privacy budget
to dominate a stronger budget using empirical scores alone. This makes explicit
the parent protocol's separation of budget as a user-facing comparison axis.
