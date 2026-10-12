# Finite Q and public POI portfolio

**Status: completed and independently verified; NOT promoted (utility gate failed).**

Frozen source/config before a new native cohort, all declared arms retained.
Primary: `finite_portfolio_g1`. Geo-I/REM remains .23/m per session, K5/L30,
public20s requests, joint0/20/40s frames and60s selection. Q allowance is
Gamma/[j(j+1)]; Gamma1 primary, Gamma0 public-pool control. There is no cap
on eight linked sessions.

- [Paired result readout](../../../docs/research/2026-10-12_finite_progress_results.md)
- [Privacy and utility proof](../../../docs/research/2026-10-12_finite_privacy_utility.md)
- [Frozen protocol](protocol.json), [results](results.json),
  [independent validation](independent_validation.json), [secondary paired controls](secondary_readout.json)
- [Native integrity](../../datasets/finite_pool_fresh_native_20261012_v1/validation.json)

TRAIN24 / SELECTION12 / TEST24 junction families. All siblings/draws stay in
one split. One private draw per training/selection trip, three per test trip;
216 evaluated trip-draws in total,144 TEST. All arms share ONE realization
of protected Geo-I history per trip-draw. Attackers fit TRAIN and select on
SELECTION before TEST. CI resamples families, not individual GPS points.

Evaluator files contain truth, protected anchors, lane IDs and score
certificates. The wire has coordinates/times and fixed all-category replies;
no private GPS, query purpose, score, fresh/reuse flag or budget ledger is
sent. Private keys and SQLite stay outside this repository. The application
returns the full locally sorted answer; Recall@5 is an evaluation metric.

The finite Q PMF is integer/rational and satisfies the stated ratio under
fresh independent uniform bits. This study uses private seeded PCG streams;
those streams and upstream Geo-I floats are NOT certified production DP.
Timing/account/IP/vehicle identifiers are outside the logical-wire theorem.
Static catalogue, same native map/generator, synthetic choices; no real-GPS,
cross-city or dynamic-provider confirmation. A bulk-download provider would
require a separate static-catalogue control and different cost contract.

Verification using only published evidence:

```sh
python -m experiments.replay_finite_evidence_20261012 --output artifacts/benchmarks/finite_pool_20261012_v1
python -m experiments.readout_finite_progress_20261012 --output artifacts/benchmarks/finite_pool_20261012_v1
```

Frozen-source replay checks the immutable source snapshots and input hashes,
uses public caches only and does not refit attackers or regenerate choices.
Fresh generation deliberately requires NEW output/cohort paths; never overwrite
these results. Promotion requires the declared -2pp noninferiority gate for
macro AND each purpose and each repeated draw, plus privacy/path checks.

The frozen copy of `experiments/scenario_v2_checks.py` preserves its original
blank line at EOF. Full `git diff --check` flags that inherited line; authored
files pass the check with `**/source_snapshot/**` excluded. Trimming an immutable
snapshot would invalidate its source hash, so its bytes are intentionally kept.
