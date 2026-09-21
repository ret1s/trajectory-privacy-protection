# Additional boundary-window audit

Exploratory reanalysis of **previously examined paper-v2** transcripts, not a new
confirmation set, defender run, or attack-training run.

- Protocol: `docs/supervisor_meeting/2026-09-26_brief/boundary_audit_protocol.md`.
- Run: `python3 -m experiments.report_boundary_audit` (standard library only).
- Verify helper behavior: `python3 -m unittest discover -s tests -p test_report_boundary_audit.py -v`.
- Source: `artifacts/benchmarks/paper_benchmark/results.json`, never modified.
- 48 input rows = 2 methods × 2 endpoint scenarios × 12 trips. Reuse each at
  0/40/80 s extra cut → 144 diagnostic rows. There are 3 simulation seeds, not
  144 independent trajectories.
- Attack functions consume retained public coordinates and the declared mask
  policy, never target labels. Source endpoint errors and POI-ID recall are
  independently recalculated before deriving the additional results.
- The envelope selects an attack at aggregate group level, not per target.
- Dropped queries remain in the original-window utility denominator, with no
  cache credit. The original hidden 60 seconds remain outside that denominator.
- No runtime/energy/network measurements are claimed. Tail clipping is offline;
  latency and causal implementation of a live boundary gate remain unmeasured.

`results.json` includes source/code/protocol hashes, all derived predictions and
errors, utility numerators/denominators, and 12 aggregate summaries. This new
artifact can be regenerated; the historical input stays immutable. The report
preserves negative results: additional clipping does not uniformly improve
measured privacy and reduces delivered service in this no-cache diagnostic.
