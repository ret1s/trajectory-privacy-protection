# Active reporting scope v2: S10 A/C

This derived view removes S10.B after the duplicate-task diagnosis and explicit
user scope decision. Frozen experiments and data are unchanged. No new fitting,
selection, transcript generation or independent confirmation occurred.

- 29 active cases overall; 14 in S1/S2/S3/S9/S10.
- Within-scenario equal case weights; equal scenario weights for overall Recall.
- Costs retain only sessions used by current cases, with full original clocks,
  service-event denominators and traffic (including the whole scheduled hour).
- Endpoint bootstrap intervals are recomputed for S9 ABC / S10 AC using family
  clusters, not copied from the historical ABC aggregate.
- `readout.json` carries source hashes and denominators; `verification.json`
  independently reconciles values to original measured privacy/service rows.

Run `python -m experiments.reaggregate_active_scope`, then
`python -m experiments.verify_active_scope` and
`python -m experiments.summarize_active_scope` from repository root.

See [the current research note](../../../docs/research/active_scope_results.md).
Historical readouts remain in their original artifact directories.
