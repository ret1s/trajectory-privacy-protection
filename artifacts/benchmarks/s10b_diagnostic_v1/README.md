# S10.B exploratory diagnosis

Read [the research note](../../../docs/research/s10b_diagnostic.md).

This is a separate, post-hoc raw-control diagnosis on a previously inspected
cohort. It is **not** a replacement for `endpoint_calendar_expanded_v1`, an
independent confirmation, or a claim that every possible attacker fails.

- `protocol.json`: declared candidates, information boundary, source hashes.
- `selection.json` / `selection_rows.json.gz`: B-specific and combined choices,
  using seven eligible historical selection families, disjoint from fit.
- `evaluation_rows.json.gz`: preserved original errors and 40 new candidate
  errors for all 25 B records (50 target trips); raw repetitions deduplicated.
- `readout.json`: raw controls, exact S6.A duplication audit, descriptive A/C
  breakdown from frozen results, and explicit finite-bank oracle diagnostic.
- `verification.json` / `input_transcript_hashes.json`: independent checks.
- `implementation_amendment.json`: reader bug fix after selection; no choices
  changed and no new holdout predictions had been computed on the failed run.

Commands from repository root, with project dependencies installed:

```sh
python -m experiments.endpoint_dataset_archive unpack
python -m experiments.diagnose_s10b select
python -m experiments.diagnose_s10b evaluate
python -m experiments.verify_s10b
pytest tests/test_prefix_destination_attack.py
```

Completed stages refuse overwrite. All original study artifacts and A/B/C
aggregate results remain unchanged. Current endpoint efficacy conclusions focus
on A/C; B remains a boundary control and belongs with the S6 forecasting work.
