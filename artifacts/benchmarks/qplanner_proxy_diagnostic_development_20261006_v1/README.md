# OLD development proxy diagnostic

Posthoc descriptive analysis only. No fresh dataset/scores, model/threshold selection, planner changes, causal/calibration claim, or Geo-I proof. All old inputs/source artifacts remain unchanged.

- `analysis_protocol.json`: declared inputs, source hashes, definitions and limitations before complete diagnostic scoring.
- `diagnostic.json`: equal-family/equal-nested-draw summaries, per-family/draw records and matched-event contrasts.
- `exact_saved_counts.json`: literal initial/subsequent public reachable-bucket and accepted-exchange counts from immutable source arrays.
- `validation.json`: 58 independent array-aggregation checks, including exact agreement with old primary readout **per purpose**.
- `cli_recheck_receipt.json`: complete repeat CLI reproduced all numerical summaries/family-draw records exactly. `cli_recheck_absolute/` retains that repeat output/source snapshot.

Diagnostic event macro averages defined purposes **within an event**. Old primary macro averages each purpose's conditional event utility **before** averaging purposes. Within-radius has 30 N/A selection events; the diagnostic does not replace primary numbers/gates.

Run focused fixtures:

```sh
python -m pytest -q tests/test_qplanner_proxy_diagnostic.py
```

Replay with the new CLI adapter into a NEW relative or absolute output path **inside this repository** (existing output is write-once):

```sh
python -m experiments.qplanner_proxy_diagnostic_cli_20261006_v2 --output artifacts/benchmarks/new_proxy_diagnostic
```

The adapter normalizes relative paths and rejects outside-repository/symlink escape **before** the immutable v1 runner writes. [The relative CLI recheck](../qplanner_proxy_diagnostic_cli_replay_20261006_v3/relative_cli_validation.json) completed with exit 0, reproduced every numeric summary/family-draw record, and rejected write-once retries unchanged. The adapter/source pins are separate from this immutable arithmetic source. Two earlier v1 `--output` display failures after completed arithmetic remain documented in `cli_recheck/` and outside-repository temporary work; they do not affect results.

Diagnostic SHA256: `167e4276615dfefbaf409daa3e57c5f6f00479f42f6a5befd087221b56654b8f`.
Source SHA256: `6ff6d37e0a21c43982594192b69b2e112d629b91a201e493ff52ed5be29b9e84`.

See [concise research note](../../../docs/research/2026-10-06_qplanner_proxy_diagnostic.md).
