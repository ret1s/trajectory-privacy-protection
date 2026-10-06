# Relative-output diagnostic CLI replay

Completed the unchanged OLD development calculation using the final [versioned CLI adapter](../../../experiments/qplanner_proxy_diagnostic_cli_20261006_v2.py). No fresh inputs/scores, new model or tuning.

```sh
python -m experiments.qplanner_proxy_diagnostic_cli_20261006_v2 --output artifacts/benchmarks/qplanner_proxy_diagnostic_cli_replay_20261006_v3
```

This output is write-once; the command now correctly refuses a repeat. `relative_cli_validation.json` records exact numeric agreement with the original diagnostic, relative CLI exit 0, unchanged failed write-once retry, and outside/symlink rejection before any write. `cli_wrapper_receipt.json` separately pins the adapter and immutable arithmetic source; both have archived snapshots.

The preceding `qplanner_proxy_diagnostic_cli_replay_20261006_v2` recorded the relative-path adapter before the outside/symlink guard was requested. Its original receipt and matching earlier wrapper snapshot are preserved. This final v3 replay pins the completed guard source.

Wrapper SHA256: `d9918453f4c8462bb4d2171f3ab7c32611e84924c1b87f1b70d081ea1e120dd6`.
Unchanged numeric source SHA256: `6ff6d37e0a21c43982594192b69b2e112d629b91a201e493ff52ed5be29b9e84`.
