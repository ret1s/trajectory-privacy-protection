# Frozen native S5/S6 primary

24 independent public fork families: 12 train, 6 selection and 6 test; 192 native SUMO parked trips. The native map is in `artifacts/datasets/future_controlled_20261005_v1/public_native.net.xml.gz`. The active fixed-window FCD is in `artifacts/datasets/future_controlled_20261005_v2/dataset.json.gz`. Neither source nor prior benchmarks was replaced.

`protocol.json` was sealed before the first protection/attacker run. `public_transcripts.json.gz` contains public Q/times, both public candidate geometries and six linked historical windows. `private_accounting.json.gz` is evaluator-only GPS-label/accounting/utility evidence; it contains no sampler keys. Keys and live SQLite state remain outside the repository. `results.json` contains selection-chosen attackers, exact native-edge predictions, destination metrics, label-permutation controls and descriptive full-bank/family results.

At the visible native turn, Raw edge accuracy and destination Hit100 are 100%; session-reset Geo-I reaches 41.7%; GeoI-Epoch8-H12 reaches 50%. Every test candidate-edge ID is absent from train. Raw before-fork accuracy of 50% is intrinsic ambiguity and does not show a privacy contribution. The attacker knows two candidate turns/destinations and six histories; previous query/joint-pair constraints are not evaluated.

The two protected branches have different total budgets: session-reset has 1.84 m⁻¹ across eight trips; GeoI-Epoch8-H12 has 0.23 m⁻¹ across eight trips. No equal-total superiority is claimed. Original L10 static conditional Recall is 97.72%/89.95%. `public_phase_readout.json` reports tail/family utility and undefined-reference coverage: 1402/1510 test windows have reachable reference POIs, 462/528 in 400–600s. Read the conditional-coverage and small-test limitations in the research note.

```bash
/private/tmp/trajectory-research-20261005-venv/bin/python -m experiments.verify_native_future
/private/tmp/trajectory-research-20261005-venv/bin/python -m pytest -q tests/test_candidate_future_attack.py
```

Verifier passed 24 families/192 native trips/12 held-out queries per task/stage; 8 unit tests passed. Rechecks run all assertions and retain existing `validation.json`. `validation_recheck.json` additionally records fixed-phase/coverage verification and preserves the first record. To record another recheck, pass `--validation-output /private/tmp/NEW-native-check.json`; existing explicit paths are refused.

See [`2026-10-05_native_future.md`](../../../docs/research/2026-10-05_native_future.md). The optional retrieval-depth development extension is separate in `../future_native_depth_20261005_v1/`; primary Q, source hashes and attack scores are unchanged.
