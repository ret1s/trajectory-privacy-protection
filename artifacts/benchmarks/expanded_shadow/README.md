# Expanded auxiliary shadow inference

Inference-only audit of seven **frozen** prior-factor protection configurations.
This is not a new defense or a SOTA leaderboard. U/U stays selected, with the
previous utility gate still unmet. The protocol was written before new scores.

## Evidence and order

1. Auxiliary data: `artifacts/datasets/urban_shadow_v1/`, sealed fourth SQLite
   release. 64 train / 16 holdout groups, four AUX schedules; SUMO+OSM only.
2. `training.json`: all 1,792 auxiliary training protection runs and seven
   expanded empirical banks. Each bank is old 360 plus 2,873 new observations.
   Forests are data-only NPZ arrays (no pickle), 128 direct and 128 residual
   trees per mechanism, with exact parameters and source hashes.
3. `validation.json` / `selection.json`: 378 old-output runs, original 103/104;
   freeze an attacker separately by metric/case plus a global choice for the
   auxiliary diagnostic. No protection method is reselected.
4. `development.json`: same 756 outputs and utility from original 201–204;
   preserve all old errors and add ten new attacks. Already inspected data,
   not fresh confirmation.
5. `holdout.json`: 448 new protected auxiliary runs, 16 unseen families;
   no training or per-profile attacker selection. This is a different sampling
   design from the nine core scenarios and must not be called their confirmation.
6. `verification.json`: independently reconstructed banks, sklearn forest
   refits, inference/errors, all original-field parity, selection, native lane
   transitions and 280 deterministic auxiliary full/prefix replays.
7. `readout.json` / `results_tables.tex`: exact selected scores, exploratory
   envelopes, all shared attacks, four paired-family changes, auxiliary-profile
   transfer and evaluator-only geographic-support diagnostics.

No exact likelihood, calibrated posterior, stronger Geo-I theorem or full
S4–S8 protection is claimed. More attacks cannot reduce the **exploratory**
maximum Hit, but a new validation-selected attack can generalize worse.

## Reproduce a new audit

Install `requirements.txt` in the existing environment, preserving pinned
SUMO/network/POI inputs and all earlier artifacts. scikit-learn is pinned to
1.7.1. The experiment freezes code hashes, so modify pinned scientific code
only for an explicitly new evidence cycle.

Use a fresh output directory; keep the order below. Development and holdout
may run concurrently **after** training and validation/selection finish.

```bash
venv/bin/python -m experiments.run_expanded_shadow --phase training --output build/shadow-replay
venv/bin/python -m experiments.run_expanded_shadow --phase validation --output build/shadow-replay
venv/bin/python -m experiments.run_expanded_shadow --phase development --output build/shadow-replay
venv/bin/python -m experiments.run_expanded_shadow --phase holdout --output build/shadow-replay
venv/bin/python -m experiments.verify_expanded_shadow --output build/shadow-replay
venv/bin/python -m experiments.export_expanded_shadow --output build/shadow-replay
```

Canonical evidence was checked/exported with:

```bash
venv/bin/python -m experiments.verify_expanded_shadow
venv/bin/python -m experiments.export_expanded_shadow
```

Both accept `--output` and default to this canonical folder. They write
exclusive receipts/readouts; do not rerun over sealed receipts. To recheck the
canonical evidence without overwriting its receipt, use
`--receipt build/shadow-recheck/verification.json` with the verifier (fresh path).
Machine timings are recorded for diagnostics but are not controlled speed
comparisons while other verification tasks share the CPU.

Detailed interpretation and open publication gates:
`docs/reviews/verification_expanded_shadow.md`,
`docs/research/expanded_shadow_literature_review.md`,
`thesis/expanded_shadow_comparison.tex`.
