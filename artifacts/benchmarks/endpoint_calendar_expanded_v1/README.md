# Frozen 32-family endpoint check

Results and interpretation: [endpoint_calendar_results.md](../../../docs/research/endpoint_calendar_results.md).

- `protocol.json`: 32 seeds declared after the pilot, no replacement/stopping
  by outcome; defense plans and attacker selection unchanged.
- `selection.json`: byte-identical to the pilot selection; its SHA-256 binds
  the exact trained checkpoint, which is archived in the pilot directory.
- `readout.json` / `case_results.csv`: 66 method/case rows, matched-family
  uncertainty, 15-case service utility and actual application bytes.
- `holdout_privacy.json.gz` / `holdout_service.json`: underlying observations.
- `transcripts/` / `transcript_hashes.json`: retained generation successes and
  failures; a failed generation is not perfect privacy.
- `calendar_flow_checks.json`: fixed-subscription payload independence checks.
- `calendar_ablation.json`: compare against active-epoch refresh, charging all
  public-hour traffic; 90,618 equal-utility service events checked.
- `verification.json` / `presentation.json`: independent accounting checks,
  group counts, source hashes and overlap checks against development/pilot.

After installing the project dependencies and preparing the shared OSM/SUMO
resources used by the original runner, restore exact retained inputs:

```sh
python -m experiments.endpoint_dataset_archive unpack
python -m experiments.restore_endpoint_attackers
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python -m experiments.extend_endpoint_cohort verify
```

Generation/evaluation stages refuse to overwrite completed evidence. Readout
and verification can be rerun. Do not call `prepare` over this sealed protocol.
The dataset archive and trained checkpoint both round-trip to their original
SHA-256 hashes; neither restore command changes model parameters.
