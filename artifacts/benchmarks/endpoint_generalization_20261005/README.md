# Locked endpoint generalization, 2026-10-05

GeoI-Endpoint20 (`scale025_L20`) was fixed from the previous development loop.
This extension does not tune/reselect the defense. It re-fits and selects
method-adapted attackers on train 701–708 / selection 709–710 using distinct
session/rep randomization, then evaluates all predeclared unused within-study
families 1205–1232 once. Groups 1201–1204 were previously examined and excluded.

The same Geo-I backbone is preserved: epsilon test/release=0.0025/m,
session cap=0.0575/m, K=5, L=20, 60-second GPS-read pacing and zero delay.
Same-L20 and L10 plain controls plus raw GPS are retained.

On 28 groups / 56 trips / 112 runs per method:

- Endpoint20 Recall=96.64%, versus 98.94% for plain L20; byte essentially equal.
- Primary single-trip S10 MAE=1,337m versus 901m: paired-family delta +436m,
  bootstrap95 [313,563]. S10 Hit100 ties at 0; Hit500 delta interval includes 0.
- All 1,308 service ticks published immediately for each method.
- Utility tail: p10=92.12%, minimum=77.61%, six of 112 runs below90%.
- These are existing simulated traces and a reconstructed public map, not
  independent real-world confirmation or faithful paper reproduction.

Read the [research note](../../../docs/research/2026-10-05_endpoint_generalization.md)
for complete results, uncertainty and the secondary linked-trip stress.

| File | Meaning |
|---|---|
| `protocol.json`, `protocol.sha256` | Full fixed population, seed derivation, defense, metrics, fit/select split and source hashes |
| `evaluator_randomness.json.gz` | Synthetic experiment key, evaluator-only; excluded from public events and attacker features |
| `selection.json` | Frozen per-method/per-loss attacker choices; defense unchanged |
| `development_summary.json` | Independent-RNG train/selection calibration; no heldout defense tuning |
| `fit-*.json.gz`, `selection-*.json.gz`, `test-*.json.gz` | Public event coordinates/times plus clearly separated evaluator target, seed and service summaries |
| `heldout_attack_rows.json.gz` | All single-trip decoder errors |
| `heldout_linked_rows.json.gz` | Account-linked pair stress, each target keeps its own GPS truth |
| `heldout.json`, `readout.json` | Equal-family estimates, paired intervals and utility tails |
| `verification.json` | Recomputed service/byte/predictions, caps, seeds, targets and full-population checks |
| `attackers.pkl.gz`, `model_archive.json` | Lossless trusted local models with compressed/uncompressed hashes |
| `sources.tar.gz`, `source_archive.json` | Lossless source snapshot |
| `failed_exact_common_endpoint_attempt/` | Training-only failed target check, protocol/code/failure; no test labels opened |

Only the `events` field and permitted observable close enter attacker inference.
Session IDs, evaluator seeds, master key, true GPS, Z and private branches do
not. Seed derivation is HMAC(master, session ID, rep), independent of method
ordering. Matched methods share a stream for the same trip/rep; unrelated trips
never reset to the same stream. Experiment key material is included solely for
reproducing synthetic evidence; it is not a production privacy secret.

The linked secondary assumes account linkability. Two Endpoint20 sessions
compose to at most 0.115/m; a per-session bound is not a historical-user bound.
The finite learned bank can select a decoder that generalizes worse with more
inputs. Linked results do not establish that linkage improves privacy.
Current features are permutation invariant. Persistent Q IDs and slot-order
attacks are not separately evaluated; the coordinate array still has order.
The results do not cover every attacker using the full public protocol.

Independent verification reads compressed models directly:

```bash
python -m experiments.verify_endpoint_generalization artifacts/benchmarks/endpoint_generalization_20261005
```

The Python/library versions are in `runtime.json`. To restore the locally
created model if required by a runner, verify its lossless archive:

```bash
python -m experiments.endpoint_noise_archive artifacts/benchmarks/endpoint_generalization_20261005
```

For a new locked-config run, choose a fresh directory. New prepare creates
private evaluator randomization; fit/select completes before test opens labels:

```bash
python -m experiments.endpoint_generalization all --out /private/tmp/endpoint-generalization-new
```

Existing protocol, attacker selection and full heldout result refuse replacement.
No old dataset, benchmark evidence or previous report was overwritten.
