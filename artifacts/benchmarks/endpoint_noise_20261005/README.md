# Endpoint noise development, 2026-10-05

Two bounded loops, each with frozen defense and method-adapted attacker
selection before its within-study family holdout. This is a full-city
reconstructed-map development study, not a rerun of the original network,
independent real-world evidence, or a faithful paper reproduction.

The selected round-two configuration, **GeoI-Endpoint20** (`scale025_L20`),
uses epsilon 0.0025/m for each test/release, effective cap 0.0575/m, K=5,
public response L=20 and no suppression/delay. All 170 heldout input ticks
are published immediately. Mean Recall=96.11%; lowest run=85.92%.
All configurations, including utility failures and unsuccessful phase tuning,
are retained. Read the [research note](../../../docs/research/2026-10-05_endpoint_noise.md)
for matched-L controls, uncertainty and the limits of the finite attack bank.

| Folder/file | Contents |
|---|---|
| `round1/` | Epsilon scales 1, 1/2, 1/4, 1/8 at L10; fit 701–708 / select 709–710 / test 711–712; no stronger-noise candidate passed Recall ≥90% |
| `round2/` | Scales 1, 1/2, 1/4 × L10/20/40, first-phase candidate, delay control; same fit/select; fresh within-study test 1201–1204 |
| `protocol.json` | Predeclared methods, selection rule, seeds and source SHA-256 |
| `development_summary.json` | Every candidate result, including failed utility gates |
| `selection.json` | Frozen defense and per-method/per-loss attack choices |
| `fit-*.json.gz`, `selection-*.json.gz`, `test-*.json.gz` | Coordinates and actual public timing, service/accounting summaries; endpoint targets explicitly evaluator-only |
| `*_attack_rows.json.gz` | Every attack error; selected losses calculated independently |
| `heldout.json`, `readout.json` | Family-balanced results, paired uncertainty and zero-hit caution |
| `verification.json` | Independent service, byte, attacker prediction, clock, cap and hash checks |
| `attackers.pkl.gz`, `model_archive.json` | Lossless trusted local training model; compressed/uncompressed hashes |
| `sources.tar.gz`, `source_archive.json` | Lossless source snapshot, including transitively imported research modules; pinned runner/engine files checked against protocol |
| `public_reconstructed.net.xml.gz`, `public_resources.json` | Public full-city reconstruction and provenance; original turns unknown |
| `runtime.json` | Python/library versions and native SUMO import check |

The dataset endpoints are simulated trajectory endpoints, not real homes.
Public transcripts do not include target labels, internal Z or privacy ledgers;
the combined evidence files intentionally contain evaluator-only target fields.
Attacker APIs receive only the `events` field and permitted observable close.
Group/seed repeats do not count as independent people.
Fixed mechanism seeds are common random numbers in this study. Intervals are
conditional on those two draws, not a deployment guarantee; independently
randomized session keys and cross-session budget composition require a separate
linked-trip validation. Do not reuse the same production RNG seed per trip.

To independently recalculate both studies using the repository's Python
environment (dependencies in `runtime.json`):

```bash
python -m experiments.verify_endpoint_noise artifacts/benchmarks/endpoint_noise_20261005/round1
python -m experiments.verify_endpoint_noise artifacts/benchmarks/endpoint_noise_20261005/round2 --depth
```

The verifier reads compressed models directly after checking their hash.
For a runner that requires an unpacked local model, restore it losslessly:

```bash
python -m experiments.endpoint_noise_archive artifacts/benchmarks/endpoint_noise_20261005/round1
python -m experiments.endpoint_noise_archive artifacts/benchmarks/endpoint_noise_20261005/round2
```

For a new complete run, choose fresh directories. Round two's prepare step
reads the specified round-one protocol, not its heldout metrics. Historical
runner snapshots are retained; current runners additionally record generation
exceptions before stopping and allow an explicit protocol path:

```bash
python -m experiments.endpoint_noise_loop all --out /private/tmp/endpoint-noise-new-round1
python -m experiments.endpoint_noise_depth_loop all --out /private/tmp/endpoint-noise-new-round2 --round-one-protocol /private/tmp/endpoint-noise-new-round1/protocol.json
```

Existing sealed outputs refuse replacement. To reproduce a historical code
revision exactly, inspect `sources.tar.gz`; do not extract over an active
checkout. Fit, selection and heldout source data are existing archived datasets,
so this is only a within-study holdout, not unseen future data confirmation.
