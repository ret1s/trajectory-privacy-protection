# Frozen legacy Geo-I Q for response-depth confirmation

This artifact generated the **sole `legacy_l10` protected Q method**, plus a raw
synthetic control, for the separate L30−L20 utility confirmation. L30 and the
test criterion were frozen **before fresh generation and scoring**. No old
development keys or private realization were imported. New OS-generated keys
and client checkpoint state remain exclusively outside the repository.

There are **108 family/draw blocks**:24 TRAIN families ×1 draw,12 SELECTION
families ×1 draw and24 TEST families ×3 nested draws. Each block has8 sessions.
K=5, protected supplier reads every60s, and effective epoch epsilon is0.23/m
(8 caps of0.02875/m; nominal epoch budget0.24/m). The Q planner retains its
historical L10 signatures; this base service evaluation uses L20 per category
per Q. The adopted L30 only changes the later server-response depth.

The independent [base certificate](validation.json) passed all108 bundles,
54,344 public event/wire rows,27,172 ledger steps,108,688 raw/protected utility
windows and27,172 objective records. It replayed directed Q feasibility,
received service, causal static cache, fixed private-input/anchor/ledger tapes
and conditional utility arithmetic. These totals include all splits.

The finite method-adapted attacker bank is fitted on TRAIN, with its selectors
and model files made durable using SELECTION before TEST predictions. S9/S10
use ordered/invariant/Hungarian track features, ExtraTrees64, kNN1/kNN5 and
geometric/history decoders. S5/S6 use candidate geometry and ExtraTrees96 with
two public alternatives, six linked history sessions and causal query prefixes.
The certificate checks12 models/selectors,17,472 probability-bank records and
588,672 endpoint-error records. It verifies these implementations and outputs;
it does not certify privacy against all attackers or a new privacy theorem.

The **60s interval applies to the protected Geo-I supplier/ledger**. Utility
uses synthetic ground-truth GPS at each public event and the actual destination
as a local exact-ranking oracle. It does not measure physical GPS energy or
validate ranking using only60s sensor fixes.

Public provenance is retained in `protocol.json`, `freeze.json`,
`execution_protocol.json`, `generation.json`, `resources.json`,
`source_snapshot/`, `execution_source_snapshot/`, `families/`,
`attack_selection.json`, `attack_readout.json` and `attack_models/`. Six optional
public cache files are pinned: `native.net.xml`, `reference5.npz`, `reply10.npz`,
`reply20.npz`, `belief-0.01.npz` and `belief-0.00125.npz`. They contain public
map/service/belief tables, can be reconstructed from pinned public inputs and
source, and are distinct from the excluded private key/checkpoint directories.
The verifier reconstructs the public resources independently; the original
cache and private RNG keys are not required to audit saved outputs.

Reverify without replacing the saved certificate:

```bash
python -m experiments.verify_qplanner_depth_base_q_20261006_v2 \
  artifacts/benchmarks/qplanner_depth_base_q_generalization_20261006_v1 \
  --validation-output /private/tmp/qplanner_base_q_validation_NEW.json
```

The subsequent [fresh response-depth readout](../qplanner_response_depth_generalization_20261006_v1/README.md)
and [recommended Geo-I / REM, L=30 configuration](../qplanner_response_depth_generalization_20261006_v1/recommended_configuration.json)
report the verified24-family TEST utility/cost result. This remains synthetic
same-map/static-service evidence, with no equivalent-cost baseline superiority
or publication-readiness claim. Fresh controls are not a new seed search.
