# Ordered Q endpoint challenge — post-inspection development

This artifact reuses the complete frozen streams from
[endpoint_generalization_20261005](../endpoint_generalization_20261005/README.md).
The 28 families 1205–1232 were already inspected. These scores are a **new
development diagnostic**, not an independent heldout confirmation. No GPS,
defense configuration, Q multiset, public clock, POI reply or utility was
changed. Four methods × two publication views × 112 trip/rep streams were
scored. The earlier coordinate-set result remains a fixed-bank result.

The intervention independently privately permutes each event's coordinates;
candidate IDs are absent in both attack views. Original internal Q order
remains available only to the engine. This closes explicit positional labels,
but does not establish geometric unlinkability or account/IP/identity anonymity.

The expanded bank includes per-slot endpoint/OLS 2/3/6 with observable-boundary
and 0/30/60/120-second horizon estimates; learned ordered sequence features
(ExtraTrees64, kNN1/kNN5); Hungarian nearest/velocity canonical track banks;
and the original coordinate-set bank as a separately reported control. Fit uses
701–708 only, loss-specific selection 709–710 only, separately for each
method/view/attack group. No attacker is replaced using test errors.

The result shows **no Endpoint20 MAE gain from shuffling**. Its full-bank
S9/S10 MAE is 2199.66/1337.16 m in both views. Geometry-associated features
and predictions remain exactly invariant. PlainL20 S10 increases only
13.96 m after shuffle, paired family bootstrap 95% [-18.59, 47.63].

Expanded attacker selection generalizes poorly: PlainL20 S10 ordered
`observed_slots_tree` has selection MAE 470.26 m but inspected-test MAE
1677.01 m; shuffled `geometry_velocity_tree` has 558.56/1690.98 m. The old
`median_ols6_120s` control has 671.96/901.48 m. Test expanded-minus-control
is +775.53 m [505.50, 1059.27] and +789.49 m [515.82, 1070.74]. Adding
attacks cannot weaken an optimal attacker, but finite selection on two
families can choose a less accurate heldout decoder.

Consequently S10 Endpoint20 minus PlainL20 MAE is +435.67 m [312.57, 563.42]
in the unchanged old bank, but -339.86 m [-623.68, -54.28] ordered and
-353.82 m [-635.73, -65.82] shuffled in the selected expanded bank. Both
must be retained. This evidence does not establish broad robust superiority.
The raw positive control stays MAE 0 and Hit100 100%; it validates the scoring
pipeline, not exhaustive attacker strength.

`protocol.json` and `protocol.sha256` seal the challenge, input streams and
feature/runner sources. `selection.json` pins the compressed trusted local
model and all loss-specific decoders. `development_summary.json` and
`selection_rows.json.gz` retain all selection outcomes; `diagnostic.json`
contains every group/view result and paired family uncertainty;
`diagnostic_rows.json.gz` retains every individual bank error.
`readout.json` gives the original-bank equality checks and selection/test
MAE for every frozen decoder, including all 28 per-family errors.

`evaluator_shuffle_randomness.json.gz` is the synthetic reproducibility key,
kept outside public features. It uses HMAC(session, rep, event) and is separate
from Geo-I RNG. Production needs its own private random source. The original
session RNG fix and privacy accounting are unchanged: Endpoint20 has a
0.0575/m ideal-kernel per-session cap, not a lifetime/global identity budget.

`verification.json` records recomputation of 478,464 actual bank predictions,
320 metrics, 10,464 event views and 177,408 invariant/geometric equality
checks. All original clock/multiset checks include repeated coordinates.
The seven new tests cover permutation invariance, crossing tracks, private
event substreams, unchanged inputs, coordinate multiplicity and rejection of
private fields from feature APIs. The 28 targeted endpoint tests pass.

`runtime.json`, `sources.tar.gz`/`source_archive.json`, and
`attackers.pkl.gz`/`model_archive.json` preserve the runtime and exact code/model.
Source inputs remain in the referenced prior artifact, with every file SHA
pinned here. Models are trusted locally generated Python pickle; verify the
recorded SHA before restoring them.

Run from the repository with the recorded environment:

```bash
ENDPOINT_PYTHON=/private/tmp/trajectory-research-20261005-venv/bin/python
$ENDPOINT_PYTHON -m pytest -q tests/test_ordered_endpoint_attacks.py
$ENDPOINT_PYTHON -m experiments.verify_endpoint_order artifacts/benchmarks/endpoint_order_20261005
$ENDPOINT_PYTHON -m experiments.endpoint_order_readout artifacts/benchmarks/endpoint_order_20261005
```

To regenerate into a **new** output directory (never overwrite evidence):

```bash
$ENDPOINT_PYTHON -m experiments.endpoint_order_challenge all --out /private/tmp/endpoint-order-new-run
$ENDPOINT_PYTHON -m experiments.verify_endpoint_order /private/tmp/endpoint-order-new-run
$ENDPOINT_PYTHON -m experiments.endpoint_order_readout /private/tmp/endpoint-order-new-run
```

A new run samples a new shuffle master; the archived key and frozen attackers
allow exact verification of this retained run. Public map resources are built
by `experiments/public_research_resources.py` from all 9,138 archived public
polylines, at 40 m state spacing. Original Beijing SUMO lane/turn resources
were absent; the reconstructed 8 m/s road assumptions remain a development
limitation. See the [research explanation](../../../docs/research/2026-10-05_endpoint_order.md).
