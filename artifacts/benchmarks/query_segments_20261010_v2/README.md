# Frozen multistep Geo-I / Q-segment evidence

Status: **completed and independently verified; NOT promoted**. The primary
`segment_g1` arm misses the prespecified utility noninferiority gate. This is a
privacy/utility tradeoff study, not a leaderboard claiming universal dominance.

- [Vietnamese result readout](../../../docs/research/2026-10-10_multistep_results.md)
- [Mathematical argument](../../../docs/research/2026-10-10_multistep_privacy_utility.md)
- [Frozen protocol](protocol.json), [results](results.json),
  [independent validation](independent_validation.json)
- [Native cohort validation](../../datasets/query_segment_fresh_native_20261010_v2/validation.json)

Sources/config were frozen before the new cohort was generated. All seven
arms use ONE .23/m per-session Geo-I history per trip, K5/L30, GPS>=60s and
public20s queries. The prototype's Gamma is dimensionless and has no linked-
session reset guarantee. Tested frontier: .5/1/2; primary Gamma1 fixed before
TEST. TRAIN24 / SELECTION12 / TEST24 families; evaluate120 native trips with
one secret Q draw each. Same native map/generator, synthetic choices; no
cross-city, real-GPS, dynamic-provider or repeated-private-draw confirmation.

The `families` files separate attacker-visible events from evaluator-only GPS,
Geo-I anchors/ledgers, sampled lane states and local certificates. Attackers
receive events and public background only. Models are fitted on TRAIN; loss-
specific selections and model hashes are saved before TEST predictions.
Private keys and SQLite files remain outside the repository.

Independent review checks126,000 directed track transitions and15,120 public
support decisions, every prefix allowance, utility calculations, attacker
receipts and the adoption gate. All15,120 decisions report a degraded public
floor: do not claim hard QoS. The point-mass likelihood audit is numerical;
ideal-real proofs cover arbitrary beliefs, but the float/PCG64 sampler remains
uncertified.

```sh
python -m experiments.replay_query_segment_evidence
```

Replay uses the immutable experimental source snapshot. Later maintenance of
`calibrated_beta` (exact binary-rational zero/calibration) does not alter the
segment kernel or experiment's utility-certificate function. The relevant
ASTs and both hashes are recorded in [postfreeze maintenance](postfreeze_maintenance.json).
Direct verification with current sources deliberately rejects source drift;
use frozen-source replay instead. No result or cohort was overwritten.

The failed v1 declaration is retained separately: a copied source-pin filename
was wrong before any cohort or defense score existed. See
[retained failure](../query_segments_20261010_v1/failure.json).
