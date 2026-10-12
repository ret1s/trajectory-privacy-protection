# Balanced Geo-I with public exploration and local session cache

Status: **completed, independently verified, balanced research candidate**.
The original -2pp baseline noninferiority gate is **NOT passed**. Canonical
engine and prior evidence are unchanged; this is not production certification.

- [Candidate configuration](balanced_candidate.json).
- [Vietnamese result report](../../../docs/research/2026-10-12_balanced_cache_results.md).
- [Privacy/utility proof and assumptions](../../../docs/research/2026-10-12_balanced_cache_proof.md).
- [Frozen protocol](protocol.json), [results](results.json),
  [independent validation](independent_validation.json), [paired controls](secondary_readout.json).
- [Native dataset validation](../../datasets/cache_fresh_native_20261012_v2/validation.json).

Primary `cache_explore_g1`: unchanged Geo-I/REM .23/m per session; K5/L30 per
category; public20s requests, joint three-frame selection every60s, Gamma1.
Public base masses1+4*(potential new POIs); support uses map and prior randomized
Q/replies. Only bounded private scores see protected Geo-I belief. Static
metadata is accumulated locally for the public session; private QuerySpec/GPS
never selects which responses are cached or triggers network requests.

ALL seven arms have current,60s epoch and session caches. Primary service
Recall is95.548%, family95%CI[93.329,97.444]%; matched old pool guard92.569%,
paired improvement2.980pp[2.077,3.904]pp. Baseline session Recall99.995%;
shortfall4.447pp[-6.665,-2.553]pp. Gamma0 achieves96.060%; a Geo-I-belief
service advantage over public/prior controls is not confirmed. Raw current-
frame Recall is87.941%; do not transfer session-cache scores to an expiring
response contract. App returns the full local sorted answer, not top5.

20 TRAIN /8 SELECTION /20 TEST junction families,176 evaluated trip-draws,
120 TEST; three TEST draws per native trip. All siblings stay in one split.
ONE protected Geo-I realization is shared per trip-draw; primary/static/guard
initial actions are coupled. Models fit TRAIN, select on SELECTION and are
saved before TEST predictions. Every comparison observes one arm in isolation.
Private sampling keys and SQLite files remain outside the repository.

Public geometry after excluding144 prior groups supplied49/60 required groups
in the failed declaration. V2 was separately frozen with48 groups, same seed
and eligibility, BEFORE defense evaluation. [Failure](../cache_balance_20261012_v1/failure.json)
and native archives are retained. Smaller selection/test size and same-map,
same-generator synthetic data limit generalization; no real-GPS/cross-city claim.

Reproducible evidence audit (no secret keys, refit or resampling):

```sh
python -m experiments.replay_balanced_cache_evidence --output artifacts/benchmarks/cache_balance_20261012_v2
python -m experiments.readout_balanced_cache_20261012
python -m experiments.verify_cache_native_20261012_v2_corrected
```

The source snapshot retains the original archive-count verifier assertion360
from its60-group predecessor. The separate corrected verifier checks6*48=288;
[correction](../../datasets/cache_fresh_native_20261012_v2/native_verifier_correction.json)
changes no source/generator/defense after freezing and uses no benchmark score.
The inherited EOF blank in frozen `scenario_v2_checks.py` is retained byte for
byte; authored files pass whitespace checks excluding immutable snapshots.

Math scope: exact finite-Q PMF under fresh independent uniform bits, logical
wire, public clocks and trusted local checkpoints; conditional upstream Geo-I
spatial guarantee. Experimental PCG, hardware timing and upstream Geo-I floats
are not certified. Cache utility assumes valid static metadata; volatile status/
opening/availability TTL is not extended. A bulk-catalogue service is a different
contract and needs an appropriate control. Historical proofs/results are kept.
