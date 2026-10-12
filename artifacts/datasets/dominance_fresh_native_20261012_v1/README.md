# Fresh native dominance-motion cohort

Source/config frozen before generation. Public seed2026101243;24 TRAIN,
12 SELECTION and24 TEST independent junction families; balanced6/8m/s;
8 native SUMO trips/family,312,480 archived1Hz fixes,240 public calibration
trips. Original24 and first12/10 development60 junction groups are excluded;
public route Jaccard>=.8 is rejected without reading protection scores.
New families are group-disjoint from those development families. They are
not claimed disjoint from every historical cohort in this repository.

[Native validation](validation.json), [frozen generator protocol](protocol.json)
and [source manifests](manifest.json) retain archived native traces and hashes.
[Compatibility receipt](native_validator_compatibility.json) records why the
original24-group verifier needed a separate84-group integrity verifier;
frozen generator and method were not changed. That new verifier checks
all original conditions, the additional pinned groups and route overlap.

```sh
python -m experiments.verify_dominance_native_20261012 --output artifacts/datasets/dominance_fresh_native_20261012_v1
```

Same public map/generator and complete418-service-POI catalogue. Synthetic
choices and private HMAC key outside repo; no interpolated GPS or
winner-directed filtering. This is not cross-city or real-device evidence.
