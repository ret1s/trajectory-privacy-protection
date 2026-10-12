# Native48-group balanced-cache cohort

20 TRAIN /8 SELECTION /20 TEST;384 native SUMO trips,249,984 archived1Hz
fixes,192 counterfactual native calibration trips and288 source archives.
Original24 plus two development cohorts120 junction groups are excluded;
public route Jaccard>=.8 rejected. Families are grouped by distinct fork
junctions; no claim of exclusion from every historical cohort in the repo.

[Protocol](protocol.json), [manifest](manifest.json), [native validation](validation.json),
[archive-count verifier correction](native_verifier_correction.json).

```sh
python -m experiments.verify_cache_native_20261012_v2_corrected
```

Public geometry seed2026101245 and method were frozen before generation.
Cohort sizing followed a [public-supply failure](../cache_fresh_native_20261012_v1/generation_failure.json)
with49/60 groups; only split counts changed to20/8/20. Eligibility, source map
and method/privacy settings were retained. V1 had no defense/attacker readout.
OS choice keys remain outside repo; all actual native trajectories are retained.
Same public map/generator, synthetic choices; no interpolation/winner filtering,
real-GPS, population representativeness or cross-city confirmation claim.
