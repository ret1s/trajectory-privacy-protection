# Fresh native finite-pool cohort

Frozen source/config; seed2026101241;60 junction families (24/12/24),
480 native SUMO trips,312,4801Hz fixes and240 public calibration trips.
Original24 junction groups and near-duplicate routes are excluded. Same
public map/generator, synthetic choices; no real-GPS/cross-city claim.

[Protocol](protocol.json), [native validation](validation.json),
[manifest](manifest.json). Protection/attacker scores were not used to
select routes or edit GPS. Choice keys remain outside the repository.

```sh
python -m experiments.verify_qplanner_fresh_native_20261006 --output artifacts/datasets/finite_pool_fresh_native_20261012_v1
```
