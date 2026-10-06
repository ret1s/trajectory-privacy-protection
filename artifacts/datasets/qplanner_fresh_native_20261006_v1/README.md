# Preserved failed first fresh-cohort attempt

The sealed first generator used public simulator seed `20261006071`, which
SUMO rejects as outside its signed 32-bit integer range. It screened 683 public
fork candidates, retained **zero families**, and produced no trajectory dataset
or protection/attacker score. 402 candidates had no eligible public separated
branch pair; the remaining 281 failed simulator option parsing.

[protocol.json](protocol.json), source snapshots,
[public_eligibility.json](public_eligibility.json) and
[generation_failure.json](generation_failure.json) are preserved. The private
choice master remained outside the repository and is not included here.

The corrected [v2 cohort](../qplanner_fresh_native_20261006_v2/README.md) uses a
valid public simulator seed, adds a range guard, and keeps the scientific
eligibility/split/workload design. No accepted target or score from this failed
attempt was replaced or used to choose v2.
