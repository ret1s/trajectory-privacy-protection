# Interrupted worker recovery — 2026-09-08

The first three workers were interrupted before completing any training phase.
No validation/development score, shadow model or training artifact was saved.
Their empty output directories are preserved under the ignored build directory.
One preflight cache with integer-versus-float metadata is also preserved; no
scientific parameter was changed to make its hash match.

Execution changes before restarting the same declared scientific grid:

- Save each completed record as an exclusive hashed checkpoint in ignored
  cache storage. The key hashes the complete non-timing provenance, including
  production code, protocol, dataset and source identities. Resume verifies
  content and exact records before reusing rows. Completed phase evidence is
  never overwritten; cached timing describes the worker that produced it.
- Reuse immutable public map/SCC/travel/belief resources within a worker rather
  than reconstructing them for every short trajectory. Recreate the exact two
  RNG streams from the original seed and reset all session state. Tests compare
  multiple sessions and repeats with separately instantiated models; the
  independent verifier replays fresh instances.
- Per-run initialization time now measures RNG reseeding, not cold map/runtime
  construction. Record public runtime construction once. Separate sequential
  profiling still uses fresh instances and reports initialization separately.

These changes improve reliability and remove repeated public preprocessing;
they do not change the candidate methods, source data, privacy budget, response
depth, attack bank or selection policy. Cached graph computation is not private
cross-session state. No performance scores were used to choose these changes.
