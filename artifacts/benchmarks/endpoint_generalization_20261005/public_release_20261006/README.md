# Seed-free synthetic evaluator bundles — 06/10/2026

These nine NEW copies omit private `rng_seed_evaluator_only` fields. Q/events,
labels, grouping, time and all other decoded JSON values match the unchanged
original protected bundles. [Manifest](manifest.json) records original/export
hashes and removal counts. These bundles include evaluator truth; only `events`
is the simulated attacker-visible view. No metric or attack prediction changed.

Original bundles and three RNG master files remain local and are ignored by
Git. Original sealed protocols/receipts retain their original hashes; these
export hashes do not replace them. Full exact sampler/order/linked-attacker
verification needs the private originals. Do not point old full verifiers at
this export and describe it as an exact private recheck.

[Export script](../../../../experiments/export_endpoint_public_evidence_20261006.py)
refuses existing output directories, validates a round trip, checks removal
and verifies the original bytes remain unchanged. Export to a NEW path for a
later independently reviewed release. Existing archived readouts and model
checkpoints remain available; the new copies permit inspection without exposing
the session sampler state.

[Release boundary and archive prerequisites](../../../../docs/reviews/2026-10-06_release_boundary.md).
