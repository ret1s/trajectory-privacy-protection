# Retained first numerical audit

This first development run completed with finite outputs, but NumPy 2 / macOS
Accelerate `matmul` emitted floating-point runtime warnings during integration.
It is preserved with its original source snapshot and numerical results.

The current authoritative diagnostic is
[`../query_bundle_kernel_20261010_v2/`](../query_bundle_kernel_20261010_v2/).
Version 2 uses direct `einsum` summation, runs with RuntimeWarning treated as an
error, and repeats the unchanged protocol and full parameter frontier. No
trajectory, data sample, parameter or library construction was changed to make
the scores better. This run is not a new moving-trajectory benchmark.

The current script pins the v2 source; do not expect current-source verification
to accept this historical v1 snapshot. Its own frozen snapshot is retained for
inspection and replay.
