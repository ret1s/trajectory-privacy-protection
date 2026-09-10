# Rejected simulation build — not benchmark input

This first build is preserved for audit. The independent native check failed:
u309_03, t=426→427 s, displacement=10.866513956312815 m >10.1 m.
Do not import this release into the benchmark DB or use it for protection scores.

The replacement is `../urban_fresh_v2/`: same 12 seeds, consistent 3-second
continuous lane-changing, unchanged physical acceptance thresholds. See
`thesis/notes/fresh_simulation_amendment.md`. No FCD point was manually edited.
