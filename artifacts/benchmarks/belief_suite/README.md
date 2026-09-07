# Belief-weighted BR-lane diagnostic

Proposed-method improvement, not a new SOTA-comparator ranking or default.
Five fixed variants, K3/5, one paired RNG replicate, nine S1.A–S3.C cases,
two families per validation/confirmation role: 360 runs in total.

- `validation.json` / `.sha256`: first-stage scores and private replay state.
- `selection.json` / `.sha256`: configuration and attack selection frozen
  **before** confirmation. Both K choose `belief24` as an infeasible fallback;
  no configuration satisfies Recall>=90% on every validation case.
- `confirmation.json` / `.sha256`: all second-stage variants, with separately
  identified validation-selected attackers and exploratory envelopes.
- `verification.json`: native-graph, arithmetic, replay and prefix checks.
- `aggregate.json`, `results_tables.tex`: reproducible thesis lookup tables.

Public context: 102,123 lane states,418 POIs; latent belief:2,076 grid states.
REM emission normalizers still use all102,123 output states. Same-coordinate
emission combines noisy reuse with fresh sampling and coordinate multiplicity.
The latent grid, synthetic occupancy prior and temporal motion are approximate.
No additional GPS read or future route is allowed in postprocessing.

All JSON results are **evaluator-private**, including anchors, belief summaries
and ground truth. The `public` field of each run is the isolated LSP transcript.
The existing web app/defaults are intentionally unchanged.

```bash
# Run in order; existing frozen results cannot be overwritten.
venv/bin/python -m experiments.run_belief_suite --phase validation --output tmp/belief_rebuild
venv/bin/python -m experiments.run_belief_suite --phase confirmation --output tmp/belief_rebuild
venv/bin/python -m experiments.verify_belief_suite --output tmp/belief_rebuild --replay
venv/bin/python -m experiments.render_belief_suite --directory tmp/belief_rebuild

# Verify the tracked release, with original pinned data dependencies available.
venv/bin/python -m experiments.verify_belief_suite --replay
venv/bin/python -m experiments.render_belief_suite --check
```

Protocols: `thesis/notes/belief_suite_v2_protocol.md` and
`thesis/notes/belief_evaluation_protocol.md`. Findings, caveats, literature and
next-cycle suggestions: `docs/reviews/verification_belief_suite.md`.

Do not merge these scores with earlier experiment leaderboards: families,
case coverage and prior changed. Compare paired variants within this release.
All-ten-scenario dataset coverage is not all-ten-scenario protection evidence.
After inspecting confirmation, any further tuning requires a new declared cycle.
