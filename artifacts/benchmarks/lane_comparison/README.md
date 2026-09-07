# Lane-state development comparison

This is a **development experiment, not a final SOTA leaderboard**. It uses the
frozen SUMO/OSM scenario suite, without GeoLife, and does not replace the earlier
`paper_benchmark` artifacts or automatically add a web-dashboard track.

- `results.json` / `results.sha256`: public releases, **private evaluator truth
  and state indices**, per-query utility, validation-selected attacks and scores.
- `audit.json`: additional report-family attack envelope, explicitly exploratory
  rather than held-out attacker selection. Also contains projection diagnostics.
- `results_tables.tex`: the two derived tables included in the canonical thesis.

Six configurations: unprotected, uniform real-containing sets, DLS,
enhanced-DLS, BR-lane with projected inputs, BR-lane with raw GPS. DLS variants
are explicit road-coordinate adaptations of Niu et al. (2014), not new SOTA
papers. Existing neural-paper comparators are not claimed as trained here.

Cases S1.A/S2.B/S3.A only; source family 91 estimates a spatial prior, 92 selects
attackers, 93 reports development results. K=3/5, two RNG replicates, six input
records / 40 retained events, 144 rows. Report-family trajectories are related;
there is only one family per role, not a statistically independent test cohort.

## Reproduce

Run from the repository root with the existing venv and the scenario-suite
network plus pinned OSM cache available. Use a fresh destination to preserve
the checked release; timings vary by machine and load.

```sh
venv/bin/python -m experiments.run_lane_comparison --output tmp/lane_rebuild
venv/bin/python -m experiments.summarize_lane_comparison --results tmp/lane_rebuild/results.json
venv/bin/python -m experiments.verify_lane_comparison --results tmp/lane_rebuild/results.json --replay
venv/bin/python -m pytest -o addopts= -q tests
```

Verify the tracked release:

```sh
venv/bin/python -m experiments.verify_lane_comparison --replay
```

Generation source hashes, input hashes, catalogue hash, RNG seeds, service POIs,
truncation and attack selection are stored. The verifier independently uses
NetworkX for native-road checks against the SciPy implementation. Replay compares
all public releases and evaluator states exactly, all POI details, and every
prefix. Timings are deliberately excluded from replay equality.

The lane service is shared inside this experiment but differs from the older
junction service. Do not combine numeric scores across the two. Read
`docs/reviews/verification_lane_comparison.md` for findings, limitations and the
next development priorities. Never serve these result files directly as an
attacker-visible response.
