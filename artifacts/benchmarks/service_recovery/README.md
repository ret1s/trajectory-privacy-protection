# Prior / corridor development ablation

This study follows the failed independent service-cover confirmation at
commit `28ef61e`. **It does not contain new independent confirmation.** The
historical `confirmation` split of SQLite v3 stays unchanged; this experiment
explicitly assigns families 201–204 the role `reused_development`.

The dataset itself is unchanged: SUMO + OSM only, pinned release/hash. No new
DB release or update-log row is warranted for a change to protection methods.
The diagnostic holds old dummy histories fixed; full-run evaluation does not.
These two kinds of results must not be conflated.

## Protocol and artifacts

- [Diagnostic declaration](../../../thesis/notes/service_recovery_protocol.md)
- [Candidate declaration](../../../thesis/notes/service_recovery_candidate_protocol.md)
- `diagnostic.json`: all 227 fixed-history event interventions, K=5, replicate 1.
- `training.json`: original three method rows/models plus three new methods,
  training only on 101/102. Original rows are preserved exactly, not regenerated.
- `validation.json`, `selection.json`: 103/104, frozen candidate/attack choices.
- `development.json`: all six methods on reused 201–204; three paired RNG seeds.
- `readout.json`, `results_tables.tex`: exact aggregates, category denominators,
  paired family differences, overhead and caveats.
- `verification.json`: independent score/corridor checks and new-method replays.

The experiment is limited to K=5 and nine S1–S3 cases. Evaluator files contain
synthetic truth and internal anchors: never publish a whole artifact as the
attacker-visible transcript. Only a row's `public` object is that transcript.

## Reproduce

From the source commit, use the existing canonical diagnostic/cache or run
`venv/bin/python -m experiments.diagnose_service_cover` in a fresh worktree.
The diagnostic refuses to overwrite existing evidence. Then run stages in order
with a fresh output directory:

```bash
venv/bin/python -m experiments.run_recovery_cover --phase training --output /private/tmp/recovery-replay
venv/bin/python -m experiments.run_recovery_cover --phase validation --output /private/tmp/recovery-replay
venv/bin/python -m experiments.run_recovery_cover --phase development --output /private/tmp/recovery-replay
```

`venv/bin/python -m experiments.verify_recovery_cover --replay` verifies the
committed canonical artifacts, including every score, directed transition,
corridor and diagnostic service answer. It replays one of three RNG replicates
for all NEW methods, plus their causal prefixes. Historical rows are checked
for exact equality with the previously verified artifacts.

Result byte hashes contain execution times and therefore change on rerun.
Compare public outputs, inputs, anchors, states and scores rather than demanding
timing-contaminated JSON hashes be identical across runs. Baseline timings were
reused, so timing comparisons are not a controlled cross-method speed study.
