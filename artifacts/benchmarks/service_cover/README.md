# Joint service-cover diagnostic

New method: `ServiceCoverLaneDummy`, a postprocessor of the existing bounded
private-anchor mechanism. All runs read **SQLite release urban-scenarios-v3**
with an explicit semantic SHA256; there is no `latest` lookup or JSON fallback.

Study order: training (101/102), validation and frozen selection (103/104),
confirmation (201–204). Four methods, K=3/5, three paired RNG repetitions,
nine S1–S3 subcases. `prior_cover` is an input-independent negative control
conditional on the public schedule, not an eligible proposed-method candidate.
No S4–S10 protection results are implied.

Files contain **synthetic evaluator truth/internal anchors**. Do not expose a
complete result JSON to an adversary or a public dashboard. Only each row's
`public` object implements the LSP-visible transcript contract.

## Reproduction

The existing output directory is immutable for experiment stages. Reproduce
into a fresh path, sequentially, from the frozen source commit:

```bash
venv/bin/python -m experiments.run_service_cover --phase training \
  --output /private/tmp/service-cover-replay
venv/bin/python -m experiments.run_service_cover --phase validation \
  --output /private/tmp/service-cover-replay
venv/bin/python -m experiments.run_service_cover --phase confirmation \
  --output /private/tmp/service-cover-replay
venv/bin/python -m experiments.verify_service_cover --replay \
  --output /private/tmp/service-cover-replay
```

Scientific arrays are deterministic in the pinned environment; execution times
are not. Thus result byte hashes change during reruns even when scientific
outputs match. Compare public outputs, inputs, anchors, scores and selections,
not timing-contaminated result file hashes across machines.

`verification.json` checks every input, score and emitted transition. With
`--replay`, it regenerates replicate 1 for every phase/family/case/method/K and
checks its causal prefix; replicates 2/3 still receive input, score, anchor,
budget and transition verification. `readout.json` and `results_tables.tex`
come from `experiments.export_service_cover` and keep the metric denominators.

Further context:

- [Prespecified protocol](../../../thesis/notes/service_cover_protocol.md)
- [Literature critique](../../../docs/research/service_cover_literature_review.md)
- [Verification and next steps](../../../docs/reviews/verification_service_cover.md)

Latency is a shared development-machine measurement, not a controlled mobile
benchmark. The mathematical privacy bound is inherited and numerically loose
at the chosen B. Shadow kNN is not an optimal adversary; this is not evidence
that the thesis already beats faithful SOTA implementations or is submission-ready.
