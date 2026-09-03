# Claude Code repository guide

This is a research repository. Preserve the distinction between a theorem,
an executable integration test, and empirical evidence.

## Read first

1. `README.md` — current commands and directory map.
2. `docs/research/problem_formulation.md` — system/output/threat contract.
3. `benchmark/README.md` — comparator fidelity and missing components.
4. `docs/reviews/README.md` — current verification entry points.
5. `thesis/main.tex` — canonical final-thesis source.

## Active boundaries

- `core/`: active REM-family mechanisms, road graph, and protocol only.
- `benchmark/engines/`: algorithmic implementations.
- `benchmark/methods/`: benchmark adapters and evidence cards.
- `experiments/`: reproducible CLI orchestration; do not trigger it from HTTP.
- `web/`: active benchmark dashboard and thesis simulator.
- `archive/`: historical material; never import it from active code.
- `outputs/`: current generated experiments; `output/pdf/`: curated PDFs.

The matching `benchmark/engines/*` and `benchmark/methods/*` names are
intentional. Do not collapse them without preserving the algorithm/evidence
separation.

## Required checks

```bash
venv/bin/python -m tests.run_all
venv/bin/python -m pytest -q tests
venv/bin/python -m pip check
```

When changing the dummy benchmark, run a quick job with every output sent to
`/private/tmp` so committed evidence is not overwritten. When changing the
thesis, compile `thesis/main.tex` with XeLaTeX into a temporary build directory,
then copy only the reviewed PDF to `output/pdf/graduation_thesis.pdf`.

## Claim safety

- Do not call paper adaptations “faithful reproductions” unless the strict gate
  passes and evidence parity exists.
- Do not compare privacy metrics across incompatible output contracts as one
  ranking.
- Do not describe displacement, DD, realism, or POI utility as privacy by
  themselves.
- Keep attacker-visible data separate from evaluator-only truth.
- Keep per-event Geo-I distinct from trajectory/window privacy and composition.
- Keep historical review files immutable; add a new review for a new commit.

The archived Internship 2 implementation is useful as provenance, not as the
current formal mechanism.
