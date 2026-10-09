# Exact public-map Q-bundle diagnostic

This is an opt-in one-fresh-REM static development diagnostic, not a replacement
for the frozen S1/S2/S3/S9/S10 moving-trajectory benchmarks.

The Geo-I release is unchanged. A joint exponential kernel randomizes a public
bundle's query positions and cardinality, using a protected belief, public POI
coverage and public query cost. Inputs contain no private intent or trajectory.

The audit integrates all 66,189 native-map REM outputs, 36 publicly selected
candidate positions and 48 public bundles. It publishes every predeclared
beta in 0/1/2/4/8 under uniform and 80%-skewed priors. L=30, reference k=5,
release epsilon=0.01/m, cost weight=0.25. The deterministic K5 control optimizes
the same bundle library; it is not the full legacy motion-feasible planner.
The client belief and attacker use the same prior/REM model in this diagnostic;
this does not validate the existing approximate mobility filter's calibration.

At the illustrative beta=2 with uniform prior:

| Kernel | Expected Recall@5 | Mean K | Optimal finite-grid guessing accuracy |
|---|---:|---:|---:|
| Deterministic fixed-K5 library control | 99.10% | 5.000 | 33.91% |
| Randomized fixed-K5 | 90.64% | 5.000 | 3.09% |
| Randomized K3/5/7 | 89.12% | 4.993 | 3.09% |

Dynamic cardinality has not demonstrated a useful advantage over randomized
K5 here. The contribution with evidence is controlled randomization and its
joint likelihood/posterior bounds; dynamic K remains exploratory.

With the 80%-skewed prior, PQB's optimal guessing accuracy remains 80%: prior
knowledge is not erased. These numbers are not identity accuracy, Hit100,
moving-trajectory Recall, a SOTA comparison, or a pure-DP finite-precision
implementation certificate. Ideal-real proofs and numerical audits are separate.

`protocol.json` fixes configurations and hashes; `source_snapshot/` preserves
all declared public inputs and sources. `results.json` publishes the complete
frontier, expected cost, maximum posterior, optimal guessing probability and
likelihood checks. A second full integration is compared in `validation.json`.

```sh
python -W error::RuntimeWarning -m pytest tests/test_probabilistic_query_bundle.py
python -m experiments.query_bundle_audit_20261010 --verify
python -W error::RuntimeWarning -m experiments.query_bundle_audit_20261010 \
  --output /tmp/query-bundle-audit-replay --work /tmp/query-bundle-public-cache
```

The script refuses to overwrite evidence directories. The earlier v1 run is
retained because its BLAS runtime emitted numerical warnings; v2 directly sums
with `einsum`, treats RuntimeWarning as an error and preserves the same study
design. It makes no privacy superiority claim for the deployed engine.

Full mathematics and scope:
[`docs/research/2026-10-10_probabilistic_query_bundles.md`](../../../docs/research/2026-10-10_probabilistic_query_bundles.md).
