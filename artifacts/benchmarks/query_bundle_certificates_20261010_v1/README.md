# PQB privacy and actual-utility certificate audit

Opt-in, exact **one-fresh-REM static development** audit on the same public
native map/POIs as the preceding bundle study. It does not replace any frozen
moving-trajectory results or switch the active K5 engine.

The public certificate computes
`epsilon_Q = beta/2 * max_(a,a') range_x(g(x,a)-g(x,a'))/(1+lambda)`.
It bounds arbitrary protected-belief pairs and includes observed cardinality.
No true-GPS-on-grid assumption is needed for that kernel privacy bound.
Actual utility on/off the proxy grid remains a separate question.

An offline public coverage floor removes low-coverage bundles using ALL public
proxy states, rather than selecting support from the current user's GPS/Z.
Empty support is reported as infeasible. The nearest-proxy utility floor does
not certify fastest/radius/detour or a physical user's off-grid ranking.

At beta=2, uniform prior, same K5 and same reply L30:

| Public library | Expected Recall@5 | Ideal epsilon_Q bound | Bayes optimal grid guessing |
|---|---:|---:|---:|
| Original K5 library | 90.64% | 0.4800 | 3.09% |
| K5 with public floor0.75 | 92.51% | 0.2933 | 2.93% |

The second library has only2 bundles and its actual proxy floor is76.67%.
This is a development result on a finite public36-state utility grid. It is
not a cross-map, moving-family, identity or SOTA claim. Fixed K5 remains a
useful option; variable K is not necessary for either theorem.

The full30-case table retains modes K5/K3/5/7; public floors0/.75/.8/.9;
literal beta2/public epsilon1 calibration; matched uniform/skewed80 priors,
and client-uniform/truth-skewed80 mismatch. All three infeasible library/floor
pairs are retained. Floor.8 leaves only K7 in the dynamic-capable mode, with
40% more query positions than K5. Strict float comparisons do not round a
coverage value0.799999... up to a certified.8 floor.

Utility certificates include surrogate regret, TV belief mismatch, proxy-error
allowances and actual expected utility. The mismatch audit computes TV under
its declared finite model; it does not measure the production filter's
calibration. Private-belief certificates stay local.

`protocol.json` freezes configurations, baseline protocol hash and source pins;
`source_snapshot/` preserves sources/inputs; `results.json` contains complete
rows, public floor failures and inspected local certificates. No Monte Carlo
draw or fitted attacker is involved. `validation.json` records checks and an
independent full reexecution. Ideal-real proofs and float verification are
explicitly separated; no finite-precision pure-DP certificate is claimed.

```sh
python -W error::RuntimeWarning -m pytest tests/test_query_bundle_bounds.py \
  tests/test_probabilistic_query_bundle.py tests/test_belief_lane.py \
  tests/test_response_aware_belief.py tests/test_session_budget.py
python -m experiments.query_bundle_certificate_audit_20261010 --verify
python -W error::RuntimeWarning -m experiments.query_bundle_certificate_audit_20261010 \
  --output /tmp/query-bundle-certificate-replay
```

Full derivations and interpretation:
[`privacy and utility proof`](../../../docs/research/2026-10-10_query_bundle_privacy_utility.md).
