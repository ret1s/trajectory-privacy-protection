# Fake-query insertion: common-service adaptation

The comparator implements the causal sequence in Sections 4.1–4.2 of
[Location privacy protection in continuous LBSs: enhancing anonymity via fake queries](https://link.springer.com/article/10.1007/s44443-025-00438-z).
It advances a dummy reference, selects companion movements by path similarity,
inserts dummy-only query events, and uses a single-query DLS fallback when a
real query cannot continue the current set. Reference candidates are tried in
historical-weight order. An infeasible fake set is suppressed and counted.

The runnable adaptation uses disjoint population history in place of unavailable
personal query logs. Historical 100-m cells are constrained by directed SUMO
lane travel. Path features are relative displacement length and heading; local
zero-distance handling is explicit. The locked configuration has K=5, at most
two fake queries per real event, waits uniformly from 5–15 seconds, at most
24 candidates per old point, and similarity threshold 0.75. It does not inspect
future true positions. All six service categories are requested at every
published coordinate under the common workload; paper-specific query-content
semantics are not reproduced.

The evaluator counts all emitted queries and response bytes. Real/fake flags
and genuine-event indices remain outside attacker inputs. Fake events within
an allowed observation interval remain visible; S1's single-event access does
not turn into a longer observation window. These choices do not establish the
paper's indistinguishability theorem, its ASR values, or complete reproduction.
