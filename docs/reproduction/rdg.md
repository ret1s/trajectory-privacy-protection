# RDG: common-service adaptation

The executable comparator in `benchmark/paper_comparators.py` follows
Algorithm 6 in the [authors' manuscript](https://arxiv.org/abs/1805.06104):
build a DLS candidate pool, add candidates greedily by the entropy of normalized
max-product path weights, and propagate the forward posterior between queries.
The implementation distinguishes the max-product selection objective from the
sum-product posterior update. Tests compare these operations numerically.

The pinned source for this implementation is the 2018 manuscript; this does
not establish parity with every detail or table of the final TMC publication.

Local choices are explicit: SUMO lane states replace paper grid locations;
query and transition statistics use disjoint background trips on 100-m cells;
additive transition smoothing is 0.1; K=5 and the dummy pool contains 20 points.
The initial set uses the existing DLS implementation with 50 subset trials.
Public candidate identifiers are event-local, and the real member is shuffled.
The probability-order optimisation is tested to preserve the original DLS
pool and seeded output, including ties and catalogue boundaries.

This is a **paper adaptation**, not a reproduced paper result. The common
attacker bank includes empirical-transition and movement-based Viterbi attacks.
