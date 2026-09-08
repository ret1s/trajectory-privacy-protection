# Contribution and privacy–utility–cost review — 2026-09-08

## Research claim being tested

The candidate problem is **online, road-reachable dummy query-set selection for
recovering useful POI results under a protected location history and explicit
service costs**. The intended gain is not simply increasing fake-point distance:
it is recovering the true local top-5 while resisting public-transcript inference.

The new candidate adds category-saturated coverage and bounded single-track
exchanges. These are testable design hypotheses, not a verified first-in-literature
algorithm or a statement that all ten threat targets are protected.

## Closest primary work and what cannot be claimed as ours

| Source | Established idea relevant here | Distinction / necessary test |
|---|---|---|
| Chatzikokolakis, ElSalamouny, Palamidessi, *Efficient Utility Improvement for Location Privacy*, PoPETs 2017(4):308–328 | Bayesian remapping improves location-obfuscation utility. | Protected-history filtering/remapping itself is not novel. Compare selectors using identical private anchors, public map and prior. |
| Oya, Troncoso, Pérez-González, *Back to the Drawing Board*, ACM CCS 2017:1959–1972 | Average inference error alone can conceal important differences; examine additional privacy and quality criteria. | Report Hit and MAE separately, plus minimum-case/individual-category utility. A capped expectation is not a worst-case quality guarantee. |
| Oya, Troncoso, Pérez-González, *Rethinking Location Privacy for Unknown Mobility Behaviors*, IEEE EuroS&P 2019:416–431 | Models fixed from training data can fail for different mobility; the paper studies adaptive profile estimation. | Our finite filter and uniform prior are approximate, not a new solution to unknown mobility. Current same-map development does not establish distributional robustness. |
| Krause, McMahan, Guestrin, Gupta, *Robust Submodular Observation Selection*, JMLR 9:2761–2801, 2008 | Truncating and summing submodular objectives is central to saturation-based robust selection. | Category capping is inherited. Our fixed cap plus partition-greedy and three exchanges is **not SATURATE**, and cannot borrow its bicriteria guarantee. |
| Călinescu et al., *Maximizing a Monotone Submodular Function Subject to a Matroid Constraint*, SIAM J. Comput. 40(6):1740–1766, 2011 | Matroid-constrained submodular optimization and the distinction between ordinary and continuous greedy. | Ordinary greedy gives the classical 1/2 bound. No 1−1/e claim for this implementation; local exchange only preserves/improves its own per-step objective. |
| Liu, Peng, Zhou, *A dummy-based location privacy protection scheme with semantic correlation of moving paths*, JKSUCIS 38:478, 2026 | Learned semantic/spatio-temporal sequence consistency for real-plus-dummy query sets. | Our dummy-only contract lacks an included true query, so utility must be measured after retrieval. Service-category balance does not establish semantic plausibility or resistance to their attack model. |

Primary sources inspected:

- https://petsymposium.org/popets/2017/popets-2017-0051.pdf
- https://simonoya.com/files/oya-2017-11-ccs.pdf
- https://simonoya.com/files/oya-2019-06-eurosp.pdf
- https://www.jmlr.org/papers/volume9/krause08b/krause08b.pdf
- https://chekuri.cs.illinois.edu/papers/submod_max_sicomp.pdf
- https://link.springer.com/article/10.1007/s44443-026-00899-w

This is a targeted closest-work review, not an exhaustive novelty clearance.
The semantic-correlation work is a journal article published 4 June 2026, not a
Q1 conference. Its original metrics/contract remain in thesis Chapter 4; the
new study is an internal matched ablation, not a replacement SOTA leaderboard.

## Falsifiable contribution map

1. **Formulation:** an ordered K-query set must remain reachable from its own
   public previous states. Its value is the union of response IDs, not K
   independent distances from a private point. This formalizes this thesis's
   service task; do not assert prior literature never considered related sets.
2. **Candidate method:** preferentially allocate remaining query coverage to
   under-covered categories and refine early greedy assignments. The exchange
   ablation isolates optimizer quality from the change of objective.
3. **Evaluation framework:** keep private budget, K, response depth, references,
   attacker training and selection fixed within comparisons; report resource
   frontiers and disaggregate difficult cases. This is a reproducible operational
   specification, not a new universal location-privacy standard.

Failure tests: if expected objective improves but actual recall falls, the
surrogate/prior or future reachability is inadequate. If cap hurts privacy,
matching the formal ledger does not excuse the empirical loss. If top-10
improves all methods similarly, credit the retrieval allowance, not the proposed
selector. If lower leakage needs substantially more requests or bytes, report
that price. If advantages occur only on already reviewed families, retain the
development label and do not claim generalization.

## Interpretation boundaries

- Recall@5 measures overlap with the five closest reachable reference POIs;
  result completeness measures cardinality, not exact correctness.
- Fixed-depth public-catalogue replies are a deterministic function of the
  query set. More results need not add information in this model, but consume
  more response bytes. This stops being automatic if depth depends on private
  GPS, the client reveals its ranking, or metadata exposes private intent.
- The service queries all six categories and assumes local road-distance
  ranking. This is not a paid production LSP integration, and ID-list JSON bytes
  are not actual HTTP/POI-detail payload or mobile latency.
- Same B is an ideal inherited guarantee, not equal absolute privacy. The
  three budgets give exp(12), exp(24), exp(48) at 100 m under the session bound;
  these remain loose. The code does not certify floating-point DP.
- The 80-family auxiliary release is already sealed. Use only its designated
  64 training families here; preserve the 16-family holdout. Reused 201–204
  remain development even though their historical SQL split says confirmation.

## Next decision

Promote no algorithm from mean recall alone. First inspect the fixed-budget,
fixed-depth ablations and validation-feasible choices, including all negative
case/category deltas and inference envelopes. Any promising configuration
requires a separately declared validation/confirmation cycle before a paper
can claim superiority or robust protection.
