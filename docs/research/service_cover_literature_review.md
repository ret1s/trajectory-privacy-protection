# Service-cover: related work and adversarial review — 2026-09-08

## Bottom line

The current defensible contribution is an **integration to be tested**: a
budgeted protected-history belief drives joint, directed-reachable dummy query
selection for an explicit POI task. Neither Bayesian remapping nor greedy set
coverage is new. A positive small-city experiment cannot establish conference
readiness. This note records a targeted review, not an exhaustive systematic
review, and separates original-paper facts from our interpretation.

## Primary sources and what they change

| Work | Verified source content | Consequence for this thesis |
|---|---|---|
| Chatzikokolakis, ElSalamouny, Palamidessi, PoPETs 2017(4), pp. 308–328 | Sec. 3: postprocessing preserves Geo-I; posterior expected-loss minimization gives a Bayesian remap. Sec. 4 cautions about fitting/evaluating a prior on the same people. | We must not claim Bayesian utility recovery as new. Our approximate filter and per-step multi-query feasible set differ from their optimal single-output remap. Reuse only training-family occupancy; test on unseen simulation families. |
| Călinescu, Chekuri, Pál, Vondrák, SIAM J. Comput. 40(6), 2011, pp. 1740–1766 | Introduction/§2: weighted coverage is submodular; ordinary greedy under a matroid has a 1/2 bound. Their stronger result uses continuous greedy. | One dummy state per persistent track is a partition constraint. The bound is 1/2 for our fixed per-step feasible set, **not** 1−1/e for ordinary greedy and not an entire-trajectory bound. |
| Oya, Troncoso, Pérez-González, ACM CCS 2017, pp. 1959–1972 | Abstract/§1: optimizing mean inference error alone can choose mechanisms with undesirable privacy/quality properties; multiple criteria are needed. | Keep Hit@100m and worst-case-by-scenario service recall alongside MAE; add mechanism-specific shadow attacks and a prior-only control. These are not a replacement for calibrated posterior/entropy analysis. |
| Buchholz et al., PoPETs 2024(3), pp. 75–93 | Publisher abstract: five design goals for synthetic trajectory publication, stressing explicit privacy unit; existing models do not meet every requirement. | Useful methodological warning, but publication of a trajectory dataset is not our online LBS task. Do not transfer their model rankings or numerical utility results. |
| Liu, Peng, Zhou, JKSUCIS 38, 478 (2026) | §6.3: ASR is defined by an identification-probability threshold 1/K. §6.4 writes DER using similarity, then describes counting dummies passing semantic constraints. §6.5 measures generation delay. | Their ASR is not our Hit@100m complement; passing semantic filters is not measured POI recall. Any reproduction needs an explicit operational resolution of the DER description. Do not infer Geo-I from their filtering parameter epsilon. |

Sources (read publisher/author sources, not secondary summaries):

- [Efficient Utility Improvement for Location Privacy](https://petsymposium.org/popets/2017/popets-2017-0051.pdf), §3 and prior discussion.
- [Matroid-constrained submodular maximization](https://chekuri.cs.illinois.edu/papers/submod_max_sicomp.pdf), pp. 1740–1742.
- [Back to the Drawing Board](https://simonoya.com/files/oya-2017-11-ccs.pdf), abstract and §1.
- [SoK: Can Trajectory Generation Combine Privacy and Utility?](https://petsymposium.org/popets/2024/popets-2024-0068.php), publisher record/abstract. Full PDF fetch was intermittent; do not treat this cycle as a fresh full-paper audit.
- [Semantic correlation of moving paths](https://link.springer.com/article/10.1007/s44443-026-00899-w), §§1, 6.3–6.6. This is a **journal** article, published June 4, 2026; no conference/quartile claim is inferred.

## Additional close work found during this cycle

**Atmaca, Biswas, Maple, Palamidessi (2024), “A Privacy-Preserving Querying
Mechanism with High Utility for Electric Vehicles,” IEEE Open Journal of
Vehicular Technology 5, 262–277, DOI 10.1109/OJVT.2024.3360302.** The author
institution's record explicitly describes approximate Geo-I plus dummy data
generation for charging-station queries, and Bayesian occupancy estimation.
This is a particularly important prior-art check: “Geo-I + dummies + useful
urban vehicle queries” is not a novel claim by itself. Its Bayesian quantity
is station occupancy, not automatically our location-belief filter. Full PDF
retrieval timed out in this cycle; a line-by-line method/metric reproduction
audit remains open. No implementation or numeric comparison is claimed here.
[Institutional record](https://wrap.warwick.ac.uk/id/eprint/183198/).

**Qiu, Liu, Pappachan, Squicciarini, Xie (2025), LR-Geo, PoPETs 2025(2), 5–22.**
Publisher abstract describes locally relevant LP geo-obfuscation, coefficient
upload/server optimization, and Benders decomposition. It is relevant non-neural
optimization literature, but not automatically a dummy-generation comparator
with our local-only trust boundary. Do not quietly replace our full-support
kernel with a secret-dependent local support and inherit its proof.
[Publisher record](https://petsymposium.org/popets/2025/popets-2025-0046.php).

**Liu, Hu, Zhou (2026), “Location privacy protection in continuous LBSs:
enhancing anonymity via fake queries,” JKSUCIS 38, 53.** Published January 3,
2026 even though its DOI contains “025”. Its abstract/introduction insert
all-dummy fake queries between real queries to disrupt path/reachability
correlation. This is closer to a future timing/query-schedule module than to
our current fixed public schedule. Treat it as a candidate comparator extension;
equal K is insufficient when methods send different numbers of query events.
[Publisher article](https://link.springer.com/article/10.1007/s44443-025-00438-z).

## Strongest reviewer objections and required evidence

1. **Incremental novelty.** Combining known primitives is insufficient by
   itself. Show what constrained online problem existing methods cannot solve,
   state an appropriately scoped theorem/objective, and compare against a
   public-prior solution. The 1/2 bound is a borrowed consequence, not new theory.
2. **Utility can be easy on a small catalogue.** Four of six categories are
   sparse. Equal-category averaging can hide errors in restaurant/cafe retrieval.
   Retain the declared primary metric and reveal category values/denominators;
   later use a new map and a declared realistic query distribution.
3. **Weak attacker = apparent privacy.** A spread-out query set may still encode
   its generator's belief. Shadow kNN can catch some patterns, but is trained on
   only two synthetic families; poor transfer does not prove security. Next:
   calibrated mechanism-likelihood inference and transfer across prior mismatch.
4. **Privacy parameter lacks practical strength.** With B=.24/m, exp(B*100) is
   exp(24), a loose bound. The present postprocessor cannot strengthen the
   inherited numeric budget. Need a predeclared budget sweep and a useful
   utility/privacy Pareto curve on new development data.
5. **One-step objective versus trajectory utility.** Greedy can consume future
   manoeuvrability. A future route-aware lookahead may use public/protected
   predictions only, not the real next segment. Verify causality and charge any
   new private decision. Include stationary/return-visit/branch failures. The
   current objective is a per-step belief-weighted surrogate, not the identical
   whole-run benchmark aggregation; all-empty latent references contribute zero
   in the objective, while measured unavailable queries are N/A.
6. **Ten scenarios are not ten solved attacks.** v3 has 30 subcase definitions
   and valid samples in their union; some train/validation cells remain zero.
   This cycle protects/evaluates S1–S3 only. Identity, destination, query intent,
   relationships and endpoints need their own adversaries and release contracts.
7. **Comparator fidelity.** Existing TransProtect/semantic routes still use
   explicit local substitutes for full neural training. Keep non-neural controls,
   but do not label this ablation as beating three faithful SOTA models.
8. **Reproducibility versus replication.** Hashes, native FCD, SQL lineage and
   replay validate one implementation. External reproduction, other urban areas,
   independent traffic families and device-level costs remain necessary.

## Next-cycle priority after inspecting the frozen results

First determine whether gains survive the prior-only and shadow controls;
inspect worst-case and dense-category recall before choosing a direction.
Do not tune on families 201–204 and then continue calling them confirmation.
If used to improve the model, mark them development and generate a new release.
Before selecting a venue, map the candidate contribution to its scope and
artifact/evaluation expectations; conference suitability is not acceptance.
