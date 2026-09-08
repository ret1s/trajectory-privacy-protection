# Prior factors and loss-aware inference: targeted literature review

Reviewed 2026-09-08. This is a targeted update, not an exhaustive systematic
survey or a claim that all 2026 work has been reproduced.

## Primary evidence and its practical consequence

1. **Chatzikokolakis, ElSalamouny, Palamidessi, PoPETs 2017**, 4:308--328,
   Sections 3/3.3. Remapping optimizes a loss under a posterior, whose usefulness
   depends on the prior. A global prior can disadvantage atypical users.
   Therefore prior adaptation/mixing is not itself our novelty. Separate
   initialization, transition/reset and public discretization effects before
   interpreting the uniform-control gain. Section 3.2 also explicitly separates
   the centroid's squared-distance optimum from ordinary Euclidean loss and
   discusses Weiszfeld's algorithm. Our finite-action choice is a limited
   diagnostic, not a replacement for that continuous optimization result.
   [Published paper](https://petsymposium.org/popets/2017/popets-2017-0051.pdf).

2. **Reza Shokri, Privacy Games, PoPETs 2015**, 2:299--315,
   Sections 3.3/3.4, 6.1 and the evaluation discussion around Figure 4.
   Optimal inference accounts for the loss function; a generic Bayesian attack
   can overestimate privacy if its decision rule ignores that loss. This
   motivates distinct decisions for Euclidean error and radius-hit success.
   Our neighbor distribution is empirical, not the exact posterior in that
   paper, and we do not reproduce its linear-program optimal protection.
   [Published paper](https://petsymposium.org/popets/2015/popets-2015-0024.pdf).
   The prespecified protocol originally cited arXiv 1402.3426v3; the final
   publication has different section numbering and is the thesis citation.

3. **Geo-indistinguishable location obfuscation with inference error bounds**,
   Journal of Complexity 91 (December 2025), article 101970,
   DOI 10.1016/j.jco.2025.101970. Publisher-indexed abstract/introduction report
   flaws in PIVE's adaptive protection-set privacy/error claims and discuss
   corrected approaches. Full-page access returned 403; no proof reproduction
   was possible this cycle. The relevant warning for us is not to assume an
   attacker must guess within the defender's candidate/protection set. Our new
   attacks can choose off-road coordinates, including circle centers, and are
   scored against raw GPS. This is an evaluation design inference, not a new
   theorem derived from the unavailable proof.
   [Publisher](https://www.sciencedirect.com/science/article/pii/S0885064X25000482).

4. **Mohammed Hasan A. and Nemi Chandra R., LAHEC**, Scientific Reports,
   published 27 August 2026, DOI 10.1038/s41598-026-66452-x. Publisher page
   explicitly labels this an accepted early version, not the final Version of
   Record. Its abstract describes diffusion/sequence-VAE dummy trajectories
   filtered by POI/population-oriented elliptical validators. This is relevant
   to novelty screening, but neither reported privacy nor numerical superiority
   has been verified here. The PDF link returned HTML, not a PDF; only publisher
   metadata and abstract were inspected. Full methods, assumptions, metrics and
   code must be obtained before promoting it to a benchmark comparator.
   [Publisher](https://www.nature.com/articles/s41598-026-66452-x).

## Critical distinctions retained in the implementation

- The public lane KDE is a density sampled at lane states. Summing it per cell
  adds a state-count effect; C uses mean density per occupied cell instead.
  This does not establish which measure matches real mobility best.
- (L,U) versus (U,U) tests initialization including fixed tie-center changes.
  (U,L) versus (U,U) tests transition weighting plus long-gap reset. The 2x2
  design cannot separately identify reset and local movement without another
  ablation. All outputs have their own previous dummy states.
- A conditional mean minimizes squared distance, not generally unsquared
  Euclidean error or radius-hit loss. Our finite-action MAE and disk decisions
  solve empirical objectives on neighbor labels; generalization can still fail.
- Radius-hit construction includes support coordinates, their mean and both
  circle centers through feasible point pairs. Coordinates need not lie on a
  road or in the dummy set. Numerical tolerance used to form disks is not
  silently added to the benchmark's strict distance <=100 m threshold.
- Causal shadow features contain only the public candidate prefix and elapsed
  time. Old full-window adversaries remain in the bank; online protection does
  not imply a server must forget previous data or cannot attack retrospectively.
- More attacks cannot lower the per-case exploratory maximum Hit, but a newly
  validation-selected attack can do worse out of distribution. Report both;
  neither value is a bound on all possible attackers.
- The actual shadow bank has 360 training observations but only **36 distinct
  XY labels**, shared across scenarios/repeats from two families. These are not
  360 independent locations or users. A richer decision rule alone cannot fix
  this narrow auxiliary-data support; broader shadow-route generation is a
  high-priority requirement before confirmation.

## Publication gates, not an acceptance promise

| Gate | Evidence still required |
|---|---|
| Specific contribution | A precise algorithmic/problem distinction from remapping, semantic dummy and query-cover work; mixing priors alone is insufficient. |
| Formal claim | Audit actual sampling, floating-point behavior, adjacency and composition; current exp(24) at 100 m is too loose to use as persuasive absolute privacy. |
| Attack validity | Likelihood-aware/sequence inference or substantially richer shadow training; calibrated uncertainty and multiple attacker side-information assumptions. |
| Utility/generalization | Resolve below-threshold cases, then locked fresh families, more than one city/catalogue, longer sessions and privacy-budget curves. |
| Fair comparison | Faithful original-paper implementations or explicitly justified adapters; equal output/trust/query-budget contracts. |
| Statistics | Independent trip/family units, stated effect sizes and uncertainty; do not treat steps or RNG replicas as independent users. |
| Reproduction | Pinned SUMO/OSM/SQLite/source manifests, negative results, replay checks, data-access boundaries and runnable commands. |

The current iteration is development evidence. A well-documented negative
factor/attack result is useful, but does not by itself establish publishability.
