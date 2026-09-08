# Prior mismatch, constrained remapping and closer dummy work

Date: 2026-09-08. Targeted primary-source review, not a systematic survey.
This follows the negative service-cover experiment; original evidence is frozen.

## What the literature actually supports

Chatzikokolakis, ElSalamouny and Palamidessi (PoPETs 2017), §3.3, explicitly
warn that a global-prior remap can hurt users whose behaviour differs from the
training population. They suggest avoiding remapping when mismatch is large;
§4.1.1 already includes a low-data fallback. Therefore neither prior-awareness
nor a safeguard around a protected output is new by itself. Their §3.1 also
discusses restricting remapping candidates near the protected output. Our
directed per-track corridor adds a different feasible-set constraint, not a new
privacy definition. [Authoritative paper](https://petsymposium.org/popets/2017/popets-2017-0051.pdf).

Atmaca et al. (IEEE OJVT 2024) was retrieved successfully this cycle. Inspected
§V.E, §VI and §VII: one AGeoI-obfuscated position plus feasible dummies; Edge
shuffling; extra travel distance (CoP), zero-CoP fraction, identifiability and
occupancy-distribution utility. Thus it is close related work, but its Edge
trust boundary and truncated kernel differ from ours. Its IBU estimates
aggregate query-location distribution, not the target's position posterior.
Counting possible paths alone does not justify equal posterior probabilities;
our evaluation must retain mechanism-aware inference.
[Published PDF](https://wrap.warwick.ac.uk/id/eprint/183198/1/WRAP-privacy-preserving-querying-mechanism-high-utility-electric-vehicles-2024.pdf).

PDF inspected: 16 pages, SHA256
`4ddd2da6d9c4f69a557439a0e3014818af6721163e7a7d8c30a9dc6c3736e118`.
This closes the earlier retrieval gap, not a full reproduction/proof audit.

## Implications for this candidate

- Keep the original full-support anchor mechanism. A restriction around its
  protected output is postprocessing; truncation around raw GPS would be a
  different mechanism requiring a different proof. No hidden rejection loop.
- Distinguish mathematical privacy from inference using an approximate mobility
  prior. A uniform-cell control removes training-occupancy dependence, not every
  mobility assumption. It may be less accurate under real population patterns.
- Keep the ordinary greedy 1/2 statement restricted to the fixed candidate
  groups. A smaller corridor can remove the unconstrained best service points.
- Do not rename conditional extra distance as an exact reproduction of CoP:
  our task returns up to five POIs in six separate workloads, with local ranking.
- A stronger utility score can make selected attacks more accurate. Evaluate
  Hit and MAE jointly with recall; no declaration of success based on recall only.

## Objections this cycle must leave visible

1. The new methods are chosen using diagnostics on the old confirmation
   families. They cannot establish independent generalization on those families.
2. Fixed-history interventions hold preceding states constant. They identify
   one-step contrasts under that history, not the effect of whole trajectories.
3. A 200 m corridor is a development design choice, not a learned optimum or a
   real-user utility constraint. Directed distance does not model traffic lights.
4. Shadow attacks still learn from only two synthetic families. Neither a zero
   observed Hit nor a worse empirical attacker proves stronger privacy.
5. The inherited B=.24/m bound is still loose at 100 m. This ablation cannot
   replace a budget sweep, proof audit, faithful comparator training or evaluation
   on other cities and protected targets.

Next confirmation should be run only after a useful candidate is fixed and
the study budget/attack/utility conditions are declared. If development fails,
keep that result instead of repeatedly querying fresh seeds for a lucky winner.
