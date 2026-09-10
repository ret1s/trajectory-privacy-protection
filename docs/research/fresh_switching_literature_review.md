# Literature review and adversarial critique — 2026-09-10

## Primary sources and the design decisions they support

1. **Oya, Troncoso, Pérez-González (IEEE EuroS&P 2019), Rethinking Location Privacy
   for Unknown Mobility Behaviors.** [Author PDF](https://simonoya.com/files/oya-2019-06-eurosp.pdf).
   Sections III–IV distinguish initial location distributions, Markov transitions
   and output-history remapping. Sections V–VII examine mismatch between design and
   actual mobility. Their adaptive profile approach has limitations under strongly
   correlated continuous use; it is not evidence that any adaptive filter improves
   privacy. Our inference: separate training, selection and new-family confirmation;
   retain stronger-attack envelopes and negative transfer. Our switching filter does
   not reproduce their profile MLE and must not be presented as their algorithm.

2. **Rabiner (Proceedings of the IEEE 77(2), 1989, pp.257–286), A Tutorial on Hidden
   Markov Models and Selected Applications in Speech Recognition.**
   [Paper PDF](https://www.cs.cornell.edu/courses/cs481/2004fa/rabiner.pdf).
   The finite forward recurrence sums previous state mass through transition
   probabilities and multiplies by observation likelihood. We use that established
   inference operation over the joint (location cell, movement mode) state. This is
   neither a new HMM algorithm nor an implementation of the Kalman IMM algorithm.
   Two modes and their constants are modeling assumptions to test, not discoveries.

3. **Eclipse SUMO: Sublane Model / Simple Continuous Lane-Change Model.**
   [Official documentation](https://eclipse.dev/sumo/docs/Simulation/SublaneModel.html).
   The default instantaneous change between lanes differs from continuous lateral
   motion. `--lanechange.duration` enables a simpler continuous model. A native
   same-seed diagnostic motivated a uniform three-second duration before scoring.
   [FCD documentation](https://eclipse.dev/sumo/docs/Simulation/Output/FCDOutput.html)
   describes the position/velocity observation output; it does not supply privacy
   threat labels, identities or user query intent. Those remain explicit synthetic
   scenario-layer constructs in this project.

## What could still invalidate a favorable result?

- **Wrong latent model:** public isotropic diffusion and two fixed modes do not model
  traffic lights, directional route preference or individual destinations. Output
  road feasibility and latent probability calibration are different properties.
- **Unfair access:** the filter must use protected anchors, not private speed or
  scenario labels. Its internal mode probabilities must not be sent to the LSP.
  Changing the public postprocessor alone does not spend another privacy budget;
  this relies on the previously stated ideal anchor assumptions.
- **Training support:** 3,233 observations still contain far fewer independent
  locations and trips. Expanding validation to six families helps selection, but
  does not approximate every adversary or remove same-city bias.
- **Repeated evaluation:** old development results may motivate the candidate but
  cannot validate it. Freeze the new protocol, select on 301–306, then score 307–312.
  Do not retune after confirmation. Report the two overlapping stopped windows and
  the predeclared exclusion sensitivity rather than claiming fully disjoint inputs.
- **Unequal exposure:** server top-10 can improve Recall without improving the
  generator. Compare methods within the same B/K/L and report reply-ID payload.
  Neither an average recall nor a larger inference error is a privacy guarantee.
- **Broad coverage claims:** 30 generated case types are dataset coverage, not 30
  proven defenses. This experiment is S1–S3 only; S4–S10 need target-specific attacks,
  sufficient class support and a separate evaluation protocol.
- **Contribution novelty:** HMM filtering, Bayesian remapping and greedy/exchange
  optimization are inherited tools. A publishable contribution would need a clear
  problem/assumption gap, justified integration, repeatable advantage, calibrated
  costs and faithful external comparators. This round can support a supervisor
  report even if it rules out the new candidate; it cannot by itself establish that
  the method exceeds SOTA or meets an unspecified conference's acceptance bar.

## Next development gate (not tuned in this experiment)

If switching fails, inspect calibration using development-only anchor likelihoods
and public directional transitions; avoid adding modes until there is evidence for
them. If top-10 passes average utility but one case fails, retain the failure and
design a distinct service-budget policy rather than retrospectively relaxing the
90% criterion. Improve sparse S5/S6/endpoint sample support using public route gates
and finer scenario observation design, then validate on another fixed cohort.
