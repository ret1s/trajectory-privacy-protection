# Formal v2 mathematical source review — 7 October 2026

**Status: PASS under the stated ideal and conditional assumptions. No remaining material mathematical issue was found in the four modules below.** These are analyses of existing primitives and service contracts, not a new privacy primitive, a production sampler certificate, a calibrated posterior guarantee, or additional empirical improvement.

## Scope, authorship and stable source pins

The boundary reviewer independently cross-reviewed accuracy, inference and the belief bridge written by the native, identity and root agents. The reviewer authored ranking robustness; its author checks are distinguished from the identity agent's **independent read-only ranking cross-review**, which also returned PASS at the same source hash. Owners confirmed source completion; root's final bridge adjustment only identifies the public slack parameter and keeps the theorem statement together across a page break.

| Reviewed source | SHA-256 |
|---|---|
| `thesis/current_formal_accuracy.tex` | `06df4f8f0654d5230a13a969a0e45e10914f7f111337acbb85b5201bd6f27bf4` |
| `thesis/current_formal_inference.tex` | `b07255e86c447d9142743124e71045af420d836889a720dfbcdac4a4bc2a47c0` |
| `thesis/current_formal_ranking_robustness.tex` | `138963ee3cf727e4e437f8bd5cf8f9a1a01de8df7c6179646108d3b988c6a784` |
| `thesis/current_formal_belief_bridge.tex` | `0297177f143b77a890cf9ac1e776192338191c4934677bce0d23ffe0f4aa0054` |

Reused proof dependencies remain:

| Source | SHA-256 |
|---|---|
| `thesis/current_formal_privacy.tex` | `b3a6f09db962ea0211addc5166ca6495aaef0004e8a4f80ed9b3b0dae3b6dd3a` |
| `thesis/current_formal_service.tex` | `db15e58820f63ae35fdcb3aaf73132cc9fc328250c9becb6351336d8d4fa58a1` |

This receipt authenticates mathematical source, not the final PDF release. Final rendering/layout reviews must separately identify their PDF bytes. The previous proof/PDF review remains an immutable review of its earlier release; this note does not overwrite or silently extend it.

Final integration bytes were also authenticated: `build/thesis/main.pdf`, SHA-256 `d3d51ad7f97995cc098b3ba7a0546e679214a36501c137e885b8d0ca14e3f920`. Root reports 95 pages and completed PDF QA; the identity reviewer reports independent final ranking/bridge source and page reviews. Those visual reviews are separate from this mathematical source review.

## Accuracy: finite support, reuse and adaptive reads

- The REM CDF and quantile correctly sum **every support ID**, including coordinate multiplicities. The distribution is map/input dependent; a planar polar-Laplace quantile cannot be substituted for it.
- The geometric tail bound follows from an inside-weight lower bound `n_a exp(-u a/2)` and outside-weight upper bound `n_>r exp(-u r/2)`, since `B/(A+B)` increases with B and decreases with A. The sufficient-radius rearrangement is correct; `n_>r` still depends on r, while replacing it with `|V|-n_a` gives a looser explicit bound.
- The exact post-read mixture is `q(d) 1[d>r] + (1-q(d)) T_x(r)`, conditional on a fixed pre-read history/input and independent fresh REM/test randomness. For `r≥theta`, the bound `T_x(r)+b(r)(1-T_x(r))` follows in both d≤r and d>r cases. Combining component failures yields `alpha_R+alpha_T-alpha_R alpha_T`; assigning each component the full requested failure rate would not prove that rate for the mixture.
- After a skipped read, `error_t≤error_last+w(last,t)` is a **pathwise measured-input displacement** statement. The graph's or estimator's 8 m/s cap does not establish that input assumption; true speed plus bounded sensor errors requires the additional 2b term. Pacing 60 s is a minimum read interval, not a maximum anchor age after budget denial.
- Adaptive last-read selection is handled correctly. A uniform conditional bound before each allowed read gives `Pr(read_i and bad_i)≤alpha_0 Pr(read_i)`. Summing over the finite public event list and using a pathwise read-count ceiling M gives `Pr(any bad allowed read)≤M alpha_0`. There is no invalid transfer of a fixed-read confidence level to the read selected retrospectively as the last one.
- These statements concern internal Z and measured GPS inputs. They do not bound every Q, physical localization error, future road feasibility, POI Recall or an executable float distribution. A common confidence bound additionally needs uniform input-domain tail control and an anchor-age/displacement hypothesis.

## Inference: decision loss, prior and composition

- A bilateral likelihood-ratio bound `exp(±alpha)` gives `TV≤tanh(alpha/2)` by the pointwise density inequality and integration. Balanced optimal Bayes success is `(1+TV)/2≤logistic(alpha)`.
- For arbitrary prior p, the **expected** optimal MAP success bound `max{p,1-p,logistic(alpha)}` is correct. Both decision-event and complement constraints define the four-vertex feasible polygon; evaluating its linear success objective gives the result, including alpha = 0 and degenerate priors. This is not the posterior bound at a chosen rare event.
- For M hypotheses, the posterior expression `k pi_j/(k pi_j+1-pi_j)` requires an **all-pair** bound. Under uniform prior, summing over disjoint decision regions proves the expected success bound `k/(k+M-1)`, including non-atomic observations. A ball of trace radius r gives diameter at most 2r, not r. M counts hypotheses, not the number K of Q coordinates or identities.
- Finite-epoch composition correctly assumes uniform conditional kernels at the **same full extended protected history**, then multiplies their likelihood-ratio factors. It is not inferred merely from a guarantee for a newly initialized engine or from the shorter server-visible history. Trace correlation does not invalidate that conditional proof, but raw retained state, secret-dependent resets, reused/non-independent entropy or an unanalysed initialization would require another argument. New epochs do not erase observations already retained.
- The sensor corollary is valid under its explicit coupling: if `D_infinity(measured,measured')≤a D_infinity(true,true')` almost surely, integrating the protected-input kernel yields factor `exp(C a D_infinity(true,true'))`. The mechanism entropy must be independent of the sensor law and samples must be valid under the same public context/schedule. Common additive error in the projected plane and a common public rectangular clipping projection supply an ideal nonexpansive example; arbitrary map matching does not. The local-ranking GPS diagnostic is explicitly not evidence for this mechanism-input coupling.

## Ranking robustness: signed directed bounds and stable hits

- With `d_plus=D(v,vhat)` and `d_minus=D(vhat,v)` finite, nearest score error lies in `[-d_plus,d_minus]`, directly from the two directed triangle inequalities. Mutual reachability preserves reachable POI/destination domains. Fastest has the analogous time interval; meters are not silently converted to seconds.
- For a fixed reachable detour destination, error is `e_p-e_a`; pointwise absolute error is at most `d_plus+d_minus≤2rho`. In a POI comparison, the common destination term cancels, so **pairwise score-difference** error has the tighter bound `d_plus+d_minus`. The implementation's nonnegative clamp leaves ideal detour distances unchanged; float errors have no certified bound here.
- The generic boundary gap `gamma>2 eta_score` preserves the **set** of top-k POIs on a shared finite eligible domain. It need not preserve their internal order and is sufficient rather than necessary. The tighter pairwise bound can replace 2 eta. Each received reference POI with sufficient margin against all outsiders must survive local top-k because at most `r_ref-1≤k-1` reference competitors can precede it. This proves the partial Recall lower bound without requiring full candidate coverage or redefining the true denominator.
- Within-radius filtering needs domain stability separately from finite score stability. Outside `[r-rho,r+rho]`, membership cannot change. If that band is populated, the possible estimated domain lies within `E union B`; the stable-reference condition includes both guaranteed membership and a margin against **every possible entrant**. That yields the stated lower bound but does not legitimize returned POIs outside the true domain.
- `r_ref<k`, empty true reference N/A and empty answer zero are handled correctly. A small Euclidean GPS error does not imply small mutual directed road distance: nearby opposite lanes can require a long return path or none. The diagnostic's 15 m is per-axis controlled Gaussian sigma, not a hard coordinate-error radius. Unknown road-error/calibration conditions are not claimed as an online client certificate.
- Full reference stability restores the retrieved-reference Recall identity on nested pools. Partial stability can make only its lower bound monotone; actual Recall under the same wrong local estimate can still decrease. There is no unconditional L20/L30 monotonicity claim under local error, changing destination/catalogue/status/clock or unmatched cache policies.

## Belief bridge: what layer 2 actually optimizes

- The reference-weight row sums to 1 when at least one category has a nonempty reference, and to 0 otherwise. Finite-sum interchange proves **exactly on that public representative grid** that the current coverage objective is `F_b(Q)=E_b f_Q`, where f is category-macro nearest reference retrieval through signature L=10.
- The access map `a(ell)` is explicit. A latent lane representative and the service's coordinate-access state need not coincide. With the same static catalogue/access/ties and exact local reference ranking at `a(ell)`, reply L=30 extends signature L=10 and actual **grid nearest** Recall is at least f. This does not transfer to arbitrary continuous GPS or four-purpose pooled utility.
- An empty-reference row contributes zero surrogate mass; it does not assign benchmark empty-reference Recall zero or invent POIs. The target law mu is required to use the same grid and reference/category weighting.
- For `[0,1]`-valued f, `TV(b,mu)≤zeta` gives the uniform expectation discrepancy `|F_b-F_mu|≤zeta`, with no extra factor 2. Applying that bound before and after the inherited half guarantee gives `F_mu(Q_alg)≥max(0, .5 max_same_feasible F_mu − slack −1.5 zeta)`. The maxima share the same fixed event/history/feasible groups.
- The final source identifies sigma with delta_F/public utility_slack=.03. Calibration zeta remains **unknown and unmeasured**. The result explains a conditional service objective, not a certified posterior, numerical real-user Recall guarantee or trajectory-wide improvement.

## Independent bounded checks and numerical examples

These source-linked, deterministic inline checks supplement the analytical proofs. They create no dataset/benchmark result and do not establish production numerical certification.

- **42,363** source-ranker checks: directed nine-state strongly connected graph with asymmetric lengths/speeds; shortest-path bounds compared against NetworkX path oracles; every source-state pair, fixed-destination detour and all 512 candidate masks for radius stability. Signed absolute/pairwise bounds and the stable-hit inclusion held.
- **2,112** scalar perturbation/pool cases: bounded ±.3 score perturbations on fixed true orders, full/partial stable reference and an eligible set smaller than k. All stable covered-reference lower bounds held.
- **63** exact rational binary randomized-response/prior cases for k in {1,2,4} and priors in twentieths: TV equals `(k−1)/(k+1)` and expected MAP equals `max{p,1-p,k/(k+1)}`. The displayed single-coordinate alpha=.25 balanced bound is `.5621765008857981` (56.22%).
- **600** exact rational bridge cases satisfying the stipulated half-minus-slack premise on four candidate coverage profiles and simplex grid laws in quarters: expectation-TV bounds and the nonnegative 1.5 zeta penalty held.
- Accuracy toy independently recomputed: CDF at 1000 `.9091830641547937`; hold probability at 3000 `.01509869171115925`; reuse tail beyond 1000 `.10454441063988669`; alpha_T=.025 test radius `2596.5858188431926`; split failure `.049375`. These are a three-state mathematical illustration, not measured thesis-map quantiles.

No runtime algorithm, protected-input GPS schedule, RNG key, ledger, response depth, cohort, split, model selection or sealed artifact was changed for these derivations. No expensive benchmark or full regression suite was rerun. The conditional statements remain separate from source-linked empirical evidence and from implementation privacy claims.
