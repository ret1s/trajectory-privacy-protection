# S8 fusion diagnostic V2 — stronger attacker, fixed defense

After reading V1, the audit found that its bank could use either individual
channel and learned joint features, but lacked explicit cross-channel means
and nearest-pair/intersection estimators. This is a **post-V1 diagnostic on
inspected data**, not an untouched confirmation. Retain V1 and its scores.

Before V2 scoring, freeze the following additions: centroid/median/velocity
mixtures with target weights .25/.5/.75; closest cross-set pair midpoint;
means of the closest two/three pair midpoints. Inputs are the same two public
causal prefixes; no genuine-member labels, GPS or future coordinates enter
fusion. When the two sets both contain the same genuine GPS, proximity may
expose it. This is a legitimate adversary in the declared aligned-prefix model.

Reuse the trained V1 regressors and every V1 candidate, add the fixed fusion
bank to the three joint views, select separately for MAE and Hit100 on the same
SELECTION groups, save selection, then score TEST. Target-only stays identical.
No new defense output, synthetic choice, route, label or sample is generated.
Use the original 10-block paired bootstrap for S8 and retain every estimator.
Archive selection estimates as well as TEST predictions so the selector itself
can be independently replayed. Report V1→V2 changes, all views and raw-partner
control. The same synthetic-alignment/identity limits from V1 still apply.
