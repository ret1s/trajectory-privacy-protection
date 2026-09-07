# Frozen belief-lane evaluation grid (2026-09-07)

Recorded before any new defender scores. Dataset QA and all new mechanism unit
tests precede this run. No parameter search beyond the five rows below.

| Key | POI target | Coverage weight | Geometric center |
|---|---|---:|---|
| baseline | none | 0 | protected anchor |
| point6 | POIs at protected anchor | 6 | protected anchor |
| belief6 | expected POI coverage | 6 | protected anchor |
| belief24 | expected POI coverage | 24 | protected anchor |
| belief24_center | expected POI coverage | 24 | approximate belief mean |

All use B=.24/m, H=12, theta=200 m, K=3/5, temperature=60 m, offset=80 m,
route weight=0. One paired RNG replicate per family/record/K (seed key schema
`belief-suite-row-v1`); this is deliberately a small diagnostic, not a powered
significance study. All nine S1.A--S3.C cases, two validation families and two
confirmation families: 360 rows total. No S4--S10 attack result is claimed.

Belief: 120 m public grid; occupancy prior from every twentieth raw FCD sample
in training families 101/102, including their repeated synthetic day trips.
This is a simulator-conditioned prior, not a population mobility distribution.
Emission is the ideal .01/m release + .01/m noisy reuse kernel on the complete
102,123-state output support, summing duplicate-coordinate output probabilities.
For public dt <=120 s, use .6 stay + .4 occupancy-weighted isotropic diffusion,
sigma=max(120,4dt) m, radius=8dt+240 m. For dt>120 s, use explicit approximate
reset: .6^(dt/20) previous belief + remaining mass times training prior. This
avoids unsupported fine local extrapolation over a long unobserved trip; it is
not a calibrated directed transition model. Feasible released paths are still
constrained independently on the directed lane graph.

Cap records to their first 12 allowed observations, preserving public elapsed
times. The six S2.C records contain nine samples in visit one and three retained
samples in visit two; verify this again from realized intervals. We evaluate
location inference across those visits, not semantic home/work classification.

Run validation first and save a checksummed selection file. Choose one method
per K over all nine cases: require every case's mean Recall@5 >=.90; among
qualifiers minimize macro worst-attacker Hit100; otherwise mark infeasible and
maximize minimum-case Recall, then minimize Hit100 (round comparisons to 12
decimal digits). Freeze separate validation-selected MAE and Hit100 attackers
for each method/K/case. Confirmation requires that selection checksum and all
scientific source hashes match; the runner must not accept a new grid.

Publish every confirmation row, including nonselected controls/ablations, but
label those as diagnostic counter-evidence, not a second selection round. Macro
averages weight cases equally and, within each case, records/families equally;
events are not independent replicates. Retain both attacker-selected reporting
and exploratory maximum/minimum envelopes, never equating the two.

Report Recall@5 and complete-query rate under the unchanged 418-POI directed
service, location MAE and Hit100, K overhead, causal budget accounting, generation
latency separate from graph/context/model preparation. No formal guarantee is
inferred from attack accuracy. Ideal postprocessing preserves the previous
ideal Geo-I ledger only under its stated adjacency/fixed-time/public-prior
assumptions; floating-point code remains an approximation. No stronger formal
guarantee or improvement against all possible adversaries is asserted.

Known scientific weaknesses to retain: two families per role, one RNG replica,
same-city road/POI catalogue, heuristic rather than likelihood-aware attackers,
coarse uncalibrated belief, partially observed records, no long-horizon budget
solution, and no new cross-method SOTA ranking. Future improvement needs a new
protocol/confirmation split once these outcomes have been inspected.
