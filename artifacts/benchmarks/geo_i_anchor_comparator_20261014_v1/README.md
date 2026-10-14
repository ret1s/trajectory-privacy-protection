# Bare Geo-I/REM protected-anchor controlled ablation

Natural K1/L60 output,31 point requests, same frozen protected Z history as
GeoI-BQ; no new noise draws or private-sample filtering. Post-hoc controlled
comparison on reused native geometry, not confirmatory new data.
Mean Recall99.700%,reply16.921KB; main99.595%,82.557KB/155requests.
Strong privacy difference is reported with SAME TRAIN/SELECTION attacker bank:
S10 Hit10035.0%/MAE246.8m for Z vs0%/2124.2m for main. Other attacks and
all output/cost limitations retained. This is a same-backbone point ablation,
not an exact reproduction of the original continuous planar-Laplace paper.

[Full readout](../../../docs/research/2026-10-14_main_model.md).
Audit: `python -m experiments.verify_geo_i_anchor_comparator_20261014`.
Do not rerun the fit script into this completed write-once output.
