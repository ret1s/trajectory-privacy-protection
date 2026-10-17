# Continual Geo-I pilot: paired results

Exploratory pilot: one fresh draw per trip, all48 families, reused native first10min plus a directed-road synthetic extension to30min. The main release/report are unchanged. No automatic promotion.

All percentages below are empirical. Intervals are paired-family percentile95% bootstrap intervals; delta is against capped_g1 in percentage points. Static418-POI cache saturation can mask gains.

| Window | Arm | Session Recall (%) | Delta (pp), 95% CI | -2pp gate, all purposes |
|---|---|---:|---|---|
| native_10min | capped_g1 | 99.633 | +0.000 [+0.000, +0.000] | True |
| native_10min | continual_g1 | 99.633 | +0.000 [+0.000, +0.000] | True |
| native_10min | continual_g0 | 99.692 | +0.059 [-0.054, +0.212] | True |
| full_30min | capped_g1 | 99.873 | +0.000 [+0.000, +0.000] | True |
| full_30min | continual_g1 | 99.873 | +0.000 [+0.000, +0.000] | True |
| full_30min | continual_g0 | 99.891 | +0.018 [-0.018, +0.063] | True |
| synthetic_after_12min | capped_g1 | 99.997 | +0.000 [+0.000, +0.000] | True |
| synthetic_after_12min | continual_g1 | 99.997 | +0.000 [+0.000, +0.000] | True |
| synthetic_after_12min | continual_g0 | 99.994 | -0.003 [-0.018, +0.008] | True |

## Full30min utility by purpose and cache

| Cache | Arm | Nearest (%) | Fastest (%) | Radius (%) | Detour (%) |
|---|---|---:|---:|---:|---:|
| current | capped_g1 | 97.861 | 97.862 | 94.072 | 97.727 |
| current | continual_g1 | 97.861 | 97.862 | 94.072 | 97.727 |
| current | continual_g0 | 96.918 | 96.925 | 90.164 | 97.492 |
| known | capped_g1 | 98.007 | 98.007 | 94.423 | 97.863 |
| known | continual_g1 | 98.007 | 98.007 | 94.423 | 97.863 |
| known | continual_g0 | 97.099 | 97.106 | 90.780 | 97.672 |
| session | capped_g1 | 99.905 | 99.905 | 99.858 | 99.825 |
| session | continual_g1 | 99.905 | 99.905 | 99.858 | 99.825 |
| session | continual_g0 | 99.896 | 99.896 | 99.890 | 99.882 |

## Localization attack bank

TRAIN fits and SELECTION chooses loss-specific attacks before TEST. S2 is late-window coordinate recovery here, not semantic stop detection. Zero empirical Hit100 is not zero inference probability.

| Arm | Scenario | Hit100 (%) | MAE (m) |
|---|---|---:|---:|
| capped_g1 | S1 | 1.264 | 1712.0 |
| capped_g1 | S2 | 2.500 | 705.1 |
| capped_g1 | S3 | 1.264 | 1610.1 |
| capped_g1 | S9 | 0.000 | 1986.4 |
| capped_g1 | S10 | 0.000 | 718.9 |
| continual_g1 | S1 | 1.264 | 1712.0 |
| continual_g1 | S2 | 2.500 | 705.1 |
| continual_g1 | S3 | 1.264 | 1611.3 |
| continual_g1 | S9 | 0.000 | 1986.4 |
| continual_g1 | S10 | 0.000 | 718.9 |
| continual_g0 | S1 | 1.264 | 1712.0 |
| continual_g0 | S2 | 0.000 | 705.1 |
| continual_g0 | S3 | 1.264 | 1712.0 |
| continual_g0 | S9 | 0.000 | 1986.4 |
| continual_g0 | S10 | 0.000 | 1024.1 |

## Upstream and wire cost

| Upstream | Mean GPS updates | Mean hidden-branch spend (/m) | Public prefix bound (/m) |
|---|---:|---:|---:|
| capped | 13.20 | 0.2245 | 0.23 |
| continual | 31.00 | 0.5523 | 0.61 |

| Arm | Point requests | Request JSON KB | Reply-index JSON KB |
|---|---:|---:|---:|
| capped_g1 | 455 | 93.76 | 228.69 |
| continual_g0 | 455 | 93.77 | 228.73 |
| continual_g1 | 455 | 93.76 | 228.69 |

The continual arm has no .23/m whole-session upstream cap. The separate bounded-Q theorem retains its all-duration logical-wire bound under the documented assumptions; private adaptation decreases over time. Numerical REM/PCG are not certified pure Geo-I implementations. Byte counts omit full metadata/HTTP/TLS. This pilot does not establish real-world or arbitrary-duration high utility.

Validation: pass; Q prefix checks: 26208; directed Q transitions: 129600.
