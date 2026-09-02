# AnotherMe reproduction record

## Status

The benchmark now contains a source-mapped clean-room implementation of the
authors' public standalone Virtual Trajectory Generation Algorithm (VTGA). It
replaces the former affine relocation prototype.

It is still labelled **`paper_adaptation`**, not an official or faithful
end-to-end reproduction. The distinction is deliberate: the locally
executable VTGA stages are implemented, while the AMap-dependent virtual-user
workflow and paper-equivalent validation remain unavailable.

## Primary sources audited

- Paper: Y. Li et al., *AnotherMe: A Location Privacy Protection System Based
  on Online Virtual Trajectory Generation*, IEEE Transactions on Dependable and
  Secure Computing, DOI
  [10.1109/TDSC.2023.3314200](https://doi.org/10.1109/TDSC.2023.3314200).
- Official repository:
  [fang-zhiyou/AnotherMe](https://github.com/fang-zhiyou/AnotherMe), pinned at
  commit `0eda877b7328ee1c0b3e0cf9a9bb9d48cb24323f`.
- Standalone generator:
  [`VTGAs/gen_virtual_traj.py`](https://github.com/fang-zhiyou/AnotherMe/blob/0eda877b7328ee1c0b3e0cf9a9bb9d48cb24323f/VTGAs/gen_virtual_traj.py).
- Coordinate/routing calls:
  [`VTGAs/public_api.py`](https://github.com/fang-zhiyou/AnotherMe/blob/0eda877b7328ee1c0b3e0cf9a9bb9d48cb24323f/VTGAs/public_api.py).
- Mobile workflow:
  [`Android/app/src/main/java/com/example/contest/Utils`](https://github.com/fang-zhiyou/AnotherMe/tree/0eda877b7328ee1c0b3e0cf9a9bb9d48cb24323f/Android/app/src/main/java/com/example/contest/Utils)
  and
  [`iOS/Browser/Utils`](https://github.com/fang-zhiyou/AnotherMe/tree/0eda877b7328ee1c0b3e0cf9a9bb9d48cb24323f/iOS/Browser/Utils).

No root licence file was present at the audited revision. For that reason,
upstream source is not vendored; the local code is an independently structured
implementation with source-to-component traceability.

## Public VTGA procedure implemented

The standalone Python component accepts one real trajectory as
`(longitude, latitude, timestamp)` samples and emits one variable-length
virtual trajectory. Internally this benchmark uses `(latitude, longitude)` to
match the existing protocol, but preserves the operations and constants:

1. Compute segment lengths, total average speed, and a speed sequence rounded
   to two decimals. The upstream uses `geopy.geodesic`; the local benchmark
   uses `pyproj.Geod` with the same WGS84 ellipsoid.
2. Select AMap walking below 3 m/s, cycling from 3 through 10 m/s, and driving
   above 10 m/s.
3. Convert the real start/end from GPS/WGS84 to AMap coordinates and request
   the first navigation route.
4. Omit the first navigation sample and retain a later sample only when it is
   at least 6 m from its immediate predecessor in the original navigation
   response.
5. Insert points using `round(segment_distance / 2) - 1`, with six-decimal
   linear interpolation.
6. Detect turns from 70 to 110 degrees and replace a five-sample window on
   each side with a quadratic Bezier curve.
7. Scan forward through the dense route. Accept a sample when its distance
   from the last accepted sample differs by less than 1.1 m from
   `real_speed[i] * 3`; cycle the real speed sequence.
8. Add independent discrete longitude and latitude noise from
   `[-20, 20] / 800000` degrees and round to six decimals.
9. Assign samples at three-second intervals from the real start time.

The implementation is in
[`benchmark/engines/anotherme.py`](../../benchmark/engines/anotherme.py).
`AnotherMeGenerationTrace` retains every intermediate stage for verification.

## Benchmark adaptation

Three explicit adapters are necessary for the SUMO benchmark:

- `RoadNetworkVirtualEndpointMapper` relocates the origin by a configured
  urban distance and selects a reachable destination that best preserves the
  real origin-to-destination displacement. This is the local deterministic
  analogue of virtual-city POI mapping; it ensures the comparator does not
  simply route between the user's real endpoints.
- `RoadNetworkRouteProvider` replaces AMap with a deterministic shortest path
  on the benchmark road graph and preserves each selected
  edge's polyline geometry. The SUMO run uses its exact passenger permissions;
  other OSM graphs use an explicit highway-class filter for the selected
  walking, bicycling, or driving mode. It fails closed when no directed route
  exists, so it never reverses one-way edges. This does not reproduce AMap or
  its GCJ02 coordinate frame.
- The official generator emits a variable number of points on a new 3-second
  grid. The benchmark requires one public event per real input event, so
  `protect_run` interpolates the raw virtual trajectory onto the original
  event grid using normalized time. The raw source-shaped output is preserved
  separately and should be used for stage-level parity checks.
- The benchmark adapter rejects raw generated trajectories shorter than 20
  samples, matching the validity rule in the upstream experiment processing,
  instead of stretching a degenerate one- or two-point output.

Transport mode and raw output length depend on the secret input and are not
published in the attacker transcript. They are recorded only as evaluator-side
paper diagnostics. Likewise, utility/temporal claims for the paper's raw
three-second output are not inferred from the aligned benchmark trajectory.

The method wrapper and fail-closed evidence card are in
[`benchmark/methods/anotherme.py`](../../benchmark/methods/anotherme.py).

## Full-system workflow found in the mobile source

The Android and iOS implementations reveal a larger workflow than the
standalone endpoint-to-endpoint VTGA:

- extract stay points using approximately 200 m and 30 minutes;
- obtain a POI type for each stay point through AMap reverse geocoding;
- choose an anchor POI (the iOS variant uses the longest stay);
- search the same POI categories around a virtual-city anchor;
- select mapped virtual POIs whose distance pattern resembles the real POIs;
- route between the mapped POIs and apply speed/shape simulation;
- present virtual city/profile choices through the local mobile browser.

The published variants are not identical. The standalone Python VTGA routes
between converted real endpoints and uses 3/10 m/s mode thresholds. The
Android simulator routes through mapped POIs, computes different thresholds,
then overrides the result to driving. No repository manifest identifies which
variant and configuration produced each paper table.

The virtual-POI selection also differs across the two mobile trees. The iOS
code uses the longest-stay POI as the real anchor and, for each POI, chooses
among the first 15 AMap candidates by closest anchor-distance difference. The
Android code requests four candidates per POI type, samples 128 whole-profile
assignments, and keeps the assignment with the smallest pairwise-distance-
matrix error. This unresolved variation is why the local endpoint mapper is
labelled an adaptation rather than a faithful reconstruction.

## Inputs and validation assets audited

The official repository includes six selected T-Drive users and six selected
GeoLife user archives. `TrajectoryExp.zip` and `TrajectoryExp1.zip` contain
research scripts for MN, MLN, ADTGA, RCDT and GAN generation, plus LSTM, CNN
and TSHN detection. They do not form a one-command reproduction:

- preprocessing code contains author-machine absolute paths;
- expected `Anotherme_train.pkl`, `Anotherme_test.pkl`, model files and several
  generated pickle datasets are not shipped as a documented pipeline;
- no dependency lockfile, split manifest, live-service response snapshot,
  random-seed manifest, golden VTGA output, or paper-table verification script
  is provided;
- live AMap results and GCJ02 conversion are external mutable dependencies;
- Android/iOS energy and latency measurements require the original device and
  service setup.

Stage-level deterministic tests are in
[`tests/test_anotherme_faithful.py`](../../tests/test_anotherme_faithful.py).
They cover the constants and operation order above, route dependency injection,
fixed-seed output, benchmark alignment, source revision, and truth separation.

## Exact blockers before a faithful-reproduction claim

1. A frozen, authorised AMap coordinate-conversion/route/POI response corpus,
   or an author-confirmed local equivalent.
2. A canonical specification tying the standalone Python and mobile variants
   to the paper experiment configuration.
3. Exact virtual-city anchors, user-level train/test split, preprocessing
   parameters, random seeds, and generated AnotherMe train/test artifacts.
4. Trained recognition models or a deterministic training recipe that matches
   the paper's privacy results.
5. A reproducible mobile device/service harness for latency and energy claims.
6. Golden intermediate/output trajectories or paper-table parity evidence.

Until these are obtained, the source-mapped VTGA is suitable as a transparent
local comparator but must not be reported as a reproduced SOTA result.
