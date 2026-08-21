"""
Location-privacy mechanisms benchmarked in this thesis.

All mechanisms expose:
    reset()                      — clear per-trajectory state
    perturb(lat, lon, t=None)    — one real point in, one reported point out

Mechanisms:

1. PlanarLaplace — the canonical ε-Geo-Indistinguishability mechanism
   (Andrés et al., CCS 2013). Radius ~ Gamma(2, 1/ε) (NOT Exponential(1/ε):
   the exponential radius used in the internship-2 code does not satisfy
   Geo-I — its density ratio blows up near r=0).

2. BaselineThesis — faithful re-implementation of the internship-2 pipeline:
   planar-Laplace noise capped at the QoS radius, heuristic continuity
   smoothing (70% previous direction) with a QoS re-check against the REAL
   point, then snap to the nearest road vertex if that still meets QoS.
   Kept as the baseline to beat; its cap/re-check steps break the formal
   guarantee (they condition the output on the true location).

3. RoadExponential (REM) — our improvement, static version: exponential
   mechanism over the road-graph vertices with Euclidean score,
   P(v | x) ∝ exp(-ε/2 · d(x, v)).
   Satisfies ε-Geo-Indistinguishability on the candidate set (standard
   exponential-mechanism argument; the ε/2 absorbs the normalizer), and by
   construction every reported point lies on the road network — no
   post-hoc plausibility filtering for an attacker to exploit (cf. RAoPT,
   ACSAC 2022). This is the Euclidean-metric sibling of the
   Graph-Exponential Mechanism of Takagi et al. (Geo-Graph-
   Indistinguishability, DBSec 2019 / arXiv:2010.13449).

4. TemporalRoadExponential (T-REM) — REM plus temporal consistency: the
   candidate score is multiplied by a reachability weight that depends ONLY
   on the previously *released* point z_{t-1} (public information), never on
   the real trajectory:
       w(v) = 1                                        if d(z_prev, v) ≤ v_max·Δt + slack
       w(v) = exp(-λ · (d(z_prev, v) - v_max·Δt))      otherwise
   Because w is independent of the true location, the per-point ε-Geo-I
   bound is unchanged (both the numerator factor and the normalizer ratio
   keep the exp(ε·d(x,x')) bound), while released trajectories become
   speed-consistent — closing the correlation-based dummy-filtering attack
   that defeats the baseline's independent-noise smoothing.

Per-trajectory budget: all mechanisms spend ε per point; over T points the
trajectory-level guarantee composes to ε·T (sequential composition), which
experiments report explicitly.
"""
import numpy as np


class PlanarLaplace:
    """Pure ε-Geo-I planar Laplace (unbounded — no QoS cap, off-road outputs)."""

    name = "planar_laplace"

    def __init__(self, epsilon, rng=None):
        self.epsilon = epsilon
        self.rng = rng or np.random.default_rng()

    def reset(self):
        pass

    def _noise_xy(self):
        r = self.rng.gamma(shape=2.0, scale=1.0 / self.epsilon)
        theta = self.rng.uniform(0.0, 2.0 * np.pi)
        return r * np.cos(theta), r * np.sin(theta)

    def perturb(self, lat, lon, t=None, proj=None):
        dx, dy = self._noise_xy()
        m_per_deg_lat = 111_132.0
        m_per_deg_lon = 111_132.0 * np.cos(np.radians(lat))
        return lat + dy / m_per_deg_lat, lon + dx / m_per_deg_lon


class BaselineThesis:
    """Internship-2 pipeline: capped noise + continuity smoothing + QoS-checked
    road snap. (The alternative-road search of the original code is O(|E|)
    per point and does not change the mechanism's privacy character, so it
    is omitted here for benchmark tractability.)"""

    name = "baseline_thesis"

    def __init__(self, epsilon, qos_radius, road_network, mix_ratio=0.7, rng=None):
        self.epsilon = epsilon
        self.qos_radius = qos_radius
        self.rn = road_network
        self.mix_ratio = mix_ratio
        self.rng = rng or np.random.default_rng()
        self.reset()

    def reset(self):
        self._fake_xy = []

    def _dist(self, a, b):
        return float(np.hypot(a[0] - b[0], a[1] - b[1]))

    def perturb(self, lat, lon, t=None):
        real_xy = self.rn.point_xy(lat, lon)

        # Noise with radius sampled as in the internship-2 code (exponential)
        # and capped at the QoS radius.
        r = min(self.rng.exponential(1.0 / self.epsilon), self.qos_radius)
        theta = self.rng.uniform(0.0, 2.0 * np.pi)
        fake = (real_xy[0] + r * np.cos(theta), real_xy[1] + r * np.sin(theta))

        # Continuity smoothing against previous fake points, QoS-rechecked
        # against the real point (as in the original _ensure_trajectory_continuity).
        if len(self._fake_xy) >= 2:
            prev, prev2 = self._fake_xy[-1], self._fake_xy[-2]
            expected = (prev[0] - prev2[0], prev[1] - prev2[1])
            adjusted = (
                prev[0] + self.mix_ratio * expected[0] + (1 - self.mix_ratio) * (fake[0] - prev[0]),
                prev[1] + self.mix_ratio * expected[1] + (1 - self.mix_ratio) * (fake[1] - prev[1]),
            )
            if self._dist(real_xy, adjusted) <= self.qos_radius:
                fake = adjusted

        # Snap to nearest road vertex, keep only if QoS still holds.
        d, idx = self.rn.tree.query(fake)
        snapped = tuple(self.rn.xy[idx])
        if self._dist(real_xy, snapped) <= self.qos_radius:
            fake = snapped

        self._fake_xy.append(fake)
        flat, flon = self.rn.proj.to_latlon(fake[0], fake[1])
        return float(flat), float(flon)


class RoadExponential:
    """ε-Geo-I exponential mechanism over road-graph vertices (static)."""

    name = "road_exponential"

    def __init__(self, epsilon, road_network, cutoff_m=1500.0, rng=None):
        """
        cutoff_m bounds the candidate search radius for tractability. With
        exp(-ε/2·d) weights, mass beyond ~10/ε meters is negligible; the
        default 1500m covers ε ≥ 0.007. The cutoff region is centered on the
        true point, which technically bounds the guarantee to location pairs
        whose cutoff disks overlap (same caveat as any truncated Geo-I);
        with weights ≤ exp(-ε/2·1500) at the boundary the truncation error
        is < 1e-4 of the probability mass for the ε used here.
        """
        self.epsilon = epsilon
        self.rn = road_network
        self.cutoff_m = cutoff_m
        self.rng = rng or np.random.default_rng()

    def reset(self):
        self._last_choice = None  # vertex index emitted by the last perturb()

    def _candidate_logits(self, real_xy):
        idxs = self.rn.tree.query_ball_point(real_xy, self.cutoff_m)
        idxs = np.asarray(idxs, dtype=int)
        d = np.linalg.norm(self.rn.xy[idxs] - np.asarray(real_xy), axis=1)
        return idxs, -0.5 * self.epsilon * d

    def _sample(self, idxs, logits):
        logits = logits - logits.max()
        p = np.exp(logits)
        p /= p.sum()
        return int(self.rng.choice(idxs, p=p))

    def perturb(self, lat, lon, t=None):
        real_xy = self.rn.point_xy(lat, lon)
        idxs, logits = self._candidate_logits(real_xy)
        choice = self._sample(idxs, logits)
        self._last_choice = choice
        return self.rn.latlon(choice)


class TemporalRoadExponential(RoadExponential):
    """REM + reachability weighting w.r.t. the previously RELEASED point."""

    name = "temporal_road_exponential"

    def __init__(
        self,
        epsilon,
        road_network,
        v_max=25.0,
        slack_m=100.0,
        lam=0.02,
        cutoff_m=1500.0,
        rng=None,
    ):
        super().__init__(epsilon, road_network, cutoff_m, rng)
        self.v_max = v_max          # m/s — plausible top speed for the release
        self.slack_m = slack_m      # tolerance before the penalty kicks in
        self.lam = lam              # penalty rate per meter of excess
        self.reset()

    def reset(self):
        self._prev_release_xy = None
        self._prev_t = None

    def perturb(self, lat, lon, t=None):
        real_xy = self.rn.point_xy(lat, lon)
        idxs, logits = self._candidate_logits(real_xy)

        if self._prev_release_xy is not None:
            dt = 60.0
            if t is not None and self._prev_t is not None:
                dt = max(1.0, (t - self._prev_t).total_seconds())
            reach = self.v_max * dt + self.slack_m
            d_prev = np.linalg.norm(
                self.rn.xy[idxs] - np.asarray(self._prev_release_xy), axis=1
            )
            excess = np.maximum(0.0, d_prev - reach)
            logits = logits - self.lam * excess  # depends only on public info

        choice = self._sample(idxs, logits)
        self._last_choice = choice
        self._prev_release_xy = tuple(self.rn.xy[choice])
        if t is not None:
            self._prev_t = t
        return self.rn.latlon(choice)


class StayMemoizedREM(TemporalRoadExponential):
    """T-REM + memoization keyed to a fixed public grid — closes the averaging /
    home-inference attack (S4).

    Motivation. REM/T-REM (like every noise-per-report mechanism) leak under
    *repeated reports from the same place*: an adversary who sees many
    independent releases z_1..z_n of a static true point x can average them,
    and the estimate converges to x as n grows. This is the documented
    real-world harm behind Strava home-zone recovery (Hassan et al., USENIX
    Security 2018) and data-broker home/work fingerprinting.

    Fix (deterministic memoization, as in Android's LocationFudger persistent
    offset and LP-Doctor's cached noise, USENIX Sec 2015). Space is partitioned
    by a FIXED PUBLIC grid of cell size `grid_m`. The first time the true point
    falls in cell C, a release r_C is sampled with the T-REM rule and cached;
    every later report whose true point falls in C returns the SAME r_C. There
    is thus no fresh independent noise to average away — repeated reports from
    one cell reveal nothing beyond the single first release.

    Guarantee (stated precisely — memoization does NOT preserve Euclidean
    ε-Geo-I; see the caveat). The cell map x -> C is a deterministic PUBLIC
    partition and memoization is deterministic post-processing of one sample
    per cell, so:
      * repeats are free — a static stay-point emits N identical copies of a
        single sample, carrying the information of one release regardless of
        N, so the averaging attack gains nothing; and
      * budget composes over DISTINCT visited cells, not reports:
        ε_traj = (#distinct cells) · ε   instead of   T · ε
        (for a dwelling trace #distinct cells << T — the w-event /
        distinct-location composition view, docs/system_model_and_threats.md §2).
    Formally the released map satisfies ε-d̃X-privacy w.r.t. the CELL
    pseudometric d̃(x,x') = d(rep(cell x), rep(cell x')) — i.e. ε-Geo-I *at the
    granularity of the partition*, the graph-indistinguishability view of
    Takagi et al. (2020). It is NOT ε-Geo-I in the original Euclidean metric:
    two points ε-close across a cell boundary receive independent releases
    (boundary discontinuity, ratio up to exp(ε·cellwidth)). This is the known
    input-side-discretisation tradeoff — the reason Android's LocationFudger
    pairs snap-to-grid with a persistent random offset — and the reason one
    cannot get clean-Euclidean-Geo-I and anti-averaging memoization from the
    same knob. The memoize trigger here is the public grid alone (memoize
    every visit), so no data-dependent stay-point test leaks budget.
    Precedents: RAPPOR permanent randomized response (Erlingsson et al.,
    CCS 2014); LP-Doctor per-place cached Geo-I (Fawaz et al., USENIX Sec 2015).

    `grid_m` trades off jitter-robustness (large cell = a stationary user's GPS
    jitter stays in one cell, so memoization holds) against spatial resolution
    (small cell = distinct nearby places do not collide) and boundary-leak
    magnitude (larger cell = larger exp(ε·cellwidth) worst case). Default 60m ~
    the Strava default hidden radius / AOSP MINIMUM_ACCURACY, comfortably above
    typical urban GPS error (~10-20m). Releases snap to shared road vertices so
    a consistent value is not itself a unique fingerprint.
    """

    name = "stay_memoized_rem"

    def __init__(self, epsilon, road_network, grid_m=60.0, **kwargs):
        super().__init__(epsilon, road_network, **kwargs)
        self.grid_m = grid_m
        self.reset()

    def reset(self):
        super().reset()
        self._cache = {}            # public-grid cell -> released vertex idx
        self.distinct_releases = 0  # = number of cells that spent budget

    def _cell(self, xy):
        return (round(xy[0] / self.grid_m), round(xy[1] / self.grid_m))

    def perturb(self, lat, lon, t=None):
        real_xy = self.rn.point_xy(lat, lon)
        cell = self._cell(real_xy)

        if cell in self._cache:
            # Cache hit: reuse the earlier release (deterministic, no new
            # noise, no new budget). Still advance the release clock so later
            # moving points remain reachable from what was actually emitted.
            choice = self._cache[cell]
            self._prev_release_xy = tuple(self.rn.xy[choice])
            if t is not None:
                self._prev_t = t
            return self.rn.latlon(choice)

        # Cache miss: sample a fresh T-REM release for this new cell.
        latlon = super().perturb(lat, lon, t=t)
        self._cache[cell] = self._last_choice  # exact vertex the sampler chose
        self.distinct_releases += 1
        return latlon
