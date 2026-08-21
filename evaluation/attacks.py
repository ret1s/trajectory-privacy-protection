"""
Inference attacks used to measure privacy empirically.

Threat model: the adversary observes the released points z_1..z_T, knows the
mechanism family and its parameters (Kerckhoffs), and holds a prior over the
candidate locations (road-graph vertices near the observation). Reported
metric is the adversary's mean estimation error in meters — the standard
"expected inference error" / adversarial-error measure (Shokri et al. 2011;
also the AE metric of Takagi et al.'s Geo-Graph-Indistinguishability).

Two adversaries:

1. BayesianPointAttack — attacks each report independently:
       posterior(x | z) ∝ prior(x) · f(z | x)
   estimate = posterior mean. Uniform prior over vertices within
   `prior_radius` of z.

2. HMMTrackingAttack — exploits temporal correlation with a forward-backward
   pass over per-step candidate sets:
       transition(x_t | x_{t-1}) ∝ exp(-d(x_t, x_{t-1}) / sigma_move)
   a Laplace-kernel mobility model with mean step sigma_move = v_typ·Δt.
   This is the attack that punishes mechanisms whose noise is independent
   across time (it filters the noise like a Kalman smoother), and is the
   discrete analogue of the correlation attacks in Xiao & Xiong (CCS 2015).

Emission likelihood f(z|x): for all mechanisms we give the adversary the
planar-Laplace likelihood f(z|x) ∝ exp(-ε·d(z,x)) — exact for PlanarLaplace,
and the assumed-mechanism approximation for the capped baseline; for the
exponential mechanisms the exact emission exp(-ε/2·d) is used.
"""
import numpy as np
from scipy.spatial.distance import cdist


def precompute_lognorm(road_network, epsilon, scale=0.5, chunk=400):
    """Per-vertex log-normaliser logZ(x) = log Σ_{v∈V} exp(-scale·ε·d(x,v)) of
    the REM emission over the FULL fixed vertex set V, for every vertex x.

    Needed for the EXACT Bayesian posterior: the REM likelihood is
    f(z|x) = exp(-scale·ε·d(x,z)) / Z(x), and Z(x) is input-dependent
    (boundary vertices have fewer nearby neighbours), so omitting logZ(x)
    biases the adversary's estimate (verifier V-004). Computed once per ε in
    memory-bounded chunks via cdist. Returns an array of shape (|V|,).
    """
    xy = road_network.xy
    n = len(xy)
    logZ = np.empty(n)
    a = scale * epsilon
    for i in range(0, n, chunk):
        d = cdist(xy[i : i + chunk], xy)          # (b, n)
        m = (-a * d).max(axis=1, keepdims=True)
        logZ[i : i + chunk] = (m.ravel() + np.log(np.exp(-a * d - m).sum(axis=1)))
    return logZ


def _geometric_median(pts, w, iters=8):
    """Weiszfeld geometric median of weighted points — the estimator that
    minimises expected Euclidean distance (the loss actually reported), unlike
    the posterior mean which minimises squared distance (verifier V-004)."""
    x = (pts * w[:, None]).sum(axis=0)  # seed at the weighted mean
    for _ in range(iters):
        dist = np.linalg.norm(pts - x, axis=1)
        dist = np.maximum(dist, 1e-6)
        ww = w / dist
        x = (pts * ww[:, None]).sum(axis=0) / ww.sum()
    return x


class BayesianPointAttack:
    """Bayesian inference adversary with the EXACT REM emission over the full
    fixed vertex set and a FIXED (data-independent) uniform prior.

    posterior(x | z) ∝ prior(x) · exp(-scale·ε·d(x,z)) / Z(x)
                     = exp(-scale·ε·d(z,x) - logZ(x))      (uniform prior)

    The estimate is the geometric median of the posterior (optimal for the
    Euclidean-distance loss reported). This is exact for REM; for the
    history/state-dependent mechanisms (T-REM, SM-REM) it is the fixed
    REM-emission adversary applied uniformly — a well-specified evaluation
    adversary, NOT claimed to be each mechanism's Bayes-optimal attacker.
    Pass `lognorm` from precompute_lognorm() (shared across reports/ε)."""

    def __init__(self, road_network, epsilon, emission_scale=0.5, lognorm=None):
        self.rn = road_network
        self.epsilon = epsilon
        self.emission_scale = emission_scale
        self.logZ = (
            lognorm
            if lognorm is not None
            else precompute_lognorm(road_network, epsilon, emission_scale)
        )

    def estimate(self, z_lat, z_lon):
        z_xy = np.asarray(self.rn.point_xy(z_lat, z_lon))
        d = cdist([z_xy], self.rn.xy)[0]  # z to every vertex
        logpost = -self.emission_scale * self.epsilon * d - self.logZ
        logpost -= logpost.max()
        w = np.exp(logpost)
        w /= w.sum()
        est_xy = _geometric_median(self.rn.xy, w)
        lat, lon = self.rn.proj.to_latlon(est_xy[0], est_xy[1])
        return float(lat), float(lon)

    def error(self, real_traj, released_traj):
        """Mean distance (m) between true points and per-point estimates."""
        errs = []
        for (rlat, rlon), (zlat, zlon) in zip(real_traj, released_traj):
            elat, elon = self.estimate(zlat, zlon)
            ex, ey = self.rn.point_xy(elat, elon)
            rx, ry = self.rn.point_xy(rlat, rlon)
            errs.append(float(np.hypot(ex - rx, ey - ry)))
        return float(np.mean(errs))


class HMMTrackingAttack:
    """Correlation-aware (forward-backward) adversary. Emission includes the
    input-dependent log-normaliser logZ(x) (mechanism-aware for REM); candidate
    truncation is enlarged and its coverage of the true-location proxy is
    tracked (verifier V-004, V-008). Approximate for T-REM/SM-REM (REM-emission
    model applied uniformly), so report it as such."""

    def __init__(
        self,
        road_network,
        epsilon,
        emission_scale=0.5,
        candidate_radius=2000.0,
        max_candidates=2000,
        v_typ=8.0,
        lognorm=None,
    ):
        self.rn = road_network
        self.epsilon = epsilon
        self.emission_scale = emission_scale
        self.candidate_radius = candidate_radius
        self.max_candidates = max_candidates
        self.v_typ = v_typ  # m/s — typical urban speed for the mobility prior
        self.logZ = lognorm  # per-vertex REM log-normaliser (may be None)
        self.true_covered = 0
        self.true_total = 0

    def _candidates(self, z_xy):
        idxs = self.rn.tree.query_ball_point(z_xy, self.candidate_radius)
        idxs = np.asarray(idxs, dtype=int)
        if len(idxs) > self.max_candidates:
            d = np.linalg.norm(self.rn.xy[idxs] - z_xy, axis=1)
            idxs = idxs[np.argsort(d)[: self.max_candidates]]
        return idxs

    def error(self, real_traj, released_traj, times):
        """Mean distance between true points and forward-backward posterior
        means over the candidate lattice."""
        T = len(released_traj)
        cand, emis = [], []
        for i, (zlat, zlon) in enumerate(released_traj):
            z_xy = np.asarray(self.rn.point_xy(zlat, zlon))
            idxs = self._candidates(z_xy)
            d = np.linalg.norm(self.rn.xy[idxs] - z_xy, axis=1)
            loge = -self.emission_scale * self.epsilon * d
            if self.logZ is not None:
                loge = loge - self.logZ[idxs]  # exact input-dependent normaliser
            cand.append(idxs)
            emis.append(loge - loge.max())
            # Track whether the true-location proxy vertex survives truncation.
            true_v, _ = self.rn.nearest(real_traj[i][0], real_traj[i][1])
            self.true_total += 1
            if true_v in idxs:
                self.true_covered += 1

        # Forward pass (log domain per step, normalized).
        fwd = [None] * T
        fwd[0] = self._norm(np.exp(emis[0]))
        trans = [None] * (T - 1)
        for t in range(1, T):
            dt = max(1.0, (times[t] - times[t - 1]).total_seconds())
            sigma = max(20.0, self.v_typ * dt)
            D = np.linalg.norm(
                self.rn.xy[cand[t]][None, :, :] - self.rn.xy[cand[t - 1]][:, None, :],
                axis=2,
            )
            Tr = np.exp(-D / sigma)
            Tr /= Tr.sum(axis=1, keepdims=True)
            trans[t - 1] = Tr
            fwd[t] = self._norm(np.exp(emis[t]) * (fwd[t - 1] @ Tr))

        # Backward pass.
        bwd = [None] * T
        bwd[T - 1] = np.ones(len(cand[T - 1]))
        for t in range(T - 2, -1, -1):
            bwd[t] = self._norm(trans[t] @ (np.exp(emis[t + 1]) * bwd[t + 1]))

        errs = []
        for t in range(T):
            post = self._norm(fwd[t] * bwd[t])
            est_xy = (self.rn.xy[cand[t]] * post[:, None]).sum(axis=0)
            rx, ry = self.rn.point_xy(real_traj[t][0], real_traj[t][1])
            errs.append(float(np.hypot(est_xy[0] - rx, est_xy[1] - ry)))
        return float(np.mean(errs))

    def online_estimates(self, released_traj, times):
        """Causal (forward-only) per-step estimates — what a live adversary
        tracking the released stream would guess at each timestep. Returns a
        list of (lat, lon) estimates, one per report."""
        T = len(released_traj)
        estimates = []
        prev_idxs, prev_belief = None, None
        for t in range(T):
            zlat, zlon = released_traj[t]
            z_xy = np.asarray(self.rn.point_xy(zlat, zlon))
            idxs = self._candidates(z_xy)
            d = np.linalg.norm(self.rn.xy[idxs] - z_xy, axis=1)
            loge = -self.emission_scale * self.epsilon * d
            emis = np.exp(loge - loge.max())

            if prev_idxs is None:
                belief = self._norm(emis)
            else:
                dt = max(1.0, (times[t] - times[t - 1]).total_seconds())
                sigma = max(20.0, self.v_typ * dt)
                D = np.linalg.norm(
                    self.rn.xy[idxs][None, :, :] - self.rn.xy[prev_idxs][:, None, :],
                    axis=2,
                )
                Tr = np.exp(-D / sigma)
                Tr /= Tr.sum(axis=1, keepdims=True)
                belief = self._norm(emis * (prev_belief @ Tr))

            est_xy = (self.rn.xy[idxs] * belief[:, None]).sum(axis=0)
            lat, lon = self.rn.proj.to_latlon(est_xy[0], est_xy[1])
            estimates.append((float(lat), float(lon)))
            prev_idxs, prev_belief = idxs, belief
        return estimates

    @staticmethod
    def _norm(v):
        s = v.sum()
        return v / s if s > 0 else np.full_like(v, 1.0 / len(v))


class AveragingAttack:
    """Repeated-report / home-inference attack (scenario S4).

    Models the documented real-world harm (Strava home-zone recovery, Hassan
    et al. USENIX Sec 2018; data-broker home fingerprinting): the user reports
    n times from ONE static true location (e.g. home overnight); the adversary
    averages the released points. For any mechanism that emits fresh
    independent noise per report, the sample mean of the releases converges to
    the true point at rate O(1/sqrt(n)) (or faster after de-biasing), so the
    estimate error collapses as n grows. A mechanism that returns a CONSISTENT
    release for the same place (memoization) leaves the averaged estimate stuck
    at the single-release error — averaging buys the adversary nothing.

    Reported metric: mean distance (m) between the averaged estimate and the
    true static point, as a function of n. Higher / non-decreasing = private.
    """

    def __init__(self, road_network):
        self.rn = road_network

    def run(self, mechanism, home_lat, home_lon, n_reports, times=None):
        """Emit n_reports from the static home point through `mechanism`, then
        report the averaged-estimate error. `times` optionally supplies per-
        report timestamps (a stationary dwell); defaults to 60s spacing."""
        import datetime

        mechanism.reset()
        if times is None:
            base = datetime.datetime(2008, 10, 23, 2, 0, 0)
            times = [base + datetime.timedelta(seconds=60 * i) for i in range(n_reports)]

        releases = []
        for i in range(n_reports):
            zlat, zlon = mechanism.perturb(home_lat, home_lon, t=times[i])
            releases.append(self.rn.point_xy(zlat, zlon))
        releases = np.asarray(releases)

        home_xy = np.asarray(self.rn.point_xy(home_lat, home_lon))
        # Averaged estimate after the first k reports, for a schedule of k.
        curve = {}
        for k in (1, 2, 5, 10, 20, 50, 100):
            if k <= n_reports:
                est = releases[:k].mean(axis=0)
                curve[k] = float(np.hypot(est[0] - home_xy[0], est[1] - home_xy[1]))
        return curve
