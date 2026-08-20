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


class BayesianPointAttack:
    def __init__(self, road_network, epsilon, emission_scale=1.0, prior_radius=1200.0):
        """emission_scale: multiplier on ε in the emission (1.0 for planar
        Laplace-style mechanisms, 0.5 for the exponential mechanisms)."""
        self.rn = road_network
        self.epsilon = epsilon
        self.emission_scale = emission_scale
        self.prior_radius = prior_radius

    def estimate(self, z_lat, z_lon):
        """Posterior-mean estimate of the true location given one report."""
        z_xy = np.asarray(self.rn.point_xy(z_lat, z_lon))
        idxs = self.rn.tree.query_ball_point(z_xy, self.prior_radius)
        idxs = np.asarray(idxs, dtype=int)
        if len(idxs) == 0:
            return z_lat, z_lon
        d = np.linalg.norm(self.rn.xy[idxs] - z_xy, axis=1)
        logw = -self.emission_scale * self.epsilon * d
        logw -= logw.max()
        w = np.exp(logw)
        w /= w.sum()
        est_xy = (self.rn.xy[idxs] * w[:, None]).sum(axis=0)
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
    def __init__(
        self,
        road_network,
        epsilon,
        emission_scale=1.0,
        candidate_radius=1000.0,
        max_candidates=800,
        v_typ=8.0,
    ):
        self.rn = road_network
        self.epsilon = epsilon
        self.emission_scale = emission_scale
        self.candidate_radius = candidate_radius
        self.max_candidates = max_candidates
        self.v_typ = v_typ  # m/s — typical urban speed for the mobility prior

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
        for zlat, zlon in released_traj:
            z_xy = np.asarray(self.rn.point_xy(zlat, zlon))
            idxs = self._candidates(z_xy)
            d = np.linalg.norm(self.rn.xy[idxs] - z_xy, axis=1)
            loge = -self.emission_scale * self.epsilon * d
            cand.append(idxs)
            emis.append(loge - loge.max())

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
