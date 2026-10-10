"""DLS Algorithm 1 (Niu et al., INFOCOM 2014), on a public road catalog.

The probability-neighbor pool, random subsets and entropy maximization follow
Section IV-A / Equations (3)-(5). Road vertices replace grid cells; symmetric
ties, edge-of-list handling, smoothing and m are explicit local choices.
This is not enhanced-DLS, RDG, or a paper-dataset reproduction.
"""
import numpy as np

from core.demo_protocol import (EvaluationTruth, OutputKind, ProtectedRun,
                                PublicCandidate, PublicEvent, PublicTranscript,
                                TrajectoryPoint)


def entropy(probabilities):
    p = np.asarray(probabilities, dtype=float)
    if p.ndim != 1 or not len(p) or not np.isfinite(p).all() or np.any(p <= 0):
        raise ValueError("positive finite probabilities required")
    p = p / p.sum()
    return float(-np.dot(p, np.log2(p)))


class DLSGraph:
    name = "dls_graph_adaptation"

    def __init__(self, rn, probabilities, *, k=3, subset_trials=50, rng=None):
        self.rn, self.k, self.subset_trials = rn, int(k), int(subset_trials)
        self.q = np.asarray(probabilities, dtype=float)
        if self.k < 2 or self.k > len(rn) or self.subset_trials < 1:
            raise ValueError("2 <= K <= catalog size and positive trial count required")
        if self.q.shape != (len(rn),) or not np.isfinite(self.q).all() or np.any(self.q <= 0):
            raise ValueError("one positive public probability per road vertex required")
        self.q = self.q / self.q.sum()
        self.rng = rng if rng is not None else np.random.default_rng()

    def candidate_pool(self, true_index):
        # Split ties around truth as specified in IV-A. Sorting inside the tie
        # block by vertex index is a deterministic local convention.
        below = sorted(np.flatnonzero(self.q < self.q[true_index]), key=lambda i: (self.q[i], i))
        above = sorted(np.flatnonzero(self.q > self.q[true_index]), key=lambda i: (self.q[i], i))
        ties = [int(i) for i in np.flatnonzero(self.q == self.q[true_index]) if i != true_index]
        half = len(ties) // 2
        order = below + ties[:half] + [true_index] + ties[half:] + above
        center = order.index(true_index)
        left, right = max(0, center - self.k), min(len(order), center + self.k + 1)
        # A boundary cell cannot have K neighbors on both sides. Fill from the
        # available side, still selecting consecutive probability ranks.
        width = min(len(order), 2 * self.k + 1)
        if right - left < width:
            left = max(0, right - width)
            right = min(len(order), left + width)
        return np.array([i for i in order[left:right] if i != true_index], dtype=int)

    def select(self, true_index):
        pool = self.candidate_pool(true_index)
        best, best_entropy = None, -np.inf
        for _ in range(self.subset_trials):
            ids = np.r_[true_index, self.rng.choice(pool, self.k - 1, replace=False)]
            score = entropy(self.q[ids])
            if score > best_entropy:
                best, best_entropy = ids, score
        return best, best_entropy

    def protect_run(self, points):
        points = tuple(points)
        if not points or not all(isinstance(p, TrajectoryPoint) for p in points):
            raise ValueError("nonempty trajectory required")
        events, true_ids, representations = [], [], []
        for j, point in enumerate(points):
            true_index = int(self.rn.nearest(point.lat, point.lon)[0])
            ids, _ = self.select(true_index)
            candidates = []
            for slot, index in enumerate(self.rng.permutation(ids)):
                candidate_id = f"event_{j:04d}_candidate_{slot:04d}"
                candidates.append(PublicCandidate(candidate_id, *self.rn.latlon(int(index))))
                if index == true_index:
                    true_ids.append(candidate_id)
            events.append(PublicEvent(f"event_{j:04d}", point.timestamp_s, tuple(candidates)))
            representations.append(TrajectoryPoint(point.timestamp_s, *self.rn.latlon(true_index)))
        return ProtectedRun(
            PublicTranscript(self.name, OutputKind.REAL_PLUS_DUMMIES, tuple(events), {
                "source_doi": "10.1109/INFOCOM.2014.6848002", "k": self.k,
                "subset_trials": self.subset_trials,
                "candidate_linkage": "event_local_unlinked_sets",
                "implementation_level": "paper_adaptation",
                "domain_adaptation": "public_road_vertices_instead_of_grid_cells",
            }), EvaluationTruth(points, tuple(true_ids), tuple(representations)))
