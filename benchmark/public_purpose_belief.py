"""Public utility weights for multiple local query purposes.

This changes only the Q planner's POI coverage objective. The Geo-I emission,
latent prior, protected observations and privacy ledger are delegated unchanged.
No current GPS, private QuerySpec or private destination is a builder input.
"""
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path

import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import dijkstra

from evaluation.lane_travel import matrix


def _membership(scores, pois, categories, k):
    """Equal-category top-k membership at each public latent state."""
    n = scores.shape[0]
    result = np.zeros((n, len(pois) + 1))
    ranked = []
    nonempty = np.zeros(n, dtype=int)
    for category in categories:
        ids = np.array([i for i, p in enumerate(pois) if p['category'] == category])
        order = np.argsort(scores[:, ids], axis=1, kind='stable')[:, :k]
        selected = ids[order]
        valid = np.isfinite(np.take_along_axis(scores[:, ids], order, axis=1))
        counts = valid.sum(axis=1)
        nonempty += counts > 0
        ranked.append((selected, valid, counts))
    for selected, valid, counts in ranked:
        denom = nonempty * counts
        weights = np.divide(1., denom, out=np.zeros(n), where=denom > 0)
        rows, slots = np.where(valid)
        result[rows, selected[rows, slots]] = weights[rows]
    return result


@dataclass(frozen=True)
class PublicPurposeWeights:
    weights: object
    metadata: dict
    sha256: str


def build_public_purpose_weights(ranking, state_ids, *, public_radius_m=1000.,
                                 public_destination_states, k=5):
    """Uniform purpose mixture; detour uses a fixed public destination prior.

    Distances/time are directed. Lexical POI IDs break ties. Undefined public
    profiles have zero target mass, then each nonempty latent row is normalized.
    The approximation is a planner objective, not an intent posterior.
    """
    rn = ranking.rn
    state_ids = np.asarray(state_ids, dtype=int)
    destinations = np.asarray(public_destination_states, dtype=int)
    if (state_ids.ndim != 1 or not len(state_ids) or np.any(state_ids < 0)
            or np.any(state_ids >= len(rn)) or len(set(state_ids)) != len(state_ids)
            or destinations.ndim != 1 or not len(destinations)
            or np.any(destinations < 0) or np.any(destinations >= len(rn))
            or len(set(destinations)) != len(destinations)):
        raise ValueError('Distinct valid public latent and destination states required')
    if (not np.isfinite(public_radius_m) or public_radius_m <= 0
            or isinstance(k, bool) or int(k) != k or k < 1):
        raise ValueError('Positive public radius and integer top-k required')
    pois = tuple(sorted(ranking.pois, key=lambda p: p['id']))
    if pois != tuple(ranking.pois):
        raise ValueError('Public lexical POI indexing required')
    vertices = np.array([p['vertex'] for p in pois], dtype=int)
    # Costs from every origin to each POI, evaluated at the public latent states.
    reverse = matrix(rn).transpose().tocsr()
    all_to_poi = dijkstra(reverse, directed=True, indices=vertices).T
    distance = all_to_poi[state_ids]
    travel_time = dijkstra(matrix(rn, time=True).transpose().tocsr(),
                           directed=True, indices=vertices).T[state_ids]
    radius = distance.copy()
    radius[radius > public_radius_m] = np.inf
    nearest = _membership(distance, pois, ranking.categories, k)
    fastest = _membership(travel_time, pois, ranking.categories, k)
    within = _membership(radius, pois, ranking.categories, k)
    to_dest = dijkstra(reverse, directed=True, indices=destinations).T
    detour = np.zeros_like(nearest)
    for j in range(len(destinations)):
        direct = to_dest[state_ids, j]
        # inf-inf is not a valid detour. Exclude undefined public profiles.
        scores = np.full_like(distance, np.inf)
        valid = np.isfinite(direct)
        scores[valid] = distance[valid] + to_dest[vertices, j] - direct[valid, None]
        finite = np.isfinite(scores)
        scores[finite] = np.maximum(scores[finite], 0.)
        detour += _membership(scores, pois, ranking.categories, k) / len(destinations)
    weights = (nearest + fastest + within + detour) / 4.
    mass = weights.sum(axis=1)
    np.divide(weights, mass[:, None], out=weights, where=mass[:, None] > 0)
    assert np.all(weights[:, -1] == 0) and np.isfinite(weights).all()
    metadata = {'schema':'public-purpose-weights-v1',
        'state_ids':state_ids.tolist(), 'public_destination_states':destinations.tolist(),
        'public_destination_prior':'uniform over the declared public states',
        'public_radius_m':float(public_radius_m), 'k':int(k),
        'purposes':['nearest_distance','fastest_travel','within_radius','minimum_detour'],
        'purpose_prior':'uniform public approximation, never actual private intent',
        'undefined_profile':'zero membership, normalize remaining public objective mass',
        'builder_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    digest = hashlib.sha256(json.dumps(metadata, sort_keys=True).encode())
    digest.update(weights.astype('<f8').tobytes())
    return PublicPurposeWeights(csr_matrix(weights), metadata, digest.hexdigest())


class PurposeCoverAnchorModel:
    """Delegate the protected-location model; change only public POI weights."""
    def __init__(self, base, public_weights, *, alpha=.5):
        if (not np.isfinite(alpha) or not 0 <= alpha <= 1
                or public_weights.weights.shape != base.poi_weights.shape
                or public_weights.metadata['state_ids'] != base.state_ids.tolist()):
            raise ValueError('Matched public weights and alpha in [0,1] required')
        self.base, self.public_weights, self.alpha = base, public_weights, float(alpha)
        self.poi_weights = (base.poi_weights if alpha == 0 else
                            (1.-alpha)*base.poi_weights + alpha*public_weights.weights)
        self.sha256 = hashlib.sha256((base.sha256 + '/' + public_weights.sha256 +
                                      '/' + repr(self.alpha)).encode()).hexdigest()

    def __getattr__(self, name):
        return getattr(self.base, name)
