"""Public reference top-k profiles matched to a fixed deeper service reply.

All inputs are a public graph/catalogue, public prototype parameters and the
existing protected-anchor grid. Current GPS/QuerySpec/family targets are absent.
Directed distance, travel time, radius and detour profiles follow local-ranking
semantics. A profile is (latent state, public purpose prototype, category);
empty references remain N/A mass before conditioning, never utility zero/one.
Equivalent reference sets are coalesced exactly for efficient discrete CVaR.
"""
from dataclasses import dataclass
from fractions import Fraction
import hashlib
import json
from pathlib import Path

import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import dijkstra

from benchmark.response_aware_belief import ResponseAwareAnchorModel
from evaluation.lane_travel import matrix


def _integer(value, name):
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)) or value < 1:
        raise ValueError(f'Positive integer {name} required')
    return int(value)


@dataclass(frozen=True)
class ProfileScore:
    mean: float
    lower_tail_cvar: float
    objective: float


class PublicServiceProfiles:
    """Fixed reference sets and their public latent/prototype/category masses."""
    def __init__(self, base, context, references, profile_mass, metadata):
        self.context, self.state_ids = context, base.state_ids.copy()
        self.state_ids.setflags(write=False)
        self.reference_profiles = tuple(tuple(x) for x in references)
        self.profile_mass = csr_matrix(profile_mass)
        self.metadata = metadata
        self.reference_k = metadata['reference_k']
        rows, cols = [], []
        for i, profile in enumerate(self.reference_profiles):
            rows.extend([i]*len(profile)); cols.extend(profile)
        self.count_incidence = csr_matrix((np.ones(len(rows), dtype=np.int16), (rows, cols)),
            shape=(len(references), len(context.pois)+1))
        self.lengths = np.asarray([len(x) for x in references], dtype=np.int16)
        lengths = np.maximum(1, self.lengths)
        self.recall_incidence = self.count_incidence.multiply(1./lengths[:, None]).tocsr()
        # Includes all reference counts <=k, with exact rational level identities.
        fractions = sorted({Fraction(j, n) for n in range(1, self.reference_k+1) for j in range(n+1)})
        self.levels = np.asarray([float(x) for x in fractions])
        indices = {x: i for i, x in enumerate(fractions)}
        self.level_lookup = np.zeros((self.reference_k+1, self.reference_k+1), dtype=int)
        for n in range(1, self.reference_k+1):
            for j in range(n+1):
                self.level_lookup[n, j] = indices[Fraction(j, n)]
        flattened = context.signatures[context.access].reshape(len(context.rn), -1)
        self.response_ids = np.where(flattened >= 0, flattened, len(context.pois))
        _, self.service_profiles = np.unique(flattened, axis=0, return_inverse=True)
        digest = hashlib.sha256(json.dumps(metadata, sort_keys=True, separators=(',', ':')).encode())
        for array in (self.profile_mass.data.astype('<f8'), self.profile_mass.indices.astype('<i8'),
                      self.profile_mass.indptr.astype('<i8'), self.count_incidence.indices.astype('<i8'),
                      self.count_incidence.indptr.astype('<i8')):
            digest.update(array.tobytes())
        self.sha256 = digest.hexdigest()

    def objective(self, belief_weights, *, tail_mass=.25, risk_weight=.5):
        return VectorizedProfileObjective(self, belief_weights, tail_mass=tail_mass, risk_weight=risk_weight)


class VectorizedProfileObjective:
    """Exact mean/discrete lower-tail objective for the declared finite profiles.

    Batching uses the finite recall levels j/n, n<=reference_k, rather than
    sorting thousands of identical levels for every one-track proposal.
    This is a public-objective optimizer, not user utility or a privacy score.
    """
    def __init__(self, profiles, belief_weights, *, tail_mass=.25, risk_weight=.5):
        if (isinstance(tail_mass, bool) or isinstance(risk_weight, bool)
                or not np.isfinite(tail_mass) or not 0 < tail_mass <= 1
                or not np.isfinite(risk_weight) or not 0 <= risk_weight <= 1):
            raise ValueError('Require public 0<a<=1 and 0<=lambda<=1')
        belief = np.asarray(belief_weights, dtype=float)
        if (belief.shape != (len(profiles.state_ids),) or not np.isfinite(belief).all()
                or np.any(belief < 0) or belief.sum() <= 0):
            raise ValueError('Matched finite protected belief required')
        belief = belief/belief.sum()
        masses = np.asarray(belief @ profiles.profile_mass).ravel()
        total = masses.sum()
        self.empty_profile_mass = float(masses[profiles.lengths == 0].sum()/total)
        active = (profiles.lengths > 0) & (masses > 0)
        if not active.any():
            raise ValueError('All positive-mass reference profiles are undefined')
        self.profiles, self.tail_mass, self.risk_weight = profiles, float(tail_mass), float(risk_weight)
        self.probabilities = masses[active]/masses[active].sum()
        self.incidence = profiles.count_incidence[active].tocsr()
        self.lengths = profiles.lengths[active]
        self.poi_weights = np.asarray(self.probabilities @ profiles.recall_incidence[active]).ravel()
        self.active_profile_count = int(active.sum())

    def covered(self, selected):
        mask = np.zeros(len(self.profiles.context.pois)+1, dtype=np.int16)
        if len(selected):
            ids = np.asarray(selected, dtype=int)
            if np.any(ids < 0) or np.any(ids >= len(self.profiles.context.rn)):
                raise ValueError('Known public response states required')
            mask[self.profiles.response_ids[ids]] = 1
        mask[-1] = 0
        return mask

    def _scores(self, masks):
        hits = np.asarray(self.incidence @ masks.T, dtype=int)
        codes = self.profiles.level_lookup[self.lengths[:, None], hits]
        columns, levels = masks.shape[0], len(self.profiles.levels)
        indexes = (codes + np.arange(columns)[None, :]*levels).T.ravel()
        masses = np.bincount(indexes, weights=np.tile(self.probabilities, columns),
                             minlength=columns*levels).reshape(columns, levels)
        mean = masses @ self.profiles.levels
        previous = np.cumsum(masses, axis=1)-masses
        take = np.minimum(masses, np.maximum(0., self.tail_mass-previous))
        tail = (take @ self.profiles.levels)/self.tail_mass
        objective = (1.-self.risk_weight)*mean + self.risk_weight*tail
        return mean, tail, objective

    def score(self, selected):
        mean, tail, objective = self._scores(self.covered(selected)[None, :])
        return ProfileScore(float(mean[0]), float(tail[0]), float(objective[0]))

    def scores_replacing(self, states, others, *, batch_size=128):
        """Evaluate every candidate, retaining exact union and fractional CVaR."""
        states = np.asarray(states, dtype=int)
        batch_size = _integer(batch_size, 'batch size')
        if states.ndim != 1 or np.any(states < 0) or np.any(states >= len(self.profiles.context.rn)):
            raise ValueError('Known one-dimensional public response states required')
        result = [np.empty(len(states)) for _ in range(3)]
        base = self.covered(others)
        for start in range(0, len(states), batch_size):
            ids = states[start:start+batch_size]
            masks = np.broadcast_to(base, (len(ids), len(base))).copy()
            masks[np.arange(len(ids))[:, None], self.profiles.response_ids[ids]] = 1
            masks[:, -1] = 0
            for dest, source in zip(result, self._scores(masks)):
                dest[start:start+len(ids)] = source
        return tuple(result)


def build_public_service_profiles(base, reply20, *, public_destination_states,
                                  public_radius_m=1000., reference_k=5):
    """Build all four public purposes; never derive prototypes from trip labels.

    Radius/destination states are PUBLIC policy arguments. The caller must use
    graph-only prototypes, not an evaluator's true destinations. Server replies
    stay nearest-distance top-L per category; only local reference purposes vary.
    """
    reference_k = _integer(reference_k, 'reference top-k')
    if base.context.k != reference_k:
        raise ValueError('Reference top-k must match the unchanged base anchor context')
    ResponseAwareAnchorModel(base, reply20)  # Exact graph/catalogue/reference-prefix gate.
    if isinstance(public_radius_m, bool) or not np.isfinite(public_radius_m) or public_radius_m <= 0:
        raise ValueError('Positive fixed public radius required')
    raw = list(public_destination_states)
    if (not raw or any(isinstance(x, bool) or not isinstance(x, (int, np.integer)) for x in raw)
            or len(set(raw)) != len(raw) or min(raw) < 0 or max(raw) >= len(base.rn)):
        raise ValueError('Distinct valid PUBLIC destination states required')
    destinations = np.asarray(sorted(raw), dtype=int)
    rn, pois, categories = base.rn, reply20.pois, reply20.categories
    if not categories or not pois:
        raise ValueError('Nonempty fixed public POI catalogue required')
    # GPS and Q both use the service's public coordinate-to-road access rule.
    states = base.context.access[base.state_ids]
    unique, inverse = np.unique(states, return_inverse=True)
    vertices = np.asarray([p['vertex'] for p in pois], dtype=int)
    distances = np.empty((len(unique), len(pois)))
    times = np.empty_like(distances)
    direct = np.empty((len(unique), len(destinations)))
    distance_graph, time_graph = matrix(rn), matrix(rn, time=True)
    for start in range(0, len(unique), 32):
        cost = dijkstra(distance_graph, directed=True, indices=unique[start:start+32])
        distances[start:start+len(cost)] = cost[:, vertices]
        direct[start:start+len(cost)] = cost[:, destinations]
        elapsed = dijkstra(time_graph, directed=True, indices=unique[start:start+32])
        times[start:start+len(elapsed)] = elapsed[:, vertices]
    to_dest = dijkstra(distance_graph.transpose().tocsr(), directed=True, indices=destinations)
    category_ids = [np.asarray([i for i, p in enumerate(pois) if p['category'] == c], dtype=int)
                    for c in categories]
    radius = distances.copy(); radius[radius > public_radius_m] = np.inf
    prototypes = [('nearest_distance', distances, .25), ('fastest_travel', times, .25),
                  ('within_radius', radius, .25)]
    for j, destination in enumerate(destinations):
        scores = np.full_like(distances, np.inf)
        valid = np.isfinite(direct[:, j])
        # Match local ranker's floating operation order, including detour ties.
        scores[valid] = distances[valid] + to_dest[j, vertices] - direct[valid, j, None]
        finite = np.isfinite(scores); scores[finite] = np.maximum(0., scores[finite])
        prototypes.append((f'minimum_detour:{destination}', scores, .25/len(destinations)))
    references, indexes, rows, cols, values = [], {}, [], [], []
    empty_count = 0
    for name, costs, weight in prototypes:
        for ids in category_ids:
            order = np.argsort(costs[:, ids], axis=1, kind='stable')[:, :reference_k]
            selected = ids[order]
            finite = np.isfinite(np.take_along_axis(costs[:, ids], order, axis=1))
            unique_refs = [tuple(sorted(map(int, row[valid]))) for row, valid in zip(selected, finite)]
            for latent, at in enumerate(inverse):
                profile = unique_refs[at]
                empty_count += not profile
                if profile not in indexes:
                    indexes[profile] = len(references); references.append(profile)
                rows.append(latent); cols.append(indexes[profile]); values.append(weight/len(categories))
    mass = csr_matrix((values, (rows, cols)), shape=(len(states), len(references)))
    if not np.allclose(np.asarray(mass.sum(axis=1)).ravel(), 1., rtol=0, atol=1e-12):
        raise RuntimeError('Public prototype/category weights lost mass')
    metadata = dict(schema='public-service-profiles-v1', reference_k=reference_k,
        reply_l=reply20.k, reference_context_sha256=base.context.sha256,
        reply_context_sha256=reply20.sha256, catalogue_sha256=rn.catalogue_sha256,
        state_ids=base.state_ids.tolist(), public_destination_states=destinations.tolist(),
        public_radius_m=float(public_radius_m), purpose_prior='uniform across four purposes',
        destination_prior='uniform fixed public graph states, independent of trip',
        category_prior='uniform fixed public categories',
        empty_reference='retain N/A mass; condition objective on nonempty public profiles',
        input_profile_count=len(states)*len(prototypes)*len(categories),
        empty_input_profile_count=empty_count, coalesced_profile_count=len(references),
        server_service='nearest-distance top-L per ALL public categories',
        access_rule='public coordinate-to-state mapping, including duplicate-coordinate ambiguity',
        builder_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    return PublicServiceProfiles(base, reply20, references, mass, metadata)


class PublicProfileAnchorModel:
    """Preserve the emission/prior/filter; expose service-aligned public profiles."""
    def __init__(self, base, profiles):
        ResponseAwareAnchorModel(base, profiles.context)
        if not np.array_equal(base.state_ids, profiles.state_ids):
            raise ValueError('Profiles must use the same protected-anchor grid')
        self.base, self.profiles, self.context = base, profiles, profiles.context
        # Compatibility only; the new engine conditions complete profile mass
        # at each protected belief, rather than averaging per-state renormalizations.
        self.poi_weights = profiles.profile_mass @ profiles.recall_incidence
        self.sha256 = hashlib.sha256((base.sha256+'/'+profiles.sha256).encode()).hexdigest()

    def __getattr__(self, name):
        return getattr(self.base, name)
