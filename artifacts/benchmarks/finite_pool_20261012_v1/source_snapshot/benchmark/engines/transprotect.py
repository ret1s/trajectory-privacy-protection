"""Clean-room TransProtect components mapped directly to the source paper.

The paper separates TransProtect into two concerns:

1. a learned Node2Vec -> GCN -> causal Transformer model that estimates
   ``p(v_j | x_1, ..., x_{n-1})`` for every road-network location; and
2. a deterministic paper-defined protection core that combines that
   probability with expected travel-cost loss, retains the top-K locations,
   and applies a Geo-I mechanism only on that candidate set.

The second concern is fully executable here.  The learned probability model is
an explicit dependency: callers must supply a :class:`ProbabilityProvider`.
This prevents an untrained heuristic from being silently reported as
TransProtect.  A dependency-optional PyTorch architecture factory implements
the layers described in Sections 4.2--4.3, but the paper/repository do not
publish enough hyperparameters or a checkpoint for paper-equivalent weights.

Primary sources:

* Yadav et al., ACM SIGSPATIAL 2024, DOI 10.1145/3678717.3691211,
  arXiv:2409.09495 (especially Sections 4.1--4.4 and 5.1).
* Authors' repository https://github.com/sourabhy1797/VehiTrack at revision
  ``035684c6c666a9af7cbd9984d92300000eb65536``.

No upstream source code is copied into this module.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import math
from typing import Any, Optional, Protocol, Sequence, runtime_checkable

import networkx as nx
import numpy as np
from scipy import sparse
from scipy.optimize import linprog


LatLon = tuple[float, float]


def _positive(value: float, name: str) -> float:
    result = float(value)
    if not math.isfinite(result) or result <= 0.0:
        raise ValueError(f"{name} must be finite and positive")
    return result


def _probability_vector(values: Sequence[float], size: int, name: str) -> np.ndarray:
    result = np.asarray(values, dtype=float)
    if result.shape != (size,):
        raise ValueError(f"{name} must have shape ({size},)")
    if not np.isfinite(result).all() or np.any(result < 0.0):
        raise ValueError(f"{name} must contain finite non-negative values")
    total = float(result.sum())
    if total <= 0.0:
        raise ValueError(f"{name} must have positive mass")
    return result / total


def _square_finite_matrix(values: Sequence[Sequence[float]], name: str) -> np.ndarray:
    result = np.asarray(values, dtype=float)
    if result.ndim != 2 or result.shape[0] != result.shape[1] or not result.size:
        raise ValueError(f"{name} must be a non-empty square matrix")
    if not np.isfinite(result).all():
        raise ValueError(f"{name} must contain only finite values")
    return result


@runtime_checkable
class ProbabilityProvider(Protocol):
    """Causal location model required by the TransProtect paper core."""

    @property
    def implementation_name(self) -> str:
        ...

    def predict_proba(self, history_indices: Sequence[int]) -> np.ndarray:
        """Return ``p(v_j | history)`` for every graph vertex."""

        ...


@runtime_checkable
class UtilityLossProvider(Protocol):
    """Scalable provider of Equation (13) losses over the location domain."""

    @property
    def implementation_name(self) -> str:
        ...

    @property
    def n_locations(self) -> int:
        ...

    def losses(self, real_index: int) -> np.ndarray:
        """Return utility loss for every release given one real location."""

        ...


@dataclass
class ArrayProbabilityProvider:
    """Deterministic provider for imported predictions and parity tests.

    Row ``n`` is used after observing ``n`` historical locations.  It is useful
    for consuming predictions exported by an external training pipeline; it is
    not itself a learned TransProtect model.
    """

    distributions: Sequence[Sequence[float]]
    implementation_name: str = "externally_supplied_probability_matrix"

    def __post_init__(self) -> None:
        matrix = np.asarray(self.distributions, dtype=float)
        if matrix.ndim != 2 or not matrix.size:
            raise ValueError("distributions must be a non-empty 2D matrix")
        if not np.isfinite(matrix).all() or np.any(matrix < 0.0):
            raise ValueError("distributions must be finite and non-negative")
        totals = matrix.sum(axis=1, keepdims=True)
        if np.any(totals <= 0.0):
            raise ValueError("every distribution must have positive mass")
        self.distributions = matrix / totals

    def predict_proba(self, history_indices: Sequence[int]) -> np.ndarray:
        row = len(tuple(history_indices))
        if row >= len(self.distributions):
            raise ValueError(
                "no externally supplied probability row for history length "
                f"{row}"
            )
        return np.asarray(self.distributions[row], dtype=float).copy()


class CausalTransitionProbabilityProvider:
    """Fitted, deterministic benchmark proxy when paper weights are absent.

    This is deliberately named and reported as a first-order proxy, not as the
    paper's Transformer.  It learns ``p(x_n | x_(n-1))`` from a caller-supplied
    *training* corpus with additive smoothing.  The sparse count table scales
    to city graphs without allocating an ``N x N`` transition matrix.
    """

    implementation_name = "empirical_markov_proxy_not_transprotect_transformer"

    def __init__(
        self,
        n_locations: int,
        *,
        smoothing: float = 1e-6,
        backoff_weight: float = 0.1,
    ) -> None:
        self.n_locations = int(n_locations)
        if self.n_locations <= 0:
            raise ValueError("n_locations must be positive")
        self.smoothing = _positive(smoothing, "smoothing")
        self.backoff_weight = float(backoff_weight)
        if not math.isfinite(self.backoff_weight) or not 0.0 <= self.backoff_weight <= 1.0:
            raise ValueError("backoff_weight must be finite and in [0, 1]")
        self._global_counts = np.zeros(self.n_locations, dtype=float)
        self._transition_counts: dict[int, dict[int, float]] = {}
        self.training_sequence_count = 0
        self.training_transition_count = 0

    def fit(
        self, training_trajectories: Sequence[Sequence[int]]
    ) -> "CausalTransitionProbabilityProvider":
        self._global_counts.fill(0.0)
        self._transition_counts.clear()
        self.training_sequence_count = 0
        self.training_transition_count = 0
        for raw_trajectory in training_trajectories:
            trajectory = tuple(int(index) for index in raw_trajectory)
            if not trajectory:
                continue
            if any(index < 0 or index >= self.n_locations for index in trajectory):
                raise IndexError("training location is outside the graph domain")
            self.training_sequence_count += 1
            for index in trajectory:
                self._global_counts[index] += 1.0
            for previous, current in zip(trajectory, trajectory[1:]):
                row = self._transition_counts.setdefault(previous, {})
                row[current] = row.get(current, 0.0) + 1.0
                self.training_transition_count += 1
        if self.training_sequence_count == 0:
            raise ValueError("at least one non-empty training trajectory is required")
        return self

    def predict_proba(self, history_indices: Sequence[int]) -> np.ndarray:
        if self.training_sequence_count == 0:
            raise RuntimeError("fit must be called before predict_proba")
        history = tuple(int(index) for index in history_indices)
        global_counts = self._global_counts + self.smoothing
        global_probabilities = global_counts / global_counts.sum()
        if history:
            previous = history[-1]
            if previous < 0 or previous >= self.n_locations:
                raise IndexError("history location is outside the graph domain")
            row = self._transition_counts.get(previous)
            if row:
                counts = np.full(self.n_locations, self.smoothing, dtype=float)
                for index, count in row.items():
                    counts[index] = count
                row_probabilities = counts / counts.sum()
                return (
                    (1.0 - self.backoff_weight) * row_probabilities
                    + self.backoff_weight * global_probabilities
                )
        return global_probabilities

    def most_frequent_locations(self, count: int) -> tuple[int, ...]:
        """Return deterministic target proxies learned from the training corpus."""

        if self.training_sequence_count == 0:
            raise RuntimeError("fit must be called before selecting targets")
        count = int(count)
        if not 1 <= count <= self.n_locations:
            raise ValueError("count must fit the location domain")
        indices = np.arange(self.n_locations, dtype=int)
        order = np.lexsort((indices, -self._global_counts))
        return tuple(int(index) for index in order[:count])

    def visit_prior(self, location_indices: Sequence[int]) -> np.ndarray:
        """Return the empirical target prior for explicitly selected targets."""

        if self.training_sequence_count == 0:
            raise RuntimeError("fit must be called before estimating a target prior")
        indices = np.asarray(tuple(int(index) for index in location_indices), dtype=int)
        if indices.ndim != 1 or not len(indices):
            raise ValueError("location_indices must be a non-empty vector")
        if np.any(indices < 0) or np.any(indices >= self.n_locations):
            raise IndexError("target location is outside the graph domain")
        counts = self._global_counts[indices]
        if float(counts.sum()) <= 0.0:
            return np.full(len(indices), 1.0 / len(indices))
        return counts / counts.sum()


@dataclass(frozen=True)
class DenseTravelCostUtility:
    """Equation (13) provider for small paper/reproduction domains."""

    travel_cost_matrix: Sequence[Sequence[float]]
    target_prior: Sequence[float]
    implementation_name: str = "dense_all_target_equation_13"
    n_locations: int = field(init=False)

    def __post_init__(self) -> None:
        costs = _square_finite_matrix(self.travel_cost_matrix, "travel_cost_matrix")
        prior = _probability_vector(
            self.target_prior, costs.shape[0], "target_prior"
        )
        object.__setattr__(self, "travel_cost_matrix", costs)
        object.__setattr__(self, "target_prior", prior)
        object.__setattr__(self, "n_locations", costs.shape[0])

    def losses(self, real_index: int) -> np.ndarray:
        return expected_travel_cost_loss(
            self.travel_cost_matrix, self.target_prior, real_index
        )


class TargetTravelCostUtility:
    """Scalable Equation (13) utility over an explicit POI/target set.

    Only an ``N x M`` cost table is retained, where ``M`` is the number of
    application targets.  For directed roads, one Dijkstra search from each
    target on the reversed graph obtains costs from every candidate to that
    target.  This is equivalent to Equation (13) without an ``N x N`` matrix.

    If ``target_indices`` is omitted, evenly spaced graph vertices are used as
    a deterministic *benchmark proxy* for unavailable POI data.  Such a run
    must not be presented as the paper's LBS target distribution.
    """

    def __init__(
        self,
        road_network: Any,
        *,
        target_indices: Optional[Sequence[int]] = None,
        target_prior: Optional[Sequence[float]] = None,
        max_proxy_targets: int = 32,
        edge_weight: str = "length",
        unreachable_multiplier: float = 2.0,
    ) -> None:
        self.rn = road_network
        self.n_locations = len(road_network)
        if self.n_locations <= 0:
            raise ValueError("road_network must be non-empty")
        if target_indices is None:
            count = min(max(1, int(max_proxy_targets)), self.n_locations)
            targets = np.linspace(
                0, self.n_locations - 1, num=count, dtype=int
            )
            targets = np.asarray(list(dict.fromkeys(targets.tolist())), dtype=int)
            self.implementation_name = (
                "equation_13_with_deterministic_graph_target_proxy"
            )
        else:
            targets = np.asarray(target_indices, dtype=int)
            self.implementation_name = "equation_13_with_explicit_lbs_targets"
        if targets.ndim != 1 or not len(targets):
            raise ValueError("target_indices must be a non-empty vector")
        if len(set(targets.tolist())) != len(targets):
            raise ValueError("target_indices must be unique")
        if np.any(targets < 0) or np.any(targets >= self.n_locations):
            raise IndexError("target index is outside the road-network domain")
        self.target_indices = targets
        self.target_prior = (
            np.full(len(targets), 1.0 / len(targets))
            if target_prior is None
            else _probability_vector(target_prior, len(targets), "target_prior")
        )
        self.edge_weight = str(edge_weight)
        self.unreachable_multiplier = _positive(
            unreachable_multiplier, "unreachable_multiplier"
        )
        self.unreachable_pair_count = 0
        self.total_pair_count = self.n_locations * len(self.target_indices)
        self.unreachable_fraction = 0.0
        self.disconnection_penalty: float | None = None
        self.cost_to_targets = self._compute_costs()

    def _compute_costs(self) -> np.ndarray:
        graph = self.rn.graph
        reversed_graph = graph.reverse(copy=False) if graph.is_directed() else graph
        node_ids = self.rn.node_ids.tolist()
        node_to_index = {node: index for index, node in enumerate(node_ids)}
        costs = np.full(
            (self.n_locations, len(self.target_indices)), np.nan, dtype=float
        )
        for column, target_index in enumerate(self.target_indices):
            target_node = node_ids[int(target_index)]
            lengths = nx.single_source_dijkstra_path_length(
                reversed_graph, target_node, weight=self.edge_weight
            )
            for node, value in lengths.items():
                index = node_to_index.get(node)
                if index is not None and math.isfinite(float(value)):
                    costs[index, column] = float(value)
        finite = costs[np.isfinite(costs)]
        if not len(finite):
            raise ValueError(
                f"no finite shortest-path costs using edge weight {self.edge_weight!r}"
            )
        # Disconnected target pairs have no finite Equation (13) value.  A
        # deterministic penalty keeps the benchmark runnable and is exposed in
        # implementation_name; faithful Rome/SF reproduction should instead
        # use the paper's connected target domain.
        penalty = max(float(finite.max()), 1.0) * self.unreachable_multiplier
        missing = np.isnan(costs)
        self.unreachable_pair_count = int(np.count_nonzero(missing))
        self.unreachable_fraction = (
            self.unreachable_pair_count / self.total_pair_count
        )
        if missing.any():
            self.disconnection_penalty = float(penalty)
            costs = np.where(np.isfinite(costs), costs, penalty)
            self.implementation_name += "_with_disconnection_penalty"
        return costs

    def losses(self, real_index: int) -> np.ndarray:
        real_index = int(real_index)
        if not 0 <= real_index < self.n_locations:
            raise IndexError("real_index is outside the location domain")
        return (
            np.abs(self.cost_to_targets - self.cost_to_targets[real_index][None, :])
            @ self.target_prior
        )


def expected_travel_cost_loss(
    travel_cost_matrix: Sequence[Sequence[float]],
    target_prior: Sequence[float],
    real_index: int,
) -> np.ndarray:
    """Evaluate Equation (13) for every candidate location.

    ``travel_cost_matrix[i, l]`` is the shortest travel cost from location
    ``i`` to target ``l``.  The result for candidate ``j`` is

    ``sum_l q_l * abs(c[real,l] - c[j,l])``.

    The absolute discrepancy is explicit in the authors' released MATLAB
    evaluation scripts (``dist_diff = abs(obf_dist - real_dist)``).
    """

    costs = _square_finite_matrix(travel_cost_matrix, "travel_cost_matrix")
    n_locations = costs.shape[0]
    prior = _probability_vector(target_prior, n_locations, "target_prior")
    real_index = int(real_index)
    if not 0 <= real_index < n_locations:
        raise IndexError("real_index is outside the location domain")
    return np.abs(costs - costs[real_index][None, :]) @ prior


def transprotect_weighted_scores(
    probability_scores: Sequence[float],
    utility_losses: Sequence[float],
    alpha: float,
) -> np.ndarray:
    """Compute the Section 4.4 score ``h[j] + alpha / Delta_c[j]``.

    A zero-loss location receives infinite score, the mathematical limit of
    the paper equation.  In particular, this keeps the real location in the
    candidate set because its utility discrepancy with itself is zero.
    """

    probabilities = np.asarray(probability_scores, dtype=float)
    losses = np.asarray(utility_losses, dtype=float)
    if probabilities.ndim != 1 or losses.shape != probabilities.shape:
        raise ValueError("probability_scores and utility_losses must be aligned vectors")
    probabilities = _probability_vector(
        probabilities, len(probabilities), "probability_scores"
    )
    if not np.isfinite(losses).all() or np.any(losses < 0.0):
        raise ValueError("utility_losses must be finite and non-negative")
    alpha = _positive(alpha, "alpha")
    inverse_loss = np.divide(
        alpha,
        losses,
        out=np.full_like(losses, np.inf, dtype=float),
        where=losses > 0.0,
    )
    return probabilities + inverse_loss


def select_top_k_candidates(
    probability_scores: Sequence[float],
    utility_losses: Sequence[float],
    *,
    k: int,
    alpha: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Return the paper's top-K candidate indices and their weighted scores.

    Equal scores are resolved by ascending location index, making imported
    predictions reproducible across platforms.  This is equivalent to the
    paper's min-heap result apart from its unspecified tie policy.
    """

    scores = transprotect_weighted_scores(probability_scores, utility_losses, alpha)
    k = int(k)
    if not 1 <= k <= len(scores):
        raise ValueError("k must be between 1 and the number of locations")
    indices = np.arange(len(scores), dtype=int)
    order = np.lexsort((indices, -scores))
    selected = order[:k]
    return selected, scores[selected]


def restricted_laplace_probabilities(
    real_index: int,
    candidate_indices: Sequence[int],
    privacy_distance_matrix: Sequence[Sequence[float]],
    epsilon: float,
) -> np.ndarray:
    """Planar-Laplace weights restricted to TransProtect's candidate set.

    The implementation follows the paper and the authors' released
    ``obfmatrix_generator_Laplace.m``: weight candidate ``j`` by
    ``exp(-epsilon * d(real,j))`` and normalize within the allowed set.  The
    caller is responsible for using matching distance/epsilon units.
    """

    distances = _square_finite_matrix(
        privacy_distance_matrix, "privacy_distance_matrix"
    )
    if np.any(distances < 0.0):
        raise ValueError("privacy distances must be non-negative")
    real_index = int(real_index)
    if not 0 <= real_index < distances.shape[0]:
        raise IndexError("real_index is outside the location domain")
    candidates = np.asarray(candidate_indices, dtype=int)
    if candidates.ndim != 1 or not len(candidates):
        raise ValueError("candidate_indices must be a non-empty vector")
    if len(set(candidates.tolist())) != len(candidates):
        raise ValueError("candidate_indices must be unique")
    if np.any(candidates < 0) or np.any(candidates >= distances.shape[0]):
        raise IndexError("candidate index is outside the location domain")
    epsilon = float(epsilon)
    if not math.isfinite(epsilon) or epsilon < 0.0:
        raise ValueError("epsilon must be finite and non-negative")
    log_weights = -epsilon * distances[real_index, candidates]
    log_weights -= float(np.max(log_weights))
    weights = np.exp(log_weights)
    return weights / weights.sum()


def solve_geo_ind_lp(
    utility_loss_matrix: Sequence[Sequence[float]],
    privacy_distance_matrix: Sequence[Sequence[float]],
    epsilon: float,
    *,
    real_prior: Optional[Sequence[float]] = None,
) -> np.ndarray:
    """Solve the paper's LP Geo-I obfuscation matrix on one candidate domain.

    Rows are real locations and columns are released locations.  The objective
    minimizes expected utility loss; constraints impose
    ``z[i,k] <= exp(epsilon*d(i,j))*z[j,k]`` in both directions and every row
    sums to one.  This is a clean-room equivalent of the equations in Section
    2 and the released ``LPObfuscationMatrx.m``.
    """

    utility = _square_finite_matrix(utility_loss_matrix, "utility_loss_matrix")
    distances = _square_finite_matrix(
        privacy_distance_matrix, "privacy_distance_matrix"
    )
    if utility.shape != distances.shape:
        raise ValueError("utility and privacy-distance matrices must have equal shape")
    if np.any(utility < 0.0) or np.any(distances < 0.0):
        raise ValueError("utility losses and privacy distances must be non-negative")
    epsilon = float(epsilon)
    if not math.isfinite(epsilon) or epsilon < 0.0:
        raise ValueError("epsilon must be finite and non-negative")

    n = utility.shape[0]
    prior = (
        np.full(n, 1.0 / n)
        if real_prior is None
        else _probability_vector(real_prior, n, "real_prior")
    )
    objective = (prior[:, None] * utility).reshape(-1)

    n_constraints = n * n * (n - 1)
    a_ub = sparse.lil_matrix((n_constraints, n * n), dtype=float)
    row = 0
    for i in range(n):
        for j in range(i + 1, n):
            try:
                factor = math.exp(epsilon * float(distances[i, j]))
            except OverflowError as exc:
                raise ValueError(
                    "epsilon * distance overflows; use consistent units or "
                    "rescale the privacy domain"
                ) from exc
            if not math.isfinite(factor):
                raise ValueError(
                    "epsilon * distance overflows; use consistent units or "
                    "rescale the privacy domain"
                )
            for released in range(n):
                a_ub[row, i * n + released] = 1.0
                a_ub[row, j * n + released] = -factor
                row += 1
                a_ub[row, j * n + released] = 1.0
                a_ub[row, i * n + released] = -factor
                row += 1

    a_eq = sparse.lil_matrix((n, n * n), dtype=float)
    for i in range(n):
        a_eq[i, i * n : (i + 1) * n] = 1.0
    result = linprog(
        objective,
        A_ub=a_ub.tocsr(),
        b_ub=np.zeros(n_constraints),
        A_eq=a_eq.tocsr(),
        b_eq=np.ones(n),
        bounds=(0.0, 1.0),
        method="highs",
    )
    if not result.success or result.x is None:
        raise RuntimeError(f"Geo-I LP failed: {result.message}")
    matrix = np.asarray(result.x, dtype=float).reshape(n, n)
    matrix[matrix < 0.0] = 0.0
    matrix /= matrix.sum(axis=1, keepdims=True)
    return matrix


def projected_distance_matrix(road_network: Any) -> np.ndarray:
    """Pairwise projected distance for a :class:`core.road_network.RoadNetwork`."""

    xy = np.asarray(road_network.xy, dtype=float)
    if xy.ndim != 2 or xy.shape[1] != 2 or not len(xy):
        raise ValueError("road_network.xy must be a non-empty (N,2) matrix")
    delta = xy[:, None, :] - xy[None, :, :]
    return np.linalg.norm(delta, axis=2)


class TransProtectEngine:
    """Executable paper-core TransProtect with injected causal predictions.

    This engine never estimates model probabilities heuristically.  A caller
    must provide an explicit probability source.  It may be a genuine learned
    architecture/checkpoint or the clearly labelled local Markov proxy above;
    either way, the remaining reproduction gap is machine-testable.
    """

    name = "transprotect_engine"
    implementation_level = "paper_adaptation"

    def __init__(
        self,
        road_network: Any,
        *,
        probability_provider: ProbabilityProvider,
        utility_provider: Optional[UtilityLossProvider] = None,
        travel_cost_matrix: Optional[Sequence[Sequence[float]]] = None,
        target_prior: Optional[Sequence[float]] = None,
        candidate_k: int = 10,
        alpha: float = 10_000.0,
        epsilon: float = 0.005,
        obfuscator: str = "laplace",
        privacy_distance_matrix: Optional[Sequence[Sequence[float]]] = None,
        force_include_real_on_tie: bool = True,
        rng: Optional[np.random.Generator] = None,
    ) -> None:
        if len(road_network) == 0:
            raise ValueError("road_network must contain at least one vertex")
        if not isinstance(probability_provider, ProbabilityProvider):
            raise TypeError("probability_provider must implement predict_proba")
        self.rn = road_network
        self.n_locations = len(road_network)
        self.probability_provider = probability_provider
        if utility_provider is not None and (
            travel_cost_matrix is not None or target_prior is not None
        ):
            raise ValueError(
                "provide utility_provider or dense travel_cost_matrix/target_prior, not both"
            )
        if utility_provider is None:
            if travel_cost_matrix is None or target_prior is None:
                raise ValueError(
                    "utility_provider is required unless both dense "
                    "travel_cost_matrix and target_prior are supplied"
                )
            utility_provider = DenseTravelCostUtility(
                travel_cost_matrix, target_prior
            )
        if not isinstance(utility_provider, UtilityLossProvider):
            raise TypeError("utility_provider must implement losses(real_index)")
        if int(utility_provider.n_locations) != self.n_locations:
            raise ValueError("utility_provider must match the road-network domain")
        self.utility_provider = utility_provider
        self.candidate_k = int(candidate_k)
        if not 1 <= self.candidate_k <= self.n_locations:
            raise ValueError("candidate_k must fit the road-network domain")
        self.alpha = _positive(alpha, "alpha")
        self.epsilon = float(epsilon)
        if not math.isfinite(self.epsilon) or self.epsilon < 0.0:
            raise ValueError("epsilon must be finite and non-negative")
        if obfuscator not in {"laplace", "lp"}:
            raise ValueError("obfuscator must be 'laplace' or 'lp'")
        self.obfuscator = obfuscator
        self.force_include_real_on_tie = bool(force_include_real_on_tie)
        self.privacy_distance_matrix = (
            None
            if privacy_distance_matrix is None
            else _square_finite_matrix(
                privacy_distance_matrix, "privacy_distance_matrix"
            )
        )
        if self.privacy_distance_matrix is not None and self.privacy_distance_matrix.shape != (
            self.n_locations,
            self.n_locations,
        ):
            raise ValueError("privacy_distance_matrix must match the road network")
        self.rng = rng if rng is not None else np.random.default_rng()
        self.last_candidate_sets: list[tuple[int, ...]] = []
        self.last_output_indices: list[int] = []
        self.last_output_utility_losses: list[float] = []
        self.last_forced_real_membership: list[bool] = []

    def _sample_laplace(self, real_index: int, candidates: np.ndarray) -> int:
        if self.privacy_distance_matrix is not None:
            distances = self.privacy_distance_matrix[real_index, candidates]
        else:
            distances = np.linalg.norm(
                np.asarray(self.rn.xy[candidates], dtype=float)
                - np.asarray(self.rn.xy[real_index], dtype=float),
                axis=1,
            )
        log_weights = -self.epsilon * distances
        log_weights -= float(np.max(log_weights))
        probabilities = np.exp(log_weights)
        probabilities /= probabilities.sum()
        return int(self.rng.choice(candidates, p=probabilities))

    def _candidate_utility_matrix(self, candidates: np.ndarray) -> np.ndarray:
        rows = [
            self.utility_provider.losses(int(real))[candidates]
            for real in candidates
        ]
        return np.asarray(rows, dtype=float)

    def _candidate_privacy_distances(self, candidates: np.ndarray) -> np.ndarray:
        if self.privacy_distance_matrix is not None:
            return self.privacy_distance_matrix[np.ix_(candidates, candidates)]
        xy = np.asarray(self.rn.xy[candidates], dtype=float)
        return np.linalg.norm(xy[:, None, :] - xy[None, :, :], axis=2)

    def _sample_lp(self, real_index: int, candidates: np.ndarray) -> int:
        matches = np.flatnonzero(candidates == int(real_index))
        if len(matches) != 1:
            raise RuntimeError(
                "the actual location must occur once in its TransProtect candidate set"
            )
        utility = self._candidate_utility_matrix(candidates)
        distances = self._candidate_privacy_distances(candidates)
        matrix = solve_geo_ind_lp(utility, distances, self.epsilon)
        row = matrix[int(matches[0])]
        return int(self.rng.choice(candidates, p=row))

    def protect_indices(self, real_indices: Sequence[int]) -> list[int]:
        real = [int(index) for index in real_indices]
        if not real:
            raise ValueError("real_indices must be non-empty")
        if any(index < 0 or index >= self.n_locations for index in real):
            raise IndexError("real location is outside the road-network domain")
        self.last_candidate_sets = []
        self.last_output_indices = []
        self.last_output_utility_losses = []
        self.last_forced_real_membership = []
        history: list[int] = []
        for real_index in real:
            probabilities = _probability_vector(
                self.probability_provider.predict_proba(tuple(history)),
                self.n_locations,
                "model probability scores",
            )
            utility = np.asarray(self.utility_provider.losses(real_index), dtype=float)
            if utility.shape != (self.n_locations,):
                raise ValueError(
                    "utility provider must return one loss per graph location"
                )
            candidates, _ = select_top_k_candidates(
                probabilities,
                utility,
                k=self.candidate_k,
                alpha=self.alpha,
            )
            forced_membership = False
            if real_index not in candidates:
                if not self.force_include_real_on_tie:
                    raise RuntimeError(
                        "Equation (13) produced more zero-loss ties than K and "
                        "the paper does not specify a tie rule that guarantees "
                        "actual-location membership"
                    )
                # A zero-loss real point has the mathematical maximum score.
                # If an unspecified top-K tie policy nevertheless excluded it,
                # the local adapter replaces the final tied member so the
                # candidate-restricted obfuscator remains defined.  The event
                # is recorded for evaluator-only diagnostics.  The policy is
                # public configuration, but whether it fired depends on the
                # secret real index and must not enter the attacker transcript.
                candidates = candidates.copy()
                candidates[-1] = real_index
                forced_membership = True
            output = (
                self._sample_laplace(real_index, candidates)
                if self.obfuscator == "laplace"
                else self._sample_lp(real_index, candidates)
            )
            self.last_candidate_sets.append(tuple(int(i) for i in candidates))
            self.last_output_indices.append(output)
            self.last_output_utility_losses.append(float(utility[output]))
            self.last_forced_real_membership.append(forced_membership)
            history.append(real_index)
        return list(self.last_output_indices)

    def protect_trajectory(
        self,
        points: Sequence[Any],
        times: Optional[Sequence[Any]] = None,
    ) -> list[LatLon]:
        points = tuple(points)
        if not points:
            raise ValueError("points must be non-empty")
        if times is not None and len(tuple(times)) != len(points):
            raise ValueError("times must align with points")
        real_indices: list[int] = []
        for point in points:
            if hasattr(point, "lat") and hasattr(point, "lon"):
                lat, lon = point.lat, point.lon
            else:
                lat, lon = point[0], point[1]
            index, _ = self.rn.nearest(float(lat), float(lon))
            real_indices.append(index)
        return [self.rn.latlon(index) for index in self.protect_indices(real_indices)]

    def protect_run(self, real_trajectory: Sequence[Any]):
        from core.demo_protocol import TrajectoryPoint, make_replacement_run

        real = tuple(real_trajectory)
        if not real or not all(isinstance(point, TrajectoryPoint) for point in real):
            raise TypeError("protect_run expects non-empty TrajectoryPoint values")
        protected = self.protect_trajectory(real)
        replacement = tuple(
            TrajectoryPoint(point.timestamp_s, lat, lon)
            for point, (lat, lon) in zip(real, protected)
        )
        return make_replacement_run(
            self.name,
            real,
            replacement,
            public_parameters={
                "implementation_level": self.implementation_level,
                "implementation_origin": "benchmark.engines.transprotect.TransProtectEngine",
                "probability_provider": self.probability_provider.implementation_name,
                "training_sequence_count": int(
                    getattr(self.probability_provider, "training_sequence_count", 0)
                ),
                "training_transition_count": int(
                    getattr(self.probability_provider, "training_transition_count", 0)
                ),
                "probability_smoothing": float(
                    getattr(self.probability_provider, "smoothing", 0.0)
                ),
                "probability_backoff_weight": float(
                    getattr(self.probability_provider, "backoff_weight", 0.0)
                ),
                "utility_provider": self.utility_provider.implementation_name,
                "utility_target_indices": ",".join(
                    str(int(value))
                    for value in getattr(self.utility_provider, "target_indices", ())
                ),
                "utility_target_prior": ",".join(
                    f"{float(value):.12g}"
                    for value in getattr(self.utility_provider, "target_prior", ())
                ),
                "utility_target_selection": str(
                    getattr(
                        self.utility_provider,
                        "target_selection_rule",
                        "externally_supplied",
                    )
                ),
                "utility_edge_weight": str(
                    getattr(self.utility_provider, "edge_weight", "dense_matrix")
                ),
                "utility_edge_weight_unit": (
                    "metres" if getattr(self.utility_provider, "edge_weight", None) == "length" else "provider_defined"
                ),
                "utility_unreachable_fraction": float(
                    getattr(self.utility_provider, "unreachable_fraction", 0.0)
                ),
                "utility_disconnection_penalty": getattr(
                    self.utility_provider, "disconnection_penalty", None
                ),
                "candidate_k": self.candidate_k,
                "alpha": self.alpha,
                "epsilon": self.epsilon,
                "epsilon_unit": "m^-1",
                "obfuscator": self.obfuscator,
                "candidate_selection_depends_on_current_secret": True,
                "privacy_claim_scope": "candidate_restricted_sampling_only",
                "end_to_end_geo_i_guarantee": False,
                "zero_loss_tie_policy": (
                    "force_real_membership_replace_last"
                    if self.force_include_real_on_tie
                    else "fail_on_real_exclusion"
                ),
            },
        )


def build_torch_transprotect_model(
    adjacency: Sequence[Sequence[float]],
    node2vec_embeddings: Sequence[Sequence[float]],
    *,
    num_heads: int,
    num_transformer_layers: int,
    num_gcn_layers: int = 1,
    dropout: float = 0.0,
    train_node2vec_embeddings: bool = False,
):
    """Build the paper-described GCN + masked-Transformer architecture.

    PyTorch is imported lazily because it is an optional reproduction
    dependency.  Node2Vec embeddings are supplied rather than fabricated: the
    paper omits its walk/window/negative-sampling settings and the official
    repository publishes neither the training code nor learned embeddings.

    ``num_heads``, layer counts and dropout are mandatory/explicit because the
    paper does not report them.  Consequently, constructing this architecture
    alone is not evidence of paper-equivalent reproduction.
    """

    try:
        import torch  # pyright: ignore[reportMissingImports]
        from torch import nn
    except ImportError as exc:  # pragma: no cover - optional heavyweight path
        raise RuntimeError(
            "PyTorch is required for the learned TransProtect architecture; "
            "the paper reports PyTorch 2.1"
        ) from exc

    adjacency_np = _square_finite_matrix(adjacency, "adjacency")
    embeddings_np = np.asarray(node2vec_embeddings, dtype=np.float32)
    if embeddings_np.ndim != 2 or embeddings_np.shape[0] != adjacency_np.shape[0]:
        raise ValueError("node2vec_embeddings must have one row per graph node")
    embedding_dim = int(embeddings_np.shape[1])
    if embedding_dim <= 0:
        raise ValueError("node2vec_embeddings must have positive dimensionality")
    if int(num_heads) <= 0 or embedding_dim % int(num_heads):
        raise ValueError("num_heads must divide the embedding dimension")
    if int(num_transformer_layers) <= 0 or int(num_gcn_layers) <= 0:
        raise ValueError("layer counts must be positive")
    if not 0.0 <= float(dropout) < 1.0:
        raise ValueError("dropout must be in [0,1)")

    graph = np.maximum(adjacency_np, adjacency_np.T)
    graph = graph + np.eye(len(graph), dtype=float)
    degree = graph.sum(axis=1)
    if np.any(degree <= 0.0):
        raise ValueError("normalized GCN adjacency has a zero-degree node")
    inv_sqrt = np.diag(1.0 / np.sqrt(degree))
    normalized = inv_sqrt @ graph @ inv_sqrt

    class _TorchTransProtectModel(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.node_embeddings = nn.Parameter(
                torch.as_tensor(embeddings_np.copy()),
                requires_grad=bool(train_node2vec_embeddings),
            )
            self.register_buffer(
                "normalized_adjacency",
                torch.as_tensor(normalized, dtype=torch.float32),
            )
            self.gcn_layers = nn.ModuleList(
                nn.Linear(embedding_dim, embedding_dim, bias=False)
                for _ in range(int(num_gcn_layers))
            )
            layer = nn.TransformerEncoderLayer(
                d_model=embedding_dim,
                nhead=int(num_heads),
                dim_feedforward=4 * embedding_dim,
                dropout=float(dropout),
                batch_first=True,
            )
            self.transformer = nn.TransformerEncoder(
                layer, num_layers=int(num_transformer_layers)
            )
            self.location_head = nn.Linear(embedding_dim, len(graph))

        @staticmethod
        def _positional_encoding(length: int, dim: int, device: Any):
            positions = torch.arange(length, device=device).float()[:, None]
            dimensions = torch.arange(0, dim, 2, device=device).float()
            scale = torch.exp(-math.log(10_000.0) * dimensions / dim)
            result = torch.zeros((length, dim), device=device)
            result[:, 0::2] = torch.sin(positions * scale)
            if dim > 1:
                result[:, 1::2] = torch.cos(positions * scale[: result[:, 1::2].shape[1]])
            return result

        def graph_embeddings(self):
            values = self.node_embeddings
            for layer in self.gcn_layers:
                values = torch.sigmoid(
                    self.normalized_adjacency @ layer(values)
                )
            return values

        def forward(self, real_location_ids: Any, padding_mask: Any = None):
            """Predict each x_n from x_1..x_(n-1), never from x_n itself."""

            if real_location_ids.ndim != 2:
                raise ValueError("real_location_ids must have shape (batch,time)")
            graph_embeddings = self.graph_embeddings()
            batch, length = real_location_ids.shape
            shifted = torch.zeros(
                (batch, length, embedding_dim),
                dtype=graph_embeddings.dtype,
                device=real_location_ids.device,
            )
            if length > 1:
                shifted[:, 1:, :] = graph_embeddings[real_location_ids[:, :-1]]
            shifted = shifted + self._positional_encoding(
                length, embedding_dim, real_location_ids.device
            )[None, :, :]
            causal_mask = torch.triu(
                torch.ones((length, length), dtype=torch.bool, device=shifted.device),
                diagonal=1,
            )
            encoded = self.transformer(
                shifted,
                mask=causal_mask,
                src_key_padding_mask=padding_mask,
            )
            return self.location_head(encoded)

    return _TorchTransProtectModel()


def fit_torch_transprotect_model(
    model: Any,
    training_trajectories: Sequence[Sequence[int]],
    *,
    epochs: int,
    batch_size: int = 50,
    learning_rate: float = 0.001,
    seed: int = 0,
    device: str = "cpu",
) -> list[float]:
    """Fit the clean-room architecture with the paper's cross-entropy loss.

    Batch size 50 and initial learning rate 0.001 are the reported settings.
    Epoch count, Adam optimizer, shuffle seed and stopping policy are explicit
    adaptation choices because the paper does not disclose them.  Returned
    epoch losses should be archived with every trained checkpoint.
    """

    try:
        import torch  # pyright: ignore[reportMissingImports]
        from torch.nn import functional as functional  # pyright: ignore[reportMissingImports]
    except ImportError as exc:  # pragma: no cover - optional heavyweight path
        raise RuntimeError("PyTorch 2.1-compatible runtime is required") from exc

    epochs = int(epochs)
    batch_size = int(batch_size)
    if epochs <= 0 or batch_size <= 0:
        raise ValueError("epochs and batch_size must be positive")
    learning_rate = _positive(learning_rate, "learning_rate")
    trajectories = [tuple(int(index) for index in item) for item in training_trajectories]
    trajectories = [item for item in trajectories if item]
    if not trajectories:
        raise ValueError("at least one non-empty training trajectory is required")
    n_locations = int(model.location_head.out_features)
    if any(
        index < 0 or index >= n_locations
        for trajectory in trajectories
        for index in trajectory
    ):
        raise IndexError("training location is outside the model domain")

    torch.manual_seed(int(seed))
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    rng = np.random.default_rng(int(seed))
    epoch_losses: list[float] = []
    for _ in range(epochs):
        order = rng.permutation(len(trajectories))
        total_loss = 0.0
        total_labels = 0
        model.train()
        for start in range(0, len(order), batch_size):
            batch = [trajectories[int(i)] for i in order[start : start + batch_size]]
            max_length = max(map(len, batch))
            inputs = torch.zeros(
                (len(batch), max_length), dtype=torch.long, device=device
            )
            labels = torch.full(
                (len(batch), max_length), -100, dtype=torch.long, device=device
            )
            padding_mask = torch.ones(
                (len(batch), max_length), dtype=torch.bool, device=device
            )
            for row, trajectory in enumerate(batch):
                length = len(trajectory)
                values = torch.as_tensor(trajectory, dtype=torch.long, device=device)
                inputs[row, :length] = values
                labels[row, :length] = values
                padding_mask[row, :length] = False
            optimizer.zero_grad(set_to_none=True)
            logits = model(inputs, padding_mask=padding_mask)
            loss = functional.cross_entropy(
                logits.reshape(-1, n_locations),
                labels.reshape(-1),
                ignore_index=-100,
            )
            loss.backward()
            optimizer.step()
            label_count = sum(map(len, batch))
            total_loss += float(loss.detach().cpu()) * label_count
            total_labels += label_count
        epoch_losses.append(total_loss / total_labels)
    return epoch_losses


class TorchTransProtectProbabilityProvider:
    """Expose a fitted clean-room PyTorch model through the causal protocol."""

    implementation_name = "cleanroom_node2vec_gcn_transformer_unvalidated"

    def __init__(self, model: Any, *, device: str = "cpu") -> None:
        try:
            import torch  # pyright: ignore[reportMissingImports]
        except ImportError as exc:  # pragma: no cover - optional heavyweight path
            raise RuntimeError("PyTorch 2.1-compatible runtime is required") from exc
        self._torch = torch
        self.model = model.to(device)
        self.device = device
        self.n_locations = int(model.location_head.out_features)

    def predict_proba(self, history_indices: Sequence[int]) -> np.ndarray:
        history = tuple(int(index) for index in history_indices)
        if any(index < 0 or index >= self.n_locations for index in history):
            raise IndexError("history location is outside the model domain")
        # The final placeholder is never embedded by the model: forward() shifts
        # the input right, so this position depends exactly on `history`.
        sequence = history + (0,)
        values = self._torch.as_tensor(
            [sequence], dtype=self._torch.long, device=self.device
        )
        self.model.eval()
        with self._torch.no_grad():
            logits = self.model(values)[0, len(history)]
            probabilities = self._torch.softmax(logits, dim=-1)
        return probabilities.detach().cpu().numpy().astype(float)


__all__ = [
    "ArrayProbabilityProvider",
    "CausalTransitionProbabilityProvider",
    "DenseTravelCostUtility",
    "ProbabilityProvider",
    "TargetTravelCostUtility",
    "TorchTransProtectProbabilityProvider",
    "TransProtectEngine",
    "UtilityLossProvider",
    "build_torch_transprotect_model",
    "expected_travel_cost_loss",
    "fit_torch_transprotect_model",
    "projected_distance_matrix",
    "restricted_laplace_probabilities",
    "select_top_k_candidates",
    "solve_geo_ind_lp",
    "transprotect_weighted_scores",
]
