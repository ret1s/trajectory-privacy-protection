"""Evidence card and benchmark adapter for the clean-room TransProtect core."""

from __future__ import annotations

from typing import Any, Mapping, Sequence

import numpy as np

from benchmark.contracts import (
    ComponentMapping,
    ComponentStatus,
    ImplementationLevel,
    MethodCard,
    MethodCardMixin,
    SourceReference,
)
from benchmark.engines.transprotect import (
    CausalTransitionProbabilityProvider,
    TargetTravelCostUtility,
    TransProtectEngine,
)
from core.demo_protocol import OutputKind


TRANSPROTECT_SOURCE = SourceReference(
    citation=(
        "Yadav et al., Protecting Vehicle Location Privacy with "
        "Contextually-Driven Synthetic Location Generation, ACM SIGSPATIAL 2024"
    ),
    doi="10.1145/3678717.3691211",
    repository_url="https://github.com/sourabhy1797/VehiTrack",
    repository_revision="035684c6c666a9af7cbd9984d92300000eb65536",
)


TRANSPROTECT_CARD = MethodCard(
    method_id="transprotect_adaptation",
    display_name="TransProtect — clean-room paper core",
    output_kind=OutputKind.REPLACEMENT_TRAJECTORY,
    implementation_level=ImplementationLevel.PAPER_ADAPTATION,
    source=TRANSPROTECT_SOURCE,
    source_mapping=(
        ComponentMapping(
            component="causal all-location probability contract",
            status=ComponentStatus.IMPLEMENTED,
            implementation=(
                "benchmark.engines.transprotect.ProbabilityProvider.predict_proba"
            ),
            evidence=(
                "The provider returns p(v_j | x_1..x_(n-1)); the engine passes "
                "only prior true indices, matching paper Section 4.3."
            ),
        ),
        ComponentMapping(
            component="expected travel-cost utility loss",
            status=ComponentStatus.IMPLEMENTED,
            implementation=(
                "benchmark.engines.transprotect.expected_travel_cost_loss; "
                "benchmark.engines.transprotect.TargetTravelCostUtility"
            ),
            evidence=(
                "Implements Equation (13), including the absolute travel-cost "
                "difference used in the authors' released MATLAB scripts. The "
                "target-based form scales as N x M rather than N x N."
            ),
        ),
        ComponentMapping(
            component="dependency-light fitted benchmark predictor",
            status=ComponentStatus.ADAPTED,
            implementation=(
                "benchmark.engines.transprotect."
                "CausalTransitionProbabilityProvider"
            ),
            evidence=(
                "A sparse first-order transition model makes local/SUMO runs "
                "end-to-end, while its public name explicitly says it is not "
                "the TransProtect Transformer."
            ),
        ),
        ComponentMapping(
            component="utility-adjusted top-K candidate ranking",
            status=ComponentStatus.ADAPTED,
            implementation=(
                "benchmark.engines.transprotect.select_top_k_candidates"
            ),
            evidence=(
                "Implements the Section 4.4 score h[j] + alpha/Delta_c. Because "
                "the paper does not specify its zero-loss tie rule, the local "
                "adapter deterministically forces real membership when needed "
                "and records every such event."
            ),
        ),
        ComponentMapping(
            component="candidate-restricted Laplace and LP obfuscation",
            status=ComponentStatus.IMPLEMENTED,
            implementation=(
                "benchmark.engines.transprotect.restricted_laplace_probabilities; "
                "benchmark.engines.transprotect.solve_geo_ind_lp"
            ),
            evidence=(
                "Implements the integrations specified in Sections 4.1 and 5.1 "
                "and checks the bidirectional epsilon-Geo-I LP inequalities."
            ),
        ),
        ComponentMapping(
            component="Node2Vec-GCN-masked-Transformer architecture",
            status=ComponentStatus.ADAPTED,
            implementation=(
                "benchmark.engines.transprotect.build_torch_transprotect_model; "
                "benchmark.engines.transprotect.fit_torch_transprotect_model"
            ),
            evidence=(
                "Clean-room optional PyTorch construction follows Equations "
                "(9)--(12), causal history and sinusoidal position encoding. "
                "Unreported head/layer/dropout choices remain caller inputs."
            ),
        ),
        ComponentMapping(
            component="paper-equivalent learned weights",
            status=ComponentStatus.MISSING,
            implementation="not available from the primary sources",
            evidence=(
                "The audited official revision publishes no Node2Vec/GCN/"
                "Transformer trainer, configuration, embeddings or checkpoint."
            ),
        ),
        ComponentMapping(
            component="paper dataset split and reported-table parity",
            status=ComponentStatus.MISSING,
            implementation="not reproduced",
            evidence=(
                "The repository contains Rome/SF data and MATLAB intermediates "
                "but not the sampled trajectory IDs, complete path-consistent "
                "pipeline, or expected TransProtect model predictions."
            ),
        ),
        ComponentMapping(
            component="VehiTrack end-to-end attack parity",
            status=ComponentStatus.MISSING,
            implementation="not reproduced",
            evidence=(
                "The repository has MATLAB attack scripts/intermediates but no "
                "complete documented environment or trained Post-LSTM artifact "
                "with which to validate Tables 1--2."
            ),
        ),
    ),
    adaptation_summary=(
        "The paper-defined utility ranking and candidate-restricted Laplace/LP "
        "mechanisms are executable and accept genuine learned probabilities. "
        "The official learned model/checkpoint and table-level parity remain "
        "unavailable, so this must not be labelled a faithful reproduction."
    ),
    validation_evidence=(
        "tests/test_transprotect_faithful.py: deterministic paper-equation and protocol checks",
    ),
)


class TransProtectAdaptation(MethodCardMixin, TransProtectEngine):
    """Benchmark adapter for the strongest currently supportable reproduction."""

    name = TRANSPROTECT_CARD.method_id
    method_card = TRANSPROTECT_CARD

    @classmethod
    def from_road_network(
        cls,
        road_network: Any,
        *,
        training_trajectories: Sequence[Sequence[Any] | Mapping[str, Any]] = (),
        candidate_k: int = 10,
        target_count: int = 8,
        alpha: float = 10_000.0,
        epsilon: float = 0.005,
        probability_smoothing: float = 1e-6,
        probability_backoff_weight: float = 0.1,
        rng: np.random.Generator | None = None,
    ) -> "TransProtectAdaptation":
        """Build the bounded local comparator without using evaluation truth.

        Coordinate trajectories should come from a disjoint SUMO training
        population.  If none are supplied, directed road edges form an
        explicit topology-only proxy.  Neither fallback is represented as the
        paper's unavailable Node2Vec--GCN--Transformer weights.
        """

        indexed: list[tuple[int, ...]] = []
        for raw in training_trajectories:
            values = raw.get("points", ()) if isinstance(raw, Mapping) else raw
            locations: list[int] = []
            for point in values:
                if isinstance(point, (int, np.integer)):
                    index = int(point)
                    if not 0 <= index < len(road_network):
                        raise IndexError("training location is outside the graph")
                else:
                    if hasattr(point, "lat") and hasattr(point, "lon"):
                        lat, lon = point.lat, point.lon
                    else:
                        lat, lon = point[0], point[1]
                    index, _distance = road_network.nearest(float(lat), float(lon))
                locations.append(index)
            if locations:
                indexed.append(tuple(locations))

        provider = CausalTransitionProbabilityProvider(
            len(road_network),
            smoothing=probability_smoothing,
            backoff_weight=probability_backoff_weight,
        )
        if indexed:
            provider.fit(indexed)
            provider.implementation_name += "_trained_on_disjoint_sumo_background"
        else:
            node_to_index = {
                (node.item() if hasattr(node, "item") else node): index
                for index, node in enumerate(road_network.node_ids)
            }
            edge_sequences = [
                (node_to_index[u], node_to_index[v])
                for u, v in road_network.graph.edges()
                if u in node_to_index and v in node_to_index
            ]
            if not edge_sequences:
                edge_sequences = [(index,) for index in range(len(road_network))]
            provider.fit(edge_sequences)
            provider.implementation_name += "_trained_on_graph_edges_only"

        target_count = min(max(1, int(target_count)), len(road_network))
        targets = provider.most_frequent_locations(target_count)
        target_prior = provider.visit_prior(targets)
        utility = TargetTravelCostUtility(
            road_network,
            target_indices=targets,
            target_prior=target_prior,
            edge_weight="length",
        )
        target_label = (
            "disjoint_background_visit_target_proxy"
            if indexed
            else "graph_degree_target_proxy"
        )
        utility.target_selection_rule = (
            "top_visit_count_from_disjoint_background"
            if indexed
            else "top_visit_count_from_directed_graph_edges"
        )
        utility.implementation_name = utility.implementation_name.replace(
            "explicit_lbs_targets", target_label
        )
        return cls(
            road_network,
            probability_provider=provider,
            utility_provider=utility,
            candidate_k=min(max(1, int(candidate_k)), len(road_network)),
            alpha=alpha,
            epsilon=epsilon,
            obfuscator="laplace",
            rng=rng,
        )

    def protect_run(self, real_trajectory: Sequence[Any]):
        return self._attach_method_card(super().protect_run(real_trajectory))


__all__ = [
    "TRANSPROTECT_CARD",
    "TRANSPROTECT_SOURCE",
    "TransProtectAdaptation",
]
