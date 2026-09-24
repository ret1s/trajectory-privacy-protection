"""GPS-independent fixed remote-query cover; a service-contract control.

The fit interface accepts only deployment-public data. Every session sends the
same coordinates. The zero coordinate-information statement is conditional on
the public region, catalogue and request clock, not a timing/identity claim.
"""
import networkx as nx
import numpy as np
from benchmark.engines.fair_cover import CoverageObjective, exchange_refine
from benchmark.engines.quotient_cover import reduce_groups
from benchmark.engines.service_cover import greedy_cover
from benchmark.response_aware_belief import ResponseAwareAnchorModel


def fit_public_cover(rn, reference_model, reply, k):
    if isinstance(k, bool) or int(k) != k or k < 1:
        raise ValueError('Positive integer query count required')
    model = ResponseAwareAnchorModel(reference_model, reply)
    viable_nodes = max(nx.strongly_connected_components(rn.graph), key=len)
    viable = np.array([i for i, node in enumerate(rn.node_ids) if node in viable_nodes])
    weights = np.asarray(model.prior @ model.poi_weights).ravel()
    objective = CoverageObjective(reply.signatures, reply.access, weights)
    center = np.average(model.xy, axis=0, weights=model.prior)
    def ties(j, ids):
        return np.zeros(len(ids)), np.linalg.norm(rn.xy[ids]-center, axis=1)
    _, profiles = np.unique(reply.signatures.reshape(len(reply.signatures), -1),
                            axis=0, return_inverse=True)
    reduced = reduce_groups([viable], profiles[reply.access], ties)[0]
    groups = [reduced]*int(k)
    greedy, _ = greedy_cover(groups, objective.marginal, ties)
    selected, history = exchange_refine(groups, greedy, objective, ties, 3)
    return {'k': int(k), 'states': selected, 'coordinates': [rn.latlon(i) for i in selected],
            'public_expected_reference_coverage': objective.value(selected),
            'objective_history': history, 'viable_states': len(viable),
            'service_equivalence_classes': len(reduced), 'reference_model_sha256': reference_model.sha256,
            'reply_context_sha256': reply.sha256, 'private_reads': 0, 'coordinate_privacy_bound': 0.}
