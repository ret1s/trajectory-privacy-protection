"""End-to-end opt-in adapter; GPS and private purpose are LOCAL-only inputs.

The upstream Geo-I engine supplies belief. PersistentQueryBudget owns sampling
and segment replay; this adapter emits only the established K5/L30 public wire
schema. Geo-I resume/accounting remains the upstream engine's responsibility.
"""
from benchmark.query_purpose import PurposeIndependentCoverClient
from benchmark.public_segment_scores import local_sorted_pois


class SegmentCoverClient:
    def __init__(self, rn, policy, ledger, categories, poi_count):
        self.rn, self.policy, self.ledger = rn, policy, ledger
        self.client = PurposeIndependentCoverClient(categories, poi_count, k=5, response_l=30)

    def step(self, timestamp_s, protected_belief, server):
        frames = 3 if self.ledger.policy['joint'] else 1
        def select(previous, epsilon, rng):
            return self.policy.select(previous, epsilon, rng, belief=protected_belief, frames=frames)
        states, certificate = self.ledger.frame(timestamp_s, select)
        wire = self.client.step(timestamp_s, tuple(self.rn.latlon(s) for s in states), server)
        # Certificate is private/local and is never part of server(request).
        return wire, certificate

    @staticmethod
    def local_answer(ranking, true_state, received_mask, private_query):
        return local_sorted_pois(ranking, true_state, received_mask, private_query)
