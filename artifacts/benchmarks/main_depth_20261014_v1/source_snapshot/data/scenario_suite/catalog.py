"""Declared cases, including gates not yet supported by the generator.

These are experimental operationalizations, not new attacks or coverage claims.
Keep unavailable cases in the denominator rather than quietly dropping them.
"""

CASES = [
    ("S1.A", "branching_location", "Current location on an edge with >=2 legal passenger successors", "niu2014dls"),
    ("S1.B", "single_exit_location", "Current location on an edge with exactly one legal successor", "niu2014dls"),
    ("S1.C", "rare_poi_context", "Location with a rare POI context; requires a pinned POI rarity model", "liu2026semantic"),
    ("S2.A", "short_scheduled_stop", "Actual scheduled stop lasting >=20 s; sample every 5 s", "chatzikokolakis2014predictive"),
    ("S2.B", "long_scheduled_stop", "Actual scheduled stop lasting >=120 s; sample every 20 s", "chatzikokolakis2014predictive"),
    ("S2.C", "return_to_stop", "Two realized scheduled stops on the same lane separated by movement", "shaham2021viterbi"),
    ("S3.A", "moving_sequence", ">=60 s of moving FCD, >=3 distinct external edges; 20 s queries", "shaham2021viterbi"),
    ("S3.B", "forced_corridor", "Moving sequence with >=50% sampled edges having one legal successor", "yadav2024transprotect"),
    ("S3.C", "sparse_queries", "Same realized moving sequence sampled every 60 s", "shaham2021viterbi"),
    ("S4.A", "same_person_same_device", "Separate, non-overlapping sessions by the same person and device", "demontjoye2013unique"),
    ("S4.B", "same_person_new_device", "Same person, different device; person linkage and device linkage differ", "beresford2004mixzones"),
    ("S4.C", "different_person_shared_device", "Different people reuse a device in non-overlapping sessions", "beresford2004mixzones"),
    ("S5.A", "next_turn_at_fork", "Prefix ends before a transition with >=2 legal successors; label next external edge", "ziebart2008maxent"),
    ("S5.B", "forced_next_turn", "Exactly one legal successor; structural lower-uncertainty control", "ziebart2008maxent"),
    ("S5.C", "shared_prefix_fork", "Two planned routes share a prefix then diverge; realized choices checked", "ziebart2008maxent"),
    ("S6.A", "shared_prefix_destinations", "Different completed destinations with the same planned prefix", "ziebart2008maxent"),
    ("S6.B", "nearby_destinations", "Completed alternative destinations separated by <=500 m", "ziebart2008maxent"),
    ("S6.C", "routine_vs_rare_destination", "Requires a multi-day activity schedule with specified destination frequencies", "ziebart2008maxent"),
    ("S7.A", "plaintext_intent_control", "One explicit sensitive query; location-only protection cannot conceal plaintext", "wu2021fakequery"),
    ("S7.B", "constant_query_cover_control", "Six-category reference cover, with true category evaluator-only", "wu2021fakequery"),
    ("S7.C", "sequence_intent", "Three synthetic intents share a first query but have different continuations", "wu2021fakequery"),
    ("S8.A", "partial_companionship", "Declared companion pair shares a route prefix then separates; >=10 s within 100 m", "olteanu2017interdependent"),
    ("S8.B", "full_route_companionship", "Declared companion pair uses the same route; >=80% overlapping timestamps within 100 m", "olteanu2017interdependent"),
    ("S8.C", "incidental_proximity", "No declared companionship but >=10 s within 100 m: a hard negative", "olteanu2017interdependent"),
    ("S9.A", "single_exit_origin", "First recorded edge has one legal exit; first 60 s withheld", "dhondt2022endpoint"),
    ("S9.B", "merging_origins", "Different origins merge onto a common suffix; disclose only from merge", "dhondt2022endpoint"),
    ("S9.C", "repeated_origin", "Two completed trips from the same simulated origin; mask both first 60 s", "dhondt2022endpoint"),
    ("S10.A", "single_access_endpoint", "Final recorded edge has one legal incoming edge; last 60 s withheld", "dhondt2022endpoint"),
    ("S10.B", "diverging_endpoints", "Common route prefix leads to different endpoints; disclose before divergence", "dhondt2022endpoint"),
    ("S10.C", "repeated_endpoint", "Two completed trips to the same simulated endpoint; mask both last 60 s", "dhondt2022endpoint"),
]

TARGETS = {"S1": "current_location", "S2": "stop_location", "S3": "past_trajectory",
           "S4": "person_and_device_linkage", "S5": "next_external_edge", "S6": "destination",
           "S7": "query_intent", "S8": "companionship_and_location", "S9": "origin", "S10": "endpoint"}


def catalogue():
    return [{"case_id": key, "scenario": key.split('.')[0], "name": name,
             "eligibility": gate, "literature_key": ref, "target": TARGETS[key.split('.')[0]],
             "attack_evaluated": False, "protection_evaluated": False}
            for key, name, gate, ref in CASES]
