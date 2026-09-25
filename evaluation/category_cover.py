"""Public, category-specific query cover for a bounded point POI API.

This planner consumes only road geometry and the static POI catalogue. A plan
is fixed before opening evaluation journeys or live availability. Query count
means category-coordinate pairs, not distinct coordinates or HTTP requests.
"""
import numpy as np


def greedy_category_cover(profiles, targets, budget=None):
    """Greedy macro-category set cover, with stable input-order tie breaking.

    profiles[c] is a list of dictionaries with `state` and static top-L `ids`.
    targets[c] includes ALL publicly reachable POIs, not observed user top-k.
    No approximation-optimality or minimum-cardinality claim is made.
    """
    if budget is not None and (isinstance(budget, bool) or int(budget) != budget or budget < 1):
        raise ValueError('Positive integer category-query budget required')
    if len(profiles) != len(targets):
        raise ValueError('One profile collection per target category required')
    remaining = [set(t) for t in targets]
    denominators = [max(1, len(t)) for t in targets]
    queries = []
    while any(remaining) and (budget is None or len(queries) < budget):
        best, best_gain = None, 0.
        for category, candidates in enumerate(profiles):
            for candidate in candidates:
                gain = len(remaining[category].intersection(candidate['ids'])) / denominators[category]
                if gain > best_gain + 1e-15:
                    best = (category, candidate)
                    best_gain = gain
        if best is None:
            break
        category, candidate = best
        new = remaining[category].intersection(candidate['ids'])
        remaining[category].difference_update(new)
        queries.append({'category_index': category, 'state': int(candidate['state']),
                        'static_ids': list(candidate['ids']), 'new_ids': sorted(new)})
    return {'queries': queries, 'category_queries': len(queries),
            'uncovered_ids': [sorted(s) for s in remaining],
            'target_counts': [len(s) for s in targets],
            'full_catalogue_cover': not any(remaining), 'private_reads': 0}


def public_category_profiles(rn, ranking, candidate_states, response_l=10):
    """Use actual coordinate-to-road access, including coincident lane ties."""
    if isinstance(response_l, bool) or int(response_l) != response_l or response_l < 1:
        raise ValueError('Positive integer response depth required')
    access = sorted({int(rn.nearest(*rn.latlon(int(s)))[0]) for s in candidate_states})
    if not access:
        raise ValueError('Public candidate set must not be empty')
    profiles, targets = [], []
    for section in ranking.slices:
        # Reference clients may be anywhere in the public road catalogue.
        target = np.unique(ranking.rank[:, section])
        targets.append([int(i) for i in target if i >= 0])
        rows = ranking.rank[np.asarray(access), section][:, :int(response_l)]
        _, first = np.unique(rows, axis=0, return_index=True)
        profiles.append([{'state': access[int(j)],
                          'ids': [int(i) for i in rows[j] if i >= 0]}
                         for j in sorted(first.tolist())])
    return profiles, targets


def attach_coordinates(plan, rn, ranking, response_l=10):
    result = dict(plan)
    result['queries'] = [dict(q, category=ranking.categories[q['category_index']],
                              coordinate=list(rn.latlon(q['state']))) for q in plan['queries']]
    result['distinct_coordinates'] = len({tuple(q['coordinate']) for q in result['queries']})
    result['response_L'] = response_l
    return result


def query_category_plan(plan, server, epoch):
    """Client receives only requested categories, not other cached server data."""
    replies = []
    for query in plan['queries']:
        category = query['category_index']
        # The simulator computes all categories; serialize and reveal only one.
        ids = server.query(query['state'], epoch)[category]
        replies.append(list(ids))
    return replies


def reply_mask(replies, size):
    known = np.zeros(size, dtype=bool)
    for ids in replies:
        if any(i < 0 or i >= size for i in ids):
            raise ValueError('Unknown POI ID')
        known[ids] = True
    return known
