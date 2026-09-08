"""Fixed-reference retrieval depth: service utility AND actual reply payload.

The reference/client answer stays top-5. Only the public per-query reply depth
changes. All categories are queried; no hidden category is uploaded or chosen.
"""
import json

import numpy as np


def evaluate_retrieval(service, context, public, truth, depths=(5, 10), client_k=5):
    if (context.rn is not service.rn or len(truth) != len(public['events'])
            or not len(truth) or int(client_k) != client_k or client_k < 1
            or not depths or any(int(d) != d or d < client_k or d > context.k for d in depths)
            or len(set(depths)) != len(depths)):
        raise ValueError('Aligned queries, fixed client k and available unique reply depths required')
    by_id = [p['id'] for p in context.pois]
    results = {str(d): [] for d in depths}
    for event, point in zip(public['events'], truth):
        actual_state, _ = service.rn.nearest(*point)
        distances = service.distances(point)  # local ranking; never attacker input
        states = [service.rn.nearest(c['lat'], c['lon'])[0] for c in event['candidates']]
        if not states:
            raise ValueError('At least one query candidate required')
        for category_index, category in enumerate(context.categories):
            ref_ids = context.signatures[actual_state, category_index, :client_k]
            reference = [by_id[i] for i in ref_ids if i >= 0]
            for depth in depths:
                replies = [[by_id[i] for i in context.signatures[s, category_index, :depth] if i >= 0]
                           for s in states]
                union = set().union(*map(set, replies))
                returned = sorted(union & distances.keys(), key=lambda p: (distances[p], p))[:client_k]
                complete = len(returned) == len(reference) if reference else None
                extra = (max(0., float(np.mean([distances[p] for p in returned]) -
                                      np.mean([distances[p] for p in reference]))) if complete else None)
                results[str(depth)].append({'event_id': event['event_id'], 'category': category,
                    'reference': reference, 'replies': replies, 'returned': returned,
                    'recall': len(set(reference) & set(returned))/len(reference) if reference else None,
                    'complete': complete, 'extra_distance_m': extra,
                    'request_count': len(states), 'reply_items': sum(map(len, replies)),
                    'response_id_json_bytes': len(json.dumps(replies, separators=(',', ':')).encode())})
    output = {}
    for depth, rows in results.items():
        eligible = [r for r in rows if r['recall'] is not None]
        extra = [r['extra_distance_m'] for r in eligible if r['extra_distance_m'] is not None]
        output[depth] = {'poi_rows': rows,
            'poi_recall_at_5': float(np.mean([r['recall'] for r in eligible])) if eligible else None,
            'poi_complete_rate': float(np.mean([r['complete'] for r in eligible])) if eligible else None,
            'poi_extra_distance_m': float(np.mean(extra)) if extra else None,
            'evaluable_queries': len(eligible), 'empty_references': len(rows) - len(eligible),
            'requests_per_event': sum(r['request_count'] for r in rows)/len(truth),
            'reply_items_per_event': sum(r['reply_items'] for r in rows)/len(truth),
            'response_id_bytes_per_event': sum(r['response_id_json_bytes'] for r in rows)/len(truth)}
    return output


def pareto_ids(rows, axes):
    """Exploratory finite-grid non-dominance; axes=(field, +1 minimize/-1 max).

    Decimal rounding avoids declaring a 1-ulp difference a scientific win.
    Missing metrics are errors, not silently favorable values.
    """
    if not axes or len({r['id'] for r in rows}) != len(rows):
        raise ValueError('Distinct IDs and explicit metric directions required')
    if any(direction not in (-1, 1) for _, direction in axes):
        raise ValueError('Metric directions must be -1 or +1')
    scores = np.array([[round(r[key], 12)*direction for key, direction in axes] for r in rows])
    if not np.isfinite(scores).all():
        raise ValueError('Finite comparable metrics required')
    return [r['id'] for i, r in enumerate(rows)
            if not any(np.all(s <= scores[i]) and np.any(s < scores[i]) for s in scores)]
