"""Static bulk/local application control; no protected engine or network calls.

The caller supplies public POI records and local query inputs. A full catalogue
is usable only under an explicit fixed-version, bulk-access service contract.
Raw position/destination are local utility inputs, never bulk request fields.
"""
import hashlib
import json
import math
from numbers import Integral

import numpy as np

from benchmark.query_purpose import QueryPurpose, QuerySpec


def compact_bytes(value):
    return json.dumps(value, separators=(',', ':'), ensure_ascii=False,
                      allow_nan=False).encode('utf-8')


def catalogue_payload(pois, *, epoch_id, public_start_s):
    """One coordinate-free public request and full coordinate-bearing response.

    Bytes are actual compact UTF-8 JSON lengths, not HTTP/TLS or timing. The
    public graph/map and POI access/ranking computation are separate inputs;
    their download/CPU/GNSS costs are not included in this payload estimate.
    """
    if (not isinstance(epoch_id, str) or not epoch_id or
            isinstance(public_start_s, bool) or not isinstance(public_start_s, (int, float)) or
            not math.isfinite(public_start_s) or public_start_s < 0):
        raise ValueError('Public nonempty epoch ID and finite start required')
    records = [{name: p[name] for name in ('id', 'category', 'lat', 'lon')} for p in pois]
    if not records or any(not isinstance(p['id'], str) or not p['id'] or
                          not isinstance(p['category'], str) or not p['category'] or
                          not np.isfinite([p['lat'], p['lon']]).all() or
                          not -90 <= p['lat'] <= 90 or not -180 <= p['lon'] <= 180
                          for p in records):
        raise ValueError('Finite public POI records required')
    if len({p['id'] for p in records}) != len(records):
        raise ValueError('Distinct public POI IDs required')
    records.sort(key=lambda p: p['id'])
    version = hashlib.sha256(compact_bytes(records)).hexdigest()
    request = {'schema': 'static_catalogue_prefetch_v1', 'catalogue_version': version,
               'epoch_id': epoch_id, 'timestamp_s': float(public_start_s)}
    response = {'catalogue_version': version, 'results': records}
    return {'catalogue_version': version, 'request': request, 'response': response,
            'request_bytes': len(compact_bytes(request)),
            'response_bytes': len(compact_bytes(response)), 'poi_count': len(records)}


def _mask(local, values):
    ids = tuple(values)
    if any(isinstance(i, bool) or not isinstance(i, Integral) or not 0 <= i < local.n for i in ids):
        raise ValueError('Known integer POI candidate IDs required')
    result = np.zeros(local.n, dtype=bool)
    result[list(ids)] = True
    return result


def evaluate_local_purposes(local, state, candidate_ids, *, destination_state,
                            radius_m=1000., k=5):
    """Same four-purpose full-catalogue reference for every candidate mask.

    Every answer uses exact existing directed-distance/travel-time/detour
    definitions with lexical POI-ID ties. Empty references remain N/A.
    destination_state is private LOCAL input; this function emits no request.
    Returned indices refer to local.pois and duplicate records count once.
    """
    if not candidate_ids or any(not isinstance(name, str) or not name for name in candidate_ids):
        raise ValueError('Named candidate pools required')
    masks = {name: _mask(local, ids) for name, ids in candidate_ids.items()}
    rows = []
    for purpose in QueryPurpose:
        for category in local.categories:
            query = QuerySpec(purpose, category, k=k,
                radius_m=radius_m if purpose == QueryPurpose.WITHIN_RADIUS else None,
                destination_state=destination_state if purpose == QueryPurpose.MIN_DETOUR else None)
            scores = local.scores(state, query)
            eligible = [i for i, p in enumerate(local.pois) if p['category'] == category
                        and math.isfinite(scores[i])]
            eligible.sort(key=lambda i: (float(scores[i]), local.pois[i]['id']))
            reference = eligible[:query.k]
            answers = {name: [i for i in eligible if mask[i]][:query.k] for name, mask in masks.items()}
            rows.append({'purpose': purpose.value, 'category': category, 'reference': reference,
                'returned': answers, 'recall': {name: len(set(reference) & set(answer))/len(reference)
                    if reference else None for name, answer in answers.items()}})
    return rows
