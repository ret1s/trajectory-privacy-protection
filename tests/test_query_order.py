from collections import Counter
from itertools import permutations
import json

import numpy as np

from benchmark.query_order import PrivateOrderCoverClient
from benchmark.query_purpose import PurposeIndependentCoverClient


def test_order_shuffle_preserves_Q_contents_reply_union_and_byte_cost():
    coordinates = [(39.9,116.001),(39.9,116.002),(39.9,116.003)]
    original = list(coordinates)
    def server(request):
        return [[int(round((request['coordinate'][1]-116.)*1000))-1]]
    plain = PurposeIndependentCoverClient(['cafe'],3,k=3)
    private = PrivateOrderCoverClient(['cafe'],3,k=3,rng=np.random.default_rng(4))
    a = plain.step(0,coordinates,server)
    b = private.step(0,coordinates,server)
    assert coordinates == original
    # Identity permutations are valid; do not force a changed order each time.
    assert sorted(a['requests'],key=lambda q:q['coordinate']) == sorted(b['requests'],key=lambda q:q['coordinate'])
    assert np.array_equal(a['known'],b['known'])
    assert len(json.dumps([a['requests'],a['replies']])) == len(json.dumps([b['requests'],b['replies']]))
    assert all(set(q)=={'schema','timestamp_s','coordinate','categories','response_l','epoch'} for q in b['requests'])
    orders = [tuple(tuple(q['coordinate']) for q in b['requests'])]
    for t in range(1,7):
        result = private.step(t,coordinates,server)
        orders.append(tuple(tuple(q['coordinate']) for q in result['requests']))
    assert len(set(orders)) > 1


def test_uniform_permutation_distribution_does_not_reveal_internal_slot_labels():
    # Exhaustive finite support: the published-order law is identical for every
    # internal permutation. This does not prevent matching by coordinate paths.
    coordinates = ((39.9,116.001),(39.9,116.002),(39.9,116.003))
    distributions=[]
    for internal in permutations(coordinates):
        distributions.append(Counter(tuple(internal[i] for i in p) for p in permutations(range(3))))
    assert all(law == distributions[0] for law in distributions)
    assert len(distributions[0]) == 6 and set(distributions[0].values()) == {1}


def test_private_order_changes_no_request_clock_or_per_epoch_cache_rule():
    client = PrivateOrderCoverClient(['cafe'],3,k=2,rng=np.random.default_rng(5))
    coordinates=[(39.9,116.001),(39.9,116.002)]
    a=client.step(0,coordinates,lambda request:[[0]])
    b=client.step(20,coordinates,lambda request:[[1]])
    c=client.step(60,coordinates,lambda request:[[2]])
    assert [len(x['requests']) for x in (a,b,c)] == [2,2,2]
    assert np.flatnonzero(b['known']).tolist() == [0,1]
    assert np.flatnonzero(c['known']).tolist() == [2]
