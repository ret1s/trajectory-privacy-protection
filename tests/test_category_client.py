import numpy as np
import pytest
from benchmark.category_client import PublicCategoryClient


PLAN = {'queries': [{'category_index': 0, 'category': 'cafe', 'coordinate': [0., 1.]},
                    {'category_index': 1, 'category': 'clinic', 'coordinate': [0., 2.]}]}


def test_epoch_expiry_and_public_refresh_schedule():
    client = PublicCategoryClient(PLAN, 4, refresh_once_per_epoch=True)
    def server(q):
        return [q['category_index'] + 2*(q['epoch'] % 2)]
    a, b, c = [client.step(t, server) for t in (0, 59, 60)]
    assert [len(r['requests']) for r in (a, b, c)] == [2, 0, 2]
    assert np.flatnonzero(b['known']).tolist() == [0, 1]
    assert np.flatnonzero(c['known']).tolist() == [2, 3]
    b['known'][:] = False
    assert np.flatnonzero(client.step(61, server)['known']).tolist() == [2, 3]
    with pytest.raises(ValueError):
        client.step(60, server)


def test_different_private_reranking_does_not_change_wire_transcript():
    clients = [PublicCategoryClient(PLAN, 4) for _ in range(2)]
    histories = [[], []]
    for t in (0, 20, 60):
        for i, client in enumerate(clients):
            r = client.step(t, lambda q: [q['category_index']])
            # Distinct private users select different returned POIs locally.
            local_choice = np.flatnonzero(r['known'])[i]
            assert local_choice == i
            histories[i].append((r['requests'], r['replies']))
    assert histories[0] == histories[1]


def test_external_mutation_cannot_change_the_frozen_plan():
    from copy import deepcopy
    plan = deepcopy(PLAN)
    client = PublicCategoryClient(plan, 4)
    plan['queries'][0]['coordinate'][0] = 99.
    r = client.step(0, lambda q: [])
    assert r['requests'][0]['coordinate'] == (0., 1.)
