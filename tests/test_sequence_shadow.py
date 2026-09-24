from copy import deepcopy
import numpy as np
import pytest
from evaluation.sequence_shadow import current_features, endpoint_features, EmpiricalShadow


class Projection:
    def point_xy(self, lat, lon):
        return lon*1000., lat*1000.


def public():
    return {'events': [{'timestamp_s': i*20.+812., 'event_id': f'hidden_original_{i}',
                        'candidates': [{'candidate_id': str(j), 'lat': i*.01, 'lon': j*.02}
                                       for j in range(5)]} for i in range(4)]}


def test_features_ignore_private_metadata_and_absolute_clock():
    a = public(); b = deepcopy(a)
    b['evaluation_truth'] = {'endpoint_xy': [-1e9, 1e9]}
    for e in b['events']:
        e['timestamp_s'] += 1e6
        e['event_id'] = 'unrelated'
        e['target_index'] = 999999
    for fun in (current_features, lambda p, r: endpoint_features(p, r, 'S9'),
                lambda p, r: endpoint_features(p, r, 'S10')):
        aa, bb = fun(a, Projection()), fun(b, Projection())
        assert all(np.array_equal(x, y) for x, y in zip(aa, bb))


def test_single_snapshot_has_no_fabricated_hidden_history():
    p = public(); p['events'] = p['events'][-1:]
    x, center = current_features(p, Projection())
    assert x.shape == (1, 32) and center.shape == (1, 2)
    assert np.array_equal(x[:, :10], x[:, 10:20])
    assert np.array_equal(x[:, :10], x[:, 20:30])
    assert np.array_equal(x[:, -2:], [[0., 0.]])
    with pytest.raises(ValueError):
        endpoint_features(p, Projection(), 'S1')


def test_shadow_queries_do_not_refit_to_target_or_query_batch():
    rng = np.random.default_rng(17)
    x = rng.normal(size=(80, 5)); centers = rng.normal(size=(80, 2))*100.
    y = centers+np.c_[x[:, 0], x[:, 1]]*20.
    model = EmpiricalShadow(x, y, centers)
    query, c = x[30:35]+.001, centers[30:35]
    together = model.predict(query, c, decisions=False)
    alone = model.predict(query[:1], c[:1], decisions=False)
    assert all(np.allclose(together[a][:1], alone[a]) for a in alone)
    assert np.isfinite(np.concatenate(list(together.values()))).all()
