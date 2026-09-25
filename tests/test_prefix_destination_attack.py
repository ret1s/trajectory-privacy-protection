import copy

import numpy as np
import pytest
from scipy.spatial import cKDTree

from evaluation.prefix_destination_attack import PrefixDestinationAttack, prefix_features


class PublicMap:
    xy = np.array([[x, y] for x in range(-1000, 3001, 100) for y in range(-1000, 3001, 100)])
    tree = cKDTree(xy)

    @staticmethod
    def point_xy(a, b):
        return np.array([a, b], float)


def events(points, offset=0):
    return [{'timestamp_s': offset+20*i, 'coordinates':[list(p)]} for i,p in enumerate(points)]


def test_relative_frame_is_translation_and_rotation_invariant():
    points = np.array([[0.,0.],[40.,10.],[80.,50.],[100.,90.]])
    rotation = np.array([[0.,-1.],[1.,0.]])
    a = prefix_features(events(points), PublicMap())
    b = prefix_features(events(points @ rotation.T + [300.,700.]), PublicMap())
    np.testing.assert_allclose(a[1], b[1], atol=1e-12)
    np.testing.assert_allclose(rotation @ a[3], b[3], atol=1e-12)


def test_hidden_metadata_and_absolute_clock_do_not_enter_prediction():
    examples = [events([[0, i*100], [100, i*100], [200, i*100]]) for i in range(6)]
    model = PrefixDestinationAttack(examples, [[700, i*100+300] for i in range(6)], PublicMap())
    clean = examples[0]
    poisoned = copy.deepcopy(clean)
    for e in poisoned:
        e.update(timestamp_s=e['timestamp_s']+99999, destination=[-9999,-9999],
                 family_id='heldout-secret', remaining_time_s=321, future_route=[7,9])
    a, b = model.predict(clean), model.predict(poisoned)
    assert a.keys() == b.keys()
    for key in a:
        np.testing.assert_array_equal(a[key], b[key])


def test_stationary_single_event_is_finite():
    features = prefix_features(events([[200,300]]), PublicMap())
    assert all(np.all(np.isfinite(f)) for f in features)


def test_raw_only_contract_rejects_multiple_candidates():
    with pytest.raises(ValueError, match='single-coordinate'):
        prefix_features([{'timestamp_s':0,'coordinates':[[0,0],[1,1]]}], PublicMap())
