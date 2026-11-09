import numpy as np
from evaluation.correlated_channel_fusion import fusion_estimates
from evaluation.remaining_scenarios import xy


def tape(points):return [dict(timestamp_s=0.,coordinates=points)]


def test_shared_real_member_is_recoverable_without_private_member_labels():
    real=[40.,116.3]
    a=tape([real,[40.01,116.32],[39.98,116.29]])
    b=tape([[39.97,116.34],real,[40.02,116.3]])
    assert np.allclose(fusion_estimates(a,b,0)['fusion_nearest_pair'],xy([real])[0])


def test_singleton_fusion_is_average_and_set_order_invariant():
    a=tape([[40.,116.3],[40.01,116.31]])
    b=tape([[40.001,116.301],[40.02,116.32]])
    c=fusion_estimates(a,b,0)
    for e in a+b:e['coordinates'].reverse()
    d=fusion_estimates(a,b,0)
    assert all(np.allclose(c[k],d[k]) for k in c)
    assert np.allclose(c['fusion_centroid_0.5'],np.mean(xy(a[0]['coordinates']+b[0]['coordinates']),axis=0))


def test_fusion_does_not_read_future_points():
    a=tape([[40.,116.3]]);b=tape([[40.01,116.31]])
    base=fusion_estimates(a,b,0)
    a.append(dict(timestamp_s=20.,coordinates=[['future garbage']]))
    assert all(np.array_equal(v,fusion_estimates(a,b,0)[k]) for k,v in base.items())
