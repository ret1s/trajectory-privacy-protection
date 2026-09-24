from tests.test_belief_lane import fixture
from evaluation.static_cache import StaticPublicPoiCache
from evaluation.lane_travel import LanePoiService

def test_local_public_context_answers_exact_static_service_without_remote_events():
    rn,context,_=fixture()
    cache=StaticPublicPoiCache(context)
    reference=LanePoiService(rn,list(context.pois),k=context.k)
    for state in range(len(rn)):
        for category in context.categories:
            assert cache.query(*rn.latlon(state),category)==reference.query(rn.latlon(state),category)
    assert cache.remote_events==()
    assert cache.index_bytes>0
