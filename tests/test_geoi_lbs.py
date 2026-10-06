from types import SimpleNamespace

from benchmark.geoi_lbs import GeoILbsClient
from benchmark.query_purpose import QueryPurpose, QuerySpec, PurposeIndependentCoverClient
from tests.test_query_purpose import fixture


def test_answering_different_purposes_never_calls_network_or_engine():
    ranking = fixture()
    coordinates = [(39.9,116.0001)]
    protected = []
    sent = []
    def protect(lat,lon,t):
        protected.append((lat,lon,t))
        return coordinates
    engine = SimpleNamespace(rn=ranking.rn,k=1,protect_step=protect)
    client = GeoILbsClient(engine,PurposeIndependentCoverClient(['cafe'],3,k=1),ranking)
    client.protect_and_fetch(0,39.9,116.,lambda request:sent.append(request) or [[0,1]])
    client.answer(QuerySpec(QueryPurpose.NEAREST,'cafe'),39.9,116.)
    client.answer(QuerySpec(QueryPurpose.FASTEST,'cafe'),39.9,116.)
    client.answer(QuerySpec(QueryPurpose.MIN_DETOUR,'cafe',destination_state=3),39.9,116.)
    assert len(sent) == len(protected) == 1
    assert set(sent[0]) == {'schema','timestamp_s','coordinate','categories','response_l','epoch'}
