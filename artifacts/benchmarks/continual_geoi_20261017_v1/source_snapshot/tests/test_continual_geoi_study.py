import numpy as np

from benchmark.query_purpose import QueryPurpose
from benchmark.query_segments import PublicSegmentLibrary
from evaluation.lane_travel import SparseTravel
from experiments import continual_geoi_study_20261017 as study
from tests.test_lane_comparison import road


def test_stress_continuation_preserves_native_prefix_and_directed_paths():
    rn=road()
    library=PublicSegmentLibrary(rn,[(0,5,10,15,20),(10,15,20,25,29)])
    native=[dict(time_s=t,lat=rn.latlon(5)[0],lon=rn.latlon(5)[1]) for t in range(601)]
    a=study.continuation(native,'public-test-session',rn,library)
    assert a==study.continuation(native,'public-test-session',rn,library)
    assert [p['t'] for p in a]==list(range(0,1801,20))
    assert all(p['source']=='reused_native_GPS' and p['gps']==list(rn.latlon(5)) for p in a[:31])
    assert all(p['source']=='synthetic_directed_road_continuation' for p in a[31:])
    travel=SparseTravel(rn)
    assert all(b['state'] in travel.reachable(a['state'],20.) for a,b in zip(a[30:],a[31:]))


def test_paired_readout_uses_all_purposes_families_and_declared_windows():
    families=[]
    for i in range(2):
        utility={}
        for method in study.METHODS:
            events=[]
            for t in range(0,1801,20):
                value=.9 if method!='continual_g1' or t<=600 else .8
                score={p.value:dict(recall5=value) for p in QueryPurpose}
                events.append(dict(t=t,current=score,known=score,session=score))
            utility[method]=events
        families.append(dict(family_id=str(i),split='test',trips=[dict(utility=utility)]))
    results=study.utility_readout(families)
    assert len(results)==len(study.WINDOWS)*3*len(study.METHODS)
    first=next(r for r in results if (r['window'],r['cache'],r['method'])==('native_10min','session','continual_g1'))
    assert first['metrics']['macro']['paired_delta']==0
    late=next(r for r in results if (r['window'],r['cache'],r['method'])==('synthetic_after_12min','session','continual_g1'))
    assert np.isclose(late['metrics']['macro']['paired_delta'],-.1)
    assert not late['utility_gate']
    assert all(v['paired_families']==2 for v in late['metrics'].values())
