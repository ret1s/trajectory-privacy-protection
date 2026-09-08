import numpy as np
import pytest

from benchmark.public_poi_context import PublicPoiContext
from core.demo_protocol import TrajectoryPoint
from benchmark.engines.service_cover import ServiceCoverLaneDummy
from evaluation.retrieval_frontier import evaluate_retrieval, pareto_ids
from evaluation.research_protocol import utility_metrics
from evaluation.lane_travel import LanePoiService
from tests.test_belief_lane import fixture


def test_reference_fixed_depth_monotonicity_and_original_parity():
    rn,context,belief=fixture()
    service=LanePoiService(rn,list(context.pois),k=5)
    deeper=PublicPoiContext(LanePoiService(rn,list(context.pois),k=10))
    points=tuple(TrajectoryPoint(i*5.,0.,.0001+i*.0001) for i in range(4))
    public=ServiceCoverLaneDummy(rn,belief_model=belief).protect_run(points).to_attacker_dict()
    truth=[(p.lat,p.lon) for p in points]
    values=evaluate_retrieval(service,deeper,public,truth)
    old=utility_metrics(service,public,truth)
    assert values['5']['poi_recall_at_5'] == pytest.approx(old['poi_recall_at_5'])
    assert values['5']['poi_complete_rate'] == pytest.approx(old['poi_complete_rate'])
    for a,b in zip(values['5']['poi_rows'],values['10']['poi_rows']):
        assert a['reference']==b['reference']
        if a['recall'] is not None: assert b['recall'] >= a['recall']
        assert b['reply_items'] >= a['reply_items']
        assert b['response_id_json_bytes'] >= a['response_id_json_bytes']
    assert values['5']['requests_per_event']==3*len(service.categories)
    with pytest.raises(ValueError): evaluate_retrieval(service,deeper,public,truth,(4,))


def test_pareto_direction_ties_and_cost():
    rows=[dict(id='a',recall=.9,hit=.1,cost=100),dict(id='b',recall=.8,hit=.1,cost=100),
          dict(id='c',recall=.95,hit=.1,cost=200),dict(id='a_tie',recall=.9,hit=.1,cost=100)]
    assert pareto_ids(rows,[('recall',-1),('hit',1),('cost',1)])==['a','c','a_tie']
    with pytest.raises(ValueError): pareto_ids(rows,[('recall',0)])


def test_validation_selection_does_not_silently_promote_infeasible_method():
    from experiments.run_coverage_frontier import METHODS,select
    summaries=[dict(method=m,case_id=f'S1.{c}',mae_by_attack={'a':100,'b':200},
        hit_by_attack={'a':.1,'b':.2},envelope_hit100=.2,
        utility={'5':{'poi_recall_at_5':.89},'10':{'poi_recall_at_5':.91}})
        for m in METHODS for c in ('A','B','C')]
    result=select(summaries)
    assert result['method_selection_by_depth']['5']['chosen'] is None
    assert not result['method_selection_by_depth']['5']['utility_feasible']
    assert result['method_selection_by_depth']['10']['utility_feasible']
    assert all(v=={'mae':'a','hit':'b'} for v in result['attackers'].values())
