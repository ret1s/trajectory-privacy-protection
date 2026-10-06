import copy
import gzip
import json
import numpy as np
import pytest
from pyproj import Transformer
from evaluation.candidate_future_attack import (candidate_features,geometric_bank,
    CandidateFutureAttack,forecast_metrics,normalize)

PROJECT=Transformer.from_crs(4326,32650,always_xy=True)
BACK=Transformer.from_crs(32650,4326,always_xy=True)
X,Y=PROJECT.transform(116.32,40.)


def context():
    return {'choices':[{'edge_id':f'new_edge_{i}','via_lane_ids':[f':turn{i}_0'],
        'via_shape_xy':[[X-10,Y],[X,Y],[X+30,Y+sign*30]],
        'outgoing_shape_xy':[[X+30,Y+sign*30],[X+100,Y+sign*100]],
        'destination_xy':[X+600,Y+sign*600],'destination_edge_id':f'public_destination_{i}'}
        for i,sign in enumerate((1,-1))]}


def public(points,times=None):
    times=times or [float(i*20) for i in range(len(points))]
    events=[]
    for i,((dx,dy),t) in enumerate(zip(points,times)):
        lon,lat=BACK.transform(X+dx,Y+dy)
        events.append({'event_id':f'e{i}','timestamp_s':t,
                       'candidates':[{'candidate_id':'q0','lat':lat,'lon':lon}]})
    return {'events':events}


def histories():
    return [public([(-40,0),(600,600 if i!=2 else -600)],times=[0.,600.]) for i in range(6)]


def rows():
    result=[]
    for family in range(1,5):
        for target in range(2):
            result.append({'family_id':f'native-{family:02d}','choice_index':target,
                'query':public([(-30,0),(-10,0),(15,15 if target==0 else -15)]),
                'histories':histories(),'public_context':context(),
                'destination_role':'routine' if target==0 else 'rare',
                'destination_xy':context()['choices'][target]['destination_xy']})
    return result


def test_candidate_geometry_decodes_visible_turn_without_training_edge_ids():
    for target in range(2):
        query=rows()[target]['query']
        result=geometric_bank(query,context())
        assert np.argmax(result['curve_mean_0.1'])==target
        assert all(np.isclose(p.sum(),1.) for p in result.values())


def test_literal_road_identifiers_are_not_numeric_features():
    query=rows()[0]['query'];original=context();renamed=copy.deepcopy(original)
    for c in renamed['choices']:
        c['edge_id']='completely_unseen_'+c['edge_id']
        c['destination_edge_id']='a_new_identifier'
        c['via_lane_ids']=[':fresh_privatefree_public_turn']
    assert np.array_equal(candidate_features(query,original,histories()),
                          candidate_features(query,renamed,histories()))


def test_context_and_history_reject_secret_fields():
    query=rows()[0]['query']
    for private in ('routine_choice','choice_index','selected_route','target_gps'):
        bad=context();bad[private]=0
        with pytest.raises(ValueError):candidate_features(query,bad,histories())
        bad=context();bad['choices'][0][private]=0
        with pytest.raises(ValueError):candidate_features(query,bad,histories())
    bad=histories();bad[0]['events'][0]['candidates'][0]['is_real']=True
    with pytest.raises(ValueError):candidate_features(query,context(),bad)
    with pytest.raises(ValueError):candidate_features(query,context(),histories()[:5])


def test_candidate_order_changes_probability_order_without_fitting_new_classes():
    query=rows()[0]['query'];a=context();b={'choices':list(reversed(a['choices']))}
    assert np.allclose(candidate_features(query,a,histories())[::-1],candidate_features(query,b,histories()))
    model=CandidateFutureAttack(rows(),use_history=True)
    left,right=model.predict(query,a,histories()),model.predict(query,b,histories())
    assert all(np.allclose(left[n][::-1],right[n]) for n in left)


def test_shared_prefix_balanced_targets_are_intrinsically_chance():
    data=rows()[:2]
    for row in data:row['query']=public([(-40,0),(-20,0)])
    first=geometric_bank(data[0]['query'],data[0]['public_context'],data[0]['histories'])
    second=geometric_bank(data[1]['query'],data[1]['public_context'],data[1]['histories'])
    for name in first:
        assert np.array_equal(first[name],second[name])
        score=forecast_metrics(data,[first[name],second[name]])
        assert score['exact_candidate_edge_accuracy']==.5 and score['balanced_accuracy']==.5


def test_five_routine_history_trips_do_not_make_balanced_query_prior_five_sixths():
    data=rows()[:2]
    for row in data:row['query']=public([(-40,0),(-20,0)])
    p=geometric_bank(data[0]['query'],context(),histories())['history_prior']
    assert np.argmax(p)==0
    score=forecast_metrics(data,[p,p])
    assert score['routine_accuracy']==1. and score['rare_accuracy']==0.
    assert score['exact_candidate_edge_accuracy']==.5


def test_forecast_error_and_probability_metrics_have_correct_units():
    data=rows()[:2]
    score=forecast_metrics(data,[np.array([1.,0.]),np.array([0.,1.])])
    assert score['destination_mae_m']==0. and score['destination_hit100']==1.
    assert score['brier']==0. and score['log_loss']==0.
    with pytest.raises(ValueError):normalize([-1.,2.])
    with pytest.raises(ValueError):forecast_metrics(data,[[.2,.2],[.5,.5]])


def test_forecast_row_extraction_never_uses_future_query_coordinates(tmp_path,monkeypatch):
    import experiments.future_sumo_eval as runner
    monkeypatch.setattr(runner,'OUT',tmp_path)
    query=public([(-40,0),(-20,0),(20,20)],times=[0.,20.,40.])
    group={'public_clocks':{'shared_fork_t':20.,'turn_visible_t':40.},
           'public_context':context(),'streams':{'raw':histories()+[query,copy.deepcopy(query)]}}
    sessions=[{} for _ in range(6)]+[{'choice_index':i,'destination_role':'routine' if i==0 else 'rare',
        'destination_xy':context()['choices'][i]['destination_xy'],
        'next_edge_labels':{'shared_fork_t':f'new_edge_{i}','turn_visible_t':f'new_edge_{i}'}} for i in range(2)]
    truth={'family_id':'native-01','split':'test','evaluator_sessions':sessions}
    def write():
        (tmp_path/'public_transcripts.json.gz').write_bytes(gzip.compress(json.dumps({'groups':[group]}).encode()))
        (tmp_path/'private_accounting.json.gz').write_bytes(gzip.compress(json.dumps({'rows':[truth]}).encode()))
    write();before=runner.forecast_rows('raw','shared_fork')
    group['streams']['raw'][6]['events'][-1]['candidates'][0]['lat']=89.
    write();after=runner.forecast_rows('raw','shared_fork')
    assert before[0]['query']==after[0]['query'] and len(after[0]['query']['events'])==2
