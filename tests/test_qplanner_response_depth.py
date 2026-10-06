"""Independent response-prefix, replay-boundary and fixed-gate contracts."""
from copy import deepcopy
import gzip
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from experiments import qplanner_response_depth_20261006 as depth


def fixture_objects():
    pois = tuple(dict(id=f'p{i:02d}',category='food',lat=float(i),lon=0.) for i in range(60))
    signatures = np.array([[list(range(20,40))+list(range(20))+list(range(40,60))], [list(range(60))]],dtype=np.int32)
    rn = SimpleNamespace(nearest=lambda lat,lon: (int(lat),0.))
    context = SimpleNamespace(rn=rn,pois=pois,categories=('food',),k=60,
        signatures=signatures,access=np.arange(2),sha256='public-test-context')
    # The independent true ranking is lexical order p00..p59. First20
    # protected replies contain none of the true top5; first30 contain all5.
    ranking = SimpleNamespace(categories=('food',), scores=lambda state,spec: np.arange(60,dtype=float))
    evaluators = {L:depth.common.UtilityEvaluator(rn,depth.PrefixContext(context,L),ranking) for L in depth.DEPTHS}
    return context,rn,evaluators


def frozen_fixture():
    context,rn,evaluators = fixture_objects(); streams = {'raw':[],depth.METHOD:[]}
    utilities,wire,sessions,specs = [],[],[],[]
    for slot in range(8):
        dep = slot*1500; specs.append({'depart_s':dep}); raw=[]; public=[]
        for j,t in enumerate(range(0,601,20)):
            fields=dict(family_id='f1',split='selection',draw=1,slot=slot,method=depth.METHOD,event_id=f'e{j:04d}',t=t)
            raw.append(dict(event_id=fields['event_id'],timestamp_s=float(t),candidates=[dict(candidate_id='q0000',lat=1.,lon=0.)]))
            public.append(dict(event_id=fields['event_id'],timestamp_s=float(t),candidates=[dict(candidate_id=f'q{i:04d}',lat=0.,lon=0.) for i in range(5)]))
            for cache in ('current','static_epoch_cache'):
                scores={p:dict(recall5=0.,completion=1.,reference_category_count=1,all_category_count=1,
                    overlap_total=0,reference_poi_total=5) for p in depth.common.PURPOSES}
                utilities.append(dict(fields,cache=cache,purposes=scores,available_count=20))
            # Build the original application payload independently of replay.
            response=[{k:context.pois[i][k] for k in ('id','category','lat','lon')} for i in range(20,40)]
            request=dict(timestamp_s=dep+t,lat=0.,lon=0.,categories=['food'],L=20)
            wire.append(dict(fields,requests=5,request_bytes=5*len(json.dumps(request,separators=(',',':')).encode()),
                reply_bytes=5*len(json.dumps({'results':response},separators=(',',':'),ensure_ascii=False).encode()),
                reply_poi_ids_by_Q=[list(range(20,40)) for _ in range(5)]))
        streams['raw'].append({'events':raw}); streams[depth.METHOD].append({'events':public})
        sessions.append({'ledger':{depth.METHOD:dict(spent_per_m=.01,private_reads=1,anchors=[{'z':'protected'}],step_ms=[9.])}})
    bundle=dict(public=dict(streams=streams,public_clocks=dict(shared_fork_t=60,turn_visible_t=120)),
        evaluator_only=dict(family_id='f1',split='selection',draw=1,sessions=sessions),utility=utilities,wire=wire)
    family={'evaluator_only':{'sessions':specs}}
    return bundle,family,rn,evaluators


def test_exact_prefix_access_padding_and_invalid_depth():
    full,_,_ = fixture_objects(); full.access=np.array([1,0])
    full.signatures[1,0,50:] = -1
    for L in depth.DEPTHS:
        ctx=depth.PrefixContext(full,L)
        assert np.array_equal(ctx.query_indices(0),full.signatures[1,:,:L])
        assert np.array_equal(ctx.query_indices(1),full.signatures[0,:,:L])
        assert ctx.pois is full.pois
    for bad in (True,0,61,20.):
        with pytest.raises(ValueError): depth.PrefixContext(full,bad)


def test_replay_exact_int_json_costs_duplicate_queries_and_monotone_local_results():
    bundle,family,rn,evaluators=frozen_fixture(); before=deepcopy(bundle)
    value=depth.replay_bundle(bundle,family,rn,evaluators)
    assert bundle==before  # Existing Q, ledger, anchor and timestamps untouched.
    assert value['frozen_controls']['Q_stream_sha256']==depth.canonical_sha(bundle['public']['streams'][depth.METHOD])
    rows=value['utility']; wires=value['wire']
    assert len(rows)==8*31*4*2 and len(wires)==8*31*4
    assert {r['method'] for r in rows}=={f'service_l{L}' for L in depth.DEPTHS}
    for row in rows:
        expected=0. if row['method']=='service_l20' else 1.
        assert all(p['recall5']==expected for p in row['purposes'].values())
        assert row['available_count']==int(row['method'].removeprefix('service_l'))
    for row in wires:
        assert row['requests']==5 and isinstance(row['t'],int)
        if row['method']=='service_l20':
            original=next(w for w in bundle['wire'] if (w['slot'],w['t'])==(row['slot'],row['t']))
            assert dict(row,method=depth.METHOD)==original
    assert wires[1]['reply_bytes']>wires[0]['reply_bytes']


@pytest.mark.parametrize('tamper',['score','wire','Q','missing'])
def test_baseline_or_coordinate_tampering_cannot_pass_exact_replay(tamper):
    bundle,family,rn,evaluators=frozen_fixture()
    if tamper=='score': bundle['utility'][0]['purposes']['nearest_distance']['recall5']=.2
    elif tamper=='wire': bundle['wire'][0]['reply_bytes']+=1
    elif tamper=='Q': bundle['public']['streams'][depth.METHOD][0]['events'][0]['candidates'][0]['lat']=1.
    else: bundle['utility'].pop()
    with pytest.raises(ValueError): depth.replay_bundle(bundle,family,rn,evaluators)


def summary_fixture():
    value={}
    for L,gain,ratio in ((20,0.,1.),(30,.019,1.5),(40,.025,2.),(60,.04,3.)):
        purpose={p:dict(family_values={'a':.7+gain,'b':.9+gain},defined_windows=8,total_windows=10,
            defined_categories=8,total_categories=10) for p in depth.common.PURPOSES}
        purpose['equal_purpose_macro']={'family_values':{'a':.7+gain,'b':.9+gain}}
        value[str(L)]={'selection':{'current':{'all':purpose},'cost':{'requests':100,'reply_bytes':int(1000*ratio)}}}
    return value


def test_smallest_fixed_eligible_depth_and_failure_not_relaxed():
    value=summary_fixture(); selected=depth.selection_from_summary(value)
    assert selected['selected_depth']==40
    assert selected['candidates'][0]['gates']['macro_gain_at_least_2pp'] is False
    assert selected['candidates'][2]['gates']['reply_byte_ratio_at_most_2_5'] is False
    value['40']['selection']['cost']['reply_bytes']=2501
    assert depth.selection_from_summary(value)['selected_depth'] is None
    value=summary_fixture()
    value['40']['selection']['current']['all']['nearest_distance']['family_values']['a']=.69
    value['40']['selection']['current']['all']['nearest_distance']['family_values']['b']=.89
    assert depth.selection_from_summary(value)['selected_depth'] is None


def test_na_or_coverage_changes_are_rejected_not_filled_with_zero():
    value=summary_fixture()
    value['30']['selection']['current']['all']['nearest_distance']['family_values']['a']=None
    with pytest.raises(ValueError,match='complete defined'): depth.selection_from_summary(value)
    value=summary_fixture();value['30']['selection']['current']['all']['within_radius']['defined_windows']=9
    with pytest.raises(ValueError,match='coverage'):depth.selection_from_summary(value)


def test_nested_draws_are_not_counted_as_independent_families_and_na_is_preserved():
    rows=[]; wires=[]
    for L in depth.DEPTHS:
        for family,draws in (('a',(1,2,3)),('b',(1,))):
            for draw in draws:
                score=float(draw/10) if family=='a' else .9
                fields=dict(family_id=family,split='selection',draw=draw,method=f'service_l{L}',slot=0,t=0)
                purposes={p:dict(recall5=None if p=='within_radius' else score,
                    completion=None if p=='within_radius' else 1.,reference_category_count=0 if p=='within_radius' else 1,
                    all_category_count=1,overlap_total=0,reference_poi_total=0 if p=='within_radius' else 5) for p in depth.common.PURPOSES}
                for cache in ('current','static_epoch_cache'):rows.append(dict(fields,cache=cache,purposes=purposes))
                wires.append(dict(fields,requests=5,request_bytes=100,reply_bytes=1000))
    summary=depth.summarize_depth(rows,wires,['selection'])['20']['selection']['current']['all']
    assert summary['nearest_distance']['family_mean']==pytest.approx(.55)  # (.2+.9)/2, not .375.
    assert summary['within_radius']['family_mean'] is None
    assert summary['within_radius']['defined_windows']==0
    assert summary['equal_purpose_macro']['family_mean']==pytest.approx(.55)


def test_declaration_precedes_scores_is_write_once_and_binds_inputs(tmp_path,monkeypatch):
    root=tmp_path/'repo';root.mkdir();monkeypatch.setattr(depth,'ROOT',root)
    (root/'new.py').write_text('# new replay source\n')
    monkeypatch.setattr(depth,'depth_source_closure',lambda:['new.py'])
    source=root/'old';source.mkdir();(source/'families').mkdir()
    for name in ('protocol.json','protocol.sha256','generation.json','resources.json'):
        (source/name).write_text('{}')
    block=source/'families'/'f--draw1.json.gz';block.write_bytes(gzip.compress(b'{"no_score":true}'))
    old={'dataset_path':'dataset.json','dataset_sha256':'unread-dataset-pin','splits':['selection'],
        'draws_by_split':{'selection':1},'configuration':{'K':5}}
    generation={'family_files_sha256':{block.name:depth.common.sha(block)}}
    monkeypatch.setattr(depth,'audit_input',lambda path:(old,generation))
    output=root/'new';protocol=depth.declare_depth_study(source,output)
    assert protocol['selection_criterion']==depth.CRITERION and not (output/'readout.json').exists()
    assert '2pp' not in json.dumps(protocol['selection_criterion'])  # Numerical gate is .02 rate.
    depth.validate_depth_study(output)
    with pytest.raises(FileExistsError):depth.declare_depth_study(source,output)
    (root/'new.py').write_text('# modified source\n')
    with pytest.raises(ValueError,match='source/snapshot'):depth.validate_depth_study(output)


def test_development_runner_refuses_fresh_before_generation_or_scores(tmp_path):
    (tmp_path/'protocol.json').write_text(json.dumps({'splits':['train','selection','test']}))
    with pytest.raises(ValueError,match='fresh TEST'):depth.audit_input(tmp_path)
