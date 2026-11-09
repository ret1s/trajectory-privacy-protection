"""Write-once S4--S8 extension on six frozen methods; no defense generation."""
import argparse
import ast
from collections import defaultdict
import copy
import gzip
import hashlib
import json
import math
from pathlib import Path
import pickle
import platform
import traceback

import numpy as np
from sklearn.metrics import roc_auc_score, f1_score

from benchmark.session_poi_cache import SessionCoverClient
from core.research_release import load_release
from evaluation.live_comparison_attacks import LearnedAttack
from evaluation.remaining_scenarios import (prefix, xy, prefix_vector, linkage_features,
    candidate_public, FutureBank, LinkageBank, class_prediction, location_candidates,
    joint_vector, wire_digest)
from experiments import paper_comparison_20261028 as source

ROOT=source.ROOT
OUT=ROOT/'artifacts/benchmarks/remaining_scenarios_20261109_v1'
DATA=source.sealed.DATA
METHODS=source.METHODS
CONFIG=dict(date='2026-11-09',methods=list(METHODS),bootstrap_replicates=10000,
    bootstrap_seed=20261109,source_outputs_regenerated=False,
    future_phases={'S5':['before_fork','turn_window'],
                   'S6':['before_fork','turn_window','post_turn']},
    S4='route-family session association only; person/vehicle identities N/A',
    S7='conditional fixed-wire purpose/category audit, not natural intent inference',
    S8='relative-clock aligned distinct sessions; not simultaneous SUMO companionship',
    joint_views=['target_only','protected_partner','raw_partner','unrelated_partner'],
    classification_selection='group BA mean-SE, logloss mean+SE, lexical',
    regression_selection='group MAE mean+SE / Hit100 mean-SE, lexical')


def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p):return source.read(p)
def save(p,value):source.save(p,value)
def model_file(name,model):
    p=OUT/'models'/(name+'.pkl.gz');p.parent.mkdir(exist_ok=True)
    with p.open('xb') as stream:stream.write(gzip.compress(pickle.dumps(model),mtime=0))
    return sha(p)


def closure():
    pending=['experiments/remaining_scenarios_20261109.py',
        'experiments/verify_remaining_scenarios_20261109.py',
        'tests/test_remaining_scenarios_20261109.py'];found=set()
    while pending:
        name=pending.pop()
        if name in found:continue
        found.add(name);package=name[:-3].split('/')[:-1]
        def add(module):
            for candidate in (module.replace('.','/')+'.py',module.replace('.','/')+'/__init__.py'):
                if (ROOT/candidate).is_file() and candidate not in found:pending.append(candidate)
        for node in ast.walk(ast.parse((ROOT/name).read_text())):
            if isinstance(node,ast.Import):
                for item in node.names:add(item.name)
            elif isinstance(node,ast.ImportFrom):
                parent=package[:len(package)-node.level+1] if node.level else []
                module='.'.join(parent+([node.module] if node.module else []));add(module)
                for item in node.names:
                    if item.name!='*':add('.'.join(filter(None,(module,item.name))))
    found.update(('docs/research/2026-11-09_remaining_scenarios_protocol.md',
                  'requirements-paper-comparison.txt','requirements-continual-geoi.txt'))
    return sorted(found)


def freeze():
    load_release();source.validate_sources()
    if OUT.exists():raise FileExistsError('Preserve prior evidence; use a new version')
    names=closure();inputs=[DATA,source.OUT/'protocol.json',source.OUT/'generation.json',
        source.OUT/'results.json',source.OUT/'validation.json']
    p=dict(schema='remaining-scenarios-protocol-v1',config=CONFIG,
        source_sha256={n:sha(ROOT/n) for n in names},
        inputs_sha256={str(n.relative_to(ROOT)):sha(n) for n in inputs},
        frozen_before_extension_scoring=True,prior_cohort_inspected=True,
        runtime=dict(python=platform.python_version(),platform=platform.platform()),
        split_counts={'train':20,'selection':8,'test':20})
    save(OUT/'protocol.json',p)
    for n in names:
        path=OUT/'source_snapshot'/n;path.parent.mkdir(parents=True,exist_ok=True)
        path.write_bytes((ROOT/n).read_bytes())
    print('Frozen S4--S8 protocol and sources; no extension score read',flush=True)


def validate_sources():
    load_release();source.validate_sources();p=read(OUT/'protocol.json')
    assert p['config']==CONFIG
    for n,h in p['source_sha256'].items():assert sha(ROOT/n)==sha(OUT/'source_snapshot'/n)==h,n
    for n,h in p['inputs_sha256'].items():assert sha(ROOT/n)==h,n
    for n,h in read(source.OUT/'validation.json')['files_sha256'].items():
        assert sha(source.OUT/n)==h,n
    return p


def datasets():
    families=source.families();native=read(DATA)
    assert {f['family_id']:f['split'] for f in families}=={f['family_id']:f['split'] for f in native['families']}
    return families,native,{f['family_id']:f for f in native['families']}


def blocks(families):
    result={}
    for split in ('train','selection','test'):
        ids=sorted(f['family_id'] for f in families if f['split']==split)
        assert len(ids)%2==0
        for i in range(0,len(ids),2):
            group=split+'-block-'+str(i//2+1).zfill(2)
            result[ids[i]]=(group,ids[i+1]);result[ids[i+1]]=(group,ids[i])
    return result


def grouped_values(rows,values):
    groups=sorted({r['group'] for r in rows})
    return groups,np.array([np.mean([v for r,v in zip(rows,values) if r['group']==g]) for g in groups])


def summary(rows,values):
    groups,a=grouped_values(rows,values)
    ids=np.random.default_rng(CONFIG['bootstrap_seed']).integers(0,len(a),(CONFIG['bootstrap_replicates'],len(a)))
    return dict(mean=float(a.mean()),ci95=np.quantile(a[ids].mean(axis=1),[.025,.975]).tolist(),
        groups=len(a),records=len(rows),group_values=dict(zip(groups,a.tolist())))


def objective(rows,values,sign):
    _,a=grouped_values(rows,values);se=float(a.std(ddof=1)/np.sqrt(len(a)))
    return float(a.mean())+sign*se


def class_metrics(rows,predictions,selected):
    p=np.asarray(predictions[selected]);y=np.array([r['label'] for r in rows])
    threshold=float(selected.rsplit('@',1)[1]) if '@' in selected else None
    pred=class_prediction(p,threshold)
    correct=(pred==y).astype(float);loss=-np.log(np.maximum(1e-12,p[np.arange(len(y)),y]))
    out={'accuracy':summary(rows,correct),'log_loss':summary(rows,loss),
        'brier':summary(rows,(p[:,1]-y)**2),
        'macro_f1':float(f1_score(y,pred,average='macro',zero_division=0)),
        'roc_auc':float(roc_auc_score(y,p[:,1]))}
    # All task groups have balanced labels; verify rather than assume BA=accuracy.
    for group in sorted({r['group'] for r in rows}):
        labels=[r['label'] for r in rows if r['group']==group]
        assert labels.count(0)==labels.count(1)>0,group
    out['balanced_accuracy']=copy.deepcopy(out['accuracy'])
    group_auc={g:float(roc_auc_score(y[[i for i,r in enumerate(rows) if r['group']==g]],
        p[[i for i,r in enumerate(rows) if r['group']==g],1])) for g in sorted({r['group'] for r in rows})}
    out['group_mean_roc_auc']=summary(rows,[group_auc[r['group']] for r in rows])
    return out,pred


def choose_class(rows,probabilities):
    scores={}
    for name,p in probabilities.items():
        y=np.array([r['label'] for r in rows]);p=np.asarray(p)
        pred=class_prediction(p,float(name.rsplit('@',1)[1]) if '@' in name else None)
        correct=(pred==y).astype(float);loss=-np.log(np.maximum(1e-12,p[np.arange(len(y)),y]))
        scores[name]=dict(accuracy=summary(rows,correct)['mean'],
            accuracy_objective=objective(rows,correct,-1),log_loss_objective=objective(rows,loss,1))
    winner=min(scores,key=lambda n:(-round(scores[n]['accuracy_objective'],12),
        round(scores[n]['log_loss_objective'],12),n))
    return winner,scores


def save_class(name,rows,probabilities,selected,selection,extra=None):
    metrics,pred=class_metrics(rows,probabilities,selected)
    records=[{**{k:v for k,v in r.items() if k not in ('features','query','public_context')},
        'probabilities':{n:np.asarray(p[i]).tolist() for n,p in probabilities.items()},
        'selected_prediction':int(pred[i])} for i,r in enumerate(rows)]
    result=dict(task=name,status='ok',selected=selected,selection=selection,metrics=metrics,**(extra or {}))
    save(OUT/'predictions'/(name+'.json.gz'),records)
    save(OUT/'tasks'/(name+'.json'),result)
    print('Scored',name,flush=True)
    return result


def linkage(families):
    by_id={f['family_id']:f for f in families};mapping=blocks(families);results=[]
    for method in METHODS:
        sets=defaultdict(list)
        for fid,(group,other) in mapping.items():
            if fid>other:continue
            a,b=by_id[fid],by_id[other]
            by_day=lambda f,d,day:next(t for t in f['trips'] if t['draw']==d and t['native_session_id'].endswith('_day'+str(day)))
            for draw in sorted({t['draw'] for t in a['trips']}):
                for left_id,right_id,label in ((fid,fid,1),(other,other,1),(fid,other,0),(other,fid,0)):
                    l=by_day(by_id[left_id],draw,7);r=by_day(by_id[right_id],draw,8)
                    sets[a['split']].append(dict(group=group,source_families=[left_id,right_id],draw=draw,
                        label=label,pair_id=l['session_id']+'--'+r['session_id'],
                        features=linkage_features(l['events'][method],r['events'][method]).tolist()))
        x=lambda split:np.array([r['features'] for r in sets[split]])
        model=LinkageBank(x('train'),[r['label'] for r in sets['train']])
        selected,selection=choose_class(sets['selection'],model.predict(x('selection')))
        name='S4--'+method;digest=model_file(name,model)
        save(OUT/'selections'/(name+'.json'),dict(selected=selected,scores=selection,model_sha256=digest))
        results.append(save_class(name,sets['test'],model.predict(x('test')),selected,selection,
            dict(method=method,scope='route-family association proxy; person/vehicle N/A',
                test_blocks=10,test_pairs=120,chance=.5)))
    return results


def future_rows(families,native,by_id,method,scenario,phase):
    sets=defaultdict(list)
    for f in families:
        nf=by_id[f['family_id']];clock=nf['clocks']
        cut={'before_fork':clock['shared_fork_t'],'turn_window':clock['turn_visible_t'],
             'post_turn':20*math.ceil(clock['turn_visible_t']/20)}[phase]
        specs={s['session_id']:s for s in nf['evaluator_only']['sessions']}
        for trip in f['trips']:
            spec=specs[trip['native_session_id']];trace=native['traces'][trip['native_session_id']]
            label=spec['choice_index'];choice=nf['public_context']['choices'][label]
            first_edge=min(p['time_s'] for p in trace if p['edge_id']==choice['edge_id'])
            arrived=min(p['time_s'] for p in trace if p['edge_id']==choice['destination_edge_id'] and p['speed_m_s']==0)
            assert (cut<first_edge if scenario=='S5' else cut<arrived),(scenario,f['family_id'],phase)
            events=prefix(trip['events'][method],cut)
            last_raw=max(p['time_s'] for p in trace if p['time_s']<=20*math.floor(cut/20))
            raw=trace[int(last_raw)]
            sets[f['split']].append(dict(group=f['family_id'],family_id=f['family_id'],
                session_id=trip['session_id'],draw=trip['draw'],label=label,choice_index=label,
                destination_xy=xy([[trace[-1]['lat'],trace[-1]['lon']]])[0].tolist(),
                edge_id=choice['edge_id'],destination_role=spec['destination_role'],
                first_target_edge_s=first_edge,destination_arrival_s=arrived,cut_s=cut,
                raw_last_tick_s=last_raw,raw_last_tick_internal=raw['edge_id'].startswith(':'),
                observed_packets=len(events),observed_coordinates=sum(len(e['coordinates']) for e in events),
                public_context=nf['public_context'],query=candidate_public(events,cut)))
    return sets


def future(families,native,by_id):
    results=[]
    for scenario,phases in CONFIG['future_phases'].items():
        for phase in phases:
            for method in METHODS:
                sets=future_rows(families,native,by_id,method,scenario,phase)
                model=FutureBank(sets['train'])
                def predictions(split):
                    values=[model.predict(r) for r in sets[split]]
                    return {n:np.array([p[n] for p in values]) for n in values[0]}
                selected,selection=choose_class(sets['selection'],predictions('selection'))
                name=scenario+'--'+phase+'--'+method;digest=model_file(name,model)
                save(OUT/'selections'/(name+'.json'),dict(selected=selected,scores=selection,model_sha256=digest))
                p=predictions('test');winner=np.argmax(p[selected],axis=1)
                errors=[float(np.linalg.norm(np.array(r['public_context']['choices'][k]['destination_xy'])-r['destination_xy']))
                        for r,k in zip(sets['test'],winner)]
                for r,error in zip(sets['test'],errors):r['selected_destination_error_m']=error
                results.append(save_class(name,sets['test'],p,selected,selection,dict(method=method,scenario=scenario,
                    phase=phase,chance=.5,destination_MAE_m=summary(sets['test'],errors),
                    destination_Hit100=summary(sets['test'],np.array(errors)<=100),
                    last_raw_tick_internal_fraction=float(np.mean([r['raw_last_tick_internal'] for r in sets['test']])))))
    return results


def regression_selection(rows,predictions):
    target=np.array([r['target_xy'] for r in rows]);scores={}
    for name,p in predictions.items():
        error=np.linalg.norm(np.asarray(p)-target,axis=1)
        scores[name]=dict(mae=objective(rows,error,1),hit100=objective(rows,error<=100,-1))
    chosen=dict(mae=min(scores,key=lambda n:(round(scores[n]['mae'],12),n)),
                hit100=min(scores,key=lambda n:(-round(scores[n]['hit100'],12),n)))
    return chosen,scores


def joint_rows(families,native,by_id,method,view):
    mapping=blocks(families);lookup={f['family_id']:f for f in families};sets=defaultdict(list)
    for f in families:
        group,other=mapping[f['family_id']];cut=20*math.floor(by_id[f['family_id']]['clocks']['shared_fork_t']/20)
        for t in f['trips']:
            partner=next(v for v in f['trips'] if v['draw']==t['draw'] and v['native_session_id']!=t['native_session_id'])
            partner_family=f['family_id']
            if view=='unrelated_partner':
                partner_family=other
                partner=next(v for v in lookup[other]['trips'] if v['draw']==t['draw'] and
                    v['native_session_id'].split('_day')[-1]!=t['native_session_id'].split('_day')[-1])
            target_trace=native['traces'][t['native_session_id']]
            if view!='unrelated_partner':
                other_trace=native['traces'][partner['native_session_id']]
                assert [(p['lat'],p['lon']) for p in target_trace if p['time_s']<=cut]==[(p['lat'],p['lon']) for p in other_trace if p['time_s']<=cut]
            target_events=prefix(t['events'][method],cut)
            partner_events=prefix(partner['events']['raw' if view=='raw_partner' else method],cut)
            point=target_trace[int(cut)]
            feature=prefix_vector(target_events,cut) if view=='target_only' else joint_vector(target_events,partner_events,cut)
            geometry={k:v.tolist() for k,v in location_candidates(target_events,cut).items()}
            if view!='target_only':geometry.update({'partner_'+k:v.tolist() for k,v in location_candidates(partner_events,cut).items()})
            sets[f['split']].append(dict(group=group,family_id=f['family_id'],partner_family_id=partner_family,
                session_id=t['session_id'],partner_session_id=partner['session_id'],draw=t['draw'],cut_s=cut,
                target_xy=xy([[point['lat'],point['lon']]])[0].tolist(),features=feature.tolist(),
                target_features=prefix_vector(target_events,cut).tolist(),geometry=geometry,
                observed_packets=len(target_events)+(0 if view=='target_only' else len(partner_events)),
                observed_coordinates=sum(len(e['coordinates']) for e in target_events)+(0 if view=='target_only' else sum(len(e['coordinates']) for e in partner_events))))
    return sets


def joint(families,native,by_id):
    results=[]
    for method in METHODS:
        base_model=None;base_records=None
        for view in CONFIG['joint_views']:
            sets=joint_rows(families,native,by_id,method,view)
            x=lambda split:np.array([r['features'] for r in sets[split]])
            y=np.array([r['target_xy'] for r in sets['train']]);model=LearnedAttack(x('train'),y)
            if view=='target_only':base_model=model
            def predict(split):
                result={'learned_'+k:v for k,v in model.predict(x(split)).items()}
                result.update({'target_only_'+k:v for k,v in base_model.predict(np.array([r['target_features'] for r in sets[split]])).items()})
                for key in sets[split][0]['geometry']:result[key]=np.array([r['geometry'][key] for r in sets[split]])
                return result
            selected,selection=regression_selection(sets['selection'],predict('selection'))
            name='S8--'+view+'--'+method;digest=model_file(name,(model,base_model))
            save(OUT/'selections'/(name+'.json'),dict(selected=selected,scores=selection,model_sha256=digest))
            predictions=predict('test');target=np.array([r['target_xy'] for r in sets['test']])
            errors={n:np.linalg.norm(p-target,axis=1) for n,p in predictions.items()}
            records=[{**{k:v for k,v in r.items() if k not in ('features','target_features','geometry')},
                'estimates':{n:p[i].tolist() for n,p in predictions.items()},
                'errors':{n:float(e[i]) for n,e in errors.items()}} for i,r in enumerate(sets['test'])]
            result=dict(task=name,method=method,view=view,status='ok',selected=selected,selection=selection,
                metrics={'MAE_m':summary(records,errors[selected['mae']]),
                         'Hit100':summary(records,errors[selected['hit100']]<=100)},
                mean_observed_packets=float(np.mean([r['observed_packets'] for r in records])),
                mean_observed_coordinates=float(np.mean([r['observed_coordinates'] for r in records])))
            if view=='target_only':base_records=records;base_selected=selected
            assert [r['session_id'] for r in records]==[r['session_id'] for r in base_records]
            result['paired_vs_target_only']={}
            for metric,key in [('MAE_m','mae'),('Hit100','hit100')]:
                a=np.array([r['errors'][selected[key]] for r in records])
                b=np.array([r['errors'][base_selected[key]] for r in base_records])
                delta=a-b if key=='mae' else (a<=100).astype(float)-(b<=100).astype(float)
                result['paired_vs_target_only'][metric]=summary(records,delta)
            save(OUT/'predictions'/(name+'.json.gz'),records);save(OUT/'tasks'/(name+'.json'),result)
            results.append(result);print('Scored',name,flush=True)
    return results


def vulnerability(digests,labels):
    counts=defaultdict(lambda:defaultdict(int))
    for d,y in zip(digests,labels):counts[d][str(y)]+=1
    return sum(max(v.values()) for v in counts.values())/len(labels)


def content(families):
    categories=sorted({p['category'] for p in read(ROOT/'artifacts/benchmarks/research_loop/resources.json')['pois_used']})
    assert len(categories)==6
    results=[]
    for method in METHODS:
        records=[]
        for f in families:
            for t in f['trips']:
                events=prefix(t['events'][method],600.)
                # Actual client interface; deterministic provider probe, not POI utility replay.
                client=SessionCoverClient(categories,418,k=len(events[0]['coordinates']),response_l=60,
                    session_token='public-probe',static_catalogue_id='fixed-public-catalogue')
                requests=[];replies=[]
                for event in events:
                    wire=client.step(event['timestamp_s'],event['coordinates'],lambda request:[[] for _ in categories])
                    requests.extend(wire['requests']);replies.extend(wire['replies'])
                protected=[];explicit=[];ps=[];cs=[];joint=[]
                for purpose in range(4):
                    for category in range(6):
                        # All private variants enter only the local task; network API has no such argument.
                        protected.append(wire_digest(requests,replies))
                        control=[{**r,'private_purpose':purpose,'private_category':categories[category]} for r in requests]
                        explicit.append(wire_digest(control,replies));ps.append(purpose);cs.append(category);joint.append((purpose,category))
                record=dict(group=f['family_id'],split=f['split'],session_id=t['session_id'],method=method,
                    equal_wire=len(set(protected))==1,request_count=len(requests),counterfactuals=24,
                    protected_hashes=protected,explicit_hashes=explicit,
                    frozen_reply_masks_sha256=hashlib.sha256(json.dumps(t['masks'][method],sort_keys=True).encode()).hexdigest(),
                    purpose_vulnerability=vulnerability(protected,ps),category_vulnerability=vulnerability(protected,cs),
                    joint_vulnerability=vulnerability(protected,joint),
                    explicit_joint_vulnerability=vulnerability(explicit,joint))
                assert record['equal_wire'] and record['explicit_joint_vulnerability']==1
                records.append(record)
        test=[r for r in records if r['split']=='test'];name='S7--'+method
        result=dict(task=name,method=method,status='ok',scope=CONFIG['S7'],
            metrics={k:summary(test,[r[k] for r in test]) for k in ('purpose_vulnerability','category_vulnerability',
                'joint_vulnerability','explicit_joint_vulnerability')},
            all_streams=len(records),test_streams=len(test),equal_wire_rate=float(np.mean([r['equal_wire'] for r in records])),
            conditional_variants_total=24*len(records),provider_scope='deterministic empty-response interface probe; frozen reply-mask signatures authenticated')
        save(OUT/'predictions'/(name+'.json.gz'),records);save(OUT/'tasks'/(name+'.json'),result)
        results.append(result);print('Scored',name,flush=True)
    return results


def run():
    validate_sources()
    if (OUT/'results.json').exists():raise FileExistsError('Completed extension must not be overwritten')
    families,native,by_id=datasets();results=[];failures=[]
    for name,operation in [('S4',lambda:linkage(families)),('S5-S6',lambda:future(families,native,by_id)),
                           ('S7',lambda:content(families)),('S8',lambda:joint(families,native,by_id))]:
        try:results.extend(operation())
        except Exception as error:
            failure=dict(task=name,type=type(error).__name__,message=str(error),traceback=traceback.format_exc())
            failures.append(failure);save(OUT/'failures'/(name+'.json'),failure)
            print('RETAINED FAILURE',name,str(error),flush=True)
    validate_sources()
    expected=6+30+6+24
    result=dict(schema='remaining-scenarios-results-v1',protocol_sha256=sha(OUT/'protocol.json'),
        tasks=results,failures=failures,expected_tasks=expected,completed_tasks=len(results),
        full_coverage=len(results)==expected and not failures,
        scope=CONFIG,old_paper_comparison_sha256=sha(source.OUT/'results.json'),
        limitations={'S4_person_identity':'N/A: independent person labels absent',
                     'S4_vehicle_identity':'N/A: physical-vehicle labels absent',
                     'S6_history':'N/A: six history sessions have no current frozen protected outputs',
                     'S7_route_correlated_intent':'N/A: no human-intent ground truth',
                     'S8_actual_companionship':'N/A: aligned separate runs, not joint SUMO simulation'})
    save(OUT/'results.json',result)
    save(OUT/'output_manifest.json',dict(files_sha256={str(p.relative_to(OUT)):sha(p) for p in sorted(OUT.rglob('*'))
        if p.is_file() and 'source_snapshot' not in p.parts and p.name not in ('output_manifest.json','validation.json')}))
    print('Completed',len(results),'/',expected,'tasks; failures',len(failures),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('stage',choices=('freeze','run'))
    args=parser.parse_args();globals()[args.stage]()
