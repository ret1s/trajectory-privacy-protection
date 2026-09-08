"""Independent v3 split gates and native SUMO evidence; no builder invocation."""
import argparse
from collections import Counter
from copy import deepcopy
import json
from pathlib import Path
import xml.etree.ElementTree as ET

from data.sumo_demo import _load_sumolib
from experiments.verify_scenario_suite import ROOT, sha, meters
from experiments.scenario_v2_checks import verify_data

DEFAULT=ROOT/'artifacts/datasets/urban_scenarios_v3/dataset.json'


def summary(data):
    cases=Counter(r['case_id'] for r in data['records'])
    return {'families':len(data['families']),
            'sessions':sum(len(f['sessions']) for f in data['families']),
            'raw_fcd_samples':sum(map(len,data['traces'].values())),
            'scenario_records':len(data['records']),'generated_subcases':len(cases),
            'declared_subcases':len(data['catalogue']),
            'generated_scenarios':len({r['scenario'] for r in data['records']}),
            'by_scenario':dict(Counter(r['scenario'] for r in data['records'])),
            'by_case':dict(cases)}


def verify_new_gates(data):
    assert data['schema']=='urban-scenario-suite-v2'
    assert data['purpose']=='scenario_challenge_dataset_not_protection_evidence'
    assert data['summary']==summary(data)
    records=data['records']
    assert len({r['record_id'] for r in records})==len(records)
    assert {f['seed']:f['split'] for f in data['families']}=={
        101:'development_train',102:'development_train',103:'development_validation',
        104:'development_validation',201:'confirmation',202:'confirmation',
        203:'confirmation',204:'confirmation'}
    families={f['family_id']:f for f in data['families']}
    sessions={s['session_id']:s for f in data['families'] for s in f['sessions']}
    pois=data['public_poi_context']['pois']
    counts=Counter(p['category'] for p in pois)
    assert counts==data['public_poi_context']['category_counts']
    for c in data['catalogue']:
        n=sum(r['case_id']==c['case_id'] for r in records)
        assert c['generated_records']==n and not c['attack_evaluated'] and not c['protection_evaluated']
        assert c['data_status']==('generated' if n else 'specified_not_generated')
    for r in records:
        if r['case_id'] not in {'S1.C','S6.C'}:
            continue
        f=families[r['family_id']]
        assert r['split']==f['split'] and r['scenario']==r['case_id'].split('.')[0]
        assert len(r['session_ids'])==len(r['observed_indices'])
        for sid,idx in zip(r['session_ids'],r['observed_indices']):
            assert sid in {s['session_id'] for s in f['sessions']}
            assert idx==sorted(set(idx)) and 0<=idx[0]<=idx[-1]<len(data['traces'][sid])
        if r['case_id']=='S1.C':
            assert len(r['session_ids'])==1 and len(r['observed_indices'][0])==1
            i=r['labels']['target_index']
            assert i==r['observed_indices'][0][0]
            poi=next(p for p in pois if p['id']==r['evidence']['poi_id'])
            assert counts[poi['category']]/len(pois)<=.05
            assert r['evidence']['category']==poi['category']
            assert abs(r['evidence']['category_share']-counts[poi['category']]/len(pois))<1e-12
            d=meters(data['traces'][r['session_ids'][0]][i],poi)
            assert d<=150 and abs(d-r['evidence']['distance_m'])<.01
        else:
            assert len(r['session_ids'])==7 and r['labels']['target_slot']==6
            assert r['evidence']['history_days']==list(range(1,7))
            ss=[sessions[s] for s in r['session_ids']]
            assert [s['day_index'] for s in ss[:6]]==list(range(1,7))
            assert ss[-1]['day_index']==r['evidence']['query_day'] in (7,8)
            for key in ('person_id','device_id','physical_vehicle_id'):
                assert len({s[key] for s in ss})==1
            ts=[data['traces'][s['session_id']] for s in ss]
            assert all(a[-1]['time_s']<b[0]['time_s'] for a,b in zip(ts,ts[1:]))
            # Realized route endpoints, not planned frequency labels.
            endpoints=[t[-1]['edge_id'] for t in ts[:6]]
            freq=Counter(endpoints)
            assert sorted(freq.values())==[1,5]
            actual='routine' if freq[ts[-1][-1]['edge_id']]==5 else 'rare' if freq[ts[-1][-1]['edge_id']]==1 else None
            assert actual==r['labels']['destination_class']
            assert r['labels']['historical_destination_counts']=={'routine':5,'rare':1}
            idx=r['observed_indices'][6]
            target=r['labels']['target_index']
            assert target==len(ts[-1])-1 and idx[-1]<target
            cut=next(i for i,p in enumerate(ts[-1]) if p['edge_id']==f['fork_edge'])
            assert idx[-1]<=cut and cut-idx[-1]<20
            for t,indices in zip(ts[:6],r['observed_indices'][:6]):
                assert indices[0]==0 and len(t)-1-indices[-1]<20


def verify(path=DEFAULT,raw=False):
    path=Path(path)
    assert sha(path)==path.with_suffix('.sha256').read_text().strip()
    d=json.loads(path.read_text())
    for p,h in d['source_sha256'].items():
        assert sha(ROOT/p)==h,p
    assert sha(ROOT/d['network']['path'])==d['network']['sha256']
    verify_new_gates(d)
    # Existing independently written v1 assertions apply to ALL raw traces and
    # unchanged cases. Only the two new case definitions are checked above.
    old={**d,'schema':'urban-scenario-suite-v1','purpose':'development_dataset_not_protection_evidence',
         'records':[r for r in d['records'] if r['case_id'] not in {'S1.C','S6.C'}],
         'catalogue':deepcopy(d['catalogue'])}
    old['summary']=summary(old)
    for c in old['catalogue']:
        n=old['summary']['by_case'].get(c['case_id'],0)
        c.update(generated_records=n,data_status='generated' if n else 'specified_not_generated')
    network=_load_sumolib().net.readNet(str(ROOT/d['network']['path']),withInternal=True)
    audit=verify_data(old,network)
    checked=runs=0
    if raw:
        for f in d['families']:
            seen=set()
            for run in f['simulation_runs']:
                runs+=1
                for p,h in run['source_files'].items():
                    assert sha(ROOT/p)==h and f['source_files'][p]==h
                files={Path(p).name:ROOT/p for p in run['source_files']}
                vehicles={v.attrib['id']:v for v in ET.parse(files['vehicle_routes.xml']).getroot().findall('vehicle')}
                assert set(vehicles)==set(run['session_ids']) and not seen&set(vehicles)
                seen.update(vehicles)
                for sid,v in vehicles.items():
                    s=next(s for s in f['sessions'] if s['session_id']==sid)
                    assert v.attrib==f['actual_vehicles'][sid]
                    assert v.find('route').attrib['edges'].split()==s['route_edges']
                    assert not any(e.startswith(':') for e in s['route_edges'])
                stops=ET.parse(files['stops.xml']).getroot().findall('stopinfo')
                for r in d['records']:
                    if r['scenario']!='S2' or r['session_ids'][0] not in vehicles:
                        continue
                    sid=r['session_ids'][0]; trace=d['traces'][sid]
                    for a,b in r['labels']['stop_intervals']:
                        assert any(s.attrib['id']==sid and s.attrib['lane']==trace[a]['lane_id'] and
                            float(s.attrib['started'])-2<=trace[a]['time_s']<=float(s.attrib['ended']) and
                            float(s.attrib['started'])<=trace[b]['time_s']<=float(s.attrib['ended'])+2 for s in stops)
                offsets=Counter()
                for _,step in ET.iterparse(files['fcd.xml'],events=('end',)):
                    if step.tag!='timestep': continue
                    for v in step.findall('vehicle'):
                        a=v.attrib; sid=a['id']; p=d['traces'][sid][offsets[sid]]
                        assert p['time_s']==float(step.attrib['time'])
                        for key,xml in [('lon','x'),('lat','y'),('speed_m_s','speed'),('lane_pos_m','pos'),('angle_deg','angle')]:
                            assert p[key]==float(a[xml])
                        assert p['lane_id']==a['lane']
                        offsets[sid]+=1; checked+=1
                    step.clear()
                assert all(offsets[s]==len(d['traces'][s]) for s in vehicles)
            assert seen=={s['session_id'] for s in f['sessions']}
    return {**d['summary'],'dataset_sha256':sha(path),'raw_runs_checked':runs,'raw_fcd_points_compared':checked,
            'external_transitions_checked':audit['external_transitions_checked'],
            'max_1s_displacement_m':audit['max_1s_displacement_m'],
            'scientific_sources_checked':len(d['source_sha256']),
            'by_family':{f['family_id']:len({r['case_id'] for r in d['records'] if r['family_id']==f['family_id']}) for f in d['families']}}


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--dataset',type=Path,default=DEFAULT)
    p.add_argument('--raw',action='store_true')
    p.add_argument('--output',type=Path)
    a=p.parse_args(); result=verify(a.dataset,a.raw)
    if a.output:
        a.output.parent.mkdir(parents=True,exist_ok=True)
        a.output.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))

