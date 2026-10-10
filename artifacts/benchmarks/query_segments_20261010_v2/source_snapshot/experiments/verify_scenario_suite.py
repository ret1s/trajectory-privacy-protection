"""Independent schema, FCD, temporal, label and case-gate audit of the suite.

Does not regenerate records or call build_records. No inference/privacy claim.
Run with --raw to additionally compare every FCD point against SUMO XML.
"""
import argparse
from collections import Counter, defaultdict
import hashlib
import json
import math
from pathlib import Path
import xml.etree.ElementTree as ET

from data.sumo_demo import _load_sumolib

ROOT=Path(__file__).resolve().parents[1]
DEFAULT=ROOT/'artifacts/datasets/urban_scenarios_v1/dataset.json'


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def meters(a,b):
    r=math.pi*6371000/180
    return math.hypot((a['lon']-b['lon'])*r*math.cos((a['lat']+b['lat'])*math.pi/360),
                      (a['lat']-b['lat'])*r)


def outgoing(edge):
    return {e.getID() for e,links in edge.getOutgoing().items() if not e.getID().startswith(':') and e.allows('passenger') and
            any(c.getFromLane().allows('passenger') and c.getToLane().allows('passenger') for c in links)}


def verify_data(data, network):
    assert data['schema']=='urban-scenario-suite-v1'
    assert data['purpose']=='development_dataset_not_protection_evidence'
    sessions={s['session_id']:s for f in data['families'] for s in f['sessions']}
    assert len(sessions)==sum(len(f['sessions']) for f in data['families'])
    assert sessions.keys()==data['traces'].keys()
    families={f['family_id']:f for f in data['families']}
    resources=defaultdict(list)
    edges_checked=0
    fcd_skipped_edges=0
    max_step=0.
    for sid,t in data['traces'].items():
        assert len(t)>1
        assert all(math.isclose(b['time_s']-a['time_s'],1,abs_tol=.001) for a,b in zip(t,t[1:])),sid
        route=sessions[sid]['route_edges']
        assert all(nxt in outgoing(network.getEdge(prev)) for prev,nxt in zip(route,route[1:])),sid
        for p in t:
            assert set(p)=={'time_s','lat','lon','speed_m_s','lane_id','edge_id','lane_pos_m','angle_deg'}
            assert all(math.isfinite(p[k]) for k in ['time_s','lat','lon','speed_m_s','lane_pos_m','angle_deg'])
            lane=network.getLane(p['lane_id'])
            assert lane.getEdge().getID()==p['edge_id'] and lane.allows('passenger')
            assert -.01 <= p['lane_pos_m'] <= lane.getLength()+.05
            assert 0 <= p['speed_m_s'] <= 8.01 and 0 <= p['angle_deg'] <= 360
            assert 116.29 <= p['lon'] <= 116.36 and 39.96 <= p['lat'] <= 40.02, (sid,p)
        ext=[p['edge_id'] for p in t if not p['edge_id'].startswith(':')]
        seen=[ext[0]]+[b for a,b in zip(ext,ext[1:]) if a!=b]
        # A sub-second short edge may be absent in 1 Hz FCD. Require monotone
        # route membership, and verify the complete route in vehroute XML below.
        cursor=-1
        for edge in seen:
            new_cursor=route.index(edge,cursor+1)
            fcd_skipped_edges+=new_cursor-cursor-1
            cursor=new_cursor
        assert seen[0]==route[0] and cursor==len(route)-1
        edges_checked+=len(route)-1
        for a,b in zip(t,t[1:]):
            d=meters(a,b)
            max_step=max(max_step,d)
            assert d <= 10.1,(sid,'apparent jump',d)
        for key in ['person_id','device_id','physical_vehicle_id']:
            resources[key,sessions[sid][key]].append((t[0]['time_s'],t[-1]['time_s'],sid))
    for resource,periods in resources.items():
        periods.sort()
        assert all(a[1]<b[0] for a,b in zip(periods,periods[1:])),(resource,'overlapping assignment')
    for field in ['person_id','device_id','physical_vehicle_id']:
        split_sets=defaultdict(set)
        for f in data['families']:
            split_sets[f['split']].update(s[field] for s in f['sessions'])
        sets=list(split_sets.values())
        assert all(not a&b for i,a in enumerate(sets) for b in sets[i+1:])
    unique=set()
    case_counts=Counter()
    for r in data['records']:
        assert r['record_id'] not in unique
        unique.add(r['record_id'])
        case_counts[r['case_id']]+=1
        f=families[r['family_id']]
        assert r['split']==f['split'] and r['scenario']==r['case_id'].split('.')[0]
        assert len(r['session_ids'])==len(r['observed_indices'])>=1
        assert all(s in {v['session_id'] for v in f['sessions']} for s in r['session_ids'])
        ts=[data['traces'][s] for s in r['session_ids']]
        obs=r['observed_indices']
        for s,t,idx in zip(r['session_ids'],ts,obs):
            assert idx==sorted(set(idx)) and 0<=idx[0]<=idx[-1]<len(t)
            assert float(f['actual_vehicles'][s]['arrival'])>=0
            assert t[-1]['time_s'] < float(f['actual_vehicles'][s]['arrival'])
        case=r['case_id']
        label,e=r['labels'],r['evidence']
        t=ts[0]
        if r['scenario']=='S1':
            assert len(obs[0])==1 and label['target_index']==obs[0][0]
            n=len(outgoing(network.getEdge(t[obs[0][0]]['edge_id'])))
            assert n==e['legal_successors']
            assert n>=2 if case=='S1.A' else n==1
        elif r['scenario']=='S2':
            assert len(label['stop_intervals'])==(2 if case=='S2.C' else 1)
            minimum=120 if case=='S2.B' else 20
            for a,b in label['stop_intervals']:
                assert t[b]['time_s']-t[a]['time_s']>=minimum
                assert all(p['speed_m_s']<=.05 and p['lane_id']==e['same_lane'] for p in t[a:b+1])
                assert max(meters(t[a],p) for p in t[a:b+1])<=1
            assert all(any(a<=i<=b for a,b in label['stop_intervals']) for i in obs[0])
            if case=='S2.C':
                assert any(p['speed_m_s']>1 for p in t[label['stop_intervals'][0][1]+1:label['stop_intervals'][1][0]])
        elif r['scenario']=='S3':
            assert label['target_indices']==obs[0]
            assert all(p['speed_m_s']>.05 for p in t[obs[0][0]:obs[0][-1]+1])
            assert t[obs[0][-1]]['time_s']-t[obs[0][0]]['time_s']>=60
            if case=='S3.B':
                ext=[t[i]['edge_id'] for i in obs[0] if not t[i]['edge_id'].startswith(':')]
                ratio=sum(len(outgoing(network.getEdge(edge)))==1 for edge in ext)/len(ext)
                assert ratio>=.5 and math.isclose(ratio,e['single_successor_fraction'])
        elif r['scenario']=='S4':
            a,b=[sessions[s] for s in r['session_ids']]
            assert label['same_person']==(a['person_id']==b['person_id'])
            assert label['same_device']==(a['device_id']==b['device_id'])
            assert (label['same_person'],label['same_device'])=={'S4.A':(True,True),'S4.B':(True,False),'S4.C':(False,True)}[case]
            assert ts[0][-1]['time_s']<ts[1][0]['time_s']
        elif r['scenario']=='S5':
            future=label.get('future_indices',[label.get('future_index')])
            targets=label.get('next_edges',[label.get('next_edge')])
            for trace,idx,j,target in zip(ts,obs,future,targets):
                assert j>idx[-1] and trace[j]['edge_id']==target
                n=next(i for i in range(idx[-1]+1,len(trace)) if not trace[i]['edge_id'].startswith(':'))
                assert j==n and target in outgoing(network.getEdge(trace[idx[-1]]['edge_id']))
            if case=='S5.C':
                assert len(set(targets))==2
                prefixes=[]
                for sid in r['session_ids']:
                    route=sessions[sid]['route_edges']
                    prefixes.append(route[:route.index(e['fork_edge'])+1])
                assert prefixes[0]==prefixes[1]
            else:
                n=len(outgoing(network.getEdge(t[obs[0][-1]]['edge_id'])))
                assert n>=2 if case=='S5.A' else n==1
        elif r['scenario']=='S6':
            assert len(ts)==2 and meters(ts[0][-1],ts[1][-1])>0
            assert all(j==len(trace)-1 and j>idx[-1] for trace,idx,j in zip(ts,obs,label['target_indices']))
            if case=='S6.B':
                assert meters(ts[0][-1],ts[1][-1])<=500
        elif r['scenario']=='S7':
            assert len(label['true_queries'])==len(obs[0]) and label['intent_source']=='synthetic'
            assert set(label['true_queries'])<={'cafe','clinic','fuel','hospital','pharmacy','restaurant'}
            if case=='S7.C':
                assert len(obs[0])==3 and label['true_queries'][0]=='pharmacy'
        elif r['scenario']=='S8':
            other={p['time_s']:(i,p) for i,p in enumerate(ts[1])}
            aligned=[(i,other[p['time_s']][0]) for i,p in enumerate(t) if p['time_s'] in other]
            close=[(i,j) for i,j in aligned if meters(t[i],ts[1][j])<=100]
            assert list(map(list,close))==label['proximity_pairs']
            assert e['aligned_timestamps']==len(aligned) and e['within_100m_samples']==len(close)
            ratio=len(close)/len(aligned)
            assert math.isclose(ratio,e['proximity_fraction'])
            longest=current=0
            last_time=None
            for i,j in close:
                current=current+1 if last_time is not None and t[i]['time_s']-last_time==1 else 1
                longest=max(longest,current-1)
                last_time=t[i]['time_s']
            assert longest==e['max_consecutive_close_s'] and longest>=10
            assert label['declared_companions']==(case!='S8.C')
            assert ratio>=.8 if case=='S8.B' else ratio<1 if case=='S8.A' else True
        elif r['scenario'] in {'S9','S10'}:
            origin=r['scenario']=='S9'
            targets=label.get('target_indices',[label.get('target_index')])
            for trace,idx,j in zip(ts,obs,targets):
                assert j==(0 if origin else len(trace)-1) and j not in idx
                if 'mask_s' in e:
                    assert trace[idx[0]]['time_s']-trace[0]['time_s']>=60 if origin else trace[-1]['time_s']-trace[idx[-1]]['time_s']>=60
                    expected=[i for i,p in enumerate(trace) if (p['time_s']>=trace[0]['time_s']+60 if origin else p['time_s']<=trace[-1]['time_s']-60)]
                    assert idx[0]==expected[0] and expected[-1]-idx[-1]<20
            if case.endswith('.A'):
                edge=network.getEdge(t[0 if origin else -1]['edge_id'])
                n=len(outgoing(edge)) if origin else sum(edge.getID() in outgoing(prev) for prev in edge.getIncoming() if not prev.getID().startswith(':'))
                assert n==1
            if case.endswith('.B'):
                paths=[]
                for sid,trace,idx in zip(r['session_ids'],ts,obs):
                    route=sessions[sid]['route_edges']
                    boundary=trace[idx[0] if origin else idx[-1]]['edge_id']
                    pos=route.index(boundary)
                    paths.append(route[pos:] if origin else route[:pos+1])
                assert paths[0]==paths[1]
                assert meters(ts[0][0 if origin else -1],ts[1][0 if origin else -1])>0
            if case.endswith('.C'):
                j=0 if origin else -1
                assert sessions[r['session_ids'][0]]['route_edges'][j]==sessions[r['session_ids'][1]]['route_edges'][j]
                # Last recorded points need not coincide: the final partial
                # second before arrival is not a sampled FCD point.
                assert meters(ts[0][j],ts[1][j])<=10.1
    expected={'families':len(families),'sessions':len(sessions),'raw_fcd_samples':sum(map(len,data['traces'].values())),
              'scenario_records':len(unique),'generated_subcases':len(case_counts),'declared_subcases':len(data['catalogue']),
              'generated_scenarios':len({r['scenario'] for r in data['records']}),
              'by_scenario':dict(Counter(r['scenario'] for r in data['records'])),'by_case':dict(case_counts)}
    assert data['summary']==expected
    for c in data['catalogue']:
        assert c['generated_records']==case_counts[c['case_id']]
        assert not c['attack_evaluated'] and not c['protection_evaluated']
        assert c['data_status']==('generated' if c['generated_records'] else 'specified_not_generated')
    return {**expected,'external_transitions_checked':edges_checked,'short_edges_between_fcd_samples':fcd_skipped_edges,
            'max_1s_displacement_m':max_step}


def verify(path=DEFAULT,raw=False):
    path=Path(path)
    assert sha(path)==path.with_suffix('.sha256').read_text().strip()
    data=json.loads(path.read_text())
    for p,h in data['source_sha256'].items():
        assert sha(ROOT/p)==h,f'stale scientific source: {p}'
    net_path=ROOT/data['network']['path']
    assert sha(net_path)==data['network']['sha256']
    network=_load_sumolib().net.readNet(str(net_path),withInternal=True)
    result=verify_data(data,network)
    checked=0
    if raw:
        for f in data['families']:
            for p,h in f['source_files'].items():
                assert sha(ROOT/p)==h
            route_file=next(ROOT/p for p in f['source_files'] if p.endswith('/vehicle_routes.xml'))
            vehicles={v.attrib['id']:v for v in ET.parse(route_file).getroot().findall('vehicle')}
            for s in f['sessions']:
                v=vehicles[s['session_id']]
                assert v.attrib==f['actual_vehicles'][s['session_id']]
                assert v.find('route').attrib['edges'].split()==s['route_edges']
            stops_file=next(ROOT/p for p in f['source_files'] if p.endswith('/stops.xml'))
            stops=ET.parse(stops_file).getroot().findall('stopinfo')
            for r in data['records']:
                if r['family_id']!=f['family_id'] or r['scenario']!='S2':
                    continue
                sid=r['session_ids'][0]
                trace=data['traces'][sid]
                for a,b in r['labels']['stop_intervals']:
                    assert any(s.attrib['id']==sid and s.attrib['lane']==trace[a]['lane_id'] and
                               float(s.attrib['started'])-2<=trace[a]['time_s']<=float(s.attrib['ended']) and
                               float(s.attrib['started'])<=trace[b]['time_s']<=float(s.attrib['ended'])+2 for s in stops)
            fcd=next(ROOT/p for p in f['source_files'] if p.endswith('/fcd.xml'))
            offsets=Counter()
            for _,step in ET.iterparse(fcd,events=('end',)):
                if step.tag!='timestep':
                    continue
                for v in step.findall('vehicle'):
                    a=v.attrib
                    sid=a['id']
                    p=data['traces'][sid][offsets[sid]]
                    assert p['time_s']==float(step.attrib['time'])
                    for key,xml in [('lon','x'),('lat','y'),('speed_m_s','speed'),('lane_pos_m','pos'),('angle_deg','angle')]:
                        assert p[key]==float(a[xml])
                    assert p['lane_id']==a['lane']
                    offsets[sid]+=1
                    checked+=1
                step.clear()
            assert all(offsets[s['session_id']]==len(data['traces'][s['session_id']]) for s in f['sessions'])
    return {**result,'raw_fcd_points_compared':checked,'scientific_sources_checked':len(data['source_sha256'])}


def scientific_payload(data):
    """Exclude only XML timestamps/paths and raw provenance, not observations."""
    keys=['schema','source','versions','source_sha256','traces','records','rejections','catalogue','summary']
    payload={key:data[key] for key in keys}
    payload['families']=[{k:v for k,v in f.items() if k not in {'sumo_command','sumo_stderr','source_files'}} for f in data['families']]
    payload['network']={k:data['network'][k] for k in ['osm_sha256','bbox_lon_lat']}
    return payload


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset',type=Path,default=DEFAULT)
    parser.add_argument('--raw',action='store_true')
    parser.add_argument('--compare',type=Path,help='Verify a separate rebuild and compare every scientific field')
    parser.add_argument('--reference',type=Path,help='Compare against a checksummed reference bundle without requiring its original XML caches')
    args=parser.parse_args()
    result=verify(args.dataset,args.raw)
    if args.compare:
        verify(args.compare,args.raw)
        assert scientific_payload(json.loads(args.dataset.read_text()))==scientific_payload(json.loads(args.compare.read_text()))
        result['scientific_replay_identical']=True
    if args.reference:
        assert sha(args.reference)==args.reference.with_suffix('.sha256').read_text().strip()
        reference=json.loads(args.reference.read_text())
        for path,expected in reference['source_sha256'].items():
            assert sha(ROOT/path)==expected
        assert scientific_payload(json.loads(args.dataset.read_text()))==scientific_payload(reference)
        result['scientific_reference_identical']=True
    print(json.dumps(result,indent=2))
