"""Fresh native SUMO fork/history cohort; no protection-score route selection.

The public reconstructed geometry is compiled into a new native network with
explicit internal turns. Its IDs are authoritative only for this new cohort.
Public calibration of BOTH possible routes chooses target-independent clocks;
private query choices are assigned afterwards. Original traces remain untouched.
"""
from pathlib import Path
from itertools import combinations
import argparse
import gzip
import hashlib
import json
import math
import random
import shutil
import subprocess
import xml.etree.ElementTree as ET
import numpy as np
import sumolib
from data.scenario_suite.mobility import parse_fcd,successors,predecessors

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'artifacts/datasets/future_controlled_20261005_v1'
WORK=Path('/private/tmp/trajectory-future-native-v1')
ORIGINAL_PUBLIC=Path('/private/tmp/trajectory-research-20261005-public-map/public_reconstructed.net.xml')
SEED=2026100517


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def save(path,payload):
    path=Path(path)
    if path.exists():raise FileExistsError(f'Preserve completed evidence: {path}')
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(payload,indent=2,allow_nan=False)+'\n')


def split(index):return 'train' if index<=12 else 'selection' if index<=18 else 'test'


def via_lanes(net,fork,target):
    connections=sorted(fork.getOutgoing()[target],key=lambda c:c.getViaLaneID())
    connection=next(c for c in connections if c.getFromLane().allows('passenger') and c.getToLane().allows('passenger'))
    lanes=[];lane_id=connection.getViaLaneID()
    for _ in range(8):
        if not lane_id:break
        lane=net.getLane(lane_id)
        if not lane.getEdge().getID().startswith(':'):break
        lanes.append(lane)
        outgoing=sorted(lane.getOutgoing(),key=lambda c:(c.getToLane().getEdge().getID()!=target.getID(),c.getToLane().getID()))
        if not outgoing:break
        nxt=outgoing[0];lane_id=nxt.getViaLaneID() or nxt.getToLane().getID()
    return lanes


def tangent(edge,first=True):
    s=np.asarray(edge.getShape(),dtype=float)
    a=s[min(2,len(s)-1)]-s[0] if first else s[-1]-s[max(0,len(s)-3)]
    return a/max(1e-9,np.linalg.norm(a))


def prefix(fork,rng):
    path=[fork];length=fork.getLength()
    for _ in range(18):
        if length>=700. and len(path)>=3:return list(reversed(path))
        choices=[e for e in predecessors(path[-1]) if e not in path and e.getLength()>=15.]
        if not choices:break
        rng.shuffle(choices)
        # Stable straightness choice after seeded shuffle; public only.
        edge=max(choices,key=lambda e:float(tangent(e,False)@tangent(path[-1],True)))
        path.append(edge);length+=edge.getLength()
    raise ValueError('no_public_prefix700m')


def tail(branch,past,rng):
    path=[branch];length=branch.getLength()
    for _ in range(24):
        if length>=1000. and len(path)>=3:return path
        choices=[e for e in successors(path[-1]) if e not in path and e not in past and e.getLength()>=15.]
        if not choices:break
        rng.shuffle(choices)
        forward=[e for e in choices if float(tangent(path[-1],False)@tangent(e,True))>-.5]
        choices=forward or choices
        edge=rng.choice(choices[:min(4,len(choices))]);path.append(edge);length+=edge.getLength()
    raise ValueError('no_public_tail1000m')


def route_proposal(net,fork,attempt):
    rng=random.Random(SEED+attempt*100)
    choices=[e for e in successors(fork) if sum(l.getLength() for l in via_lanes(net,fork,e))>=12.]
    pairs=[(a,b) for a,b in combinations(choices,2) if float(tangent(a)@tangent(b))<.6]
    rng.shuffle(pairs)
    common=prefix(fork,rng)
    for a,b in pairs[:16]:
        try:
            tails=[tail(e,common,rng) for e in (a,b)]
            destinations=[p[-1].getShape()[-1] for p in tails]
            if math.dist(*destinations)<600.:continue
            if any(sum(e.getLength() for e in common+t)>4000. for t in tails):continue
            choices=sorted(zip((a,b),tails),key=lambda v:v[0].getID())
            return {'fork_edge':fork.getID(),'common_route':[e.getID() for e in common],
                'routes':[[e.getID() for e in common+t] for _,t in choices],
                'public_choices':[{'edge_id':e.getID(),
                    'via_lane_ids':[l.getID() for l in via_lanes(net,fork,e)],
                    'via_shape_xy':[list(p) for l in via_lanes(net,fork,e) for p in l.getShape()],
                    'outgoing_shape_xy':[list(p) for p in e.getShape()],
                    'destination_xy':list(t[-1].getShape()[-1]),'destination_edge_id':t[-1].getID()}
                    for e,t in choices]}
        except ValueError:continue
    raise ValueError('no_public_separated_branches')


def simulate(net_path,network,specs,directory,seed):
    directory.mkdir(parents=True,exist_ok=True)
    root=ET.Element('routes')
    ET.SubElement(root,'vType',id='controlled',vClass='passenger',maxSpeed='8',speedFactor='1',
                  speedDev='0',sigma='0',length='4',minGap='2')
    for s in specs:
        v=ET.SubElement(root,'vehicle',id=s['session_id'],type='controlled',
            depart=str(s['depart_s']),departLane='0',departPos='5',departSpeed='8')
        ET.SubElement(v,'route',edges=' '.join(s['route_edges']))
    routes=directory/'routes.rou.xml';ET.ElementTree(root).write(routes,encoding='utf-8',xml_declaration=True)
    fcd,vehicle_routes=directory/'fcd.xml',directory/'vehicle_routes.xml'
    executable=Path(shutil.which('sumo') or Path(__import__('sys').executable).parent/'sumo')
    command=[str(executable),'--net-file',str(net_path),'--route-files',str(routes),
        '--seed',str(seed),'--begin','0','--end',str(max(s['depart_s'] for s in specs)+1500),
        '--step-length','1','--device.fcd.period','1','--fcd-output',str(fcd),
        '--fcd-output.geo','true','--fcd-output.attributes','id,x,y,speed,lane,pos,angle',
        '--vehroute-output',str(vehicle_routes),'--vehroute-output.write-unfinished','true',
        '--time-to-teleport','-1','--no-step-log','true','--no-warnings','true']
    result=subprocess.run(command,check=True,capture_output=True,text=True,timeout=180)
    traces=parse_fcd(fcd,network)
    vehicles={v.attrib['id']:v.attrib for v in ET.parse(vehicle_routes).getroot().findall('vehicle')}
    if set(traces)!=set(s['session_id'] for s in specs) or any(float(v.get('arrival',-1))<0 for v in vehicles.values()):
        raise ValueError('native_sumo_incomplete_trip')
    if any(float(v['arrival'])-float(v['depart'])>=1200 for v in vehicles.values()):
        raise ValueError('trip_exceeds_declared_isolation_gap')
    source={p.name:sha(p) for p in (routes,fcd,vehicle_routes)}
    return traces,{'command':command,'stderr':result.stderr,'files':source,'arrivals':vehicles}


def clocks_from_public_calibration(proposal,traces,network):
    relative=[]
    for branch in range(2):
        trace=traces[f'calibration_{branch}'];origin=trace[0]['time_s']
        relative.append([{**p,'time_s':p['time_s']-origin} for p in trace])
    fork=network.getEdge(proposal['fork_edge']);end=fork.getLength()-80.
    clocks=[{int(p['time_s']) for p in t if p['edge_id']==fork.getID() and p['lane_pos_m']<=end} for t in relative]
    common=sorted(clocks[0]&clocks[1])
    if not common:raise ValueError('no_common_prefork_clock')
    ambiguous=common[-1]
    internal=[{int(p['time_s']) for p in t if p['lane_id'] in proposal['public_choices'][b]['via_lane_ids']}
              for b,t in enumerate(relative)]
    overlap=sorted(internal[0]&internal[1])
    if not overlap:raise ValueError('no_common_internalturn_clock')
    visible=overlap[len(overlap)//2]
    # Exact raw shared-prefix equality is an explicit control eligibility rule
    # on public calibration, not a protection/attacker result filter.
    first={int(p['time_s']):p for p in relative[0] if p['time_s']<=ambiguous}
    other={int(p['time_s']):p for p in relative[1] if p['time_s']<=ambiguous}
    if set(first)!=set(other) or any(first[t]['lat']!=other[t]['lat'] or first[t]['lon']!=other[t]['lon'] for t in first):
        raise ValueError('public_calibration_prefix_not_identical')
    return {'shared_fork_t':ambiguous,'turn_visible_t':visible,
            'internal_overlap_s':len(overlap),'raw_shared_prefix_identical':True}


def build(output=OUT,work=WORK):
    output,work=Path(output),Path(work)
    if (output/'dataset.json.gz').exists():raise FileExistsError('Preserve fresh cohort evidence')
    output.mkdir(parents=True,exist_ok=True);work.mkdir(parents=True,exist_ok=True)
    protocol={'schema':'native-future-cohort-protocol-v1','seed':SEED,'family_count':24,
        'splits':{'train':[f'native-{i:02d}' for i in range(1,13)],
                  'selection':[f'native-{i:02d}' for i in range(13,19)],
                  'test':[f'native-{i:02d}' for i in range(19,25)]},
        'public_planning':'all public fork edges>=180m; two native internal-turn choices>=12m; '
            'direction dot<.6; shared prefix>=700m; branch tails>=1000m; endpoints separated>=600m',
        'clock_calibration':'both alternative public routes before private choice; last shared fork sample80m before end, '
            'middle clock shared by both native internal turn traversals; reject nonidentical prefork calibration',
        'history':'six trips,5routine1rare; routine branch selected independently per family',
        'querydays':'day7/day8 one routine one rare, privately randomized order; no dayindex in attacker API',
        'target_priors':'balanced query diagnostic; historyfrequency5/6 is not test target prior',
        'private_choice_rng':'evaluator_only, separate from public route planner',
        'isolation':'departgap1500seconds; sigma0 speed8; no cross-vehicle traffic interactions',
        'eligibility':'public map/calibration only, before any protection run or attack score',
        'scope':'new controlled native SUMO map; exact native road IDs for this cohort, not original Beijing network'}
    save(output/'protocol.json',protocol)
    net_path=work/'native_cohort.net.xml'
    executable=Path(shutil.which('netconvert') or Path(__import__('sys').executable).parent/'netconvert')
    command=[str(executable),'--sumo-net-file',str(ORIGINAL_PUBLIC),'--output-file',str(net_path),
             '--no-internal-links','false','--junctions.limit-turn-speed','8','--no-warnings','true']
    compiled=subprocess.run(command,check=True,capture_output=True,text=True,timeout=120)
    network=sumolib.net.readNet(str(net_path),withInternal=True)
    forks=sorted([e for e in network.getEdges(withInternal=False) if e.allows('passenger') and
                  e.getLength()>=180. and len(successors(e))>=2],key=lambda e:e.getID())
    random.Random(SEED).shuffle(forks)
    families=[];public_calibrations=[];rejections=[];traces={}
    for attempt,fork in enumerate(forks[:160]):
        if len(families)==24:break
        try:
            proposal=route_proposal(network,fork,attempt)
            specs=[{'session_id':f'calibration_{b}','route_edges':route,'depart_s':b*1500.}
                   for b,route in enumerate(proposal['routes'])]
            calibration,run=simulate(net_path,network,specs,work/f'calibration_{attempt:03d}',SEED+attempt)
            clocks=clocks_from_public_calibration(proposal,calibration,network)
        except (ValueError,subprocess.CalledProcessError,subprocess.TimeoutExpired) as exc:
            rejections.append({'public_fork':fork.getID(),'attempt':attempt,'reason':str(exc)});continue
        index=len(families)+1;family=f'native-{index:02d}'
        # Private synthetic choices never enter context/features/road construction.
        hidden=random.Random(SEED+100000+index);routine=hidden.randrange(2)
        query=[routine,1-routine];hidden.shuffle(query)
        pattern=[routine,routine,1-routine,routine,routine,routine]+query
        sessions=[{'session_id':f'{family}_day{day}','day':day,'depart_s':(day-1)*1500.,
            'route_edges':proposal['routes'][branch],'choice_index':branch,
            'destination_role':'routine' if branch==routine else 'rare'} for day,branch in enumerate(pattern,1)]
        actual,actual_run=simulate(net_path,network,sessions,work/family,SEED+index)
        for s in sessions:
            trace=actual[s['session_id']];origin=trace[0]['time_s']
            for p in trace:p['time_s']-=origin
            for stage,t in [('shared_fork',clocks['shared_fork_t']),('turn_visible',clocks['turn_visible_t'])]:
                p=next((p for p in trace if p['time_s']==t),None)
                if p is None:raise ValueError('native_clock_not_in_actual_query_trace')
                if stage=='turn_visible' and p['lane_id'] not in proposal['public_choices'][s['choice_index']]['via_lane_ids']:
                    raise ValueError('native_query_no_longer_in_declared_internalturn')
        traces.update(actual)
        families.append({'family_id':family,'split':split(index),'public_attempt':attempt,
            'fork_edge':fork.getID(),'clocks':clocks,'public_context':{'choices':proposal['public_choices']},
            'evaluator_only':{'routine_choice':routine,'sessions':sessions,
                              'actual_run':actual_run,'calibration_run':run}})
        public_calibrations.append({'family_id':family,'proposal':proposal,'clocks':clocks,
            'calibration_fcd':calibration,'source':run})
        print('Native cohort accepted',family,'fork',fork.getID(),'clocks',clocks,flush=True)
    if len(families)!=24:raise RuntimeError(f'Only{len(families)}/24 eligible public forks; no defender score examined')
    compressed=output/'public_native.net.xml.gz'
    with compressed.open('wb') as f:f.write(gzip.compress(net_path.read_bytes(),mtime=0))
    pins={'artifacts/benchmarks/paper_benchmark/results.json':sha(ROOT/'artifacts/benchmarks/paper_benchmark/results.json'),
          'experiments/public_research_resources.py':sha(ROOT/'experiments/public_research_resources.py'),
          'data/scenario_suite/mobility.py':sha(ROOT/'data/scenario_suite/mobility.py'),
          str(Path(__file__).relative_to(ROOT)):sha(Path(__file__))}
    payload={'schema':'fresh-native-future-sumo-cohort-v1','source':'native SUMO1Hz FCD, publicly reconstructed map; synthetic intent/destination choices',
        'protocol_sha256':sha(output/'protocol.json'),'source_sha256':pins,
        'network':{'compressed_path':str(compressed.relative_to(ROOT)),'compressed_sha256':sha(compressed),
            'native_sha256':sha(net_path),'compile_command':command,'compile_stdout':compiled.stdout,
            'external_edges':len(network.getEdges(withInternal=False)),
            'internal_edges':sum(e.getID().startswith(':') for e in network.getEdges()),
            'source_reconstructed_network_sha256':sha(ORIGINAL_PUBLIC),
            'turns_authoritative_for_this_new_network':True,'original_lane_turn_parity':False},
        'families':families,'traces':traces,'public_calibrations':public_calibrations,
        'public_eligibility_rejections':rejections,'family_count':len(families),'session_count':len(traces)}
    target=output/'dataset.json.gz'
    target.write_bytes(gzip.compress(json.dumps(payload,separators=(',',':'),allow_nan=False).encode(),mtime=0))
    save(output/'manifest.json',{'dataset_sha256':sha(target),'family_count':len(families),'session_count':len(traces),
        'network':payload['network'],'protocol_sha256':payload['protocol_sha256'],'source_sha256':pins,
        'public_eligibility_rejection_count':len(rejections)})
    print('Saved native cohort',target,len(traces),'unmodified native FCD traces',flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,default=OUT)
    parser.add_argument('--workdir',type=Path,default=WORK)
    args=parser.parse_args();build(args.output,args.workdir)
