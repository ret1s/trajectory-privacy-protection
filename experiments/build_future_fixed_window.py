"""Native parked SUMO history with a public fixed observation window.

Preserves v1 traces. Real SUMO stops keep each vehicle on its destination lane
until650s, so the public 0..600s history window exists for both route choices.
No GPS interpolation/extension is used and no protection score is consulted.
"""
from pathlib import Path
import gzip
import json
import shutil
import subprocess
import sys
import xml.etree.ElementTree as ET
import sumolib
from data.scenario_suite.mobility import parse_fcd
from experiments.build_future_sumo_cohort import ROOT,SEED,sha,save

SOURCE=ROOT/'artifacts/datasets/future_controlled_20261005_v1'
OUT=ROOT/'artifacts/datasets/future_controlled_20261005_v2'
WORK=Path('/private/tmp/trajectory-future-native-fixed-v2')


def simulate(net_path,network,specs,directory,seed):
    directory.mkdir(parents=True,exist_ok=True)
    xml=ET.Element('routes')
    ET.SubElement(xml,'vType',id='controlled',vClass='passenger',maxSpeed='8',speedFactor='1',
                  speedDev='0',sigma='0',length='4',minGap='2')
    for s in specs:
        depart=s['depart_s']
        vehicle=ET.SubElement(xml,'vehicle',id=s['session_id'],type='controlled',
            depart=str(depart),departLane='0',departPos='5',departSpeed='8')
        ET.SubElement(vehicle,'route',edges=' '.join(s['route_edges']))
        end=network.getEdge(s['route_edges'][-1]).getLanes()[0]
        ET.SubElement(vehicle,'stop',lane=end.getID(),endPos=str(end.getLength()-5),
                      until=str(depart+650),parking='false')
    routes=directory/'routes.rou.xml';ET.ElementTree(xml).write(routes,encoding='utf-8',xml_declaration=True)
    fcd,vehicles=directory/'fcd.xml',directory/'vehicle_routes.xml'
    exe=Path(shutil.which('sumo') or Path(sys.executable).parent/'sumo')
    argv=[str(exe),'--net-file',str(net_path),'--route-files',str(routes),'--seed',str(seed),
        '--begin','0','--end',str(max(s['depart_s'] for s in specs)+1400),
        '--step-length','1','--device.fcd.period','1','--fcd-output',str(fcd),
        '--fcd-output.geo','true','--fcd-output.attributes','id,x,y,speed,lane,pos,angle',
        '--vehroute-output',str(vehicles),'--vehroute-output.write-unfinished','true',
        '--time-to-teleport','-1','--no-step-log','true','--no-warnings','true']
    result=subprocess.run(argv,check=True,capture_output=True,text=True,timeout=180)
    traces=parse_fcd(fcd,network)
    arrivals={v.attrib['id']:v.attrib for v in ET.parse(vehicles).getroot().findall('vehicle')}
    assert set(traces)==set(s['session_id'] for s in specs)
    assert all(0<=float(a['arrival'])-float(a['depart'])<1400 for a in arrivals.values())
    for trace in traces.values():
        start=trace[0]['time_s']
        for p in trace:p['time_s']-=start
        point=next(p for p in trace if p['time_s']==600.)
        assert point['speed_m_s']==0.,'600s public history endpoint must be native parked FCD'
    return traces,{'command':argv,'stderr':result.stderr,'arrivals':arrivals,
                  'source_files':{p.name:sha(p) for p in (routes,fcd,vehicles)}}


def main():
    if (OUT/'dataset.json.gz').exists():raise FileExistsError('Preserve native fixed-window evidence')
    OUT.mkdir(parents=True,exist_ok=True);WORK.mkdir(parents=True,exist_ok=True)
    original=json.loads(gzip.decompress((SOURCE/'dataset.json.gz').read_bytes()))
    save(OUT/'protocol.json',{'schema':'native-fixed-history-window-protocol-v1',
        'source_cohort_sha256':sha(SOURCE/'dataset.json.gz'),
        'before_any_attack_score':True,'same_public_forks_choices_and_family_splits':True,
        'native_dynamics':'SUMO sigma0 speed8; real destination-lane stop until650s',
        'public_history_window_s':[0.,600.],'history_clock':'0,20,...,600 same for every history',
        'query_clocks':'same public two-route calibration fork/turn clocks as v1, queryprefix only',
        'scope':'New native parked FCD, not edits/extensions of v1 GPS; existing sources immutable'})
    compressed=ROOT/original['network']['compressed_path'];assert sha(compressed)==original['network']['compressed_sha256']
    net_path=WORK/'native.net.xml';net_path.write_bytes(gzip.decompress(compressed.read_bytes()))
    assert sha(net_path)==original['network']['native_sha256']
    network=sumolib.net.readNet(str(net_path),withInternal=True)
    families=[];all_traces={}
    for i,f in enumerate(original['families'],1):
        specs=f['evaluator_only']['sessions']
        actual,run=simulate(net_path,network,specs,WORK/f['family_id'],SEED+i)
        for s in specs:
            trace=actual[s['session_id']]
            for stage,t in [('shared_fork',f['clocks']['shared_fork_t']),('turn_visible',f['clocks']['turn_visible_t'])]:
                point=next(p for p in trace if p['time_s']==t)
                if stage=='turn_visible':assert point['lane_id'] in f['public_context']['choices'][s['choice_index']]['via_lane_ids']
        query=[actual[s['session_id']] for s in specs if s['day'] in (7,8)]
        t=f['clocks']['shared_fork_t']
        a=[(p['lat'],p['lon']) for p in query[0] if p['time_s']<=t]
        b=[(p['lat'],p['lon']) for p in query[1] if p['time_s']<=t]
        assert a==b,'No private-route adaptive observation clock'
        families.append({**{k:v for k,v in f.items() if k!='evaluator_only'},
            'evaluator_only':{**f['evaluator_only'],'actual_run':run}})
        all_traces.update(actual)
        print('Native fixed-window FCD',f['family_id'],len(actual),'trips; parked at600s',flush=True)
    payload={'schema':'fresh-native-fixed-history-cohort-v1','protocol_sha256':sha(OUT/'protocol.json'),
        'source_cohort_sha256':sha(SOURCE/'dataset.json.gz'),
        'source_sha256':{str(Path(__file__).relative_to(ROOT)):sha(Path(__file__)),
                        'data/scenario_suite/mobility.py':sha(ROOT/'data/scenario_suite/mobility.py')},
        'network':original['network'],'families':families,'traces':all_traces,
        'family_count':len(families),'session_count':len(all_traces),
        'public_calibration_source':str((SOURCE/'dataset.json.gz').relative_to(ROOT)),
        'scope':'Native SUMO parked destination history, public fixed600s observation window; not original Beijing network parity'}
    path=OUT/'dataset.json.gz';path.write_bytes(gzip.compress(json.dumps(payload,separators=(',',':'),allow_nan=False).encode(),mtime=0))
    save(OUT/'manifest.json',{'dataset_sha256':sha(path),'protocol_sha256':payload['protocol_sha256'],
        'source_cohort_sha256':payload['source_cohort_sha256'],'source_sha256':payload['source_sha256'],
        'family_count':len(families),'session_count':len(all_traces),'network':payload['network']})
    print('Saved fixed-window native cohort',path,flush=True)


if __name__=='__main__':main()
