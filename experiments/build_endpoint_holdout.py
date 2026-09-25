"""New, predeclared SUMO families; never filter by privacy or utility outcomes."""
from collections import Counter
from pathlib import Path
import json
import xml.etree.ElementTree as ET

from data.scenario_suite.mobility import route_xml,parse_fcd
from data.scenario_suite_v2 import SCHEMA
from data.scenario_suite_v2.design import plan,rarity_context
from data.scenario_suite_v2.records import build_records,catalogue
from data.sumo_demo import resolve_sumo_toolchain,_sumo_environment,_run,_version,_load_sumolib
from experiments.build_scenario_suite_v3 import summary
from experiments.run_live_paper_comparison import write_json
from experiments.research_loop_resources import ROOT,CACHE,sha

OUT=ROOT/'artifacts/benchmarks/endpoint_calendar_v1'
DATA=ROOT/'artifacts/datasets/endpoint_holdout_v1'


def build():
    protocol=json.loads((OUT/'protocol.json').read_text())
    amendment=json.loads((OUT/'construction_amendment.json').read_text())
    assert amendment['protocol_sha256']==sha(OUT/'protocol.json')
    assert amendment['revised_builder_sha256']==sha(Path(__file__))
    if (DATA/'dataset.json').exists():raise FileExistsError('Do not overwrite held-out evidence')
    work=CACHE/'endpoint_holdout_simulations'
    resources=json.loads((CACHE/'resources.json').read_text());net_path=CACHE/'beijing.net.xml'
    assert sha(net_path)==resources['net_sha256']
    network=_load_sumolib().net.readNet(str(net_path),withInternal=True)
    context=rarity_context(resources['pois_used']);tc=resolve_sumo_toolchain();env=_sumo_environment(tc)
    source_files=['data/scenario_suite/mobility.py','data/scenario_suite/records.py',
                  'data/scenario_suite_v2/design.py','data/scenario_suite_v2/records.py']
    bundle={'schema':SCHEMA,'purpose':'new families for endpoint/calendar check; same city and generator',
        'protocol_sha256':sha(OUT/'protocol.json'),'source':'SUMO+OSM_only; synthetic endpoints, not homes',
        'source_sha256':{f:sha(ROOT/f) for f in source_files},
        'versions':{'sumo':_version(tc.sumo,env),'netconvert':_version(tc.netconvert,env)},
        'network':{'path':str(net_path.relative_to(ROOT)),'sha256':sha(net_path),'osm_sha256':resources['osm_sha256']},
        'construction_amendment_sha256':sha(OUT/'construction_amendment.json'),
        'public_poi_context':context,'families':[],'traces':{},'records':[],'rejections':[],
        'planned_seeds':protocol['holdout_seeds'],'construction_failures':[]}
    for seed in protocol['holdout_seeds']:
        try:design=plan(network,seed,context)
        except ValueError as error:
            bundle['construction_failures'].append({'seed':seed,'error':str(error),'replacement':None})
            print(seed,'construction failed, no replacement:',error,flush=True)
            continue
        traces={};actual={};runs=[]
        for day in range(9):
            sessions=[s for s in design['sessions'] if s['day_index']==day]
            directory=work/f'seed_{seed}'/f'day_{day}';directory.mkdir(parents=True,exist_ok=True)
            routes=directory/'routes.rou.xml';xml=ET.fromstring(route_xml({'sessions':sessions}))
            for stop in xml.iter('stop'):stop.set('parking','false')
            route_text=ET.tostring(xml,encoding='unicode')
            if routes.exists():assert routes.read_text()==route_text
            else:routes.write_text(route_text)
            fcd,vr=[directory/n for n in ('fcd.xml','vehicle_routes.xml')]
            begin=0 if day==0 else min(s['depart_s'] for s in sessions)
            argv=[str(tc.sumo),'--net-file',str(net_path),'--route-files',str(routes),'--seed',str(seed+day*1000),
                '--begin',str(begin),'--end',str(max(s['depart_s'] for s in sessions)+5000),
                '--step-length','1','--device.fcd.period','1','--fcd-output',str(fcd),'--fcd-output.geo','true',
                '--fcd-output.attributes','id,x,y,speed,lane,pos,angle','--vehroute-output',str(vr),
                '--vehroute-output.write-unfinished','true','--lanechange.duration','3','--time-to-teleport','-1','--no-step-log','true']
            if fcd.exists() and vr.exists():
                # Both must be complete XML, not partial files from a crash.
                for path in (fcd,vr):ET.parse(path)
                stderr='Reused complete source files from the interrupted construction; routes verified equal'
            elif fcd.exists() or vr.exists():raise RuntimeError('Incomplete run requires explicit inspection')
            else:stderr=_run(argv,environment=env).stderr
            daily=parse_fcd(fcd,network)
            assert not traces.keys()&daily.keys();traces.update(daily)
            actual.update({v.attrib['id']:dict(v.attrib) for v in ET.parse(vr).getroot().findall('vehicle')})
            runs.append({'day':day,'command':argv,'stderr':stderr,
                         'source_files':{str(p.relative_to(ROOT)):sha(p) for p in (routes,fcd,vr)}})
        completed={sid for sid,v in actual.items() if float(v.get('arrival',-1))>=0}
        # Report completeness; do not replace failed seeds with easier ones.
        records,rejected=build_records(design,traces,completed,network,'endpoint_holdout',context)
        design.update(split='endpoint_holdout',actual_vehicles=actual,simulation_runs=runs)
        bundle['families'].append(design);bundle['traces'].update(traces);bundle['records'].extend(records)
        bundle['rejections'].extend({'seed':seed,**v} for v in rejected)
        print(seed,len(completed),'/22 completed;',len(records),'records',flush=True)
    counts=Counter(r['case_id'] for r in bundle['records'])
    bundle['catalogue']=[{**c,'generated_records':counts[c['case_id']]} for c in catalogue()]
    bundle['summary']=summary(bundle);write_json(DATA/'dataset.json',bundle);write_json(DATA/'summary.json',bundle['summary'])


if __name__=='__main__':build()
