"""Preserve development families; add four fresh SUMO confirmation families."""
import argparse
from collections import Counter
from copy import deepcopy
import json
from pathlib import Path
import xml.etree.ElementTree as ET

from data.scenario_suite.mobility import route_xml, parse_fcd
from data.scenario_suite_v2 import SCHEMA
from data.scenario_suite_v2.design import plan, rarity_context
from data.scenario_suite_v2.records import build_records, catalogue
from data.sumo_demo import resolve_sumo_toolchain, _sumo_environment, _run, _version, _sha256, _load_sumolib

ROOT=Path(__file__).resolve().parents[1]
BASE=ROOT/'artifacts/datasets/urban_scenarios_v2/dataset.json'
LANE=ROOT/'artifacts/benchmarks/lane_comparison/results.json'
SPLITS={201:'confirmation',202:'confirmation',203:'confirmation',204:'confirmation'}


def summary(bundle):
    counts=Counter(r['case_id'] for r in bundle['records'])
    return {'families':len(bundle['families']),'sessions':len(bundle['traces']),
            'raw_fcd_samples':sum(map(len,bundle['traces'].values())),
            'scenario_records':len(bundle['records']),'generated_subcases':len(counts),
            'declared_subcases':len(catalogue()),
            'generated_scenarios':len({r['scenario'] for r in bundle['records']}),
            'by_scenario':dict(Counter(r['scenario'] for r in bundle['records'])),
            'by_case':dict(sorted(counts.items()))}


def build(output,work):
    output,work=Path(output).resolve(),Path(work).resolve()
    if (output/'dataset.json').exists():
        raise FileExistsError('Use a fresh dataset destination; existing evidence is immutable')
    old=json.loads(BASE.read_text())
    previous=json.loads(LANE.read_text())
    net_path=ROOT/old['network']['path']
    assert _sha256(net_path)==old['network']['sha256']==previous['network_sha256']
    assert _sha256(ROOT/previous['service']['osm_path'])==previous['service']['osm_sha256']
    network=_load_sumolib().net.readNet(str(net_path),withInternal=True)
    context=rarity_context(previous['service']['pois_used'])
    tc=resolve_sumo_toolchain()
    env=_sumo_environment(tc)
    bundle={'schema':SCHEMA,'purpose':'scenario_challenge_dataset_not_protection_evidence',
            'source':'SUMO+OSM_only; synthetic identity, intent, activity and relationship labels',
            'base_dataset_sha256':_sha256(BASE),'public_context_source_sha256':_sha256(LANE),
            'versions':{'sumo':_version(tc.sumo,env),'netconvert':_version(tc.netconvert,env)},
            'network':deepcopy(old['network']),'public_poi_context':context,
            'families':[],'traces':{},'records':[],'rejections':[]}
    retained=[f for f in old['families'] if f['seed'] in (101,102,103,104)]
    ids={s['session_id'] for f in retained for s in f['sessions']}
    bundle['families']=deepcopy(retained)
    bundle['traces']={s:deepcopy(old['traces'][s]) for s in sorted(ids)}
    bundle['records']=[deepcopy(r) for r in old['records'] if set(r['session_ids']) <= ids]
    bundle['rejections']=[deepcopy(r) for r in old['rejections'] if r['seed'] in (101,102,103,104)]
    work.mkdir(parents=True,exist_ok=True)
    for seed,split in SPLITS.items():
        print(f'Planning and simulating family {seed} ({split})',flush=True)
        design=plan(network,seed,context)
        traces,actual,source_files,runs={}, {}, {}, []
        for day in range(9):
            sessions=[s for s in design['sessions'] if s['day_index']==day]
            directory=work/f'seed_{seed}'/f'day_{day}'
            directory.mkdir(parents=True,exist_ok=True)
            routes=directory/'routes.rou.xml'
            xml=ET.fromstring(route_xml({'sessions':sessions}))
            # A one-second route-enforcing waypoint is NOT off-road parking.
            # Parking there can displace FCD laterally for a single frame.
            for stop in xml.iter('stop'):
                if float(stop.attrib.get('duration',0))==1:
                    stop.set('parking','false')
            routes.write_text(ET.tostring(xml,encoding='unicode'),encoding='utf-8')
            fcd,vr,stops=[directory/name for name in ('fcd.xml','vehicle_routes.xml','stops.xml')]
            begin=0 if day==0 else min(s['depart_s'] for s in sessions)
            end=max(s['depart_s'] for s in sessions)+5000
            argv=[str(tc.sumo),'--net-file',str(net_path),'--route-files',str(routes),
                  '--seed',str(seed+day*1000),'--begin',str(begin),'--end',str(end),'--step-length','1',
                  '--device.fcd.period','1','--fcd-output',str(fcd),'--fcd-output.geo','true',
                  '--fcd-output.attributes','id,x,y,speed,lane,pos,angle',
                  '--vehroute-output',str(vr),'--vehroute-output.write-unfinished','true',
                  '--stop-output',str(stops),'--time-to-teleport','-1','--no-step-log','true']
            result=_run(argv,environment=env)
            daily=parse_fcd(fcd,network)
            if traces.keys() & daily.keys():
                raise ValueError('A session must not be simulated twice')
            traces.update(daily)
            actual.update({v.attrib['id']:dict(v.attrib) for v in ET.parse(vr).getroot().findall('vehicle')})
            paths=[routes,fcd,vr,stops]
            sources={str(p.relative_to(ROOT)) if p.is_relative_to(ROOT) else str(p):_sha256(p) for p in paths}
            source_files.update(sources)
            runs.append({'day':day,'session_ids':[s['session_id'] for s in sessions],
                         'command':argv,'stderr':result.stderr,'source_files':sources})
        missing={s['session_id'] for s in design['sessions']}-traces.keys()
        if missing:
            raise ValueError(f'No FCD for planned sessions: {sorted(missing)}')
        completed={sid for sid,a in actual.items() if float(a.get('arrival',-1))>=0}
        records,rejected=build_records(design,traces,completed,network,split,context)
        design.update(split=split,actual_vehicles=actual,simulation_runs=runs,source_files=source_files)
        bundle['families'].append(design)
        bundle['traces'].update(traces)
        bundle['records'].extend(records)
        bundle['rejections'].extend({'seed':seed,**r} for r in rejected)
        print(f'  {len(completed)}/{len(design["sessions"])} completed; {len(records)} records; '
              f'{len({r["case_id"] for r in records})}/30 cases',flush=True)
    counts=Counter(r['case_id'] for r in bundle['records'])
    bundle['catalogue']=[{**c,'generated_records':counts[c['case_id']],
                          'data_status':'generated' if counts[c['case_id']] else 'specified_not_generated'}
                         for c in catalogue()]
    bundle['summary']=summary(bundle)
    sources=set(old['source_sha256']) | {'evaluation/scenario_metrics.py',
            'thesis/notes/service_cover_protocol.md','experiments/build_scenario_suite_v3.py'}
    sources.update(str(p.relative_to(ROOT)) for p in (ROOT/'data/scenario_suite_v2').glob('*.py'))
    bundle['source_sha256']={p:_sha256(ROOT/p) for p in sorted(sources)}
    output.mkdir(parents=True,exist_ok=True)
    path=output/'dataset.json'
    path.write_text(json.dumps(bundle,ensure_ascii=False,separators=(',',':')))
    path.with_suffix('.sha256').write_text(_sha256(path)+'\n')
    (output/'summary.json').write_text(json.dumps(bundle['summary'],indent=2)+'\n')
    print(json.dumps(bundle['summary'],indent=2),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,default=ROOT/'artifacts/datasets/urban_scenarios_v3')
    parser.add_argument('--workdir',type=Path,default=ROOT/'cache/scenario_suite_v3')
    args=parser.parse_args()
    build(args.output,args.workdir)

