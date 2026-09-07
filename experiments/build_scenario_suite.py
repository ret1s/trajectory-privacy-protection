"""Build a versioned ten-scenario development dataset, NOT a protection benchmark.

python -m experiments.build_scenario_suite --output artifacts/datasets/urban_scenarios_v1
Existing deliverables are never overwritten without --replace. Seeds and split
roles are fixed before simulation; all views of a related family stay together.
"""
import argparse
from collections import Counter
import json
from pathlib import Path
import xml.etree.ElementTree as ET

from data.scenario_suite import SCHEMA
from data.scenario_suite.catalog import catalogue
from data.scenario_suite.mobility import plan, route_xml, parse_fcd
from data.scenario_suite.records import build_records
from data.sumo_demo import (_existing_default_osm, resolve_sumo_toolchain, _sumo_environment,
                            _run, _version, _sha256, _load_sumolib, _command_arrays, SumoSmokeConfig)

ROOT = Path(__file__).resolve().parents[1]
SPLITS = {91: 'development_train', 92: 'development_validation', 93: 'development_test'}


def build(output, work, *, replace=False):
    output, work = Path(output).resolve(), Path(work).resolve()
    if (output/'dataset.json').exists() and not replace:
        raise FileExistsError('Dataset exists: use a new output directory, or explicit --replace')
    output.mkdir(parents=True,exist_ok=True)
    work.mkdir(parents=True,exist_ok=True)
    tc=resolve_sumo_toolchain()
    env=_sumo_environment(tc)
    osm=_existing_default_osm()
    commands,files=_command_arrays(tc,osm,work,SumoSmokeConfig())
    print('Building passenger network from OSM',flush=True)
    net_result=_run(commands['netconvert'],environment=env)
    network=_load_sumolib().net.readNet(str(files['network']),withInternal=True)
    bundle={'schema':SCHEMA,'purpose':'development_dataset_not_protection_evidence',
            'source':'SUMO+OSM_only; synthetic identity, intent and relationship labels',
            'versions':{'sumo':_version(tc.sumo,env),'netconvert':_version(tc.netconvert,env)},
            'network':{'path':str(files['network'].relative_to(ROOT)) if files['network'].is_relative_to(ROOT) else str(files['network']),
                       'sha256':_sha256(files['network']),'osm_sha256':_sha256(osm),
                       'bbox_lon_lat':[116.29,39.96,116.36,40.02],'netconvert_command':list(commands['netconvert']),
                       'netconvert_stderr':net_result.stderr},
            'families':[],'traces':{},'records':[],'rejections':[]}
    for seed,split in SPLITS.items():
        print(f'Running SUMO family {seed} ({split})',flush=True)
        design=plan(network,seed)
        seed_dir=work/f'seed_{seed}'
        seed_dir.mkdir(exist_ok=True)
        routes=seed_dir/'routes.rou.xml'
        routes.write_text(route_xml(design),encoding='utf-8')
        fcd,vr,stops = [seed_dir/name for name in ['fcd.xml','vehicle_routes.xml','stops.xml']]
        argv=[str(tc.sumo),'--net-file',str(files['network']),'--route-files',str(routes),
              '--seed',str(seed),'--begin','0','--end','10000','--step-length','1',
              '--device.fcd.period','1','--fcd-output',str(fcd),'--fcd-output.geo','true',
              '--fcd-output.attributes','id,x,y,speed,lane,pos,angle',
              '--vehroute-output',str(vr),'--vehroute-output.write-unfinished','true',
              '--stop-output',str(stops),'--time-to-teleport','-1','--no-step-log','true']
        result=_run(argv,environment=env)
        traces=parse_fcd(fcd,network)
        actual={v.attrib['id']:dict(v.attrib) for v in ET.parse(vr).getroot().findall('vehicle')}
        completed={vid for vid,a in actual.items() if float(a.get('arrival',-1))>=0}
        missing={s['session_id'] for s in design['sessions']}-traces.keys()
        if missing:
            raise ValueError(f'No FCD for planned sessions: {missing}')
        records,rejected=build_records(design,traces,completed,network,split)
        design.update({'split':split,'actual_vehicles':actual,'sumo_command':argv,'sumo_stderr':result.stderr,
                       'source_files':{str(p.relative_to(ROOT)) if p.is_relative_to(ROOT) else str(p):_sha256(p)
                                       for p in [routes,fcd,vr,stops]}})
        bundle['families'].append(design)
        bundle['traces'].update(traces)
        bundle['records'].extend(records)
        bundle['rejections'].extend({'seed':seed,**r} for r in rejected)
        print(f'  {len(completed)}/{len(design["sessions"])} completed; {len(records)} records',flush=True)
    counts=Counter(r['case_id'] for r in bundle['records'])
    bundle['catalogue']=[{**c,'generated_records':counts[c['case_id']],
                           'data_status':'generated' if counts[c['case_id']] else 'specified_not_generated'} for c in catalogue()]
    bundle['summary']={'families':len(bundle['families']),'sessions':len(bundle['traces']),
                       'raw_fcd_samples':sum(map(len,bundle['traces'].values())),
                       'scenario_records':len(bundle['records']),'generated_subcases':len(counts),
                       'declared_subcases':len(catalogue()),'generated_scenarios':len({r['scenario'] for r in bundle['records']}),
                       'by_scenario':dict(Counter(r['scenario'] for r in bundle['records'])),
                       'by_case':dict(sorted(counts.items()))}
    sources=list((ROOT/'data/scenario_suite').glob('*.py'))+[Path(__file__),ROOT/'data/sumo_demo.py']
    bundle['source_sha256']={str(p.relative_to(ROOT)):_sha256(p) for p in sources}
    target=output/'dataset.json'
    target.write_text(json.dumps(bundle,ensure_ascii=False,separators=(',',':')),encoding='utf-8')
    (output/'dataset.sha256').write_text(_sha256(target)+'\n',encoding='utf-8')
    (output/'summary.json').write_text(json.dumps(bundle['summary'],indent=2)+'\n',encoding='utf-8')
    print(json.dumps(bundle['summary'],indent=2),flush=True)
    return bundle


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,default=ROOT/'artifacts/datasets/urban_scenarios_v1')
    parser.add_argument('--workdir',type=Path,default=ROOT/'cache/scenario_suite_v1')
    parser.add_argument('--replace',action='store_true')
    args=parser.parse_args()
    build(args.output,args.workdir,replace=args.replace)
