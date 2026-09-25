"""Four new SUMO families after freezing the public category plans.

Separate builder preserves historical source hashes of the expanded dataset.
"""
import argparse
from collections import Counter
from copy import deepcopy
from pathlib import Path
import xml.etree.ElementTree as ET

from data.scenario_suite.mobility import route_xml, parse_fcd
from data.scenario_suite_v2 import SCHEMA
from data.scenario_suite_v2.design import plan, rarity_context
from data.scenario_suite_v2.records import build_records, catalogue
from data.sumo_demo import resolve_sumo_toolchain, _sumo_environment, _run, _version, _load_sumolib
from experiments.build_scenario_suite_v3 import summary
from experiments.run_service_cover import ROOT, read, write, sha

OUTPUT = ROOT/'artifacts/datasets/research_loop_confirmation_v1'
PROTOCOL = ROOT/'artifacts/benchmarks/research_loop/iteration30_protocol.json'
SPLITS = {s: 'new_family_check' for s in read(PROTOCOL)['family_seeds']}


def build(output=OUTPUT, work=ROOT/'cache/research_loop_20260924/confirmation_simulations'):
    output, work = Path(output), Path(work)
    if (output/'dataset.json').exists():
        raise FileExistsError('Existing evidence is immutable')
    if work.exists():
        raise FileExistsError('Inspect the previous process/results before restarting a partial simulation')
    resources=read(ROOT/'cache/research_loop_20260924/resources.json')
    net_path=ROOT/'cache/research_loop_20260924/beijing.net.xml'
    assert sha(net_path)==resources['net_sha256']
    network=_load_sumolib().net.readNet(str(net_path),withInternal=True)
    context=rarity_context(resources['pois_used'])
    tc=resolve_sumo_toolchain();env=_sumo_environment(tc)
    sources={'experiments/build_research_loop_confirmation.py','artifacts/benchmarks/research_loop/iteration30_protocol.json','artifacts/benchmarks/research_loop/iteration29_plans.json','experiments/build_research_loop_dataset.py','data/scenario_suite_v2/design.py',
        'data/scenario_suite_v2/records.py','data/scenario_suite/mobility.py',
        'data/scenario_suite/records.py','data/sumo_demo.py'}
    bundle={'schema':SCHEMA,'purpose':'frozen_plan_new_family_check; same city and generator, not final thesis confirmation',
        'source':'SUMO+OSM_only; synthetic labels',
        'versions':{'sumo':_version(tc.sumo,env),'netconvert':_version(tc.netconvert,env)},
        'network':{'path':str(net_path.relative_to(ROOT)),'sha256':sha(net_path),
            'osm_sha256':resources['osm_sha256'],'bbox_lon_lat':[116.29,39.96,116.36,40.02]},
        'public_poi_context':context,'families':[],'traces':{},'records':[],'rejections':[],
        'source_sha256':{p:sha(ROOT/p) for p in sorted(sources)}}
    for seed, split in SPLITS.items():
        print(f'Simulating fixed family {seed} ({split})', flush=True)
        design = plan(network, seed, context)
        traces, actual, source_files, runs = {}, {}, {}, []
        for day in range(9):
            sessions = [s for s in design['sessions'] if s['day_index'] == day]
            directory = work/f'seed_{seed}'/f'day_{day}'
            directory.mkdir(parents=True, exist_ok=True)
            routes = directory/'routes.rou.xml'
            xml = ET.fromstring(route_xml({'sessions': sessions}))
            for stop in xml.iter('stop'): stop.set('parking', 'false')
            routes.write_text(ET.tostring(xml, encoding='unicode'), encoding='utf-8')
            fcd, vr, stops = [directory/n for n in ('fcd.xml', 'vehicle_routes.xml', 'stops.xml')]
            begin = 0 if day == 0 else min(s['depart_s'] for s in sessions)
            end = max(s['depart_s'] for s in sessions)+5000
            argv = [str(tc.sumo), '--net-file', str(net_path), '--route-files', str(routes),
                '--seed', str(seed+day*1000), '--begin', str(begin), '--end', str(end),
                '--step-length', '1', '--device.fcd.period', '1', '--fcd-output', str(fcd),
                '--fcd-output.geo', 'true', '--fcd-output.attributes', 'id,x,y,speed,lane,pos,angle',
                '--vehroute-output', str(vr), '--vehroute-output.write-unfinished', 'true',
                '--stop-output', str(stops), '--lanechange.duration', '3', '--time-to-teleport', '-1', '--no-step-log', 'true']
            result = _run(argv, environment=env)
            daily = parse_fcd(fcd, network)
            assert not traces.keys() & daily.keys()
            traces.update(daily)
            actual.update({v.attrib['id']: dict(v.attrib) for v in ET.parse(vr).getroot().findall('vehicle')})
            files = {str(p.relative_to(ROOT)): sha(p) for p in (routes, fcd, vr, stops)}
            source_files.update(files)
            runs.append({'day': day, 'session_ids': [s['session_id'] for s in sessions],
                'command': argv, 'stderr': result.stderr, 'source_files': files})
        assert {s['session_id'] for s in design['sessions']} <= traces.keys()
        completed = {sid for sid, a in actual.items() if float(a.get('arrival', -1)) >= 0}
        if len(completed) != 22:
            raise ValueError(f'Family {seed}: only {len(completed)}/22 completed; do not silently discard failed trips')
        records, rejected = build_records(design, traces, completed, network, split, context)
        design.update(split=split, actual_vehicles=actual, simulation_runs=runs, source_files=source_files)
        bundle['families'].append(design); bundle['traces'].update(traces)
        bundle['records'].extend(records)
        bundle['rejections'].extend({'seed': seed, **r} for r in rejected)
        print(f'  {len(completed)}/22 arrived; {len(records)} records; '
              f'{len({r["case_id"] for r in records})}/30 subcases', flush=True)
    counts = Counter(r['case_id'] for r in bundle['records'])
    bundle['catalogue'] = [{**c, 'generated_records': counts[c['case_id']],
        'data_status': 'generated' if counts[c['case_id']] else 'specified_not_generated'} for c in catalogue()]
    bundle['summary'] = summary(bundle)
    write(output/'dataset.json', bundle); write(output/'summary.json', bundle['summary'])
    print(bundle['summary'], flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', type=Path, default=OUTPUT)
    p.add_argument('--workdir', type=Path, default=ROOT/'cache/research_loop_20260924/confirmation_simulations')
    a = p.parse_args(); build(a.output, a.workdir)
