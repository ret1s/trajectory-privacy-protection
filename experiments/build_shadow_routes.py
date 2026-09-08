"""Generate auxiliary SUMO data, separately verify it before a registry import."""
import argparse
from copy import deepcopy
import json
from pathlib import Path
import xml.etree.ElementTree as ET

from data.shadow_routes import PROFILES, plans, records, summarize
from data.scenario_suite.mobility import route_xml, parse_fcd
from data.scenario_store.store import validate_bundle
from data.sumo_demo import resolve_sumo_toolchain, _sumo_environment, _run, _version, _sha256, _load_sumolib
from experiments.run_service_cover import write

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / 'artifacts/datasets/urban_shadow_v1'
WORK = ROOT / 'cache/urban_shadow_v1'
SOURCE_PATHS = ('data/shadow_routes.py', 'experiments/build_shadow_routes.py',
                'data/scenario_suite/mobility.py', 'data/sumo_demo.py',
                'thesis/notes/expanded_shadow_protocol.md',
                'thesis/notes/shadow_route_quality_amendment.md')


def build(output=OUTPUT, work=WORK):
    output, work = Path(output), Path(work)
    if (output / 'dataset.json').exists() or work.exists():
        raise FileExistsError('Use fresh data and native SUMO destinations')
    base_path = ROOT / 'artifacts/datasets/urban_scenarios_v3/dataset.json'
    base = json.loads(base_path.read_text())
    net_path = ROOT / base['network']['path']
    assert _sha256(net_path) == base['network']['sha256']
    network = _load_sumolib().net.readNet(str(net_path), withInternal=True)
    families = plans(network)
    tc = resolve_sumo_toolchain()
    env = _sumo_environment(tc)
    bundle = {'schema': 'urban-scenario-suite-v2', 'purpose': 'auxiliary_shadow_training_and_holdout_only',
        'source': 'SUMO+OSM_only; synthetic auxiliary identities',
        'base_dataset_sha256': _sha256(base_path), 'network': deepcopy(base['network']),
        'versions': {'sumo': _version(tc.sumo, env), 'netconvert': _version(tc.netconvert, env)},
        'families': families, 'traces': {}, 'records': [], 'rejections': [],
        'source_sha256': {p: _sha256(ROOT / p) for p in SOURCE_PATHS}}
    for f in families:
        directory = work / str(f['seed'])
        directory.mkdir(parents=True)
        route_path = directory / 'routes.rou.xml'
        xml = ET.fromstring(route_xml(f))
        for stop in xml.iter('stop'):
            # Auxiliary scope is on-lane stops. SUMO parking=true can move FCD
            # off the lane while retaining its lane ID and longitudinal pos.
            stop.set('parking', 'false')
        route_path.write_text(ET.tostring(xml, encoding='unicode'))
        fcd, vr, stops = [directory / p for p in ('fcd.xml', 'vehicle_routes.xml', 'stops.xml')]
        argv = [str(tc.sumo), '--net-file', str(net_path), '--route-files', str(route_path),
            '--seed', str(f['seed']), '--begin', '0', '--end', '7000', '--step-length', '1',
            '--device.fcd.period', '1', '--fcd-output', str(fcd), '--fcd-output.geo', 'true',
            '--fcd-output.attributes', 'id,x,y,speed,lane,pos,angle',
            '--vehroute-output', str(vr), '--vehroute-output.write-unfinished', 'true',
            '--stop-output', str(stops), '--time-to-teleport', '-1', '--no-step-log', 'true']
        result = _run(argv, environment=env)
        traces = parse_fcd(fcd, network)
        actual = {v.attrib['id']: dict(v.attrib) for v in ET.parse(vr).getroot().findall('vehicle')}
        expected = {s['session_id'] for s in f['sessions']}
        if set(traces) != expected or set(actual) != expected or any(float(v.get('arrival', -1)) < 0 for v in actual.values()):
            raise ValueError(f"Incomplete SUMO family {f['seed']}; do not silently discard it")
        paths = (route_path, fcd, vr, stops)
        sources = {str(p.relative_to(ROOT)) if p.is_relative_to(ROOT) else str(p): _sha256(p) for p in paths}
        f.update(actual_vehicles=actual, source_files=sources,
            simulation_runs=[{'session_ids': sorted(expected), 'command': argv, 'stderr': result.stderr, 'source_files': sources}])
        bundle['traces'].update(traces)
        bundle['records'].extend(records(f, traces))
        print(f"SUMO {f['seed']}: {sum(map(len, traces.values()))} FCD points", flush=True)
    bundle['summary'] = summarize(bundle)
    bundle['catalogue'] = [{'case_id': 'AUX.' + p, 'scenario': 'AUX',
        'generated_records': 80, 'attack_evaluated': False, 'protection_evaluated': False,
        'data_status': 'generated', 'interpretation': 'auxiliary_profile_not_core_threat_case'} for p in PROFILES]
    validate_bundle(bundle)
    write(output / 'dataset.json', bundle)
    write(output / 'summary.json', bundle['summary'])
    print(json.dumps(bundle['summary'], indent=2), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=OUTPUT)
    parser.add_argument('--workdir', type=Path, default=WORK)
    a = parser.parse_args()
    build(a.output, a.workdir)
