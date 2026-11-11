"""Export an offline teaching replay from sealed GeoI-BQ evidence.

No model fitting, defense sampling, or modification of benchmark artifacts.
All 20 TEST families, both native sessions, and all three retained draws appear.
"""
from collections import defaultdict
from pathlib import Path
import argparse
import gzip
import hashlib
import json
import math
import shutil

import numpy as np

from benchmark.query_purpose import MultiPurposeRoadRanking, QueryPurpose, QuerySpec
from core.research_release import load_release
from experiments import paper_comparison_20261028 as study

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT/'artifacts/demos/geoi_bq_20261111_v1'
PRIMARY = study.OUT
ORIGINAL = ROOT/'artifacts/benchmarks/main_depth_20261014_v1'
EXTRA = ROOT/'artifacts/benchmarks/remaining_scenarios_20261109_v1'
FUSION = ROOT/'artifacts/benchmarks/s8_fusion_20261109_v2'
ASSETS = ROOT/'web/static/thesis_demo'
METHODS = study.METHODS
LABELS = ['Raw GPS','REM-Z K1','TransProtect*','Semantic correlation*','Fake-query insertion*','GeoI-BQ v1']


def read(path):
    content = Path(path).read_bytes()
    return json.loads(gzip.decompress(content) if str(path).endswith('.gz') else content)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    content = json.dumps(value, ensure_ascii=False, separators=(',',':'), allow_nan=False).encode()
    path.write_bytes(gzip.compress(content, mtime=0) if str(path).endswith('.gz') else content)


def authenticated_sources():
    release = load_release()
    audit = read(PRIMARY/'validation.json')
    assert audit['status'] == 'pass' and audit['results_sha256'] == sha(PRIMARY/'results.json')
    for name, digest in audit['files_sha256'].items():
        assert sha(PRIMARY/name) == digest, name
    protocol = read(PRIMARY/'protocol.json')
    for name, digest in protocol['inputs_sha256'].items():
        assert sha(ROOT/name) == digest, name
    for name, digest in protocol['original_families'].items():
        assert sha(ORIGINAL/'families'/name) == digest, name
    extra_audit, fusion_audit = read(EXTRA/'validation.json'), read(FUSION/'validation.json')
    assert extra_audit['status'] == fusion_audit['status'] == 'pass'
    assert extra_audit['result_sha256'] == sha(EXTRA/'results.json')
    assert fusion_audit['results_sha256'] == sha(FUSION/'results.json')
    return release, protocol


def small_task(task):
    def trim(value):
        if isinstance(value, dict):
            return {k:trim(v) for k,v in value.items() if k not in ('group_values','family_metrics')}
        if isinstance(value,list): return [trim(v) for v in value]
        return value
    return trim(task)


def export(output=OUT):
    output = Path(output)
    release, protocol = authenticated_sources()
    print('Authenticated frozen inputs; loading public map and catalogue', flush=True)
    rn, reference, reply, _ = study.resources()
    ranking = MultiPurposeRoadRanking(reference)
    resources = read(PRIMARY/'resources.json')
    assert rn.catalogue_sha256 == resources['catalogue_sha256']
    assert len(reply.pois) == 418 and reply.sha256 == resources['reply_sha256']
    origin = rn.xy.min(axis=0)

    def xy(point):
        return [float(point[0]-origin[0]),float(point[1]-origin[1])]

    def coords(latlon): return xy(rn.point_xy(*latlon))

    roads=[]
    # One passenger lane per external edge: a drawing simplification, not a new graph.
    seen=set()
    for lane, ids in sorted(rn.lane_indices.items()):
        edge = rn.graph.nodes[ids[0]]['edge_id']
        if edge.startswith(':') or edge in seen: continue
        seen.add(edge)
        roads.append([xy(rn.xy[i]) for i in ids])
    pois=[dict(index=i,id=p['id'],category=p['category'],xy=coords([p['lat'],p['lon']]))
          for i,p in enumerate(reply.pois)]
    categories=list(ranking.categories)
    write(output/'data/map.json.gz',dict(roads=roads,pois=pois,categories=categories,
        attribution='© OpenStreetMap contributors · ODbL. SUMO native geometry; one lane drawn per external edge.',
        projection='fixed local equirectangular metres; translated origin for drawing'))

    main=read(PRIMARY/'results.json')
    summary=dict(methods=[dict(id=m,label=label,description=protocol['labels'][m]) for m,label in zip(METHODS,LABELS)],
        privacy=[small_task(t) for t in main['attackers']],
        utility=[small_task(t) for t in main['utility']],cost=main['cost'],
        comparisons=main['comparisons'],coverage=main['coverage'],native_metrics=main['native_metrics'],
        remaining=[small_task(t) for t in read(EXTRA/'results.json')['tasks']],
        fusion=[small_task(t) for t in read(FUSION/'results.json')['tasks']],
        config=protocol['config'],scope=main['scope'],limitations=main['limitations'])
    write(output/'data/results.json.gz', summary)

    sources=[PRIMARY/'results.json', PRIMARY/'protocol.json', PRIMARY/'validation.json',
        EXTRA/'results.json',EXTRA/'validation.json',FUSION/'results.json',FUSION/'validation.json',
        ROOT/study.sealed.public.NETWORK, ROOT/study.sealed.public.RESOURCES,
        ROOT/'configs/geo_i_bq_v1.json',ROOT/'artifacts/reports/graduation_thesis.pdf']
    sessions=[]; checked_cells=0; draws=0
    for family_path in sorted((PRIMARY/'families').glob('*.json.gz')):
        family=read(family_path)
        if family['split']!='test': continue
        old_path=ORIGINAL/'families'/family_path.name
        source_by_id={t['session_id']:t for t in read(old_path)['trips']}
        sources += [family_path,old_path]
        grouped=defaultdict(list)
        for trip in family['trips']: grouped[trip['native_session_id']].append(trip)
        for session_id,trips in sorted(grouped.items()):
            first=trips[0]; truth=first['truth']; destination=truth[-1]['state']
            scores=[]; orders=[]
            for point in truth:
                score_row={}; order_row={}
                for purpose in QueryPurpose:
                    spec=QuerySpec(purpose,categories[0],k=5,
                        radius_m=1000. if purpose==QueryPurpose.WITHIN_RADIUS else None,
                        destination_state=destination if purpose==QueryPurpose.MIN_DETOUR else None)
                    values=ranking.scores(point['state'],spec)
                    score_row[purpose.value]=[float(v) if math.isfinite(v) else None for v in values]
                    order_row[purpose.value]={c:sorted([i for i,p in enumerate(reply.pois)
                        if p['category']==c and math.isfinite(values[i])],key=lambda i:(float(values[i]),reply.pois[i]['id']))
                        for c in categories}
                scores.append(score_row);orders.append(order_row)
            session=dict(id=session_id,family=family['family_id'],
                times=[p['t'] for p in truth],truth=[xy(p['xy']) for p in truth],
                scores=scores,orders=orders,destination=xy(rn.xy[destination]))
            write(output/f'data/sessions/{session_id}.json.gz',session)
            draw_ids=[]
            for trip in sorted(trips,key=lambda t:t['draw']):
                assert trip['truth']==truth
                saved=source_by_id[trip['session_id']]
                assert trip['events']['geoi_bq']==saved['events']['cache_target_g1']
                streams={m:[dict(t=e['timestamp_s'],q=[coords(p) for p in e['coordinates']],
                                  coordinates=e['coordinates']) for e in trip['events'][m]] for m in METHODS}
                masks={m:[{c:np.flatnonzero(study.unpacked(v,len(pois))).tolist() for c,v in row.items()}
                          for row in trip['masks'][m]] for m in METHODS}
                # Validate every displayed aggregate against the frozen utility, including N/A.
                for m in METHODS:
                    for tick,row in enumerate(masks[m]):
                        for cache,indices in row.items():
                            available=set(indices)
                            for purpose,by_category in orders[tick].items():
                                recalls=[];count=0
                                for full in by_category.values():
                                    answer=[i for i in full if i in available];count+=len(answer)
                                    if full: recalls.append(len(set(full[:5])&set(answer[:5]))/min(5,len(full)))
                                expected=trip['utility'][m][tick][cache][purpose]
                                actual=float(np.mean(recalls)) if recalls else None
                                assert actual==expected['recall5'] and count==expected['full_sorted_answer_count']
                                checked_cells+=1
                certificates=[]
                for c in saved['Q_certificates']['cache_target_g1']:
                    keys=('block','t','allocated_epsilon_Q','reserved_prefix_epsilon_Q','alpha',
                        'public_library_size','integer_scores','exact_likelihood_ratio_upper',
                        'score_status','actual_expected_utility_lower')
                    certificates.append({k:c[k] for k in keys if k in c})
                bundle=dict(id=trip['session_id'],native_session=session_id,draw=trip['draw'],
                    streams=streams,masks=masks,utility=trip['utility'],cost=trip['cost'],
                    anchors=[coords(p) for p in saved['GeoI_anchors']],ledger=saved['GeoI_ledger'],
                    certificates=certificates)
                write(output/f'data/runs/{trip["session_id"]}.json.gz',bundle)
                draw_ids.append(trip['session_id']);draws+=1
            sessions.append(dict(id=session_id,family=family['family_id'],draws=draw_ids))
        print('Exported',family['family_id'],flush=True)
    assert len(sessions)==40 and draws==120 and len({s['family'] for s in sessions})==20
    write(output/'data/index.json.gz',dict(schema='geoi-bq-teaching-replay-v1',date='2026-11-11',
        sessions=sessions,default=sessions[0]['draws'][0],categories=categories,steps=31,
        source_scope='20 TEST families, days 7/8, all 3 retained draws; default first by ID, never by score',
        replay_only=True,protected_gps_interval_s=60,frame_interval_s=20,geo_i_cap_per_m=.23,gamma_q=1,
        all_duration_ratio_upper=2.437372,balanced_posterior_upper_percent=70.91))
    for path in ASSETS.iterdir():
        if path.is_file(): shutil.copyfile(path,output/path.name)
    shutil.copyfile(ROOT/'artifacts/reports/graduation_thesis.pdf',output/'thesis.pdf')
    source_paths=sorted(set(sources+[Path(__file__).resolve(),ROOT/'web/thesis_demo.py']+list(ASSETS.glob('*'))))
    manifest=dict(schema='geoi-bq-demo-manifest-v1',date='2026-11-11',families=20,sessions=40,draws=120,
        checked_utility_cells=checked_cells,main_pins_verified=len(release['files_sha256']),
        sampling_or_fitting=False,public_map_only_recomputed=True,
        sources_sha256={str(p.relative_to(ROOT)):sha(p) for p in source_paths},
        files_sha256={str(p.relative_to(output)):sha(p) for p in sorted(output.rglob('*'))
                      if p.is_file() and p.name!='manifest.json'})
    write(output/'manifest.json',manifest)
    print('PASS:',draws,'frozen draws;',checked_cells,'utility cells;108 pins; no defense sampling')


def check(output=OUT):
    output=Path(output);manifest=read(output/'manifest.json');load_release()
    for name,digest in manifest['sources_sha256'].items(): assert sha(ROOT/name)==digest,name
    for name,digest in manifest['files_sha256'].items(): assert sha(output/name)==digest,name
    index=read(output/'data/index.json.gz')
    assert len(index['sessions'])==40 and sum(len(x['draws']) for x in index['sessions'])==120
    assert manifest['sampling_or_fitting'] is False
    print('PASS: demo hashes,120 draws,40 sessions,108 frozen pins')


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--check',action='store_true')
    p.add_argument('--output',type=Path,default=OUT);args=p.parse_args()
    (check if args.check else export)(args.output)


if __name__=='__main__': main()
