"""Full TEST public-query overlap check; no private GPS/purpose/destination."""
from pathlib import Path
from collections import Counter
import hashlib,json,numpy as np
from experiments.future_sumo_eval import native_resources
from experiments.qplanner_study_20261006_v2 import read
from benchmark.public_poi_context import PublicPoiContext
from benchmark.multi_purpose_retrieval import PublicPurposePoiService
from evaluation.lane_travel import LanePoiService

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'artifacts/benchmarks/multi_purpose_retrieval_20261009_v2'
BASE=ROOT/'artifacts/benchmarks/qplanner_depth_base_q_generalization_20261006_v1'
CACHE=Path('/private/tmp/qplanner-response-depth-generalization-20261006-v1')
WORK=Path('/private/tmp/multi-purpose-retrieval-20261009-v1')
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def main():
    p=read(OUT/'protocol.json');resources=read(OUT/'resources.json');data=read(ROOT/p['dataset_path'])
    rn,ref,_,_,_=native_resources(data,CACHE)
    context=PublicPoiContext(LanePoiService(rn,list(ref.pois),k=60),CACHE/'public_resources/reply60.npz')
    service=PublicPurposePoiService(rn,context,np.load(WORK/'distance.npy',mmap_mode='r'),np.load(WORK/'time.npy',mmap_mode='r'),
        radius_m=1000.,destination_states=tuple(resources['public_destination_states']))
    counter=Counter();files={}
    test_families={f['family_id'] for f in data['families'] if f['split']=='test'}
    for name,pin in p['source_family_files_sha256'].items():
        if name.split('--draw')[0] not in test_families:continue
        assert sha(BASE/'families'/name)==pin;files[name]=pin
        bundle=read(BASE/'families'/name)
        for session in bundle['public']['streams']['legacy_l10']:
            for event in session['events']:
                for q in event['candidates']:
                    state=rn.nearest(q['lat'],q['lon'])[0]
                    nearest15=service.query(state,'nearest_distance',15);fastest15=service.query(state,'fastest_travel',15)
                    nearest10=service.query(state,'nearest_distance',10);radius10=service.query(state,'within_radius',10)
                    counter['public_Q_requests']+=1
                    for near,fast,base,radius in zip(nearest15,fastest15,nearest10,radius10):
                        counter['nearest15_records']+=len(near)
                        counter['fastest15_records']+=len(fast)
                        counter['nearest_fastest15_overlap_records']+=len(set(near)&set(fast))
                        counter['fastest15_new_records']+=len(set(fast)-set(near))
                        counter['radius10_records']+=len(radius)
                        counter['radius10_new_over_nearest10']+=len(set(radius)-set(base))
    assert counter['radius10_new_over_nearest10']==0
    result=dict(schema='public-template-overlap-diagnostic-v1',verifier_sha256=sha(Path(__file__)),
        protocol_sha256=sha(OUT/'protocol.json'),source_family_files_sha256=files,
        counts=dict(counter),fastest15_overlap_fraction_of_fastest=counter['nearest_fastest15_overlap_records']/counter['fastest15_records'],
        radius_new_records=0,scope='All TEST protected Q, all six categories; static public source/time scores; no private query or endpoint input')
    with (OUT/'template_overlap.json').open('x') as f:json.dump(result,f,indent=2);f.write('\n')
    print(json.dumps({k:v for k,v in result.items() if k!='source_family_files_sha256'}))
if __name__=='__main__':main()
