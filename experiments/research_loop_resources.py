"""Separate public resources for the recovered 20-Sep OSM snapshot."""
from pathlib import Path
import hashlib,json,pickle,time
import numpy as np
from data.lane_states import build_lane_states,catalogue_summary
from evaluation.scenario_metrics import read_osm_pois
from evaluation.lane_travel import LanePoiService
from benchmark.public_poi_context import PublicPoiContext
from benchmark.anchor_belief import PublicAnchorModel
ROOT=Path(__file__).resolve().parents[1]
CACHE=ROOT/'cache/research_loop_20260924'
OSM=ROOT/'data/raw/Beijing_20260920.osm.gz'
NET=CACHE/'beijing.net.xml'

def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def prepare():
    t=time.perf_counter();rn=build_lane_states(NET)
    print('Built catalogue',len(rn),flush=True)
    pois=read_osm_pois(OSM,(116.29,39.96,116.36,40.02))
    service=LanePoiService(rn,pois)
    context=PublicPoiContext(service,CACHE/'poi5.npz')
    print('Built POI context',len(context.pois),flush=True)
    _,inv,counts=np.unique(np.floor(rn.xy/120.).astype(np.int64),axis=0,return_inverse=True,return_counts=True)
    prior=1./counts[inv];prior/=prior.sum()
    belief=PublicAnchorModel(rn,context,prior,cache_path=CACHE/'belief.npz')
    metadata={'schema':'research-loop-resources-v1','source':'https://download.bbbike.org/osm/bbbike/Beijing/Beijing.osm.gz',
        'osm_sha256':sha(OSM),'net_sha256':sha(NET),'catalogue':catalogue_summary(rn),
        'context_sha256':context.sha256,'belief_sha256':belief.sha256,'pois_used':service.pois,
        'preparation_s':time.perf_counter()-t,'historical_replay':False}
    # Locally generated trusted cache; do not load pickles supplied externally.
    with (CACHE/'resources.pkl').open('wb') as f:pickle.dump((rn,service,context,belief),f)
    metadata['resource_sha256']=sha(CACHE/'resources.pkl')
    (CACHE/'resources.json').write_text(json.dumps(metadata,indent=2)+'\n')
    (ROOT/'artifacts/benchmarks/research_loop/resources.json').write_text(json.dumps(metadata,indent=2)+'\n')
    (CACHE/'resources.sha256').write_text(sha(CACHE/'resources.json')+'\n')
    print('Resources ready',metadata['preparation_s'],flush=True)

def load():
    metadata=json.loads((CACHE/'resources.json').read_text())
    if sha(CACHE/'resources.pkl')!=metadata['resource_sha256']:raise ValueError('Resource hash mismatch')
    with (CACHE/'resources.pkl').open('rb') as f:return (*pickle.load(f),metadata)
if __name__=='__main__':prepare()
