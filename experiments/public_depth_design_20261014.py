"""Choose response depth by exact PUBLIC reference mass, never private scores."""
from pathlib import Path
from fractions import Fraction
import argparse,hashlib,json
import numpy as np

from benchmark.public_poi_context import PublicPoiContext
from benchmark.defined_service_scores import DefinedPublicPurposeTable
from benchmark.exact_public_reference import ExactReferencePortfolio
from evaluation.lane_travel import LanePoiService
from experiments import query_bundle_audit_20261010 as public

ROOT=public.ROOT
OUT=ROOT/'artifacts/benchmarks/public_depth_design_20261014_v1'
WORK=Path('/private/tmp/trajectory-public-depth-20261014-v1')
CANDIDATES=(30,40,50,60)
FLOOR=Fraction(99,100)

def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()

def prepare():
    rn,reference,reply30,ids=public.resources(WORK)
    lo,hi=rn.xy.min(axis=0),rn.xy.max(axis=0)
    destinations=sorted(set(map(int,rn.tree.query([lo+(hi-lo)*[a,b] for a in [.15,.5,.85] for b in [.15,.5,.85]])[1])))
    table=DefinedPublicPurposeTable(reference,reply30,ids,destinations=destinations)
    pois=[{k:v for k,v in p.items() if k not in ('vertex','access_offset_m')} for p in json.loads((ROOT/public.RESOURCES).read_text())['pois_used']]
    rows=[];selected=None
    for depth in CANDIDATES:
        reply=reply30 if depth==30 else PublicPoiContext(LanePoiService(rn,pois,k=depth),WORK/f'reply{depth}.npz')
        portfolio=ExactReferencePortfolio(reply,reference,table)
        goals=portfolio.goals(width=24)
        qualified=[g for g in goals if portfolio.quality_exact(portfolio.pool(g))>=FLOOR]
        rows.append(dict(L=depth,exact_best_public_mean_recall=str(max(portfolio.quality_exact(portfolio.pool(g)) for g in goals)),
            qualified_initial_goals=[list(g) for g in qualified],reply_sha256=reply.sha256))
        if qualified and selected is None:selected=depth
    payload=dict(schema='public-depth-design-v1',source_sha256={str(Path(__file__).relative_to(ROOT)):sha(Path(__file__)),
        'benchmark/exact_public_reference.py':sha(ROOT/'benchmark/exact_public_reference.py')},
        public_inputs_sha256={p:sha(ROOT/p) for p in (public.NETWORK,public.RESOURCES)},
        candidates=list(CANDIDATES),K=5,beam_width=24,exact_public_mean_floor=str(FLOOR),selected_L=selected,
        latent_states=ids.tolist(),destinations=destinations,rows=rows,
        reference_domain='equal public states/defined purposes/defined destinations/defined categories; k5 reference',
        selector='smallest declared L with at least one beam goal at exact PUBLIC mean floor; fixed before new private draws',
        private_GPS_or_benchmark_results_read=False,actual_private_GPS_QoS_certified=False,
        limited_to_beam=True,globally_optimal_response_depth_claimed=False)
    OUT.mkdir(parents=True,exist_ok=True);p=OUT/'certificate.json'
    if p.exists():assert json.loads(p.read_text())==payload
    else:p.write_text(json.dumps(payload,indent=2)+'\n')
    print('Public selected L:',selected)
    for row in rows:print(row['L'],float(Fraction(row['exact_best_public_mean_recall'])),len(row['qualified_initial_goals']))
    if selected is None:raise RuntimeError('Public design failed within fixed response-depth frontier')
    return payload

if __name__=='__main__':prepare()
