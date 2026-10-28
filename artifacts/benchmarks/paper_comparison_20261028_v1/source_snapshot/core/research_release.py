"""Load the audited GeoI-BQ research reference; no production certificate claim."""
import hashlib,json
from pathlib import Path
import numpy as np
from scipy.spatial import cKDTree

ROOT=Path(__file__).resolve().parents[1]
DEFAULT=ROOT/'configs/geo_i_bq_v1.json'


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def load_release(path=DEFAULT):
    path=Path(path);spec=json.loads(path.read_text())
    if spec['schema']!='geoi-bq-research-reference-v1':raise ValueError('Supported research manifest required')
    out=ROOT/spec['evidence']['trial']
    result=json.loads((out/'results.json').read_text())
    audit=json.loads((out/'independent_validation.json').read_text())
    if not result['promotion']['utility_gate'] or audit['status']!='pass':
        raise ValueError('Main reference requires utility gate AND independent hard-check audit')
    if audit['results_sha256']!=sha(out/'results.json') or audit['protocol_sha256']!=sha(out/'protocol.json'):
        raise ValueError('Audited evidence changed')
    for name,digest in spec['files_sha256'].items():
        p=ROOT/name
        if not p.resolve().is_relative_to(ROOT):raise ValueError('Root-relative pinned source required')
        if sha(p)!=digest:raise ValueError('Research release source/input changed: '+name)
    return spec


def create_model(private_directory, *, session_token, manifest=DEFAULT, restored_engine=None):
    from benchmark.anchor_belief import PublicAnchorModel
    from benchmark.response_aware_belief import ResponseAwareAnchorModel
    from benchmark.defined_service_scores import DefinedPublicPurposeTable
    from benchmark.exact_public_reference import ExactReferencePortfolio,certified_goals
    from benchmark.targeted_cache_exploration import TargetedExploreLibrary,TargetedCachePolicy
    from benchmark.engines.paced_slack import PacedSlackProgressLaneDummy
    from benchmark.query_purpose import MultiPurposeRoadRanking
    from core.bounded_query_session import BoundedQuerySession,GeoIBoundedQPipeline
    from experiments.public_depth_resources_20261014 import resources
    spec=load_release(manifest)
    directory=Path(private_directory).resolve()
    if directory.is_relative_to(ROOT):raise ValueError('Keep private state outside the source repository')
    rn,reference,reply,ids=resources(directory/'public_resources')
    _,inverse,counts=np.unique(np.floor(rn.xy/120.).astype(np.int64),axis=0,return_inverse=True,return_counts=True)
    prior=1./counts[inverse];prior/=prior.sum()
    base=PublicAnchorModel(rn,reference,prior,spacing_m=200.,epsilon_release=.01,epsilon_test=.01,
        cache_path=directory/'public_resources'/'belief01.npz')
    model=ResponseAwareAnchorModel(base,reply)
    lo,hi=rn.xy.min(axis=0),rn.xy.max(axis=0)
    destinations=sorted(set(map(int,rn.tree.query([lo+(hi-lo)*[a,b] for a in [.15,.5,.85] for b in [.15,.5,.85]])[1])))
    table=DefinedPublicPurposeTable(reference,reply,ids,destinations=destinations)
    portfolio=ExactReferencePortfolio(reply,reference,table)
    goals=certified_goals(portfolio)
    policy=TargetedCachePolicy(TargetedExploreLibrary(rn,goals,portfolio,travel_scale_s=60),
        table,portfolio,novelty_weight=4,long_range_weight=4)
    if restored_engine is None:
        engine=PacedSlackProgressLaneDummy(rn,belief_model=model,k=5,budget=.24,horizon=12,
            theta_m=200.,read_interval_s=60.,utility_slack=.03)
    else:
        engine=restored_engine
        if not isinstance(engine,PacedSlackProgressLaneDummy) or engine.belief_model.sha256!=model.sha256:
            raise ValueError('Restore the matching FULL upstream Geo-I engine')
        expected={'budget':.24,'horizon':12,'k':5,'theta_m':200.,'read_interval_s':60.,'utility_slack':.03,'max_units':23,'unit_epsilon':.01}
        if any(getattr(engine,key)!=value for key,value in expected.items()):
            raise ValueError('Restored engine privacy/public settings differ')
    context=sha(manifest)
    session=BoundedQuerySession(directory/'session',policy=policy,session_token=session_token,
        catalogue_id=reply.sha256,categories=reply.categories,poi_count=len(reply.pois),
        configuration_id=context,gamma=1.,response_l=60)
    try:
        pipeline=GeoIBoundedQPipeline(engine,session,cKDTree(rn.xy[ids]).query(base.xy)[1])
    except Exception:
        session.close();raise
    pipeline.ranking=MultiPurposeRoadRanking(reference)
    return pipeline
