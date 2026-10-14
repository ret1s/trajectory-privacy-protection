"""Post-hoc controlled bare Geo-I comparator from SAME protected histories.

No private resampling, parameter tuning, family filtering or claims of a new
confirmation cohort. Natural single-point output K1; same L60/local ranking.
"""
from pathlib import Path
import hashlib,json,copy
import numpy as np
from benchmark.session_poi_cache import SessionCoverClient
from experiments import main_depth_study_20261014 as study

OUT=study.ROOT/'artifacts/benchmarks/geo_i_anchor_comparator_20261014_v1'
METHOD='geo_i_Z_k1'


def run():
    study.validate_sources(study.OUT)
    source=study.load_families(study.OUT)
    if OUT.exists():raise FileExistsError('Retain completed comparator; never refit/reselect on TEST')
    OUT.mkdir()
    study.save(OUT/'protocol.json',dict(schema='bare-GeoI-controlled-comparator-v1',
        source_protocol_sha256=study.sha(study.OUT/'protocol.json'),source_generation_sha256=study.sha(study.OUT/'generation.json'),
        source_sha256=study.sha(Path(__file__)),method=METHOD,K=1,L=60,
        natural_output='one already-protected Z, repeated between private updates; no fake-route continuity claim',
        same_GeoI_history=True,private_resampling=False,post_hoc=True,untouched_trajectory_confirmation=False,
        attacker='same TRAIN/SELECTION loss-specific bank as primary; freeze this implementation before fitting',
        metadata_cost='natural31 point queries; repeated-Z K5 would duplicate records at5x point-query traffic'))
    (OUT/'source_snapshot').mkdir();(OUT/'source_snapshot'/'comparator.py').write_bytes(Path(__file__).read_bytes())
    rn,reference,reply,ids=study.public.resources(study.WORK/'public_resources')
    utility=study.Utility(reference,reply);dataset=study.read(study.DATA)
    families=[]
    for family in source:
        trips=[]
        for original in family['trips']:
            trip={k:copy.deepcopy(original[k]) for k in ('session_id','native_session_id','draw','truth')}
            by_time={int(p['time_s']):p for p in dataset['traces'][trip['native_session_id']]}
            destination=rn.nearest(by_time[600]['lat'],by_time[600]['lon'])[0]
            client=SessionCoverClient(reply.categories,len(reply.pois),k=1,response_l=60,
                session_token=trip['session_id'],static_catalogue_id=reply.sha256)
            events=[];scores=[];cost=dict(requests=0,request_bytes=0,reply_bytes=0)
            def server(request):
                return [list(map(int,row[row>=0])) for row in reply.query_indices(rn.nearest(*request['coordinate'])[0])]
            for truth,anchor in zip(trip['truth'],original['GeoI_anchors']):
                t=truth['t'];wire=client.step(float(t),[anchor],server)
                events.append(dict(timestamp_s=float(t),coordinates=[anchor]))
                scores.append(dict(t=t,**{cache:utility.score(truth['state'],destination,wire[cache]) for cache in ('current','known','session')}))
                cost['requests']+=1
                for name,field in [('request_bytes','requests'),('reply_bytes','replies')]:
                    cost[name]+=sum(len(json.dumps(v,separators=(',',':')).encode()) for v in wire[field])
            trip.update(events={METHOD:events},utility={METHOD:scores},cost={METHOD:cost});trips.append(trip)
        families.append(dict(family_id=family['family_id'],split=family['split'],trips=trips))
    # All data/implementations frozen above; shared bank fits TRAIN and chooses
    # loss-specific models on SELECTION, never uses TEST labels for selection.
    study.METHODS=(METHOD,)
    attacks=study.attacks(OUT,families,rn)
    test=[f for f in families if f['split']=='test']
    bootstrap=np.random.default_rng(study.CONFIG['bootstrap_seed']).integers(0,len(test),(10000,len(test)))
    purposes=[p.value for p in study.QueryPurpose];rows=[]
    for old,new in zip([f for f in source if f['split']=='test'],test):
        row={}
        for method,f in [('bare',new),('main',old)]:
            name=METHOD if method=='bare' else study.CONFIG['primary']
            metrics={p:float(np.mean([s['session'][p]['recall5'] for t in f['trips'] for s in t['utility'][name] if s['session'][p]['recall5'] is not None])) for p in purposes}
            metrics['macro']=float(np.mean(list(metrics.values())));row[method]=metrics
        rows.append(row)
    metrics={}
    for p in purposes+['macro']:
        bare=np.array([r['bare'][p] for r in rows]);main=np.array([r['main'][p] for r in rows]);delta=main-bare
        metrics[p]=dict(bare_mean=float(bare.mean()),primary_minus_bare=float(delta.mean()),paired95=np.quantile(delta[bootstrap].mean(axis=1),[.025,.975]).tolist())
    costs={k:float(np.mean([t['cost'][METHOD][k] for f in test for t in f['trips']])) for k in ('requests','request_bytes','reply_bytes')}
    study.save(OUT/'results.json',dict(method=METHOD,metrics=metrics,cost=costs,attackers=attacks,
        scope='post-hoc controlled comparator with native output K1, shared protected Z; all prior results retained'))
    print(json.dumps(dict(metrics=metrics,cost=costs),indent=2))

if __name__=='__main__':run()
