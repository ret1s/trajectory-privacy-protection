"""Matched reply-depth frontier on newly generated full-session transcripts."""
import json
import numpy as np
from benchmark.public_poi_context import PublicPoiContext
from evaluation.lane_travel import LanePoiService
from experiments.research_loop_resources import ROOT,CACHE,load,sha
OUT=ROOT/'artifacts/benchmarks/research_loop/iteration07_slack_depth.json'
SOURCE=ROOT/'artifacts/benchmarks/research_loop/iteration07_slack.json'
DATA=ROOT/'artifacts/datasets/research_loop_development_v1/dataset.json'

def main():
    if OUT.exists():raise FileExistsError('Preserve completed evidence')
    rn,service,context,belief,metadata=load()
    deeper=PublicPoiContext(LanePoiService(rn,list(context.pois),k=10),CACHE/'poi10.npz')
    assert np.array_equal(deeper.signatures[:,:,:5],context.signatures)
    d=json.loads(DATA.read_text());source=json.loads(SOURCE.read_text());rows=[]
    ids=[p['id'] for p in deeper.pois]
    for r in source['rows']:
        trace=d['traces'][r['session_id']];points=trace[::20];start=points[0]['time_s']
        events={e['timestamp_s']:e for e in r['public']['events']}
        for L in (5,10):
            scores=[];bytes_=0;items=0
            for p in points:
                state,_=rn.nearest(p['lat'],p['lon']);event=events.get(p['time_s']-start)
                states=[rn.nearest(c['lat'],c['lon'])[0] for c in event['candidates']] if event else []
                per_category=[]
                for ci in range(len(deeper.categories)):
                    ref=set(deeper.signatures[state,ci,:5]);ref.discard(-1)
                    replies=[[ids[i] for i in deeper.signatures[s,ci,:L] if i>=0] for s in states]
                    got=set(deeper.signatures[states,ci,:L].ravel()) if states else set();got.discard(-1)
                    if ref:per_category.append(len(ref&got)/len(ref))
                    if states:bytes_+=len(json.dumps(replies,separators=(',',':')).encode());items+=sum(map(len,replies))
                if per_category:scores.append(float(np.mean(per_category)))
            recall=float(np.mean(scores))
            if L==5:assert abs(recall-r['recall_all_current_queries'])<1e-12
            rows.append({'session_id':r['session_id'],'family_id':r['family_id'],'split':r['split'],'method':r['method'],
                'reply_depth':L,'client_k':5,'recall_all_current_queries':recall,'scheduled_queries':len(points),
                'id_reply_bytes_total':bytes_,'id_reply_bytes_per_scheduled_query':bytes_/len(points),'poi_reply_items':items})
    summary=[]
    for m in sorted({r['method'] for r in rows}):
        for L in (5,10):
            group=[r for r in rows if r['method']==m and r['reply_depth']==L and r['split']=='development_validation']
            summary.append({'method':m,'reply_depth':L,'validation_sessions':len(group),'validation_families':2,
                'mean_recall':float(np.mean([r['recall_all_current_queries'] for r in group])),
                'minimum_session_recall':min(r['recall_all_current_queries'] for r in group),
                'id_bytes_per_query':float(np.mean([r['id_reply_bytes_per_scheduled_query'] for r in group]))})
    from pathlib import Path
    result={'schema':'research-loop-reply-depth-v1','source_sha256':sha(SOURCE),'dataset_sha256':sha(DATA),
        'code_sha256':sha(Path(__file__)),'context10_sha256':deeper.sha256,'scope':'same public transcripts, no new privacy claims; static POI service assumption still applies',
        'rows':rows,'summaries':summary}
    OUT.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(summary,indent=2),flush=True)
if __name__=='__main__':main()
