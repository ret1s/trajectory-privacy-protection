"""Evaluate release accounting on all S9/S10 allowed views; NOT privacy benchmark."""
from pathlib import Path
import hashlib,json
from collections import defaultdict
from core.boundary_release import BoundaryPolicy,BoundaryProtectedStream
ROOT=Path(__file__).resolve().parents[1]
class ClockOnly:
    """Fixed public placeholder: no BR, no coordinates or POI accuracy measured."""
    def reset(self): pass
    def protect_step(self,lat,lon,t):return ((0.,0.),)
def main():
    path=ROOT/'artifacts/datasets/urban_fresh_v2/dataset.json';d=json.loads(path.read_text());rows=[]
    for r in d['records']:
        if r['scenario'] not in ('S9','S10'):continue
        for sid,indices in zip(r['session_ids'],r['observed_indices']):
            times=[d['traces'][sid][i]['time_s'] for i in indices];start=times[0]
            for h,delay in [(0,0),(60,0),(0,60),(60,60)]:
                gate=BoundaryProtectedStream(ClockOnly(),BoundaryPolicy(h,delay));public=[]
                for t in times:public.extend(gate.ingest(t-start,0.,0.))
                gate.close(times[-1]-start)
                stats=gate.evaluator_summary();assert stats['input_events']==stats['head_suppressed']+stats['released_events']+stats['tail_cancelled']
                rows.append({'record_id':r['record_id'],'case_id':r['case_id'],'session_id':sid,'head_s':h,'delay_s':delay,**stats})
    grouped=defaultdict(list)
    for r in rows:grouped[(r['case_id'],r['head_s'],r['delay_s'])].append(r)
    summaries=[]
    for (case,h,delay),rr in sorted(grouped.items()):
        n=sum(x['input_events'] for x in rr);released=sum(x['released_events'] for x in rr)
        summaries.append(dict(case_id=case,head_s=h,delay_s=delay,record_views=len(rr),input_events=n,released_events=released,release_fraction=released/n))
    out=ROOT/'artifacts/benchmarks/boundary_schedule';out.mkdir(parents=True,exist_ok=True)
    result={'status':'schedule_only_no_privacy_or_POI_claim','source_sha256':hashlib.sha256(path.read_bytes()).hexdigest(),'code_sha256':hashlib.sha256((ROOT/'core/boundary_release.py').read_bytes()).hexdigest(),'scope':'additional gating on already allowed views; not full-trip utility; clock-only placeholder; close at last allowed sample; conservative tail accounting','records':len(set(r['record_id'] for r in rows)),'rows':rows,'summaries':summaries}
    (out/'results.json').write_text(json.dumps(result,indent=2)+'\n');print(result['records'],'records;',len(rows),'record-view/config rows; accounting passed')
if __name__=='__main__':main()
