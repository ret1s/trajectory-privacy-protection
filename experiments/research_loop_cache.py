"""Audit whether the present static POI contract requires remote queries."""
import json,time
import numpy as np
from evaluation.static_cache import StaticPublicPoiCache
from experiments.research_loop_resources import ROOT,load,sha,NET
OUT=ROOT/'artifacts/benchmarks/research_loop/iteration02_static_cache.json'

def main():
    if OUT.exists():raise FileExistsError('Preserve completed evidence')
    rn,service,context,belief,meta=load();cache=StaticPublicPoiCache(context)
    data=json.loads((ROOT/'artifacts/datasets/urban_fresh_v2/dataset.json').read_text())
    rows=[];times=[]
    for r in data['records']:
        if r['family_id'] not in ('family-301','family-302') or r['scenario'] not in ('S1','S2','S3','S9','S10'):continue
        for slot,(sid,indices) in enumerate(zip(r['session_ids'],r['observed_indices'])):
            for i in indices[:12]:
                p=data['traces'][sid][i]
                for category in context.categories:
                    expected=service.query((p['lat'],p['lon']),category)
                    start=time.perf_counter();got=cache.query(p['lat'],p['lon'],category);times.append((time.perf_counter()-start)*1000)
                    assert got==expected
                    if expected: rows.append({'record_id':r['record_id'],'case_id':r['case_id'],'slot':slot,
                        'fcd_index':i,'category':category,'recall':len(set(got)&set(expected))/len(expected)})
    result={'schema':'static-cache-assumption-audit-v1','scope':'exact static/public-catalogue service only; development GPS on recovered map',
        'resources':meta,'source_sha256':{p:sha(ROOT/p) for p in ('evaluation/static_cache.py','experiments/research_loop_cache.py','artifacts/datasets/urban_fresh_v2/dataset.json')},
        'records':len({r['record_id'] for r in rows}),'queries':len(rows),'all_results_equal':True,
        'macro_query_recall':float(np.mean([r['recall'] for r in rows])),
        'remote_queries':0,'public_poi_catalogue_json_bytes':len(json.dumps(context.pois,separators=(',',':')).encode()),
        'precomputed_index_bytes':cache.index_bytes,'public_network_xml_bytes':NET.stat().st_size,
        'local_lookup_mean_ms':float(np.mean(times)),'local_lookup_p95_ms':float(np.percentile(times,95)),
        'privacy_statement':'No coordinates transmitted after location-independent public preload; this does not conceal preload region, identity or other application telemetry.',
        'consequence':'Current public static-POI utility contract admits a no-location-query control. Do not claim BR dominates this control. Live server-only information requires a separately specified workload.',
        'rows':rows}
    OUT.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k not in ('resources','rows')},indent=2),flush=True)
if __name__=='__main__':main()
