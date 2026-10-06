"""Same replies/Q: compare original distance reranking with private purpose reranking."""
from pathlib import Path
from collections import defaultdict
import gzip
import hashlib
import json
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'artifacts/benchmarks/query_purpose_20261005'


def main():
    source=OUT/'snapshot/utility_rows.json.gz'
    rows=json.loads(gzip.decompress(source.read_bytes()))
    selected=json.loads((OUT/'snapshot/selection.json').read_text())['selected_L']
    nearest={(r['record_id'],r['time_s'],r['L'],r['category']):r['returned']
             for r in rows if r['purpose']=='nearest_distance'}
    grouped=defaultdict(list);violations=[]
    for row in rows:
        if row['split']!='test' or row['L']!=selected:continue
        distance_only=nearest[(row['record_id'],row['time_s'],selected,row['category'])]
        if row['reference']:
            baseline=len(set(distance_only)&set(row['reference']))/len(row['reference'])
            grouped[(row['purpose'],row['family_id'])].append((baseline,row['recall']))
        # When reference has <k items it contains EVERY eligible/available POI
        # within radius. Any additional returned ID is definitely outside.
        if row['purpose']=='within_radius' and len(row['reference'])<5:
            violations.append({'record_id':row['record_id'],'category':row['category'],
                'baseline_returned':len(distance_only),'outside_radius':len(set(distance_only)-set(row['reference'])),
                'new_outside_radius':len(set(row['returned'])-set(row['reference']))})
    results={}
    for purpose in sorted({p for p,f in grouped}):
        family={f:np.mean(vals,axis=0).tolist() for (p,f),vals in grouped.items() if p==purpose}
        values=np.array(list(family.values()));delta=values[:,1]-values[:,0]
        rng=np.random.default_rng(31007)
        boot=np.mean(delta[rng.integers(0,len(delta),size=(10000,len(delta)))],axis=1)
        results[purpose]={'distance_only_recall':float(values[:,0].mean()),
            'purpose_reranking_recall':float(values[:,1].mean()),
            'paired_delta':float(delta.mean()),'paired_family_bootstrap_95':np.percentile(boot,[2.5,97.5]).tolist(),
            'families':len(family),'family_values_distance_new':family}
    assert all(r['new_outside_radius']==0 for r in violations)
    readout={'schema':'query-purpose-local-comparison-v1','L':selected,
        'scope':'Snapshot internal development; same Q, replies, GPS and top-k, only local ranking changes',
        'purpose_results':results,'radius_exhaustive_reference_checks':{
            'rows':len(violations),'baseline_returned_items':sum(r['baseline_returned'] for r in violations),
            'baseline_definitely_outside_radius':sum(r['outside_radius'] for r in violations),
            'new_definitely_outside_radius':0},
        'traffic_change':0,'private_GPS_budget_change':0,
        'source_sha256':{str(source.relative_to(ROOT)):hashlib.sha256(source.read_bytes()).hexdigest(),
                         str(Path(__file__).relative_to(ROOT)):hashlib.sha256(Path(__file__).read_bytes()).hexdigest()},
        'limits':['three test families, bootstrap conditional on this small cohort',
                  'fastest equals distance at constant road speed',
                  'not a comparison with external paper methods']}
    (OUT/'purpose_comparison.json').write_text(json.dumps(readout,indent=2)+'\n')
    print(json.dumps(readout))


if __name__=='__main__':main()
