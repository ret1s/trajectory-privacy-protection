"""Secondary paired readout; never selects/tunes defenses or promotion gates."""
import argparse
from collections import Counter
from fractions import Fraction
from pathlib import Path
import gzip,json
import numpy as np

def read(p):
    raw=p.read_bytes();return json.loads(gzip.decompress(raw) if p.suffix=='.gz' else raw)

def analyze(output):
    protocol=read(output/'protocol.json');results=read(output/'results.json')
    primary=protocol['config']['primary'];methods=protocol['methods']
    families=[read(p) for p in sorted((output/'families').glob('*.gz'))]
    families=[f for f in families if f['split']=='test']
    purposes=list(next(iter(families[0]['trips'][0]['utility'].values()))[0]['current'])
    bootstrap=np.random.default_rng(protocol['config']['bootstrap_seed']).integers(0,len(families),(10000,len(families)))
    means={}
    for method in methods:
        rows=[]
        for f in families:
            row={p:np.mean([s['current'][p]['recall5'] for t in f['trips'] for s in t['utility'][method] if s['current'][p]['recall5'] is not None]) for p in purposes}
            row['macro']=np.mean(list(row.values()));rows.append(row)
        means[method]=rows
    comparisons=[]
    for control in methods:
        if control==primary:continue
        metrics={}
        for p in purposes+['macro']:
            diff=np.array([a[p]-b[p] for a,b in zip(means[primary],means[control])])
            metrics[p]=dict(delta=float(diff.mean()),family_paired95=np.quantile(diff[bootstrap].mean(axis=1),[.025,.975]).tolist(),minimum_family_delta=float(diff.min()))
        comparisons.append(dict(primary=primary,control=control,metrics=metrics))
    decisions=[t['Q_certificates'][primary][tick] for f in families for t in f['trips'] for tick in range(0,31,3)]
    counts=Counter(c['public_library_size'] for c in decisions)
    motion=[]
    for f in families:
        for t in f['trips']:
            frames=t['sampled_states'][primary]
            motion.append(sum(a!=b for a,b in zip(frames,frames[1:]))/30.)
    payload=dict(schema='secondary-readout-no-tuning-v1',primary=primary,bootstrap_seed=protocol['config']['bootstrap_seed'],paired_family_comparisons=comparisons,
        primary_test_decisions=len(decisions),public_library_cardinality_counts={str(k):v for k,v in sorted(counts.items())},
        singleton_fraction=counts[1]/len(decisions),positive_planning_gain_decisions=sum(Fraction(c['exact_planning_score_gain'])>0 for c in decisions),
        mean_planning_gain=float(np.mean([c['planning_score_gain'] for c in decisions])),mean_nonhold_frame_fraction=float(np.mean(motion)),
        scope='same cohort family paired; public pool/protected-belief controls; planning gain is not actual Recall gain')
    destination=output/'secondary_readout.json'
    if destination.exists():assert read(destination)==payload
    else:destination.write_text(json.dumps(payload,indent=2,allow_nan=False)+'\n')
    print(json.dumps(payload,indent=2))
    return payload

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,required=True)
    analyze(parser.parse_args().output)
