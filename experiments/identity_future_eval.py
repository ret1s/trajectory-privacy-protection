"""Bounded S4/S5/S6 diagnostic with fixed group-disjoint attacker selection.

Does not replace original evidence. S5/S6 use frozen S3 prefixes with new future
targets. S4 runs active GeoI-Slack on a separately reconstructed public graph;
the person/vehicle labels are synthetic SUMO design truth, not human identities.
"""
from pathlib import Path
from itertools import combinations
import argparse
import gzip
import hashlib
import json
import time
import numpy as np
from evaluation.identity_future import (xy_from_latlon,public_arrays,prefix_features,
    pair_features,check_group_split,classification_metrics,location_metrics,
    ClassifierBank,RegressorBank)

ROOT=Path(__file__).resolve().parents[1]
DATA=ROOT/'artifacts/datasets/research_loop_expanded_v1/dataset.json'
SCREEN=ROOT/'artifacts/benchmarks/research_loop/iteration18_expanded_screening.json'
OUT=ROOT/'artifacts/benchmarks/identity_future_20261005'
SPLITS={'train':[f'family-{i}' for i in range(701,707)],
        'selection':[f'family-{i}' for i in range(707,710)],
        'test':[f'family-{i}' for i in range(710,713)]}
SEED=20261005
S4_ROLES=('base','repeat','new_device','shared_device','companion','partial')


def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def save(path,data):
    if path.exists():raise FileExistsError(f'Preserve completed evidence: {path}')
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(data,indent=2,allow_nan=False)+'\n')


def split_of(family):return next(k for k,v in SPLITS.items() if family in v)


def protocol():
    return {'schema':'identity-future-protocol-v2','seed':SEED,'splits':SPLITS,
        'split_unit':'SUMO route family; contains every person/vehicle/session/seed of a linked family',
        'S4':{'target':'pairwise same synthetic person and same physical vehicle, separately',
            'roles':S4_ROLES,'pairs':'all unordered pairs among the six selected roles, including hard negatives',
            'observation':'60-second public relative-time clock; no network/account identity fields',
            'models':['raw','geoi_slack_reconstructed'],
            'K':5,'B':.24,'H':12,'theta_m':200.,'read_interval_s':60.,'utility_slack':.03,
            'attacker_selection':'maximum balanced accuracy, then ROC-AUC, stable name on selection only'},
        'S5':{'source':'unmodified archived S3.A public prefix',
            'target':'next external edge after prefix; first distinct non-internal SUMO edge within 60 seconds',
            'secondary_target':'GPS at prefix end +20 seconds, for next-position error',
            'no_lookahead':'only prefix events enter feature extraction'},
        'S6':{'source':'same unmodified archived S3.A prefix',
            'target':'last GPS of completed SUMO trace, not route plan or future timestamp as input'},
        'frozen_prefix_scope':'new residual-future diagnostic, not rerun of original S5/S6 eligibility records',
        'eligibility':'Independently for each target: S6 completed trace; S5 position requires +20s; '
                      'S5 exact edge additionally requires a next external edge within60s. '
                      'Unseen exact-edge classes and failed raw controls block a protection conclusion.',
        'controls':['raw observations','train-majority/train-prior','training-label permutation'],
        'attacker_bank':'ExtraTrees (96 trees, depth16, leaf2), standardized distance-weighted kNN1/5/15',
        'permutation':'training labels permuted at family/session block; every RNG repetition kept together',
        'limits':'small grouped development diagnostic; reconstructed graph is not original lane/turn graph; '
                 'no human identity, no faithful external-paper reproduction, no universal dominance claim'}


def evaluate_regression(rows,task):
    partitions={s:[r for r in rows if r['split']==s] for s in SPLITS}
    if any(not p for p in partitions.values()):raise ValueError('Missing split')
    x={s:np.array([prefix_features(r['public']) for r in p]) for s,p in partitions.items()}
    y={s:np.array([r[task] for r in p]) for s,p in partitions.items()}
    bank=RegressorBank(x['train'],y['train'],SEED)
    estimates={s:bank.predict(x[s]) for s in ('selection','test')}
    for split in estimates:
        centres=[public_arrays(r['public']) for r in partitions[split]]
        estimates[split]['public_last']=np.array([c[-1] for c,_,_ in centres])
        if task=='next_xy':
            estimates[split]['public_velocity']=np.array([c[-1]+20*(c[-1]-c[0])/max(1.,t[-1])
                                                          for c,_,t in centres])
    selected=min(estimates['selection'],key=lambda n:(location_metrics(y['selection'],estimates['selection'][n])['mae_m'],n))
    metrics={n:location_metrics(y['test'],v) for n,v in estimates['test'].items()}
    # Shuffle one label per family, preserving repetitions/observations together.
    blocks=sorted({r['family_id'] for r in partitions['train']})
    perm=np.random.default_rng(SEED).permutation(blocks)
    target={r['family_id']:r[task] for r in partitions['train']}
    permuted=np.array([target[dict(zip(blocks,perm))[r['family_id']]] for r in partitions['train']])
    negative=RegressorBank(x['train'],permuted,SEED)
    ps,pt=negative.predict(x['selection']),negative.predict(x['test'])
    pn=min(ps,key=lambda n:(location_metrics(y['selection'],ps[n])['mae_m'],n))
    return {'selected_attacker':selected,'selection_mae_m':location_metrics(y['selection'],estimates['selection'][selected])['mae_m'],
        'test':metrics[selected],'bank_test_descriptive_only':metrics,
        'permuted_training_control':{'selected_attacker':pn,'test':location_metrics(y['test'],pt[pn])},
        'test_family_rows':[{'family_id':r['family_id'],'rep':r['rep'],
            'error_m':float(np.linalg.norm(t-p))} for r,t,p in zip(partitions['test'],y['test'],estimates['test'][selected])],
        'counts':{s:len(p) for s,p in partitions.items()}}


def evaluate_classification(rows,label,*,pair=False):
    parts={s:[r for r in rows if r['split']==s] for s in SPLITS}
    features=(lambda r:pair_features(*r['public_pair'])) if pair else (lambda r:prefix_features(r['public']))
    x={s:np.array([features(r) for r in p]) for s,p in parts.items()}
    y={s:np.array([r[label] for r in p]) for s,p in parts.items()}
    bank=ClassifierBank(x['train'],y['train'],SEED)
    pred={s:bank.predict(x[s]) for s in ('selection','test')}
    prob={s:bank.probability(x[s]) if pair else {} for s in ('selection','test')}
    def metric(s,n):return classification_metrics(y[s],pred[s][n],prob[s].get(n))
    chosen=min(pred['selection'],key=lambda n:(-metric('selection',n)['balanced_accuracy'],
                                             -metric('selection',n).get('roc_auc',0.),n))
    labels,counts=np.unique(y['train'],return_counts=True);majority=labels[np.argmax(counts)]
    # Whole pair/session blocks; repetitions never get independent shuffled targets.
    keys=sorted({r['block_id'] for r in parts['train']})
    targets={r['block_id']:r[label] for r in parts['train']}
    permutation=np.random.default_rng(SEED).permutation(keys)
    shuffled={k:targets[v] for k,v in zip(keys,permutation)}
    negative=ClassifierBank(x['train'],[shuffled[r['block_id']] for r in parts['train']],SEED)
    neg={s:negative.predict(x[s]) for s in ('selection','test')}
    negprob={s:negative.probability(x[s]) if pair else {} for s in ('selection','test')}
    nc=min(neg['selection'],key=lambda n:(-classification_metrics(y['selection'],neg['selection'][n],negprob['selection'].get(n))['balanced_accuracy'],n))
    result={'selected_attacker':chosen,'selection':metric('selection',chosen),'test':metric('test',chosen),
        'bank_test_descriptive_only':{n:metric('test',n) for n in pred['test']},
        'train_majority_control':classification_metrics(y['test'],np.repeat(majority,len(y['test']))),
        'permuted_training_control':{'selected_attacker':nc,'test':classification_metrics(y['test'],neg['test'][nc],negprob['test'].get(nc))},
        'train_label_coverage_test':float(np.mean(np.isin(y['test'],y['train']))),
        'counts':{s:len(p) for s,p in parts.items()},
        'test_truth':y['test'].tolist(),'test_predictions':pred['test'][chosen].tolist()}
    if pair:
        result['test_positive_rate']=float(y['test'].mean())
        result['test_family_metrics']={f:classification_metrics(
            [r[label] for r in parts['test'] if r['family_id']==f],
            [p for r,p in zip(parts['test'],pred['test'][chosen]) if r['family_id']==f],
            [p for r,p in zip(parts['test'],prob['test'][chosen]) if r['family_id']==f])
            for f in SPLITS['test']}
    return result


def future_rows(data):
    screen=json.loads(SCREEN.read_text())
    assert screen['provenance']['dataset_sha256']==sha(DATA)
    records={r['record_id']:r for r in data['records']}
    rows=[];sources={str(SCREEN.relative_to(ROOT)):sha(SCREEN)};rejections=[]
    for item in screen['shards']:
        path=SCREEN.parent/'expanded_screening'/item['file']
        assert sha(path)==item['sha256'];sources[str(path.relative_to(ROOT))]=sha(path)
        for row in json.loads(gzip.decompress(path.read_bytes()))['rows']:
            if row['case_id']!='S3.A':continue
            record=records[row['record_id']];assert len(record['session_ids'])==1
            trace=data['traces'][record['session_ids'][0]];last=record['observed_indices'][0][-1]
            external=trace[last]['edge_id']
            future=next((p for p in trace[last+1:] if p['time_s']-trace[last]['time_s']<=60
                         and p['edge_id']!=external and not p['edge_id'].startswith(':')),None)
            if last+20>=len(trace) or future is None:
                rejections.append({'record_id':row['record_id'],'method':row['method'],'rep':row['rep'],
                    'excluded_tasks':(['S5_next20s_position'] if last+20>=len(trace) else [])+
                                     (['S5_next_edge'] if future is None else []),
                    'reason':'target_unavailable_in_fixed_future_horizon'})
            public=row['public_views'][0];public_arrays(public)
            assert len(public['events'])==len(record['observed_indices'][0])
            rows.append({'method':row['method'],'family_id':record['family_id'],
                'split':split_of(record['family_id']),'record_id':record['record_id'],'rep':row['rep'],
                'block_id':record['family_id']+'/'+record['session_ids'][0],
                'public':public,'next_edge':None if future is None else future['edge_id'],
                'next_xy':None if last+20>=len(trace) else xy_from_latlon([[trace[last+20]['lat'],trace[last+20]['lon']]])[0].tolist(),
                'destination_xy':xy_from_latlon([[trace[-1]['lat'],trace[-1]['lon']]])[0].tolist(),
                'evaluator_only':{'last_observed_index':last,'next_edge_horizon_s':None if future is None else future['time_s']-trace[last]['time_s'],
                                  'destination_horizon_s':trace[-1]['time_s']-trace[last]['time_s']}})
    return rows,sources,rejections


def run_future(data):
    rows,sources,rejections=future_rows(data)
    results={}
    for method in sorted({r['method'] for r in rows}):
        subset=[r for r in rows if r['method']==method]
        edge=evaluate_classification([r for r in subset if r['next_edge'] is not None],'next_edge')
        edge['protection_conclusion_allowed']=False
        edge['gate_failure']='No test next-edge labels are covered by training; raw exact-edge classifier also fails. Need public candidate-edge decoder.'
        results[method]={'S5_next_edge':edge,
                         'S5_next20s_position':evaluate_regression([r for r in subset if r['next_xy'] is not None],'next_xy'),
                         'S6_destination':evaluate_regression(subset,'destination_xy')}
        print('Future scored',method,flush=True)
    payload={'schema':'frozen-prefix-future-diagnostic-v2','protocol_sha256':sha(OUT/'protocol_v2.json'),
        'source_sha256':{str(DATA.relative_to(ROOT)):sha(DATA),**sources,
            str(Path(__file__).relative_to(ROOT)):sha(Path(__file__)),
            'evaluation/identity_future.py':sha(ROOT/'evaluation/identity_future.py')},
        'scope':'Residual S5/S6 future inference on frozen S3.A prefixes; split fixed before scores. '
                'Small development diagnosis, not original S5/S6 benchmark, not confirmation.',
        'repair':'v2 separates eligibility for independent tasks. The retained v1 used a combined '
                 'next-edge/20s availability filter, which unnecessarily reduced the S6 cohort.',
        'row_count':len(rows),'rows':rows,'rejections':rejections,'results':results}
    save(OUT/'future_results_v2.json',payload)


def run_linkage(data,cache_dir):
    from experiments.public_research_resources import load_public_research_resources
    from benchmark.engines.paced_slack import PacedSlackProgressLaneDummy
    rn,_,_,_,belief,_,metadata=load_public_research_resources(cache_dir)
    engine=PacedSlackProgressLaneDummy(rn,belief_model=belief,k=5,budget=.24,horizon=12,
        theta_m=200.,read_interval_s=60.,utility_slack=.03,rng=np.random.default_rng(SEED))
    transcripts=[];source_hash=sha(DATA);started=time.perf_counter()
    for f in sorted(data['families'],key=lambda f:f['family_id']):
        sessions=[s for s in f['sessions'] if s['role'] in S4_ROLES]
        assert len(sessions)==6 and set(s['role'] for s in sessions)==set(S4_ROLES)
        for session in sessions:
            sid=session['session_id'];trace=data['traces'][sid]
            indices=list(range(0,len(trace),60))
            # Session IDs choose independent public seeds, never enter attack features.
            seed=SEED+int(hashlib.sha256(sid.encode()).hexdigest()[:8],16)
            engine.anchor_rng=np.random.default_rng(seed);engine.dummy_rng=np.random.default_rng(seed+1)
            engine.reset();raw=[];protected=[]
            for j,i in enumerate(indices):
                p=trace[i];t=p['time_s']-trace[0]['time_s']
                q=engine.protect_step(p['lat'],p['lon'],t)
                raw.append({'event_id':f'e{j:04d}','timestamp_s':t,
                            'candidates':[{'candidate_id':'candidate_0000','lat':p['lat'],'lon':p['lon']}]})
                protected.append({'event_id':f'e{j:04d}','timestamp_s':t,
                            'candidates':[{'candidate_id':f'candidate_{n:04d}','lat':lat,'lon':lon} for n,(lat,lon) in enumerate(q)]})
            assert engine.spent_bound<=.23+1e-12
            assert all(r['spent_units']<=23 for r in engine.evaluator_ledger)
            transcripts.append({'family_id':f['family_id'],'session_id':sid,'split':split_of(f['family_id']),
                'role':session['role'],'raw':{'events':raw},'geoi_slack_reconstructed':{'events':protected},
                'evaluator_only':{'person_id':session['person_id'],'physical_vehicle_id':session['physical_vehicle_id'],
                    'budget_spent':engine.spent_bound,'private_reads':sum(l['private_read'] for l in engine.evaluator_ledger),
                    'ledger':engine.evaluator_ledger}})
        print('Linkage generated',f['family_id'],len(transcripts),'sessions',flush=True)
    results={};pair_rows=[]
    for method in ('raw','geoi_slack_reconstructed'):
        rows=[]
        for family in sorted(SPLITS['train']+SPLITS['selection']+SPLITS['test']):
            for a,b in combinations([t for t in transcripts if t['family_id']==family],2):
                rows.append({'family_id':family,'split':split_of(family),
                    'session_pair':[a['session_id'],b['session_id']],
                    'block_id':'/'.join([a['session_id'],b['session_id']]),
                    'public_pair':[a[method],b[method]],
                    'same_person':int(a['evaluator_only']['person_id']==b['evaluator_only']['person_id']),
                    'same_vehicle':int(a['evaluator_only']['physical_vehicle_id']==b['evaluator_only']['physical_vehicle_id'])})
        pair_rows.extend({'method':method,**{k:v for k,v in r.items() if k!='public_pair'}} for r in rows)
        results[method]={label:evaluate_classification(rows,label,pair=True) for label in ('same_person','same_vehicle')}
        print('Linkage scored',method,flush=True)
    assert sha(DATA)==source_hash
    # Preserve a separate attacker-visible file that contains no synthetic IDs or truth.
    save(OUT/'linkage_public_transcripts.json',{'schema':'public-linkage-transcripts-v1',
        'transcripts':[{'public_index':i,'raw':t['raw'],'geoi_slack_reconstructed':t['geoi_slack_reconstructed']}
                       for i,t in enumerate(transcripts)]})
    save(OUT/'linkage_results.json',{'schema':'synthetic-linkage-diagnostic-v1',
        'protocol_sha256':sha(OUT/'protocol_v2.json'),'dataset_sha256':source_hash,
        'code_sha256':sha(Path(__file__)),'features_sha256':sha(ROOT/'evaluation/identity_future.py'),
        'public_transcripts_sha256':sha(OUT/'linkage_public_transcripts.json'),
        'reconstructed_public_resources':metadata,'elapsed_s':time.perf_counter()-started,
        'scope':'Active GeoI-Slack on reconstructed public graph, synthetic SUMO person/physical vehicle labels; '
                'small family-disjoint development diagnosis, not human identification or original benchmark rerun.',
        'session_count':len(transcripts),'pair_rows':pair_rows,'results':results,'evaluator_sessions':transcripts})


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--future-only',action='store_true')
    parser.add_argument('--linkage-only',action='store_true')
    parser.add_argument('--cache-dir',type=Path,default=Path('/private/tmp/trajectory-research-20261005-public-map'))
    args=parser.parse_args()
    if args.future_only and args.linkage_only:parser.error('Choose at most one stage')
    check_group_split(*SPLITS.values());data=json.loads(DATA.read_text())
    OUT.mkdir(parents=True,exist_ok=True)
    frozen=protocol()
    if (OUT/'protocol_v2.json').exists():assert json.loads((OUT/'protocol_v2.json').read_text())==json.loads(json.dumps(frozen))
    else:save(OUT/'protocol_v2.json',frozen)
    if not args.linkage_only:run_future(data)
    if not args.future_only:run_linkage(data,args.cache_dir)


if __name__=='__main__':main()
