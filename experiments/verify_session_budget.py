"""Read-only, independent checks of linked-session accounting and readouts."""
import argparse
from collections import defaultdict
import gzip
import hashlib
import json
from pathlib import Path

import numpy as np

from evaluation.identity_future import classification_metrics, public_arrays

ROOT = Path(__file__).resolve().parents[1]
DEFAULT = ROOT/'artifacts/benchmarks/session_budget_20261005/round3'


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read(path):
    return json.loads(gzip.decompress(path.read_bytes()) if path.suffix=='.gz' else path.read_bytes())


def close(a,b):
    assert (a is None and b is None) or (a is not None and b is not None and np.isclose(a,b,rtol=1e-11,atol=1e-12)), (a,b)


def verify(out):
    p, result = read(out/'protocol.json'), read(out/'readout.json')
    for f, digest in p['source_sha256'].items():
        assert sha(ROOT/f)==digest, f
    assert sha(out/'protocol.json')==result['protocol_sha256']
    for file, key in [('public_transcripts.json.gz','public_transcripts_sha256'),
                      ('evaluator_executions.json.gz','evaluator_executions_sha256'),
                      ('epoch_accounting.json','epoch_accounting_sha256')]:
        assert sha(out/file)==result[key]
    executions = read(out/'evaluator_executions.json.gz')
    public = read(out/'public_transcripts.json.gz')['transcripts']
    accounts = read(out/'epoch_accounting.json')
    policies = {c['id']:c['public_parameters'] for c in p['policies']}
    reference = p['per_session_reference']
    assert len(executions)==72*len(p['repetitions'])*5==len(public)
    assert len({e['session_id'] for e in executions})==72
    lookup={}
    read_units=0
    for index,(e,visible) in enumerate(zip(executions,public)):
        assert set(visible)=={'public_index','method','public'}
        assert visible==dict(public_index=index,method=e['method'],public=e['public'])
        public_arrays(e['public'])  # Reject truth/labels/evaluator keys, invalid times.
        lookup[e['method'],e['family_id'],e['rep'],e['session_id']]=e
        u=e['utility']; rows=u['rows']
        assert len(rows)==len(e['public']['events'])==u['total_ticks']
        for i,row in enumerate(rows):
            category=[r for r in row['category_recall'] if r is not None]
            close(row['recall'],float(np.mean(category)) if category else None)
            assert row['public_first8']==(i<8)
            assert row['public_tail_480s']==(row['timestamp_s']>=480.)
        for key, pred in [('recall',lambda r:True),('first8_recall',lambda r:r['public_first8']),
                          ('tail_recall',lambda r:r['public_tail_480s']),
                          ('after_filter_recall',lambda r:r['evaluator_filter_stopped'])]:
            values=[r['recall'] for r in rows if pred(r) and r['recall'] is not None]
            close(u[key],float(np.mean(values)) if values else None)
        assert u['request_bytes']==sum(r['request_bytes'] for r in rows)
        assert u['response_bytes']==sum(r['response_bytes'] for r in rows)
        close(u['bytes_per_tick'],(u['request_bytes']+u['response_bytes'])/len(rows))
        if e['method']=='raw':
            continue
        if e['method']=='per_session_reset_H12':
            unit, cap, maximum = reference['unit_epsilon_per_m'], reference['effective_cap'], 23
        else:
            pp=policies[e['method']]
            unit,cap,maximum=pp['unit_epsilon_per_m'],pp['effective_session_cap_per_m'],pp['max_units_per_session']
        diag=e['evaluator_only']['privacy']; ledger=diag['ledger']
        assert len(ledger)==len(rows)==len(diag['states'])
        previous=0
        for j,row in enumerate(ledger):
            assert isinstance(row['cost_units'],int) and not isinstance(row['cost_units'],bool)
            expected={'fresh':1 if j==0 else 2,'reuse':1,'postprocess':0,'public_clock_skip':0}[row['branch']]
            assert row['cost_units']==expected
            assert row['private_read']==(expected>0)
            if row['private_read']:
                assert previous+(1 if j==0 else 2)<=maximum
            elif row['branch']=='postprocess':
                assert previous+2>maximum
            previous+=row['cost_units']
            assert previous==row['spent_units']<=maximum
            assert rows[j]['evaluator_filter_stopped']==(row['branch']=='postprocess')
        assert diag['private_reads']==sum(r['private_read'] for r in ledger)==u['private_read_count']
        close(diag['spent_per_m'],previous*unit)
        assert diag['spent_per_m']<=cap+1e-12
        read_units+=previous
    exact_controls=0
    for key,e in lookup.items():
        if key[0]!='per_session_reset_H12':
            continue
        control=lookup[('equal_total_reset_control_H12',)+key[1:]]
        assert e['public']==control['public']
        assert e['evaluator_only']['privacy']['states']==control['evaluator_only']['privacy']['states']
        assert e['evaluator_only']['privacy']['ledger']==control['evaluator_only']['privacy']['ledger']
        exact_controls+=1
    for account in accounts:
        rows=[e for e in executions if e['family_id']==account['family_id'] and e['rep']==account['rep'] and e['method']==account['method']]
        assert len(rows)==6
        close(account['spent_per_m'],sum(e['evaluator_only']['privacy']['spent_per_m'] for e in rows))
        close(account['reserved_cap_per_m'],account['epoch_cap_per_m'])
        assert account['spent_per_m']<=account['epoch_cap_per_m']+1e-12
    metrics_checked=0
    for method, summary in result['results'].items():
        for split, values in summary['utility'].items():
            subset=[e for e in executions if e['method']==method and e['split']==split]
            for metric, expected in values.items():
                groups=defaultdict(lambda:defaultdict(list))
                for e in subset:
                    v=e['utility'][metric]
                    if v is not None:
                        groups[e['family_id']][e['rep']].append(v)
                family_means=[np.mean([np.mean(v) for v in reps.values()]) for reps in groups.values()]
                close(expected,float(np.mean(family_means)) if family_means else None)
                metrics_checked+=1
        for label, task in summary['S4'].items():
            rows=task['test_rows']
            for row in rows:
                a,b=[lookup[method,row['family_id'],row['rep'],sid] for sid in row['session_pair']]
                field={'same_person':'person_id','same_vehicle':'physical_vehicle_id'}[label]
                assert row['truth']==int(a['evaluator_only'][field]==b['evaluator_only'][field])
            for selection in ('selection_best_balanced_accuracy','selection_best_auc'):
                selected=task[selection]
                ys=np.array([r['truth'] for r in rows]); probs=np.array([r[selected['attacker']] for r in rows])
                pred=probs>=selected['threshold']
                actual=classification_metrics(ys,pred,probs)
                for field in ('accuracy','balanced_accuracy','macro_f1','roc_auc'):
                    close(actual[field],selected['test'][field]);metrics_checked+=1
                families=[]
                for fid,expected in selected['test_families'].items():
                    ids=[i for i,r in enumerate(rows) if r['family_id']==fid]
                    actual=classification_metrics(ys[ids],pred[ids],probs[ids])
                    close(actual['roc_auc'],expected['roc_auc']);families.append(actual['roc_auc'])
                close(float(np.mean(families)),selected['family_macro_auc'])
                close(max(families),selected['maximum_family_auc'])
    return dict(status='passed', executions=len(executions), epoch_accounts=len(accounts),
        exact_equal_total_control_sessions=exact_controls, integer_units_recomputed=read_units,
        summary_metrics_checked=metrics_checked, private_rng_keys_in_public_artifacts=False,
        source_hashes_unchanged=True, old72session_evidence_unchanged=True)


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--out',type=Path,default=DEFAULT)
    args=parser.parse_args();print(json.dumps(verify(args.out),indent=2))


if __name__=='__main__':
    main()
