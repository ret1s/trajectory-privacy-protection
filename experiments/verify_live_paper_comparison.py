"""Independent accounting, case-coverage and boundary checks for comparison v1."""
from collections import defaultdict
import gzip
import json
from pathlib import Path
import numpy as np

from experiments.run_live_paper_comparison import OUT,protocol,METHODS,data_path,write_json
from experiments.research_loop_resources import ROOT,sha


def independent_family_mean(rows,field):
    nested=defaultdict(lambda:defaultdict(list))
    for r in rows:
        if r.get(field) is not None:nested[r['family_id']][r['record_id']].append(r[field])
    vals=[np.mean([np.mean(v) for v in rr.values()]) for rr in nested.values()]
    return float(np.mean(vals)) if vals else None


def verify():
    p=protocol();r=json.loads((OUT/'readout.json').read_text())
    assert r['protocol_sha256']==sha(OUT/'protocol.json')
    assert r['analysis_code_sha256']==sha(ROOT/'experiments/read_live_paper_comparison.py')
    attacker=json.loads((OUT/'attacker_protocol.json').read_text())
    for path,digest in attacker['source_sha256'].items():assert sha(ROOT/path)==digest,path
    assert attacker['initial_bank_sha256']==sha(OUT/'diagnostics/initial_attack_bank.json.gz')
    contracts=json.loads((OUT/'contract_checks.json').read_text())
    assert contracts['status']=='passed'
    assert contracts['protocol_sha256']==sha(OUT/'protocol.json')
    assert all(c['prefix_equal'] and c['future_perturbation_equal'] for c in contracts['prefix_checks'])
    for path,digest in r['source_sha256'].items():assert sha(ROOT/path)==digest,path
    privacy=json.loads(gzip.decompress((OUT/'privacy.json.gz').read_bytes()))
    assert privacy['protocol_sha256']==sha(OUT/'protocol.json')
    assert privacy['attacker_protocol_sha256']==sha(OUT/'attacker_protocol.json')
    assert privacy['analysis_code_sha256']==r['analysis_code_sha256']
    expected_cases={f'S{s}.{c}' for s in (1,2,3,9,10) for c in 'ABC'}
    assert len(r['summaries'])==2*9*15
    assert len(r['aggregates'])==2*9
    fits=set(p['fit_families']);validation=set(p['attack_selection_families'])
    evaluation=set(sum(p['evaluation_families'].values(),[]))
    assert not(fits&validation or fits&evaluation or validation&evaluation)
    for item in privacy['fit']:assert set(item['families'])<=fits
    checked=0;cost_checked=0;shards=0;failed=defaultdict(int);false_flags=0
    for split in ('development','new_groups'):
        service=json.loads((OUT/f'service_{split}.json').read_text())
        assert service['protocol_sha256']==sha(OUT/'protocol.json')
        for path,digest in service['transcript_sha256'].items():
            assert sha(ROOT/path)==digest,path;shards+=1
            ex=json.loads(gzip.decompress((ROOT/path).read_bytes()))
            if ex['status']!='ok':failed[ex['method']]+=1;continue
            assert ex['protocol_sha256']==sha(OUT/'protocol.json')
            t=[e['timestamp_s'] for e in ex['events']]
            assert all(a<b for a,b in zip(t,t[1:]))
            ledger={int(i):int(j) for i,j in ex['service_event_positions'].items()}
            assert set(ledger)==set(range(len(ex['clock_indices'])))
            for e in ex['events']:
                assert set(e)<= {'timestamp_s','coordinates','queries','server_states'}
                false_flags+=1
        for method in METHODS:
            aggregate=next(a for a in r['aggregates'] if a['split']==split and a['method']==method)
            for metric in ('request_bytes','response_bytes','category_queries','emissions','coordinates','distinct_coordinates','generation_ms'):
                costs=[dict(family_id=v['family_id'],record_id=v['session_id'],value=v[metric]/v['service_events']) for v in service['costs'] if v['method']==method]
                expected=independent_family_mean(costs,'value')
                assert np.isclose(expected,aggregate[metric+'_per_service_event'])
                cost_checked+=1
            ss=[s for s in r['summaries'] if s['split']==split and s['method']==method]
            assert {s['case_id'] for s in ss}==expected_cases
            for s in ss:
                for prob in p['probabilities']:
                    raw=[v for v in service['rows'] if v['method']==method and v['case_id']==s['case_id'] and v['probability']==prob]
                    expected=independent_family_mean(raw,'recall')
                    assert expected is None and s[f'recall_{prob}'] is None or np.isclose(expected,s[f'recall_{prob}'])
                    checked+=1
                raw=[v for v in privacy['rows'] if v['method']==method and v['split']==split and v['case_id']==s['case_id'] and v['status']=='ok']
                for field in ('mae_m','median_m','p90_m','hit50','hit100','hit200','hit500'):
                    for v in raw:
                        selected=v['selected'];err=np.array(v['errors'][selected['mae' if not field.startswith('hit') else field]])
                        expected=float(np.mean(err<=int(field[3:]))) if field.startswith('hit') else float({'mae_m':np.mean,'median_m':np.median,'p90_m':lambda e:np.quantile(e,.9)}[field](err))
                        assert np.isclose(v[field],expected)
                    expected=independent_family_mean(raw,field)
                    assert expected is None and s[field] is None or np.isclose(expected,s[field])
                    checked+=1
        # No suppressed failed session may be reported as perfect utility.
        assert all(v['recall'] in (None,0.) for v in service['rows'] if v['status']=='failed')
    assert set(failed)<= {'anotherme_offline'}
    write_json(OUT/'verification.json',{'status':'passed','protocol_sha256':sha(OUT/'protocol.json'),
        'readout_sha256':sha(OUT/'readout.json'),'code_sha256':sha(Path(__file__)),
        'attacker_protocol_sha256':sha(OUT/'attacker_protocol.json'),
        'contract_checks_sha256':sha(OUT/'contract_checks.json'),
        'summary_metrics_recomputed':checked,'evaluation_transcripts_checked':shards,
        'cost_aggregates_recomputed':cost_checked,
        'public_emission_whitelists_checked':false_flags,'failure_counts':dict(failed),
        'all_15_cases_per_method_split':True,'disjoint_fit_selection_evaluation_families':True,
        'scope':'Accounting and implementation-boundary verification, not proof of faithful paper reproduction or universal privacy'})
    print('Verified',checked,'metrics;',shards,'transcripts; failures',dict(failed),flush=True)


if __name__=='__main__':verify()
