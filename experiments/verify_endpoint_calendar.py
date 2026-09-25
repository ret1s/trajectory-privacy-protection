"""Independent aggregation and input-boundary audit of the endpoint study."""
from collections import defaultdict,Counter
import hashlib
import json
from pathlib import Path
import numpy as np

from experiments.endpoint_calendar_study import OUT,DATA,FIT,SELECT,METHODS,CASES,gzread,protocol,DATASETS
from experiments.research_loop_resources import ROOT,sha
from experiments.run_live_paper_comparison import protocol as original_protocol,write_json


def independent_mean(rows,key):
    pairs=defaultdict(list)
    for row in rows:
        if row.get(key) is not None:pairs[row['family_id'],row['record_id']].append(row[key])
    families=defaultdict(list)
    for (family,_),values in pairs.items():families[family].append(sum(values)/len(values))
    return sum(sum(v)/len(v) for v in families.values())/len(families) if families else None


def verify():
    p=protocol();original_protocol();sel=json.loads((OUT/'selection.json').read_text())
    read=json.loads((OUT/'readout.json').read_text());data=json.loads((DATA/'dataset.json').read_text())
    privacy=gzread(OUT/'holdout_privacy.json.gz');service=json.loads((OUT/'holdout_service.json').read_text())
    flow=json.loads((OUT/'calendar_flow_checks.json').read_text())
    assert flow['status']=='passed'
    for path,digest in flow['source_sha256'].items():assert sha(ROOT/path)==digest
    for path,digest in read['source_sha256'].items():assert sha(ROOT/path)==digest,path
    runner=sha(ROOT/'experiments/endpoint_calendar_study.py')
    assert sel['runner_sha256']==privacy['runner_sha256']==service['runner_sha256']==runner
    assert privacy['selection_sha256']==sha(OUT/'selection.json')
    assert privacy['dataset_sha256']==service['dataset_sha256']==read['dataset_sha256']==sha(DATA/'dataset.json')
    assert sel['selection_rows_sha256']==sha(OUT/'selection_rows.json.gz')
    for split,digest in sel['input_dataset_sha256'].items():assert sha(ROOT/f'artifacts/datasets/{DATASETS[split]}/dataset.json')==digest
    assert sha(ROOT/sel['model_path'])==sel['model_sha256']
    test={f['family_id'] for f in data['families']}
    assert not(FIT&SELECT or FIT&test or SELECT&test)
    assert test|{f"family-{v['seed']}" for v in data['construction_failures']}=={f'family-{s}' for s in p['holdout_seeds']}
    assert all(x['replacement'] is None for x in data['construction_failures'])
    assert all(set(f['families'])<=FIT for f in sel['fit'])
    assert all(r['family_id'] in SELECT for r in gzread(OUT/'selection_rows.json.gz'))
    # Inspect exact route overlap instead of treating seed differences as proof.
    historical_routes=set()
    for split,name in DATASETS.items():
        old=json.loads((ROOT/f'artifacts/datasets/{name}/dataset.json').read_text())
        historical_routes.update(tuple(s['route_edges']) for f in old['families'] for s in f['sessions'])
    overlap=sum(tuple(s['route_edges']) in historical_routes for f in data['families'] for s in f['sessions'])
    metrics=0;cost_metrics=0
    assert len(read['summaries'])==len(METHODS)*len(CASES)
    for summary in read['summaries']:
        m,c=summary['method'],summary['case_id']
        rows=[r for r in privacy['rows'] if r['method']==m and r['case_id']==c]
        good=[r for r in rows if r['status']=='ok']
        assert summary['attempted_record_runs']==len(rows) and summary['valid_record_runs']==len(good)
        for r in good:
            assert r['selected']==sel['selected'][m+'/'+c]
            errors=np.array(r['errors'][r['selected']['mae']])
            assert np.isclose(r['mae_m'],errors.mean())
            for radius in (50,100,200,500):assert np.isclose(r[f'hit{radius}'],np.mean(np.array(r['errors'][r['selected'][f'hit{radius}']])<=radius))
            for key,value in r['posterior_metrics'][r['selected']['posterior']].items():assert r[key]==value
        for field in ('mae_m','median_m','p90_m','hit50','hit100','hit200','hit500','log_gain_bits','nll_bits','entropy_bits','credible90_cells','credible90_coverage'):
            val=independent_mean(good,field)
            assert val is None and summary[field] is None or np.isclose(val,summary[field]);metrics+=1
        for q in p['probabilities']:
            rr=[r for r in service['rows'] if r['method']==m and r['case_id']==c and r['probability']==q]
            assert np.isclose(independent_mean(rr,'recall'),summary[f'recall_{q}']);metrics+=1
    for a in read['aggregates']:
        m=a['method']
        for field in ('request_bytes','response_bytes','emissions'):
            rows=[{'record_id':r['session_id'],'family_id':r['family_id'],'ratio':r[field]/r['service_events']} for r in service['costs'] if r['method']==m]
            assert np.isclose(independent_mean(rows,'ratio'),a[field+'_per_service_event']);cost_metrics+=1
        for q in p['probabilities']:
            per_case={c:independent_mean([r for r in service['rows'] if r['method']==m and r['case_id']==c and r['probability']==q],'recall') for c in a[f'case_recall_{q}']}
            for c,v in per_case.items():assert v is None and a[f'case_recall_{q}'][c] is None or np.isclose(v,a[f'case_recall_{q}'][c]);metrics+=1
            assert np.isclose(np.mean([v for v in per_case.values() if v is not None]),a[f'recall_{q}'])
        if m.startswith('calendar'):
            assert a['outside_interval_events']==0
            assert all(r['emissions']==60 for r in service['costs'] if r['method']==m)
    # Failure is per service request, not necessarily the whole multi-session
    # record. Its other session may succeed; retain both in the denominator.
    record_lookup={r['record_id']:r for r in data['records']}
    failed_rows=0
    for row in service['rows']:
        if row['status']!='failed':continue
        record=record_lookup[row['record_id']];good_count=0;bad_sessions=0
        for sid,ids in zip(record['session_ids'],record['observed_indices']):
            ex=gzread(OUT/'transcripts/holdout'/row['method']/f"{sid}_r{row['rep']}.json.gz")
            if ex['status']=='ok':good_count+=len(ids)
            else:bad_sessions+=1
        assert bad_sessions>0
        if row['recall'] is not None:
            assert 0<=row['recall']<=min(1.,good_count/row['eligible_events'])+1e-12
        failed_rows+=1
    transcripts=0;failed=Counter();hashes={}
    for path in sorted((OUT/'transcripts/holdout').rglob('*.gz')):
        ex=gzread(path);assert ex['protocol_sha256']==sha(OUT/'protocol.json')
        hashes[str(path.relative_to(ROOT))]=sha(path);transcripts+=1
        if ex['status']!='ok':failed[ex['method']]+=1;continue
        times=[e['timestamp_s'] for e in ex['events']];assert all(a<b for a,b in zip(times,times[1:]))
        assert set(map(int,ex['service_event_positions']))==set(range(len(ex['clock_indices'])))
    assert set(failed)<={'anotherme_offline'}
    write_json(OUT/'transcript_hashes.json',hashes)
    write_json(OUT/'verification.json',{'status':'passed','code_sha256':sha(Path(__file__)),
        'readout_sha256':sha(OUT/'readout.json'),'protocol_sha256':sha(OUT/'protocol.json'),
        'calendar_flow_checks_sha256':sha(OUT/'calendar_flow_checks.json'),
        'transcript_hashes_sha256':sha(OUT/'transcript_hashes.json'),
        'case_metrics_recomputed':metrics,'cost_aggregates_recomputed':cost_metrics,
        'failure_record_accounting_checks':failed_rows,
        'transcripts_checked':transcripts,'generation_failures':dict(failed),
        'planned_families':len(p['holdout_seeds']),'completed_families':len(test),
        'construction_failures':data['construction_failures'],'exact_route_overlaps_with_historical_sessions':overlap,
        'source_scope':'same-city synthetic endpoint check, disjoint families, local paper adaptations',
        'holds_for':'public region/subscription fixed before private activity; payload confidentiality, not IP/account/fault timing'})
    print('Verified',metrics,'case metrics,',cost_metrics,'cost aggregates;',transcripts,'transcripts; exact route overlaps:',overlap,flush=True)


if __name__=='__main__':verify()
