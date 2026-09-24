"""Check immutable iteration controls, actual SUMO completion and information boundaries."""
from pathlib import Path
import hashlib,json
ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'artifacts/benchmarks/research_loop'

def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
    data=json.loads((ROOT/'artifacts/datasets/research_loop_development_v1/dataset.json').read_text())
    assert len(data['families'])==4 and len(data['traces'])==88
    for f in data['families']:
        assert len(f['sessions'])==22
        assert all(float(f['actual_vehicles'][s['session_id']]['arrival'])>=0 for s in f['sessions'])
    for r in data['records']:
        for sid,indices in zip(r['session_ids'],r['observed_indices']):
            assert all(0<=i<len(data['traces'][sid]) for i in indices)
    checked=[];control={}
    for filename in ('iteration03_boundary.json','iteration04_filter.json','iteration05_progress.json','iteration07_slack.json','iteration08_matched.json','iteration09_switching.json'):
        p=OUT/filename
        if not p.exists():continue
        x=json.loads(p.read_text())
        for r in x['rows']:
            a=r['accounting'];assert a['input_events']==a['head_suppressed']+a['released_events']+a['tail_cancelled']
            assert a['released_events']==len(r['public']['events'])
            assert r['budget_bound'] is None or r['budget_bound']<=.24+1e-12
            if filename in ('iteration08_matched.json','iteration09_switching.json') and r['budget_bound'] is not None:assert r['budget_bound']<=.23+1e-12
            if r.get('evaluator_ledger'):
                ledger=r['evaluator_ledger'];assert sum(v['cost_units'] for v in ledger)==ledger[-1]['spent_units']<=24
                assert r['privacy_reads']==sum(v['private_read'] for v in ledger)
            if r['method'] in ('raw','core_H12'):
                key=(r['session_id'],r['method'])
                value=(r['public']['events'],r['recall_all_current_queries'],r['budget_bound'])
                if key in control:assert control[key]==value,(filename,key)
                else:control[key]=value
            text=json.dumps(r['public'])
            assert not any(k in text for k in ('evaluator_ledger','spent_units','target_index','truth'))
        checked.append({'file':filename,'sha256':sha(p),'full_session_rows':len(x['rows'])})
    case_check = None
    p=OUT/'iteration10_cases_checked.json'
    if p.exists():
        from experiments.research_loop_cases import public_view, select_and_summarize
        x=json.loads(p.read_text())
        json.dumps(x,allow_nan=False)
        original=json.loads((OUT/'iteration10_cases.json').read_text())
        assert x['code_sha256']==sha(OUT/'sources/iteration10_cases_v1.py')
        assert x['aggregation_repair']['original_artifact_sha256']==sha(OUT/'iteration10_cases.json')
        lookup={};eligibility={}
        for ex,old in zip(x['executions'],original['executions']):
            assert ex['events']==old['events']
            key=(ex['session_id'],ex['rep'],ex['method']);assert key not in lookup
            lookup[key]=ex
            assert list(map(int,ex['events']))==ex['clock_indices']
            assert ex['eligible_events']+ex['empty_reference_events']==len(ex['events'])
            flags=[u['5'] is not None for u in ex['utility_by_index'].values()]
            ekey=(ex['session_id'],ex['rep'])
            if ekey in eligibility:assert eligibility[ekey]==flags
            else:eligibility[ekey]=flags
            assert ex['budget_bound'] is None or ex['budget_bound']<=.23+1e-12
            if ex.get('evaluator_ledger'):
                ledger=ex['evaluator_ledger'];assert sum(v['cost_units'] for v in ledger)==ledger[-1]['spent_units']<=23
        records={r['record_id']:r for r in data['records']}
        for row,old in zip(x['rows'],original['rows']):
            assert row['errors']==old['errors']
            record=records[row['record_id']]
            for slot,(sid,indices) in enumerate(zip(record['session_ids'],record['observed_indices'])):
                ex=lookup[sid,row['rep'],row['method']]
                view=public_view({int(i):e for i,e in ex['events'].items()},indices)
                assert view==row['public_views'][slot]
                assert not any(k in json.dumps(view) for k in ('family_id','session_id','truth','target_index','evaluator_ledger'))
            assert row['composition_bound'] is None or abs(row['composition_bound']-.23*len(set(record['session_ids'])))<1e-12
        assert x['summaries']==select_and_summarize(x['rows'])
        case_check={'file':p.name,'sha256':sha(p),'full_session_executions':len(x['executions']),
                    'case_rows':len(x['rows']),'case_types':len({r['case_id'] for r in x['rows']}),
                    'source_sessions':len({r['session_id'] for r in x['executions']}),
                    'exact_allowed_views_checked':True,'unchanged_defender_outputs_and_privacy_errors':True}
    shadow_check=None
    p=OUT/'iteration11_shadow_attack.json'
    if p.exists():
        x=json.loads(p.read_text());training=OUT/'iteration11_shadow_training.json'
        assert x['training_sha256']==sha(training)
        assert x['source_sha256']==sha(OUT/'iteration09_switching.json')
        d=json.loads(training.read_text());assert len(d['x'])==len(d['y'])==len(d['public'])==2000
        first={}
        for row in json.loads((OUT/'iteration09_switching.json').read_text())['rows']:
            if row['method']=='raw':continue
            event=row['public']['events'][0]
            if row['session_id'] in first:assert first[row['session_id']]==event
            else:first[row['session_id']]=event
        shadow_check={'file':p.name,'sha256':sha(p),'public_simulated_shadows':2000,
                      'first_query_identical_across_protected_methods':True}
    from experiments.verify_research_loop_extended import check
    extended=check()
    from experiments.verify_research_loop_expanded import check as check_expanded
    expanded=check_expanded()
    from experiments.verify_research_loop_mobility import check as check_mobility
    mobility=check_mobility()
    from experiments.verify_research_loop_public_cover import check as check_public_cover
    public_cover=check_public_cover()
    from experiments.verify_research_loop_backbone import check as check_backbone
    backbone=check_backbone()
    from experiments.verify_research_loop_site_density import check as check_site_density
    site_density=check_site_density()
    from experiments.verify_research_loop_capped import check as check_capped
    capped=check_capped()
    from experiments.verify_research_loop_planar import check as check_planar
    planar=check_planar()
    from experiments.verify_research_loop_planar_first import check as check_planar_first
    planar_first=check_planar_first()
    from experiments.verify_research_loop_supplement import check as check_supplement
    supplement=check_supplement()
    from experiments.verify_research_loop_live_service import check as check_live_service
    live_service=check_live_service()
    live_readout=None
    if (OUT/'iteration28_readout.json').exists():
        from experiments.research_loop_live_readout import calculate as live_calculate
        assert json.loads((OUT/'iteration28_readout.json').read_text())==live_calculate()
        live_readout={'file':'iteration28_readout.json','sha256':sha(OUT/'iteration28_readout.json'),
                      'all_readout_and_family_bootstrap_recomputed':True}
    readout_check=None
    readout_path=OUT/'iteration25_26_readout.json'
    if readout_path.exists():
        from experiments.research_loop_capped_planar_readout import calculate
        assert json.loads(readout_path.read_text())==calculate()
        readout_check={'file':readout_path.name,'sha256':sha(readout_path),'all_case_readout_recomputed':True}
    result={'status':'passed','scope':'source completion, indices, accounting, budget caps, public schema and identical fixed controls; not efficacy proof',
        'completed_sumo_sessions':88,'records':len(data['records']),'iterations':checked,
        'persistent_case_views':case_check,'shadow_attack':shadow_check,
        'extended_iterations':extended,
        'expanded_development':expanded,
        'auxiliary_mobility':mobility,
        'public_cover':public_cover,
        'public_backbone':backbone,
        'repeated_site_density':site_density,
        'capped_service':capped,
        'planar_anchor':planar,
        'planar_first_query':planar_first,
        'public_supplement':supplement,
        'live_service':live_service,
        'live_service_readout':live_readout,
        'capped_planar_readout':readout_check,
        'verifier_sha256':sha(Path(__file__)),
        'extended_verifier_sha256':sha(ROOT/'experiments/verify_research_loop_extended.py'),
        'expanded_verifier_sha256':sha(ROOT/'experiments/verify_research_loop_expanded.py'),
        'mobility_verifier_sha256':sha(ROOT/'experiments/verify_research_loop_mobility.py'),
        'public_cover_verifier_sha256':sha(ROOT/'experiments/verify_research_loop_public_cover.py'),
        'backbone_verifier_sha256':sha(ROOT/'experiments/verify_research_loop_backbone.py'),
        'site_density_verifier_sha256':sha(ROOT/'experiments/verify_research_loop_site_density.py'),
        'capped_verifier_sha256':sha(ROOT/'experiments/verify_research_loop_capped.py'),
        'planar_verifier_sha256':sha(ROOT/'experiments/verify_research_loop_planar.py'),
        'planar_first_verifier_sha256':sha(ROOT/'experiments/verify_research_loop_planar_first.py'),
        'supplement_verifier_sha256':sha(ROOT/'experiments/verify_research_loop_supplement.py'),
        'live_service_verifier_sha256':sha(ROOT/'experiments/verify_research_loop_live_service.py')}
    (OUT/'verification.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))
if __name__=='__main__':main()
