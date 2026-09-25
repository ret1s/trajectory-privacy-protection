"""Independent arithmetic, cohort and input checks for the B diagnosis."""
from collections import defaultdict
import json

import numpy as np

from experiments.diagnose_s10b import OUT, OLD, ROOT, PAPERS, DATASETS, EXPANDED, check_sources
from experiments.endpoint_calendar_study import gzread
from experiments.research_loop_resources import sha
from experiments.run_live_paper_comparison import write_json


def read(path):return json.loads(path.read_text())


def score(rows, name, radius=None):
    families=defaultdict(list)
    for row in rows:
        e=np.array(row['errors'][name])
        families[row['family_id']].append(float(np.mean(e if radius is None else e<=radius)))
    return float(np.mean([np.mean(v) for v in families.values()]))


def main():
    protocol, selection, result=(read(OUT/n) for n in ('protocol.json','selection.json','readout.json'))
    check_sources(protocol)
    assert result['selection_sha256']==sha(OUT/'selection.json')
    assert result['evaluation_rows_sha256']==sha(OUT/'evaluation_rows.json.gz')
    assert result['protocol_sha256']==sha(OUT/'protocol.json')
    selected_rows, rows=(gzread(OUT/n) for n in ('selection_rows.json.gz','evaluation_rows.json.gz'))
    fit=set(selection['fit_families_present']); select=set(selection['selection_families_present'])
    evaluate={r['family_id'] for r in rows}
    assert not (fit & select or fit & evaluate or select & evaluate)
    checks=0
    for mode in ('B_specific','combined'):
        names=sorted(n for n in selected_rows[0]['errors'] if mode=='combined' or n.startswith('b_'))
        best=min(names,key=lambda n:(score(selected_rows,n),n))
        assert selection[mode]['mae']==best
        for radius in (50,100,200,500):
            best=min(names,key=lambda n:(-score(selected_rows,n,radius),score(selected_rows,n),n))
            assert selection[mode][f'hit{radius}']==best
            checks+=1
    original=read(OLD/'selection.json')['selected']['raw/S10.B']
    for mode, choices in [('original',original),('B_specific',selection['B_specific']),('combined',selection['combined'])]:
        np.testing.assert_allclose(result['raw_control'][mode]['mae_m'],score(rows,choices['mae']),rtol=0,atol=1e-10)
        for radius in (50,100,200,500):
            np.testing.assert_allclose(result['raw_control'][mode][f'hit{radius}'],score(rows,choices[f'hit{radius}'],radius),rtol=0,atol=1e-12)
            checks+=1
    old_rows=gzread(OLD/'holdout_privacy.json.gz')['rows']
    old={(r['record_id'],r['rep']):r for r in old_rows if r['method']=='raw' and r['case_id']=='S10.B'}
    for r in rows:
        a,b=old[r['record_id'],0],old[r['record_id'],1]
        assert a['errors']==b['errors'], 'Raw repetitions must be identical before dropping duplicate repetition'
        assert {k:r['errors'][k] for k in a['errors']}==a['errors'], 'Historical bank changed'
    counts={}
    inputs={}
    manifest=read(OLD/'transcript_hashes.json')
    for split,path,base,view_split in [('expanded',EXPANDED,OLD,'holdout')]+[
        (split,ROOT/f'artifacts/datasets/{name}/dataset.json',PAPERS,split) for split,name in DATASETS.items()]:
        d=read(path); six={r['family_id']:r for r in d['records'] if r['case_id']=='S6.A'}
        same,total=0,0
        for r in d['records']:
            if r['case_id']!='S10.B':continue
            total+=1
            same+=all(r[k]==six[r['family_id']][k] for k in ('session_ids','observed_indices','labels','observation_policy'))
            for sid in r['session_ids']:
                p=base/'transcripts'/view_split/'raw'/f'{sid}_r0.json.gz'
                rel=str(p.relative_to(ROOT));inputs[rel]=sha(p)
                if split=='expanded':assert inputs[rel]==manifest[rel]
        assert (same,total)==(result['specification_audits'][split]['same_as_S6A'],result['specification_audits'][split]['records'])
        counts[split]={'same_as_S6A':same,'records':total}
    frozen={(r['method'],r['case_id']):r for r in read(OLD/'readout.json')['summaries']}
    for r in result['frozen_A_C_descriptive_breakdown_not_new_confirmation']:
        for metric in ('hit100','hit200','mae_m'):
            expected=(frozen[r['method'],'S10.A'][metric]+frozen[r['method'],'S10.C'][metric])/2
            np.testing.assert_allclose(r['A_C_equal_weight'][metric],expected,rtol=0,atol=1e-12)
            checks+=1
    write_json(OUT/'input_transcript_hashes.json',inputs)
    write_json(OUT/'verification.json',{'status':'passed','readout_sha256':sha(OUT/'readout.json'),
        'source_sha256':sha(ROOT/'experiments/verify_s10b.py'),'independent_score_and_selection_checks':checks,
        'B_records':len(rows),'target_trips':sum(len(r['target_xy_evaluator_only']) for r in rows),
        'audited_duplicate_counts':counts,'fit_selection_evaluation_families_disjoint':True,
        'original_errors_unchanged':True,'raw_repetitions_equal_not_counted_twice':True,
        'raw_input_transcripts':len(inputs),'expanded_inputs_match_original_manifest':True,
        'input_transcript_hashes_sha256':sha(OUT/'input_transcript_hashes.json'),
        'limitations':'Not an independent confirmation: evaluation cohort already observed. Only 7 eligible fit and 7 selection families. No claim all attackers fail.'})
    print('Verified S10.B diagnostic:',checks,'score/selection checks,',len(inputs),'raw inputs',flush=True)


if __name__=='__main__':main()
