"""Replayable conservation, boundary, provenance and score checks for S4–S6.

Completed evidence is immutable. The default recheck retains an existing
validation.json; --validation-output writes a fresh record to a new path.
"""
import argparse
import json
from pathlib import Path
import numpy as np
from evaluation.identity_future import (public_arrays,check_group_split,classification_metrics,
                                         xy_from_latlon,prefix_features)
from experiments.identity_future_eval import ROOT,OUT,SPLITS,DATA,sha,save


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--validation-output',type=Path,
                        help='Write a new validation record here; refuse existing paths')
    args=parser.parse_args()
    output=args.validation_output if args.validation_output is not None else OUT/'validation.json'
    if args.validation_output is not None and output.exists():
        raise FileExistsError(f'Preserve completed evidence: {output}')
    checked=[]
    for name in ('future_results_v2.json','linkage_results.json','refinement_results.json','prefix_cut_results.json'):
        payload=json.loads((OUT/name).read_text())
        for path,expected in payload.get('source_sha256',{}).items():
            source=ROOT/path if (ROOT/path).exists() else OUT/path
            assert sha(source)==expected,(name,path)
        checked.append(name)
    data=json.loads(DATA.read_text());record_by_id={r['record_id']:r for r in data['records']}
    future=json.loads((OUT/'future_results_v2.json').read_text())
    cut=json.loads((OUT/'prefix_cut_results.json').read_text())
    assert len(future['rows'])==120 and len(cut['rows'])==120 and not cut['rejections']
    for r in cut['rows']:
        public_arrays(r['public']);record=record_by_id[r['record_id']]
        assert r['evaluator_only']['last_observed_index']==record['observed_indices'][0][-2]
        assert len(r['public']['events'])==len(record['observed_indices'][0])-1
        original=next(v for v in future['rows'] if (v['record_id'],v['rep'],v['method'])==
                      (r['record_id'],r['rep'],r['method']))
        assert r['public']['events']==original['public']['events'][:-1]
        trace=data['traces'][record['session_ids'][0]];target=trace[r['evaluator_only']['target_index']]
        assert np.array_equal(r['next_xy'],xy_from_latlon([[target['lat'],target['lon']]])[0])
        assert target['time_s']-trace[r['evaluator_only']['last_observed_index']]['time_s']==20.
    for method,result in cut['results'].items():
        assert result['test']['n']==6
        assert result['learned_bank']['counts']=={'train':12,'selection':6,'test':6}
    for result in future['results'].values():
        assert not result['S5_next_edge']['protection_conclusion_allowed']
        assert result['S5_next_edge']['train_label_coverage_test']==0.
        assert result['S6_destination']['test']['n']==6
    linkage=json.loads((OUT/'linkage_results.json').read_text())
    sessions=linkage['evaluator_sessions'];assert len(sessions)==72
    by_id={s['session_id']:s for s in sessions}
    check_group_split(*[[s['family_id'] for s in sessions if s['split']==split] for split in SPLITS])
    for target in ('person_id','physical_vehicle_id'):
        check_group_split(*[[s['evaluator_only'][target] for s in sessions if s['split']==split] for split in SPLITS])
    for s in sessions:
        for method in ('raw','geoi_slack_reconstructed'):public_arrays(s[method])
        truth=s['evaluator_only'];ledger=truth['ledger']
        assert sum(l['cost_units'] for l in ledger)*.01<=.23+1e-12
        assert np.isclose(sum(l['cost_units'] for l in ledger)*.01,truth['budget_spent'])
        assert all(l['spent_units']<=23 for l in ledger)
        assert len(s['raw']['events'])==len(s['geoi_slack_reconstructed']['events'])==len(ledger)
        for l,event in zip(ledger,s['geoi_slack_reconstructed']['events']):
            assert len(event['candidates'])==5
            assert l['cost_units']==(2 if l['branch']=='fresh' and event['timestamp_s']>0 else
                                     1 if l['branch'] in ('reuse','fresh') else 0)
    assert len(linkage['pair_rows'])==360
    for r in linkage['pair_rows']:
        a,b=[by_id[sid] for sid in r['session_pair']]
        assert a['family_id']==b['family_id']==r['family_id'] and a['split']==b['split']==r['split']
        assert r['same_person']==int(a['evaluator_only']['person_id']==b['evaluator_only']['person_id'])
        assert r['same_vehicle']==int(a['evaluator_only']['physical_vehicle_id']==b['evaluator_only']['physical_vehicle_id'])
    public=json.loads((OUT/'linkage_public_transcripts.json').read_text())
    assert len(public['transcripts'])==72
    for t in public['transcripts']:
        assert set(t)=={'public_index','raw','geoi_slack_reconstructed'}
        for name in ('raw','geoi_slack_reconstructed'):public_arrays(t[name])
    refined=json.loads((OUT/'refinement_results.json').read_text())
    for method,res in refined['linkage_results'].items():
        for label,r in res.items():
            truth=r['test_truth'];pred=r['test_predictions']
            recomputed=classification_metrics(truth,pred)
            assert all(np.isclose(recomputed[k],r['test'][k]) for k in ('accuracy','balanced_accuracy','macro_f1'))
            assert len(truth)==45 and sum(truth)==9
    # Empirical gate: raw controls must outperform majority and permutation for
    # each S4 target. No claim for the retained failed initial threshold run.
    raw=refined['linkage_results']['raw']
    assert raw['same_person']['test']['balanced_accuracy']>.5
    assert raw['same_vehicle']['test']['balanced_accuracy']>.5
    assert all(raw[label]['test']['balanced_accuracy']>
               raw[label]['permuted_training_control']['test']['balanced_accuracy'] for label in raw)
    assert cut['results']['raw']['test']['hit100']>0.
    record={'schema':'identity-future-validation-v1','status':'passed',
        'checked_artifacts':{p:sha(OUT/p) for p in checked},'validator_sha256':sha(Path(__file__)),
        'group_split_disjoint':True,'synthetic_people_and_vehicles_disjoint':True,
        'strict_public_feature_boundary':True,'private_ledger_conservation':True,
        'unchanged_frozen_public_prefixes':True,'fixed_cut_withheld_future_event':True,
        'S4_public_positive_controls_after_threshold_refinement':True,
        'S5_exact_edge_gate':'failed; unknown test labels and failed raw learner, no protection conclusion',
        'limits':['three dependent test route families','test reused for development diagnosis',
                  'synthetic identities only','S5 spatial proxy, not exactedge','S6 not fullfork/history suite']}
    if output.exists():
        # Re-run every assertion above, while retaining the original record and
        # its original validator hash. Never treat an old record as a recheck.
        print(f'Existing validation record retained unchanged: {output}',flush=True)
    else:
        save(output,record)
        print(f'Fresh validation record written: {output}',flush=True)
    print('Identity/future validation passed:72sessions,360pairs,120futureprefixes,120fixedcuts',flush=True)


if __name__=='__main__':main()
