"""S5 spatial proxy with a fixed cut of an archived causal public prefix.

The original S3 window often ends at the final GPS, leaving no future to score.
Remove its final public event for EVERY family before looking at target values.
This is a development repair, not new independent confirmation or exact S5-edge.
"""
import copy
import json
from pathlib import Path
from evaluation.identity_future import xy_from_latlon,public_arrays
from experiments.identity_future_eval import (ROOT,OUT,DATA,SPLITS,sha,save,
                                              evaluate_regression)
from experiments.identity_future_refine import geometry_future


def cut_rows(source,data):
    records={r['record_id']:r for r in data['records']}
    rows=[];rejections=[]
    for r in source:
        record=records[r['record_id']];indices=record['observed_indices'][0]
        if len(indices)<2:
            rejections.append({'record_id':r['record_id'],'reason':'need_two_public_events'});continue
        last=indices[-2];trace=data['traces'][record['session_ids'][0]]
        if last+20>=len(trace):
            rejections.append({'record_id':r['record_id'],'reason':'no20s_future_after_fixed_cut'});continue
        public={'events':copy.deepcopy(r['public']['events'][:-1])};public_arrays(public)
        rows.append({**{k:v for k,v in r.items() if k not in ('public','next_xy','next_edge','evaluator_only')},
            'public':public,'next_xy':xy_from_latlon([[trace[last+20]['lat'],trace[last+20]['lon']]])[0].tolist(),
            'evaluator_only':{'last_observed_index':last,'target_index':last+20,'horizon_s':20.,
                              'removed_public_events':1}})
    return rows,rejections


def main():
    save(OUT/'prefix_cut_protocol.json',{'schema':'fixed-public-prefix-cut-protocol-v1',
        'splits':SPLITS,'cut':'remove exactly final public event from every archived S3.A observation',
        'target':'GPS20seconds after newlastobservedindex','fit':'trainingfamilies only',
        'attacker_selection':'minimum selectionMAE across learned/geometric banks, stabletie byname',
        'scope':'S5 nextlocation spatialproxy, not exactnextedge, not confirmation; developmentrepair aftercontrolinspection'})
    source=OUT/'future_results_v2.json';original=json.loads(source.read_text())
    rows,rejections=cut_rows(original['rows'],json.loads(DATA.read_text()))
    geo=geometry_future(rows);result={}
    for method in sorted({r['method'] for r in rows}):
        learned=evaluate_regression([r for r in rows if r['method']==method],'next_xy')
        geometric=geo['results'][method]
        selected=min([('learned_bank',learned),('geometry_bank',geometric)],
                     key=lambda item:(item[1]['selection_mae_m'],item[0]))
        result[method]={'selected_bank':selected[0],'selected_attacker':selected[1]['selected_attacker'],
            'test':selected[1]['test'],'selection_mae_m':selected[1]['selection_mae_m'],
            'learned_bank':learned,'geometry_bank':geometric}
    save(OUT/'prefix_cut_results.json',{'schema':'fixed-prefix-next20s-diagnostic-v1',
        'protocol_sha256':sha(OUT/'prefix_cut_protocol.json'),
        'source_sha256':{'future_results_v2.json':sha(source),str(DATA.relative_to(ROOT)):sha(DATA),
            str(Path(__file__).relative_to(ROOT)):sha(Path(__file__)),
            'experiments/identity_future_eval.py':sha(ROOT/'experiments/identity_future_eval.py'),
            'experiments/identity_future_refine.py':sha(ROOT/'experiments/identity_future_refine.py'),
            'evaluation/identity_future.py':sha(ROOT/'evaluation/identity_future.py')},
        'scope':'New residual20s futurelocation proxy on unchanged causal archive prefixes; '
                'exactS5edge and originalS5eligible records remain untested.',
        'row_count':len(rows),'rows':rows,'rejections':rejections,
        'public_catalogue_points':geo['public_catalogue_points'],'results':result})


if __name__=='__main__':main()
