import numpy as np
from experiments.contribution_stress import endpoint_predictions, cross_seed_select, paired_ablation

def test_endpoint_estimators_obey_linear_motion():
    events=[{'timestamp_s':float(t),'candidates':[{'lat':0.,'lon':t/1000}]} for t in (0,20,40,60,80,100)]
    public={'events':events}
    pred=endpoint_predictions(public,'S9',0.)
    assert len(pred)==31
    # Moving east: extrapolating back 60 s must point west of first sample.
    assert pred['mean_ols3_60s'][0]<0
    assert abs(pred['mean_ols3_60s'][1])<1e-9
    assert endpoint_predictions(public,'S10',0.)['mean_ols3_60s'][0]>pred['window_mean'][0]

def test_cross_seed_selection_never_uses_held_out_errors():
    rows=[{'seed':s,'record_id':str(s),'errors':{'a':10. if s!=3 else 1000.,'b':200.},'recall_all_queries':1.} for s in (1,2,3)]
    out=cross_seed_select(rows)
    held=next(r for r in out if r['seed']==3)
    assert held['selected_mae']=='a' and held['mae_m']==1000.
    assert held['selection_seeds']==[1,2]
    rows[2]['errors']['a']=5000.
    assert next(r for r in cross_seed_select(rows) if r['seed']==3)['selected_mae']=='a'

def test_paired_identical_methods_have_zero_contrast():
    rows=[];summaries=[]
    for method in ('geometric','switching_exchange'):
        summaries.append({'method':method,'case_id':'S1.A','selected_mae_attack':'a','selected_hit_attack':'a'})
        for f in range(3):
            for rep in range(2):
                rows.append({'method':method,'case_id':'S1.A','family_id':str(f),
                    'errors_by_attack':{'a':[10+f]},'step_ms':[2.],
                    'utility':{d:{'poi_recall_at_5':.9,'response_id_bytes_per_event':20.} for d in ('5','10')}})
    out=paired_ablation({'rows':rows,'summaries':summaries},draws=100)
    for v in out['contrasts'][0]['metrics'].values():
        assert v['delta']==0 and np.array_equal(v['ci95'],[0.,0.])
