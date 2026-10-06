"""Plot consumers reject unrecorded primaries, lost N/A and cost omissions."""
import json

import pytest

from experiments import plot_qplanner_response_depth_20261006 as plot


def coverage():
    return dict(defined_windows=8,total_windows=10,defined_categories=12,total_categories=60)


def estimate(delta,*,primary=False):
    return dict(mean_difference=delta,percentile95_family_bootstrap=[delta-.005,delta+.005],
        lower_tail_mean_difference=delta,percentile95_lower_tail_difference=[delta-.01,delta+.01],
        primary_contrast=primary,family_ids=['a','b'],independent_family_clusters=2,
        family_differences={'a':delta,'b':delta},left_reference_coverage=coverage(),right_reference_coverage=coverage(),
        within_draw={'draws':{str(d):{'paired_mean_difference':delta+(d-2)*.001} for d in (1,2,3)}})


def development():
    summary={}; contrasts={}
    for L,delta,ratio in ((20,0.,1.),(30,.025,1.3),(40,.04,1.6),(60,.06,2.2)):
        cells={p:dict(family_mean=.8+delta,family_values={'a':.7+delta,'b':.9+delta},**coverage())
            for p in (*plot.PURPOSES,'equal_purpose_macro')}
        summary[str(L)]={'selection':dict(current={'all':cells},cost=dict(requests=100,request_bytes=1000,reply_bytes=int(10000*ratio)))}
        if L!=20:contrasts[plot.contrast_key(L,'selection','equal_purpose_macro')]=estimate(delta)
    saved=dict(schema='qplanner-response-depth-readout-v1',summary=summary)
    paired=dict(schema='qplanner-response-depth-paired-readout-v1',defense_selected_by_this_readout=False,
        no_private_generation=True,contrasts=contrasts)
    selection=dict(selected_depth=30,candidates=[{'depth':L} for L in (30,40,60)])
    return saved,paired,selection


def fresh():
    contrasts={plot.contrast_key(30,'test',p):estimate(.025,primary=p=='equal_purpose_macro')
        for p in (*plot.PURPOSES,'equal_purpose_macro')}
    costs=[];cells={}
    for L in (20,30):
        for f in ('a','b'):
            for d in (1,2,3):costs.append(dict(method=f'service_l{L}',split='test',family_id=f,draw=d,
                requests=100,request_bytes=1000,reply_bytes=10000 if L==20 else 13000))
        cells[f'service_l{L}--test--current--all']={'equal_purpose_macro':{
            'family_draw_values':{f:{str(d):.9 for d in (1,2,3)} for f in ('a','b')}}}
    paired=dict(schema='qplanner-response-depth-paired-readout-v1',contrasts=contrasts,
        defense_selected_by_this_readout=False,no_private_generation=True,exact_Q_clock_ledger_certificate_checked=True,
        family_draw_costs=costs,conditional_family_cells=cells,primary_criterion_result={'passes':True})
    freeze=dict(schema='qplanner-selected-response-depth-freeze-v1',fresh_depth_scores_viewed=False,selected_depth=30,baseline_depth=20)
    return paired,freeze


def test_development_retains_all_arms_and_actual_paid_cost():
    prepared=plot.prepare_development(*development())
    assert [r['depth'] for r in prepared['rows']]==[20,30,40,60]
    assert prepared['selected_depth']==30
    assert prepared['rows'][1]['macro_delta']==pytest.approx(.025)
    assert prepared['rows'][1]['reply_ratio']==1.3
    assert prepared['coverage']['within_radius']['defined_categories']==12


@pytest.mark.parametrize('tamper',['missing_arm','wrong_mean','na_denominator','fake_primary'])
def test_development_metadata_cannot_silently_change_curve(tamper):
    saved,paired,selection=development()
    if tamper=='missing_arm':selection['candidates'].pop()
    elif tamper=='wrong_mean':paired['contrasts'][plot.contrast_key(30,'selection','equal_purpose_macro')]['mean_difference']=.04
    elif tamper=='na_denominator':saved['summary']['30']['selection']['current']['all']['within_radius']['defined_categories']=13
    else:next(iter(paired['contrasts'].values()))['primary_contrast']=True
    with pytest.raises(ValueError):plot.prepare_development(saved,paired,selection)


def test_fresh_reads_saved_ci_draws_and_full_matched_bytes_without_reselection():
    paired,freeze=fresh();value=plot.prepare_fresh(paired,freeze,selected_depth=30)
    assert value['reply_ratio']==1.3
    assert value['draw_values']=={'1':.024,'2':.025,'3':.026000000000000002}
    assert plot.estimate(value['rows']['equal_purpose_macro'])==(.025,(.02,.030000000000000002))


@pytest.mark.parametrize('tamper',['viewed','different_depth','unmarked_primary','cost_omitted_both','coverage'])
def test_fresh_unfrozen_primary_or_cohort_cost_tampering_rejected(tamper):
    paired,freeze=fresh()
    if tamper=='viewed':freeze['fresh_depth_scores_viewed']=True
    elif tamper=='different_depth':freeze['selected_depth']=40
    elif tamper=='unmarked_primary':paired['contrasts'][plot.contrast_key(30,'test','equal_purpose_macro')]['primary_contrast']=False
    elif tamper=='cost_omitted_both':paired['family_draw_costs']=[r for r in paired['family_draw_costs'] if r['family_id']!='b']
    else:paired['contrasts'][plot.contrast_key(30,'test','within_radius')]['right_reference_coverage']['defined_windows']=7
    with pytest.raises(ValueError):plot.prepare_fresh(paired,freeze,selected_depth=30)


def test_na_interval_remains_undefined_and_ordered_interval_required():
    row=estimate(.02);row['mean_difference']=None;row['percentile95_family_bootstrap']=None
    assert plot.estimate(row) is None
    row['percentile95_family_bootstrap']=[0.,0.]
    with pytest.raises(ValueError,match='N/A'):plot.estimate(row)
    row=estimate(.02);row['percentile95_family_bootstrap']=[.03,.01]
    with pytest.raises(ValueError,match='Ordered'):plot.estimate(row)


def test_plot_all_panels_from_saved_synthetic_fixture_only(tmp_path):
    prepared=plot.prepare_development(*development());paired,freeze=fresh()
    prepared['fresh']=plot.prepare_fresh(paired,freeze,selected_depth=30)
    figure=plot.plot(prepared)
    target=tmp_path/'fixture.pdf';figure.savefig(target)
    import matplotlib.pyplot as plt
    plt.close(figure)
    assert target.stat().st_size>1000


def test_render_pins_actual_input_bytes_and_rejects_selection_not_validated(tmp_path):
    saved,paired,selection=development();source=tmp_path/'development';source.mkdir()
    def write(name,value):
        path=source/name;path.write_text(json.dumps(value));return plot.sha(path)
    protocol_sha=write('protocol.json',{'schema':'qplanner-response-depth-development-v1'})
    saved['protocol_sha256']=protocol_sha;readout_sha=write('readout.json',saved)
    paired_protocol_sha=write('paired_protocol.json',dict(scope='development',source_protocol_sha256=protocol_sha,
        source_readout_sha256=readout_sha))
    paired.update(source_readout_sha256=readout_sha,paired_protocol_sha256=paired_protocol_sha)
    write('paired_readout.json',paired)
    selection.update(protocol_sha256=protocol_sha,readout_sha256=readout_sha)
    write('depth_selection.json',selection)
    write('validation.json',dict(status='pass',protocol_sha256=protocol_sha,readout_sha256=readout_sha,selected_depth=30))
    target=tmp_path/'figure';manifest=plot.render(source,target)
    assert manifest['inputs']['development_readout']['sha256']==readout_sha
    assert manifest['figure_source_sha256']==plot.sha(plot.__file__)
    assert manifest['raw_coordinates_or_rng_keys_read'] is False
    assert all(plot.sha(target/name)==digest for name,digest in manifest['output_sha256'].items())
    with pytest.raises(FileExistsError):plot.render(source,target)
    selection['selected_depth']=40;write('depth_selection.json',selection)
    with pytest.raises(ValueError,match='hash chain'):plot.render(source,tmp_path/'invalid')
    assert not (tmp_path/'invalid').exists()
