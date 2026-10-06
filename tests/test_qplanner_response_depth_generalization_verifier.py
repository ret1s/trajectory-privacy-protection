"""Independent statistical-verifier fixtures; no fresh artifacts are read."""
from copy import deepcopy
import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from experiments.qplanner_paired_readout_20261006 import (
    build_readout, paired as production_paired, primary_decision, summarize_bundles,
)
from experiments.qplanner_response_depth_generalization_20261006_v2 import CRITERION
from experiments.verify_qplanner_response_depth_generalization_20261006 import (
    PURPOSES, contrasts, decision, draw_diagnostic, family_cells, paired, tail_mean,
)
from experiments import verify_qplanner_response_depth_generalization_20261006 as verifier
from experiments import qplanner_response_depth_generalization_20261006_v2 as adoption


def assert_same_numbers(left,right):
    """Allow arithmetic-order rounding while retaining exact schema/NA fields."""
    if isinstance(left,dict):
        assert left.keys()==right.keys()
        for key in left:assert_same_numbers(left[key],right[key])
    elif isinstance(left,(list,tuple)):
        assert len(left)==len(right)
        for a,b in zip(left,right):assert_same_numbers(a,b)
    elif isinstance(left,(float,np.floating)):
        assert left==pytest.approx(right,abs=1e-13,rel=0.)
    else:assert left==right


def fixture():
    rows, wires, bundles = [], [], []
    for family, base in [('a', .5), ('b', .7), ('c', .8)]:
        for draw in range(1, 4):
            local, local_wire = [], []
            for method in ('service_l20', 'service_l30'):
                for time in (0, 400):
                    wire = dict(family_id=family, split='test', draw=draw, method=method,
                        slot=0, t=time, requests=5, request_bytes=100,
                        reply_bytes=1000 if method=='service_l20' else 1300)
                    wires.append(wire);local_wire.append(wire)
                    for cache in ('current', 'static_epoch_cache'):
                        scores={}
                        for purpose in PURPOSES:
                            undefined=family=='b' and purpose=='within_radius'
                            value=None if undefined else base+(.03 if method=='service_l30' else 0.)
                            scores[purpose]=dict(recall5=value,reference_category_count=0 if undefined else 1,
                                all_category_count=2)
                        row=dict(family_id=family, split='test', draw=draw, method=method,
                            cache=cache, slot=0, t=time, purposes=scores)
                        rows.append(row);local.append(row)
            bundles.append(dict(evaluator_only=dict(family_id=family,split='test',draw=draw),
                                utility=local,wire=local_wire))
    return rows,wires,bundles


@pytest.mark.parametrize('mass', [.25, .5, 1.])
def test_fractional_rank_tail_matches_literal_weighted_order_statistic(mass):
    values=[.9,.1,.3,.8,.5,.6]
    remaining=len(values)*mass;total=0.
    for value in sorted(values):
        take=min(remaining,1.);total+=take*value;remaining-=take
        if remaining<=0:break
    assert tail_mean(values,mass)==pytest.approx(total/(len(values)*mass))
    assert tail_mean([values,values],mass)==pytest.approx([total/(len(values)*mass)]*2)


def test_empty_tail_and_paired_unknown_families_remain_na():
    assert tail_mean([]) is None
    value=paired({'a':None},{'b':.5})
    assert value['mean_difference'] is None and value['independent_family_clusters']==0
    assert value['excluded_missing_or_undefined_family_pairs']==['a','b']


def test_independent_bootstrap_matches_family_cluster_rng_and_tail_difference_not_delta_tail():
    left={'a':.9,'b':.4,'c':.6,'d':None}
    right={'a':.2,'b':.3,'c':.8,'e':.9}
    independent=paired(left,right)
    production=production_paired(left,right)
    for field in ('mean_difference','percentile95_family_bootstrap','lower_tail_mean_difference',
                  'percentile95_lower_tail_difference','left_lower_tail_mean','right_lower_tail_mean'):
        assert independent[field]==pytest.approx(production[field],abs=1e-13)
    assert independent['independent_family_clusters']==3
    assert independent['excluded_missing_or_undefined_family_pairs']==['d','e']
    assert independent['lower_tail_mean_difference']!=pytest.approx(tail_mean([left[f]-right[f] for f in ['a','b','c']]))


def test_independent_cells_nested_draws_na_coverage_cost_and_all_contrasts_match():
    rows,wires,bundles=fixture()
    cells,cost=family_cells(rows,wires,{'test':3})
    expected,expected_cost=summarize_bundles(bundles,PURPOSES,{'test':[1,2,3]},
        expected_methods=('service_l20','service_l30'),expected_caches=('current','static_epoch_cache'))
    assert_same_numbers(cells,expected)
    assert cost==expected_cost
    macro=cells['service_l30','test','current','all']['equal_purpose_macro']
    assert macro['partial_or_undefined_purpose_families']==['b']
    assert macro['complete_all_purpose_families']==['a','c']
    assert cells['service_l20','test','current','all']['within_radius']['family_values']['b'] is None
    for family in ('a','b','c'):
        assert cost['service_l20','test',family,1]['requests']==10
        assert cost['service_l30','test',family,1]['reply_bytes']==2600
    values=contrasts(cells,{'test':3},[20,30])
    generic,_=build_readout(expected,{'test':[1,2,3]},['service_l20','service_l30'],
                           'service_l20','service_l30','service_l20')
    for key,value in values.items():
        for field in ('mean_difference','percentile95_family_bootstrap','lower_tail_mean_difference',
                      'percentile95_lower_tail_difference','primary_contrast','within_draw'):
            assert_same_numbers(value[field],generic[key][field])
    assert sum(v['primary_contrast'] for v in values.values())==1
    assert_same_numbers(decision(values,30,CRITERION),primary_decision(generic,'service_l30','service_l20',CRITERION))
    assert decision(values,30,CRITERION)['passes'] is True


@pytest.mark.parametrize('fault',['duplicate_row','duplicate_wire','missing_draw','invalid_na','nan_recall','cross_split'])
def test_invalid_conditional_cluster_inputs_reject(fault):
    rows,wires,_=fixture()
    if fault=='duplicate_row':rows.append(deepcopy(rows[0]))
    elif fault=='duplicate_wire':wires.append(deepcopy(wires[0]))
    elif fault=='missing_draw':rows=[r for r in rows if not (r['family_id']=='a' and r['draw']==3)]
    elif fault=='invalid_na':rows[0]['purposes'][PURPOSES[0]]['recall5']=None
    elif fault=='nan_recall':rows[0]['purposes'][PURPOSES[0]]['recall5']=float('nan')
    else:rows[0]['split']='selection'
    with pytest.raises((AssertionError,KeyError)):
        family_cells(rows,wires,{'test':3,'selection':3})


def test_draw_sign_diagnostics_require_common_defined_family_pairs_not_three_replicates():
    left={'a':{'1':.8,'2':.4,'3':.6},'b':{'1':.8,'2':None,'3':.9}}
    right={'a':{'1':.5,'2':.5,'3':.6},'b':{'1':.5,'2':.5,'3':.5}}
    result=draw_diagnostic(left,right,[1,2,3])
    assert result['complete_family_pairs']==['a']
    assert result['sign_consistency']==dict(positive_draws=1,negative_draws=1,zero_draws=1,defined_draws=3)
    assert all(v['family_clusters']==1 for v in result['draws'].values())


@pytest.mark.parametrize('fault',['mean','lower_bound','negative_draw','zero_draw','missing_primary'])
def test_primary_requires_all_frozen_gates_and_does_not_rotate_to_other_depth(fault):
    rows,wires,_=fixture();cells,_=family_cells(rows,wires,{'test':3})
    values=contrasts(cells,{'test':3},[20,30]);key='service_l30--minus--service_l20--test--current--all--equal_purpose_macro'
    if fault=='mean':values[key]['mean_difference']=.01999
    elif fault=='lower_bound':values[key]['percentile95_family_bootstrap'][0]=0.
    elif fault in ('negative_draw','zero_draw'):
        values[key]['within_draw']['draws']['2']['paired_mean_difference']=-.01 if fault=='negative_draw' else 0.
    else:values.pop(key)
    values['service_l40--minus--service_l20--test--current--all--equal_purpose_macro']={
        'mean_difference':.99,'percentile95_family_bootstrap':[.98,1.]}
    assert decision(values,30,CRITERION)['passes'] is False


def verification_fixture(root,monkeypatch):
    out=root/'artifacts/benchmarks/verification';out.mkdir(parents=True)
    p=dict(source_sha256={},selected_depth=30,depths=[20,30],criterion=deepcopy(CRITERION),
           dataset_path='artifacts/datasets/absent-fresh.json.gz')
    for name in verifier.verification_sources(p):
        path=root/name;path.parent.mkdir(parents=True,exist_ok=True)
        path.write_bytes((verifier.ROOT/name).read_bytes())
    (out/'protocol.json').write_text(json.dumps(p))
    (out/'depth_freeze.json').write_text(json.dumps({'selected_depth':30}))
    def source_only_contract(path,*,root):
        assert Path(path)==out
        assert not (Path(root)/p['dataset_path']).exists()
        assert not (out/'readout.json').exists()
        return dict(fresh_dataset_opened_for_contract=False,fresh_metrics_opened_for_contract=False)
    monkeypatch.setattr(adoption,'depth_contract',source_only_contract)
    return out,p


def test_independent_verification_declaration_write_once_before_fresh_data_and_replay(tmp_path,monkeypatch):
    out,p=verification_fixture(tmp_path,monkeypatch)
    v=verifier.declare_verification(out,root=tmp_path)
    assert v['declaration_before_depth_replay'] is True
    assert v['fresh_dataset_or_depth_scores_opened_for_declaration'] is False
    assert v['synthetic_fixture_test_count']==20
    assert verifier.verification_contract(out,root=tmp_path)['synthetic_fixture_test_count']==20
    assert not (tmp_path/p['dataset_path']).exists() and not (out/'readout.json').exists()
    original=(out/'verification_protocol.json').read_bytes()
    with pytest.raises(FileExistsError):verifier.declare_verification(out,root=tmp_path)
    assert (out/'verification_protocol.json').read_bytes()==original


def test_independent_verification_rejects_tampered_sources_scope_and_late_declaration(tmp_path,monkeypatch):
    for fault in ('source','snapshot','protocol','criterion','after_replay'):
        root=tmp_path/fault;out,p=verification_fixture(root,monkeypatch)
        if fault=='after_replay':
            (out/'replay_started.json').write_text('{}')
            with pytest.raises(AssertionError):verifier.declare_verification(out,root=root)
            assert not (out/'verification_protocol.json').exists()
            continue
        verifier.declare_verification(out,root=root)
        if fault in ('source','snapshot'):
            directory=root if fault=='source' else out/'verification_source_snapshot'
            (directory/verifier.THIS).write_bytes(b'changed independent audit code')
        elif fault=='protocol':
            p['depths']=[20,40];(out/'protocol.json').write_text(json.dumps(p))
        else:
            path=out/'verification_protocol.json';v=json.loads(path.read_text())
            v['criterion']['minimum_absolute_mean_gain']=.01
            path.write_text(json.dumps(v))
            (out/'verification_protocol.sha256').write_text(hashlib.sha256(path.read_bytes()).hexdigest()+'\n')
        with pytest.raises(AssertionError):verifier.verification_contract(out,root=root)
