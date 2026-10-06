from copy import deepcopy
from types import SimpleNamespace

import pytest

from benchmark.geoi_lbs import GeoILbsClient
from benchmark.query_purpose import PurposeIndependentCoverClient, QueryPurpose, QuerySpec
from experiments.verify_geoi_purpose_refinement import (attack_choices,attack_summary,balanced_mean,
    allowed_test_methods,selection_from_summaries,utility_summary)
from tests.test_query_purpose import fixture


def test_verifier_equal_family_weights_and_separate_attack_losses():
    rows=[dict(family_id='one',status='ok',errors=dict(a=110.,b=0.)) for _ in range(50)]
    rows.append(dict(family_id='two',status='ok',errors=dict(a=110.,b=1000.)))
    choice=attack_choices(rows)
    assert choice['mae']=='a' and choice['hit100']=='b'
    result=attack_summary(rows,choice)
    assert result['mae_m']==110. and result['hit100']==.5
    assert balanced_mean(rows,lambda r:r['errors']['b'])==500.
    with pytest.raises(AssertionError,match='Failed attacks'):
        attack_choices(rows+[dict(family_id='three',status='empty_transcript',errors={})])


def row(worst=.94,mae=1000.,bytes_=1000.):
    return dict(utility=dict(worst_purpose_recall=worst),bytes_per_input=bytes_,
                attacks={s:dict(mae_m=mae,hit100=0.) for s in ('S9','S10')})


def test_verifier_rejects_utility_gain_when_privacy_or_cost_guards_fail():
    configs=[dict(id='base',L=20),dict(id='privacy_bad',L=20),dict(id='cost_bad',L=40)]
    summaries=dict(base=row(),privacy_bad=row(.97,800.),cost_bad=row(.98,1000.,1600.))
    selected,gates=selection_from_summaries(configs,'base',summaries)
    assert selected=='base'
    assert gates['privacy_bad']['gates']['minimum_gain1pp']
    assert not gates['privacy_bad']['gates']['selection_endpoint_guard']
    assert not gates['cost_bad']['gates']['public_cost_guard']
    # Cost exception must be >=5pp, not merely any positive utility change.
    summaries['cost_bad']=row(1.,1000.,1600.)
    assert selection_from_summaries(configs,'base',summaries)[0]=='cost_bad'


def test_verifier_withheld_method_scope_allows_only_selected_matched_control_and_raw():
    configs=[dict(id='alpha000_L20',L=20),dict(id='alpha050_L40',L=40),dict(id='alpha000_L40',L=40)]
    assert allowed_test_methods('alpha050_L40','alpha000_L20',configs)=={
        'alpha050_L40','alpha000_L20','alpha000_L40','raw'}
    assert 'alpha100_L10' not in allowed_test_methods('alpha000_L20','alpha000_L20',configs)


def test_verifier_keeps_empty_purpose_reference_out_of_score_denominator():
    rows=[]
    for purpose in ('nearest_distance','fastest_travel','within_radius','minimum_detour'):
        rows.extend([dict(method='m',split='test',family_id='one',purpose=purpose,recall=.8,invalid_returned_items=0),
                     dict(method='m',split='test',family_id='one',purpose=purpose,recall=None,invalid_returned_items=0),
                     dict(method='m',split='test',family_id='two',purpose=purpose,recall=.6,invalid_returned_items=0)])
    result=utility_summary(rows,'m','test')
    assert result['empty_reference_rows']==4 and result['eligible_rows']==8
    assert result['mean_recall']==pytest.approx(.7)


def test_all_four_local_purposes_and_multiple_private_parameters_add_no_traffic():
    ranking=fixture();calls=[];protect=[]
    def mechanism(lat,lon,t):
        protect.append((lat,lon,t));return [(39.9,116.0001),(39.9,116.0002)]
    engine=SimpleNamespace(rn=ranking.rn,k=2,protect_step=mechanism)
    client=GeoILbsClient(engine,PurposeIndependentCoverClient(['cafe'],3,k=2,response_l=20),ranking)
    fetch=client.protect_and_fetch(0.,39.9,116.,lambda request:calls.append(deepcopy(request)) or [[0,1,2]])
    before=deepcopy((fetch['requests'],fetch['replies'],calls,protect))
    queries=[QuerySpec(QueryPurpose.NEAREST,'cafe'),QuerySpec(QueryPurpose.FASTEST,'cafe'),
             QuerySpec(QueryPurpose.WITHIN_RADIUS,'cafe',radius_m=90.),
             QuerySpec(QueryPurpose.WITHIN_RADIUS,'cafe',radius_m=1000.),
             QuerySpec(QueryPurpose.MIN_DETOUR,'cafe',destination_state=3),
             QuerySpec(QueryPurpose.MIN_DETOUR,'cafe',destination_state=0)]
    for query in queries:
        for gps in [(39.9,116.),(39.9,116.0003)]:
            client.answer(query,*gps)
    assert (fetch['requests'],fetch['replies'],calls,protect)==before
    assert len(calls)==2 and len(protect)==1
    assert all(set(request)=={'schema','timestamp_s','coordinate','categories','response_l','epoch'} for request in calls)
