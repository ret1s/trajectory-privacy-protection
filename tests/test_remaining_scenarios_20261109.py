import copy

import numpy as np
import pytest

from evaluation.remaining_scenarios import prefix, prefix_vector, linkage_features, candidate_public, xy
from experiments import remaining_scenarios_20261109 as study


def events():
    return [dict(timestamp_s=t,coordinates=[[40+t/100000,116.3],[40.001+t/100000,116.301]])
            for t in (0,7,20,40)]


def test_prefix_keeps_fake_packets_and_does_not_inspect_future_coordinates():
    tape=events();tape[-1]['coordinates']=[['not a number']]
    assert [e['timestamp_s'] for e in prefix(tape,20)]==[0,7,20]
    baseline=prefix_vector(tape,20)
    tape[-1]['coordinates']=[[float('nan'),0]]
    assert np.array_equal(prefix_vector(tape,20),baseline)


def test_private_fields_and_duplicate_timestamps_rejected():
    tape=events();tape[0]['choice_index']=1
    with pytest.raises(ValueError):prefix(tape,20)
    with pytest.raises(ValueError):prefix([events()[0],events()[0]],20)


def test_features_are_set_permutation_invariant_and_linkage_symmetric():
    a=events();b=copy.deepcopy(a)
    for event in b:event['coordinates'].reverse()
    assert np.allclose(prefix_vector(a,40),prefix_vector(b,40))
    for event in b:
        for p in event['coordinates']:p[1]+=.002
    assert np.allclose(linkage_features(a,b),linkage_features(b,a))


def test_candidate_conversion_contains_only_observable_order_and_coordinates():
    result=candidate_public(events(),20)
    assert set(result)=={'events'}
    assert result['events'][-1]['timestamp_s']==20
    assert set(result['events'][0]['candidates'][0])=={'candidate_id','lat','lon'}


def test_disjoint_linkage_blocks_do_not_cross_splits():
    families=[dict(family_id=f'{s}-{i}',split=s) for s in ('train','selection','test') for i in range(4)]
    result=study.blocks(families)
    for f,(group,partner) in result.items():
        assert partner!=f and result[partner]==(group,f)
        assert f.split('-')[0]==partner.split('-')[0]==group.split('-')[0]


def test_bootstrap_uses_groups_not_repeated_draws():
    rows=[{'group':'a'},{'group':'b'}];values=[0.,1.]
    repeated=[r for r in rows for _ in range(7)];rv=[v for v in values for _ in range(7)]
    assert study.summary(rows,values)['ci95']==study.summary(repeated,rv)['ci95']
    assert study.summary(repeated,rv)['groups']==2


def test_conditional_payload_audit_distinguishes_equal_and_explicit_wire():
    labels=[(p,c) for p in range(4) for c in range(6)]
    assert study.vulnerability(['same']*24,[p for p,c in labels])==.25
    assert study.vulnerability(['same']*24,[c for p,c in labels])==1/6
    assert study.vulnerability(['same']*24,labels)==1/24
    assert study.vulnerability([str(label) for label in labels],labels)==1


def test_full_cohort_future_labels_are_strictly_after_public_cutoff():
    families,native,by_id=study.datasets()
    for scenario,phases in study.CONFIG['future_phases'].items():
        for phase in phases:
            rows=study.future_rows(families,native,by_id,'raw',scenario,phase)
            assert {s:len(r) for s,r in rows.items()}=={'train':40,'selection':16,'test':120}
            for group in rows.values():
                for r in group:
                    assert r['query']['events'][-1]['timestamp_s']<=r['cut_s']
                    future=r['first_target_edge_s'] if scenario=='S5' else r['destination_arrival_s']
                    assert r['cut_s']<future


def test_s8_partner_raw_is_explicit_and_matches_shared_prefix_truth():
    families,native,by_id=study.datasets()
    rows=study.joint_rows(families,native,by_id,'geoi_bq','raw_partner')
    for split in rows:
        for r in rows[split]:
            assert np.linalg.norm(np.array(r['geometry']['partner_centroid'])-r['target_xy'])<1e-6
            assert r['observed_coordinates']>r['observed_packets']


def test_actual_client_does_not_accept_private_request_parameters():
    from benchmark.session_poi_cache import SessionCoverClient
    client=SessionCoverClient(['a','b'],2,k=2,response_l=60,session_token='p',static_catalogue_id='c')
    seen=[]
    def server(request):seen.append(request);return [[],[]]
    client.step(0,events()[0]['coordinates'],server)
    assert all(r['categories']==['a','b'] and r['response_l']==60 for r in seen)
    assert all(set(r)=={'schema','timestamp_s','coordinate','categories','response_l','epoch'} for r in seen)
    with pytest.raises(TypeError):client.step(20,events()[1]['coordinates'],server,private_purpose='medical')
