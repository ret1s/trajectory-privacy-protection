"""Scientific regression and deliberate-corruption tests for scenario data."""
import copy
import json
from pathlib import Path

import pytest

from data.scenario_suite.catalog import catalogue
from data.scenario_suite.records import device_view, reference_queries, runs, sample
from data.sumo_demo import _load_sumolib
from experiments.verify_scenario_suite import verify, verify_data

ROOT=Path(__file__).resolve().parents[1]
DATA=ROOT/'artifacts/datasets/urban_scenarios_v1/dataset.json'


@pytest.fixture(scope='module')
def suite():
    return json.loads(DATA.read_text())


@pytest.fixture(scope='module')
def network(suite):
    path=ROOT/suite['network']['path']
    if not path.is_file():
        pytest.skip('Integration audit requires rebuilding ignored SUMO caches')
    return _load_sumolib().net.readNet(str(path),withInternal=True)


def test_catalogue_keeps_missing_cases_in_denominator():
    c=catalogue()
    assert len(c)==len({x['case_id'] for x in c})==30
    assert {x['scenario'] for x in c}=={f'S{i}' for i in range(1,11)}


def test_raw_release_and_all_gates(network):
    result=verify(DATA,raw=True)
    assert result['generated_scenarios']==10
    assert result['raw_fcd_points_compared']==result['raw_fcd_samples']


def test_device_view_allowlist_and_future_isolation(suite):
    for r in suite['records']:
        for slot,sid in enumerate(r['session_ids']):
            before=list(device_view(r,suite['traces'],slot))
            assert before[0]['time_s']>=0
            if r['scenario']!='S8':
                assert before[0]['time_s']==0
            assert all(set(p)=={'time_s','lat','lon','query_category'} for p in before)
            assert len(before)==len(r['observed_indices'][slot])
            # Alter every unobserved coordinate, including the hidden future.
            t=copy.deepcopy(suite['traces'][sid])
            allowed=set(r['observed_indices'][slot])
            for i,p in enumerate(t):
                if i not in allowed:
                    p.update(lat=-80,lon=-170)
            assert before==list(device_view(r,{**suite['traces'],sid:t},slot))


def test_companions_keep_real_relative_timing(suite):
    for r in suite['records']:
        if r['scenario']!='S8':
            continue
        a,b=[list(device_view(r,suite['traces'],i)) for i in range(2)]
        times=[suite['traces'][s][idx[0]]['time_s'] for s,idx in zip(r['session_ids'],r['observed_indices'])]
        assert b[0]['time_s']-a[0]['time_s']==times[1]-times[0]


def test_query_cover_not_a_location_privacy_claim(suite):
    a=next(r for r in suite['records'] if r['case_id']=='S7.A')
    b=next(r for r in suite['records'] if r['case_id']=='S7.B')
    assert reference_queries(a,suite['traces'])[0]['categories']==['clinic']
    assert len(reference_queries(b,suite['traces'])[0]['categories'])==6
    changed=copy.deepcopy(b)
    changed['labels']['true_queries']=['fuel']
    assert reference_queries(b,suite['traces'])==reference_queries(changed,suite['traces'])
    for case in ['S7.A','S7.B','S7.C']:
        assert len({r['labels']['intent'] for r in suite['records'] if r['case_id']==case})==3


@pytest.mark.parametrize('corruption',['future','identity','split','duplicate','progress','jump','summary','coverage'])
def test_verifier_rejects_corrupted_evidence(suite,network,corruption):
    d=copy.deepcopy(suite)
    if corruption=='future':
        r=next(r for r in d['records'] if r['case_id']=='S5.A')
        r['labels']['future_index']=r['observed_indices'][0][-1]
    elif corruption=='identity':
        next(r for r in d['records'] if r['case_id']=='S4.B')['labels']['same_device']=True
    elif corruption=='split':
        d['records'][0]['split']='development_test'
    elif corruption=='duplicate':
        d['records'][1]['record_id']=d['records'][0]['record_id']
    elif corruption=='progress':
        next(iter(d['traces'].values()))[1]['lane_pos_m']=-30
    elif corruption=='jump':
        next(iter(d['traces'].values()))[1]['lat']+=.01
    elif corruption=='summary':
        d['summary']['raw_fcd_samples']+=1
    else:
        d['catalogue'][0]['protection_evaluated']=True
    with pytest.raises(AssertionError):
        verify_data(d,network)


def test_stops_do_not_bridge_missing_fcd():
    t=[{'time_s':i,'speed_m_s':0} for i in [0,1,2,5,6]]
    assert runs(t,lambda p:p['speed_m_s']==0)==[[0,1,2],[3,4]]
    assert sample(t,interval=2)==[0,2,3]


def test_thesis_case_table_matches_frozen_dataset(suite):
    import re
    text=(ROOT/'thesis/scenario_dataset_spec.tex').read_text()
    counts={key:int(n) for key,n in re.findall(r'^(S\d+\.[ABC]) & .*? & (\d+)\\\\',text,re.M)}
    assert len(counts)==30
    assert counts=={c['case_id']:c['generated_records'] for c in suite['catalogue']}
