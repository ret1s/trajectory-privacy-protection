"""Versioned, post-hoc diagnosis; never rewrites the frozen comparison.

Stages: select (historical families only), evaluate (previously seen expanded
cohort), report. Raw-control strengthening is exploratory, not a new holdout
confirmation. The scenario audit evaluates specification independently of wins.
"""
import argparse
from collections import defaultdict
import copy
import json
from pathlib import Path

import numpy as np

from evaluation.prefix_destination_attack import PrefixDestinationAttack
from evaluation.live_comparison_attacks import public_view
from experiments.endpoint_calendar_study import FIT, SELECT, gzread, gzwrite, select_case, family_mean
from experiments.research_loop_resources import ROOT, CACHE, load, sha
from experiments.run_live_paper_comparison import DATASETS, write_json
from experiments.research_loop_cases import target_xy

OUT = ROOT/'artifacts/benchmarks/s10b_diagnostic_v1'
OLD = ROOT/'artifacts/benchmarks/endpoint_calendar_expanded_v1'
PAPERS = ROOT/'artifacts/benchmarks/live_paper_comparison_v1'
EXPANDED = ROOT/'artifacts/datasets/endpoint_holdout_expanded_v1/dataset.json'
SOURCES = [Path(__file__), ROOT/'evaluation/prefix_destination_attack.py',
           ROOT/'evaluation/live_comparison_attacks.py', ROOT/'experiments/endpoint_calendar_study.py',
           ROOT/'data/scenario_suite/records.py', ROOT/'data/scenario_suite_v2/records.py',
           OLD/'protocol.json', OLD/'selection.json', OLD/'selection_rows.json.gz',
           OLD/'holdout_privacy.json.gz', OLD/'readout.json', EXPANDED, CACHE/'resources.json']
SOURCES += [ROOT/f'artifacts/datasets/{name}/dataset.json' for name in DATASETS.values()]


def dataset(path):
    return json.loads(path.read_text())


def raw_inputs(d, rn, base, split):
    result = []
    for record in d['records']:
        if record['case_id'] != 'S10.B':
            continue
        views, truth = [], []
        for slot, (sid, indices) in enumerate(zip(record['session_ids'], record['observed_indices'])):
            # Reuse the benchmark's permitted view and quantized raw control;
            # don't silently switch to denser/private FCD or its absolute clock.
            ex = gzread(base/'transcripts'/split/'raw'/f'{sid}_r0.json.gz')
            events, _ = public_view(ex, indices)
            views.append(events)
            truth.append(target_xy(record, slot, d['traces'][sid], rn)[0])
        result.append({k: record[k] for k in ('record_id', 'family_id', 'case_id', 'scenario')} |
                      {'events': views, 'truth': np.array(truth)})
    return result


def historical_inputs(rn):
    return [row for split, name in DATASETS.items()
            for row in raw_inputs(dataset(ROOT/f'artifacts/datasets/{name}/dataset.json'), rn, PAPERS, split)]


def fit_model(rows, rn):
    fit = [r for r in rows if r['family_id'] in FIT]
    events = [v for r in fit for v in r['events']]
    truth = np.concatenate([r['truth'] for r in fit])
    return PrefixDestinationAttack(events, truth, rn)


def merge_scores(inputs, old_rows, model):
    if isinstance(old_rows, dict):
        old_rows = old_rows['rows']
    old = {r['record_id']: r for r in old_rows if r['method']=='raw' and r['case_id']=='S10.B' and r['rep']==0}
    result = []
    for row in inputs:
        scored = copy.deepcopy(old[row['record_id']])
        np.testing.assert_allclose(scored['target_xy_evaluator_only'], row['truth'], atol=1e-8, rtol=0)
        bank = [model.predict(e) for e in row['events']]
        for name in bank[0]:
            scored['errors'][name] = np.linalg.norm(np.array([p[name] for p in bank])-row['truth'], axis=1).tolist()
        result.append(scored)
    return result


def check_sources(protocol):
    for path, digest in protocol['sources'].items():
        amendment_path = OUT/'implementation_amendment.json'
        if path == str(Path(__file__).resolve().relative_to(ROOT)) and amendment_path.exists():
            amendment = dataset(amendment_path)
            assert amendment['original_runner_sha256'] == digest
            assert amendment['unchanged_selection_sha256'] == sha(OUT/'selection.json')
            digest = amendment['revised_runner_sha256']
        assert sha(ROOT/path) == digest, path


def select():
    if (OUT/'selection.json').exists():
        raise FileExistsError('Preserve sealed diagnostic selection')
    protocol = {'schema': 's10b-diagnostic-v1', 'date': '2026-09-25',
        'status': 'exploratory_after_observing_weak_raw_control_not_independent_confirmation',
        'fit_families': sorted(FIT), 'selection_families': sorted(SELECT),
        'evaluation_families': [f'family-{i}' for i in range(1201, 1233)],
        'input': 'raw allowed relative-time prefix only, from unchanged baseline transcripts',
        'candidate_grid': {'trees': {'n_estimators':128,'leaf':[1,2,4],'max_depth':16,'seed':20260925},
            'frames': ['absolute endpoint','heading-relative displacement'], 'knn_k':[1,3,5,9],
            'B_only_priors': ['mean','median','mode100','mode200','mode500'],
            'geometry':'last observed; each candidate also projected to public roads'},
        'selection': 'min family MAE / max family Hit with MAE tie-break, using historical SELECT only',
        'no_defense_or_dataset_changes': True,
        'sources': {str(p.relative_to(ROOT)): sha(p) for p in SOURCES}}
    write_json(OUT/'protocol.json', protocol)
    rn, *_ = load()
    inputs = historical_inputs(rn)
    model = fit_model(inputs, rn)
    selected_inputs = [r for r in inputs if r['family_id'] in SELECT]
    rows = merge_scores(selected_inputs, gzread(OLD/'selection_rows.json.gz'), model)
    new_only = [{**r, 'errors': {n:e for n,e in r['errors'].items() if n.startswith('b_')}} for r in rows]
    gzwrite(OUT/'selection_rows.json.gz', rows)
    write_json(OUT/'selection.json', {'protocol_sha256':sha(OUT/'protocol.json'),
        'selection_rows_sha256':sha(OUT/'selection_rows.json.gz'),
        'fit_records':sum(r['family_id'] in FIT for r in inputs), 'selection_records':len(rows),
        'fit_families_present': sorted({r['family_id'] for r in inputs if r['family_id'] in FIT}),
        'selection_families_present': sorted({r['family_id'] for r in selected_inputs}),
        'bank_sizes': {'original':len(rows[0]['errors'])-len(new_only[0]['errors']),
                       'new_B_specific':len(new_only[0]['errors']), 'combined':len(rows[0]['errors'])},
        'combined':select_case(rows), 'B_specific':select_case(new_only)})
    print('Sealed B-only and combined selections on historical data', flush=True)


def summary(rows, selected):
    return {'families':len({r['family_id'] for r in rows}), 'records':len(rows),
        'targets':sum(len(r['target_xy_evaluator_only']) for r in rows),
        'mae_m':family_mean([{**r,'value':float(np.mean(r['errors'][selected['mae']]))} for r in rows], 'value'),
        **{f'hit{k}':family_mean([{**r,'value':float(np.mean(np.array(r['errors'][selected[f'hit{k}']]) <= k))}
                                 for r in rows], 'value') for k in (50,100,200,500)}}


def audit(d, rn):
    reference = {r['family_id']: r for r in d['records'] if r['case_id']=='S6.A'}
    durations, hidden_distances, separations, visible_separations, comparisons = [], [], [], [], []
    for r in d['records']:
        if r['case_id'] != 'S10.B':continue
        other = reference.get(r['family_id'])
        identical = other is not None and all(r[k]==other[k] for k in
            ('session_ids','observed_indices','labels','observation_policy'))
        comparisons.append({'record_id':r['record_id'],'family_id':r['family_id'],
                            'S6A_record_id':other['record_id'] if other else None,'same_task_data':identical})
        ends, last = [], []
        for sid, indices in zip(r['session_ids'], r['observed_indices']):
            trace = d['traces'][sid]
            a,b = trace[indices[-1]],trace[-1]
            last.append(rn.point_xy(a['lat'],a['lon']));ends.append(rn.point_xy(b['lat'],b['lon']))
            durations.append(b['time_s']-a['time_s'])
            hidden_distances.append(float(np.linalg.norm(np.array(last[-1])-ends[-1])))
        separations.append(float(np.linalg.norm(np.array(ends[0])-ends[1])))
        visible_separations.append(float(np.linalg.norm(np.array(last[0])-last[1])))
    def quantiles(v):return dict(zip(('min','q25','median','q75','max'), map(float,np.quantile(v,[0,.25,.5,.75,1])))) if v else None
    return {'records':len(comparisons),'same_as_S6A':sum(r['same_task_data'] for r in comparisons),
        'comparisons':comparisons,'hidden_duration_s':quantiles(durations),'hidden_straight_distance_m':quantiles(hidden_distances),
        'paired_endpoint_separation_m':quantiles(separations),'paired_last_observed_separation_m':quantiles(visible_separations),
        'not_an_indistinguishability_proof':'Full visible prefixes may differ in speed, timing and lane. Last-point proximity alone proves no such claim.'}


def evaluate():
    if (OUT/'readout.json').exists():raise FileExistsError('Preserve completed diagnostic')
    protocol, selection = dataset(OUT/'protocol.json'), dataset(OUT/'selection.json')
    check_sources(protocol)
    assert selection['protocol_sha256']==sha(OUT/'protocol.json')
    assert selection['selection_rows_sha256']==sha(OUT/'selection_rows.json.gz')
    rn, *_ = load()
    historical = historical_inputs(rn)
    model = fit_model(historical, rn)
    d = dataset(EXPANDED)
    inputs = raw_inputs(d, rn, OLD, 'holdout')
    rows = merge_scores(inputs, gzread(OLD/'holdout_privacy.json.gz'), model)
    gzwrite(OUT/'evaluation_rows.json.gz', rows)
    original = dataset(OLD/'selection.json')['selected']['raw/S10.B']
    # This oracle only diagnoses finite-bank support. It uses truth and is
    # explicitly NOT an attacker, selected score, or bound on all attacks.
    envelope = {f'hit{k}':family_mean([{**r,'value':float(np.mean(np.min(np.array(list(r['errors'].values())),axis=0)<=k))}
                for r in rows], 'value') for k in (100,200,500)}
    audits = {'expanded':audit(d,rn)}
    for split, name in DATASETS.items():audits[split]=audit(dataset(ROOT/f'artifacts/datasets/{name}/dataset.json'),rn)
    frozen = dataset(OLD/'readout.json')
    by_case = {(r['method'],r['case_id']):r for r in frozen['summaries']}
    scoped = []
    for method in ('raw','dls','rdg','transprotect_markov','semantic_poi','fake_queries','calendar30','calendar67'):
        a,c,b = (by_case[method,case] for case in ('S10.A','S10.C','S10.B'))
        scoped.append({'method':method,'A_C_equal_weight':{k:(a[k]+c[k])/2 for k in ('hit100','hit200','mae_m')},
                       'B_original':{k:b[k] for k in ('hit100','hit200','mae_m')}})
    result = {'schema':'s10b-diagnostic-readout-v1','status':protocol['status'],
        'protocol_sha256':sha(OUT/'protocol.json'),'selection_sha256':sha(OUT/'selection.json'),
        'evaluation_rows_sha256':sha(OUT/'evaluation_rows.json.gz'),
        'raw_control': {'original':summary(rows,original),'B_specific':summary(rows,selection['B_specific']),
                        'combined':summary(rows,selection['combined'])},
        'oracle_support_only_uses_truth_not_a_benchmark':envelope,'specification_audits':audits,
        'frozen_A_C_descriptive_breakdown_not_new_confirmation':scoped,
        'scope_decision': 'Retain original S10.B and ABC results; treat B as S6.A-equivalent forecasting/boundary control, not separate evidence of endpoint protection. Focus current endpoint contribution on S10.A/C.'}
    write_json(OUT/'readout.json',result)
    check_sources(protocol)
    print(json.dumps(result['raw_control'],indent=2),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('stage',choices=['select','evaluate'])
    {'select':select,'evaluate':evaluate}[parser.parse_args().stage]()
