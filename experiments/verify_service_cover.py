"""Audit DB consumption, shadow training, scoring, causality and new evidence."""
import argparse
from collections import defaultdict
import json
from pathlib import Path

import networkx as nx
import numpy as np
from scipy.sparse.csgraph import dijkstra
from scipy.spatial.distance import cdist

from core.demo_protocol import TrajectoryPoint
from data.scenario_store import ScenarioStore
from data.scenario_suite.records import device_view
from evaluation.lane_travel import matrix
from evaluation.research_protocol import utility_metrics
from experiments.run_contextual_lane import estimators, sha
from experiments.run_service_cover import ROOT, DB, RELEASE, CONTENT, METHODS, CASES, OUTPUT, prepare, generate, read
from experiments.rng_util import rng_from_key
from experiments.verify_lane_comparison import audit_native_graph
from experiments.verify_scenario_db import FROZEN


def near(a, b):
    assert np.allclose(a, b, rtol=1e-10, atol=1e-8), (a, b)


def shadow_features(public, rn):
    # Independent prefix construction; does not call the fitting implementation.
    rows = [[v / 1000. for c in e['candidates'] for v in rn.point_xy(c['lat'], c['lon'])] for e in public['events']]
    result = []
    for t, row in enumerate(rows):
        result.append(row + np.mean(rows[:t+1], axis=0).tolist() +
                      [(public['events'][t]['timestamp_s'] - public['events'][0]['timestamp_s']) / 60.])
    return np.array(result)


def verify(output=OUTPUT, replay=False):
    output = Path(output)
    for name, digest in FROZEN.items(): assert sha(ROOT / name) == digest, name
    training, validation, confirmation, selection = [read(output / f'{name}.json') for name in ('training', 'validation', 'confirmation', 'selection')]
    _, rn, service, context, prior, belief, provenance = prepare('confirmation')
    native = audit_native_graph(rn, ROOT / confirmation['network_path'])
    bundle = json.loads((ROOT / 'artifacts/datasets/urban_scenarios_v3/dataset.json').read_text())
    old = json.loads((ROOT / 'artifacts/datasets/urban_scenarios_v2/dataset.json').read_text())
    # Development observations were copied, not regenerated or selected by score.
    for f in old['families']:
        if f['seed'] > 104: continue
        assert f in bundle['families']
        for s in f['sessions']: assert old['traces'][s['session_id']] == bundle['traces'][s['session_id']]
    assert {f['seed'] for f in bundle['families']} == {101, 102, 103, 104, 201, 202, 203, 204}
    with ScenarioStore(DB) as store:
        assert store._release(RELEASE)['content_sha256'] == CONTENT
        assert store.export_bundle(RELEASE) == bundle
        for r in bundle['records']:
            for slot in range(len(r['session_ids'])):
                assert store.device_view(RELEASE, r['record_id'], slot=slot) == list(device_view(r, bundle['traces'], slot))
    assert selection['validation_sha256'] == sha(output / 'validation.json')
    assert selection['training_sha256'] == sha(output / 'training.json')
    counts = dict(rows=0, events=0, scored_rows=0, poi_queries=0, replay_rows=0,
                  prefix_checks=0, paired_anchor_groups=0, shadow_models=0, motion_transitions=0)
    records_by_phase = {}
    for p in (training, validation, confirmation, selection):
        assert p['source_sha256'] == provenance['source_sha256']
        assert p['dataset_content_sha256'] == CONTENT and p['dataset_release'] == RELEASE
        assert p['belief_model_sha256'] == belief.sha256
    for p in (training, validation, confirmation):
        phase = p['phase']; split = {'training':'development_train', 'validation':'development_validation', 'confirmation':'confirmation'}[phase]
        records = []
        for r in bundle['records']:
            if r['split'] != split or r['case_id'] not in CASES: continue
            stream = list(device_view(r, bundle['traces']))
            if r['case_id'] == 'S2.C':
                indices = r['observed_indices'][0][:12]
                assert [sum(a <= i <= b for i in indices) for a,b in r['labels']['stop_intervals']] == [9,3]
            records.append({**{k:r[k] for k in ('record_id','case_id','scenario','family_id','split')},
                'available_events':len(stream), 'points':[{'timestamp_s':x['time_s'],'lat':x['lat'],'lon':x['lon']} for x in stream[:12]]})
        assert records == p['records']
        records_by_phase[phase] = {r['record_id']:r for r in records}
    # Fully rederive each training matrix/target and standardization.
    for key, model in training['shadow_models'].items():
        name, k = key.split('/'); k = int(k)
        rows = [r for r in training['rows'] if (r['method'], r['k']) == (name, k)]
        x = np.concatenate([shadow_features(r['public'], rn) for r in rows])
        y = np.concatenate([np.array([rn.point_xy(p['lat'],p['lon']) for p in records_by_phase['training'][r['record_id']]['points']]) for r in rows])
        near(x, model['x']); near(y, model['y']); near(x.mean(axis=0), model['mean'])
        scale = x.std(axis=0); scale[scale < 1e-12] = 1.; near(scale, model['scale'])
        assert model['provenance'] == {'families':['family-101','family-102'], 'split':'development_train',
            'row_keys':[[r['record_id'],r['replicate']] for r in rows]}
        counts['shadow_models'] += 1
    transitions = defaultdict(set); prior_outputs = {}; measured = {}
    for p in (training, validation, confirmation):
        phase = p['phase']; records = records_by_phase[phase]
        expected = {(rid, m, k, rep) for rid in records for m in METHODS for k in (3,5) for rep in (1,2,3)}
        keys = [(r['record_id'],r['method'],r['k'],r['replicate']) for r in p['rows']]
        assert len(keys) == len(set(keys)) and set(keys) == expected
        assert p['training_sha256'] == (None if phase == 'training' else sha(output/'training.json'))
        assert p['selection_sha256'] == (sha(output/'selection.json') if phase == 'confirmation' else None)
        anchors = {}; summary_rows = []
        for n, row in enumerate(p['rows']):
            r = records[row['record_id']]; public = row['public']; name = row['method']; k = row['k']
            assert all(row[key] == r[key] for key in ('case_id','family_id','split'))
            seed = int(rng_from_key(r['record_id'],k,row['replicate'],schema='service-cover-row-v1').integers(0,2**31))
            assert row['rng_seed'] == seed
            assert set(public) == {'mechanism','output_kind','public_parameters','events'} and public['output_kind'] == 'dummy_only'
            assert len(public['events']) == len(r['points']) == len(row['step_ms']) == len(row['evaluator_states']) == len(row['evaluator_anchors'])
            assert min(row['step_ms']) >= 0
            for e, x, states in zip(public['events'],r['points'],row['evaluator_states']):
                assert set(e) == {'event_id','timestamp_s','candidates'} and e['timestamp_s'] == x['timestamp_s']
                assert len(e['candidates']) == k and all(set(c) == {'candidate_id','lat','lon'} for c in e['candidates'])
                assert [rn.latlon(i) for i in states] == [(c['lat'],c['lon']) for c in e['candidates']]
            near(row['spent_bound'], .01 + .02 * (len(r['points']) - 1))
            if name in ('service_cover','prior_cover'):
                assert len(row['evaluator_objective']) == len(r['points'])
                for objective in row['evaluator_objective']:
                    gains = objective['greedy_gains']
                    assert len(gains) == k and min(gains) >= 0
                    assert all(a+1e-12 >= b for a,b in zip(gains,gains[1:]))
                    near(sum(gains),objective['value']); assert objective['value'] <= 1+1e-10
                    assert len(objective['reachable_counts']) == k and min(objective['reachable_counts']) > 0
            anchor_key = r['record_id'], k, row['replicate']
            if anchor_key in anchors: assert anchors[anchor_key] == row['evaluator_anchors']
            anchors[anchor_key] = row['evaluator_anchors']
            if name == 'prior_cover':
                prior_key = k, tuple(x['timestamp_s'] for x in r['points'])
                if prior_key in prior_outputs: assert prior_outputs[prior_key] == public['events']
                prior_outputs[prior_key] = public['events']
            for t in range(1,len(r['points'])):
                dt = r['points'][t]['timestamp_s'] - r['points'][t-1]['timestamp_s']
                for u,v in zip(row['evaluator_states'][t-1],row['evaluator_states'][t]):
                    transitions[u,dt].add(v); counts['motion_transitions'] += 1
            if phase != 'training':
                truth = np.array([rn.point_xy(x['lat'],x['lon']) for x in r['points']])
                predicted = estimators(public,rn,prior,r['scenario'])
                model = training['shadow_models'][f'{name}/{k}']
                x = shadow_features(public,rn); mean,scale = np.asarray(model['mean']),np.asarray(model['scale'])
                distances = cdist((x-mean)/scale,(np.asarray(model['x'])-mean)/scale)
                indices = np.argsort(distances,axis=1,kind='stable')
                for count in (1,5,15): predicted[f'shadow_knn_{count}'] = np.asarray(model['y'])[indices[:,:count]].mean(axis=1)
                assert predicted.keys() == row['errors_by_attack'].keys()
                for a, xy in predicted.items(): near(row['errors_by_attack'][a],np.linalg.norm(xy-truth,axis=1))
                u = row['utility']; recalls = []; complete = []
                assert len(u['poi_rows']) == 6 * len(r['points'])
                for q in u['poi_rows']:
                    if q['reference']:
                        val = len(set(q['reference']) & set(q['returned'])) / len(q['reference'])
                        near(q['recall'],val); recalls.append(val)
                        flag = len(q['reference']) == len(q['returned']); assert flag == q['complete']; complete.append(flag)
                    else: assert q['recall'] is None
                near(u['poi_recall_at_5'],np.mean(recalls)); near(u['poi_complete_rate'],np.mean(complete))
                assert u['poi_evaluable_n'] == len(recalls)
                # Reissue every POI query, not merely trust stored response IDs.
                assert utility_metrics(service,public,[(x['lat'],x['lon']) for x in r['points']]) == u
                counts['scored_rows'] += 1; counts['poi_queries'] += len(u['poi_rows'])
            # One replicate per family/case/K/method: deterministic replay + prefix;
            # remaining repeats still have all scores/inputs/motion checked.
            if replay and row['replicate'] == 1:
                points = tuple(TrajectoryPoint(**x) for x in r['points'])
                again = generate(name,points,rn,context,belief,k,seed)
                for field in ('public','evaluator_states','evaluator_anchors','spent_bound'):
                    assert again[field] == row[field], (phase,n,field)
                if name in ('service_cover','prior_cover'):
                    assert again['evaluator_objective'] == row['evaluator_objective']
                short = generate(name,points[:3],rn,context,belief,k,seed)
                assert short['public']['events'] == public['events'][:3]
                counts['replay_rows'] += 1; counts['prefix_checks'] += 1
            counts['rows'] += 1; counts['events'] += len(r['points'])
            if (n+1) % 144 == 0: print(f'{phase}: {n+1}/{len(p["rows"])} audited',flush=True)
        counts['paired_anchor_groups'] += len(anchors)
        if phase == 'training': continue
        assert len(p['summaries']) == 72
        for s in p['summaries']:
            group = [r for r in p['rows'] if all(r[k] == s[k] for k in ('method','k','case_id'))]
            assert len(group) == s['families'] * s['replicates'] and s['replicates'] == 3
            assert len({r['family_id'] for r in group}) == s['families'] == (4 if phase == 'confirmation' else 2)
            for a in group[0]['errors_by_attack']:
                near(s['mae_by_attack'][a],np.mean([np.mean(r['errors_by_attack'][a]) for r in group]))
                near(s['hit_by_attack'][a],np.mean([np.mean(np.asarray(r['errors_by_attack'][a]) <= 100) for r in group]))
            near(s['audit_min_mae_m'],min(s['mae_by_attack'].values())); near(s['audit_max_hit100'],max(s['hit_by_attack'].values()))
            near(s['poi_recall'],np.mean([r['utility']['poi_recall_at_5'] for r in group]))
            near(s['poi_complete'],np.mean([r['utility']['poi_complete_rate'] for r in group]))
            near(s['step_mean_ms'],np.mean([np.mean(r['step_ms']) for r in group]))
            for category, entry in s['by_category'].items():
                values = [[q['recall'] for q in r['utility']['poi_rows'] if q['category'] == category and q['recall'] is not None] for r in group]
                near(entry['recall'],np.mean([np.mean(v) for v in values if v]))
                assert entry['valid_query_n'] == sum(map(len,values))
                assert entry['empty_reference_n'] == sum(sum(q['category'] == category and q['recall'] is None for q in r['utility']['poi_rows']) for r in group)
            key = f'{s["method"]}/{s["k"]}/{s["case_id"]}'
            if phase == 'validation':
                assert selection['attackers'][key] == {
                    'mae': min(s['mae_by_attack'],key=lambda a:(s['mae_by_attack'][a],a)),
                    'hit': min(s['hit_by_attack'],key=lambda a:(-s['hit_by_attack'][a],a))}
            else:
                a = selection['attackers'][key]
                assert s['selected_mae_attack'] == a['mae'] and s['selected_hit_attack'] == a['hit']
                near(s['selected_mae_m'],s['mae_by_attack'][a['mae']]); near(s['selected_hit100'],s['hit_by_attack'][a['hit']])
        measured[phase] = p['summaries']
    for k in (3,5):
        candidates = []
        for name in METHODS[:-1]:
            group = [s for s in measured['validation'] if (s['method'],s['k']) == (name,k)]
            candidates.append({'method':name,'min_case_recall':min(s['poi_recall'] for s in group),
                'macro_hit100':float(np.mean([s['audit_max_hit100'] for s in group]))})
        valid = [c for c in candidates if round(c['min_case_recall'],12) >= .9]
        chosen = min(valid,key=lambda c:(round(c['macro_hit100'],12),c['method'])) if valid else min(candidates,key=lambda c:(-round(c['min_case_recall'],12),round(c['macro_hit100'],12),c['method']))
        assert selection['method_selection'][str(k)] == {'chosen':chosen,'utility_feasible':bool(valid),'candidates':candidates}
    # Independent shortest paths on the same native-audited graph; one run per
    # distinct source/time pair, then check ALL emitted transitions.
    travel_matrix = matrix(rn, time=True)
    for n, ((u,dt), targets) in enumerate(transitions.items()):
        distance = dijkstra(travel_matrix,directed=True,indices=u,limit=dt+1e-8)
        assert all(distance[v] <= dt+1e-8 for v in targets)
    for r in confirmation['records'][::4]:
        x = r['points'][0]; state,_ = rn.nearest(x['lat'],x['lon'])
        distances = nx.single_source_dijkstra_path_length(rn.graph,state,weight='length')
        expected = {p['id']:distances[p['vertex']] for p in service.pois if p['vertex'] in distances}
        actual = service.distances((x['lat'],x['lon']))
        assert actual.keys() == expected.keys(); near(list(actual.values()),[expected[k] for k in actual])
    return {'verified':True, **counts, 'native_connections':native,
        'unique_motion_origins_times':len(transitions), 'replay_one_of_three_replicates':replay,
        'dataset_release':RELEASE,'dataset_content_sha256':CONTENT,
        'artifact_sha256':{name:sha(output/f'{name}.json') for name in ('training','validation','selection','confirmation')},
        'verifier_source_sha256':{name:sha(ROOT/name) for name in ('experiments/verify_service_cover.py','experiments/verify_scenario_suite_v3.py','tests/test_service_cover_evidence.py')},
        'limitations':['Same city; four confirmation families.', 'Heuristic/shadow attacks are not optimal adversaries.',
                       'Native motion is free-flow feasibility, not traffic compliance.', 'S4-S10 are not protection-evaluated here.']}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,default=OUTPUT)
    parser.add_argument('--replay',action='store_true')
    args = parser.parse_args(); result = verify(args.output,args.replay)
    (args.output/'verification.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))
