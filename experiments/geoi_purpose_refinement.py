"""Geo-I-preserving public multi-purpose Q objective and integrated LBS client.

Seal public alpha/depth candidates, select on reused development families, then
read the separate development readout. No core/location primitive is changed.
"""
import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path

import numpy as np

from benchmark.engines.endpoint_noise import EndpointNoiseProgressLaneDummy
from benchmark.geoi_lbs import GeoILbsClient
from benchmark.paper_comparators import PublicHistory
from benchmark.public_purpose_belief import build_public_purpose_weights, PurposeCoverAnchorModel
from benchmark.query_purpose import (MultiPurposeRoadRanking, PurposeIndependentCoverClient,
                                     QueryPurpose, QuerySpec)
from evaluation.endpoint_noise_attacks import EndpointShadowBank, endpoint_features, family_mean, select_attackers
from evaluation.live_poi import AvailabilityWorld, LivePointService
from experiments import endpoint_noise_loop as common
from experiments.endpoint_noise_depth_loop import resources
from experiments.public_research_resources import ROOT, sha
from experiments.rng_util import rng_from_key

DATA = ROOT/'artifacts/datasets/research_loop_expanded_v1/dataset.json'
CONFIGS = [dict(id=f'alpha{int(a*100):03d}_L{L}',alpha=a,L=L)
           for a in (0.,.5,1.) for L in (10,20,40)]
BASELINE = 'alpha000_L20'
BACKBONE = ['core/mechanisms.py','core/road_network.py','benchmark/anchor_belief.py',
    'benchmark/engines/paced_guard.py','benchmark/engines/filtered_cover.py',
    'benchmark/engines/matched_filter.py','benchmark/engines/progress_cover.py',
    'benchmark/engines/slack_progress.py','benchmark/engines/endpoint_noise.py']


def public_destinations(ranking, count=8):
    """Public farthest-point spread; never inspect a trace/destination/score."""
    ids = sorted({int(p['vertex']) for p in ranking.pois})
    chosen = [ids[0]]
    xy = ranking.rn.xy[ids]
    for _ in range(min(count,len(ids))-1):
        d = np.min(np.linalg.norm(xy[:,None,:]-ranking.rn.xy[chosen][None,:,:],axis=2),axis=1)
        chosen.append(ids[int(np.argmax(d))])
    return sorted(chosen)


def seed_for(session, rep):
    # Experiment reproducibility only; deployment must use private fresh RNG.
    return int(rng_from_key(f'{session}/{rep}',schema='geoi-purpose-refinement-v1').integers(0,2**63))


def utility_summary(rows, split, method):
    subset = [r for r in rows if r['split']==split and r['method']==method]
    values = {}
    for purpose in QueryPurpose:
        selected = [dict(r,v=r['recall']) for r in subset if r['purpose']==purpose.value and r['recall'] is not None]
        values[purpose.value] = family_mean(selected,'v')
    return {'by_purpose':values,'mean_recall':float(np.mean(list(values.values()))),
            'worst_purpose_recall':min(values.values()),
            'eligible_rows':sum(r['recall'] is not None for r in subset),
            'empty_reference_rows':sum(r['recall'] is None for r in subset),
            'invalid_returned_items':sum(r['invalid_returned_items'] for r in subset)}


def generate(session, config, rep, split, rn, beliefs, weights, ranking, world):
    seed = seed_for(session['session_id'],rep)
    method = config['id']
    raw = method=='raw'
    if raw:
        class Raw:
            k = 1
            def protect_step(self,lat,lon,t):return [(lat,lon)]
        engine = Raw(); engine.rn = rn
    else:
        belief = PurposeCoverAnchorModel(beliefs[.25,config['L']],weights,alpha=config['alpha'])
        engine = EndpointNoiseProgressLaneDummy(rn,privacy_scale=.25,budget=.24,horizon=12,
            k=5,belief_model=belief,theta_m=200.,read_interval_s=60.,utility_slack=.03,
            rng=np.random.default_rng(seed))
    retrieval = PurposeIndependentCoverClient(ranking.categories,ranking.n,k=engine.k,response_l=config['L'])
    local = MultiPurposeRoadRanking(ranking)
    client = GeoILbsClient(engine,retrieval,local)
    service = LivePointService(ranking,world,response_l=config['L'])
    events, utility = [], []
    byte_count = 0
    private_destination = rn.nearest(session['points'][-1]['lat'],session['points'][-1]['lon'])[0]
    for point in session['points']:
        calls = []
        def server(request):
            calls.append(request)
            return service.query(rn.nearest(*request['coordinate'])[0],request['epoch'])
        result = client.protect_and_fetch(point['t'],point['lat'],point['lon'],server)
        state = rn.nearest(point['lat'],point['lon'])[0]
        available = world.at_epoch(world.epoch(point['t']))  # evaluator only
        wire = json.dumps({'requests':result['requests'],'replies':[
            [[ranking.pois[i]['id'] for i in cat] for cat in reply] for reply in result['replies']]},
            separators=(',',':'),sort_keys=True)
        byte_count += len(wire.encode())
        for category in ranking.categories:
            for purpose in QueryPurpose:
                query = QuerySpec(purpose,category,radius_m=1000. if purpose==QueryPurpose.WITHIN_RADIUS else None,
                    destination_state=private_destination if purpose==QueryPurpose.MIN_DETOUR else None)
                reference = local.top(state,available,query)
                returned = client.answer(query,point['lat'],point['lon'])
                invalid = sum(not available[i] or not np.isfinite(local.scores(state,query)[i]) for i in returned)
                utility.append(dict(method=method,split=split,family_id=session['family_id'],
                    session_id=session['session_id'],rep=rep,timestamp_s=point['t'],purpose=purpose.value,
                    category=category,reference=reference,returned=returned,
                    recall=len(set(reference)&set(returned))/len(reference) if reference else None,
                    invalid_returned_items=invalid))
        assert len(calls)==engine.k  # 24 local queries add no traffic/GPS protection call.
        assert wire==json.dumps({'requests':result['requests'],'replies':[
            [[ranking.pois[i]['id'] for i in cat] for cat in reply] for reply in result['replies']]},
            separators=(',',':'),sort_keys=True)
        events.append({'timestamp_s':point['t'],'coordinates':[q['coordinate'] for q in result['requests']]})
    valid = [r['recall'] for r in utility if r['recall'] is not None]
    run = dict(method=method,split=split,family_id=session['family_id'],session_id=session['session_id'],
        rep=rep,seed=seed,config=config,close_s=session['close_s'],events=events,
        recall=float(np.mean(valid)),bytes_per_input=byte_count/len(events),
        target_xy_evaluator_only={s:list(rn.point_xy(p['lat'],p['lon'])) for s,p in
            [('S9',session['points'][0]),('S10',session['points'][-1])]},
        protected_anchors_evaluator_only=[] if raw else engine.evaluator_anchors,
        ledger_evaluator_only=[] if raw else engine.evaluator_ledger,
        spent_bound_per_m=None if raw else engine.spent_bound)
    if not raw:assert run['spent_bound_per_m']<=.0575+1e-12
    assert all(r['invalid_returned_items']==0 for r in utility)
    return run, utility


def fit_attack(runs, history, rn, method):
    banks, choices, rows = {}, {}, []
    for scenario in ('S9','S10'):
        x,seq,truth = [],[],[]
        for item in runs['fit']:
            a,b,_ = endpoint_features(item['events'],scenario,rn,history,observable_close_s=item['close_s'])
            x.append(a[0]);seq.append(b[0]);truth.append(item['target_xy_evaluator_only'][scenario])
        banks[scenario] = EndpointShadowBank(x,seq,truth)
        scored = common.bank_rows(runs['selection'],scenario,banks[scenario],rn,history)
        choices[scenario] = select_attackers(scored)
        rows.extend(scored)
    return banks,choices,common.score_summary(rows,choices),rows


def select_candidate(summaries):
    """Public, predeclared gates; never called with test results."""
    base = summaries[BASELINE]
    candidates = []
    for config in CONFIGS:
        s = summaries[config['id']]
        utility_ok = s['utility']['worst_purpose_recall'] >= .90
        gain = s['utility']['worst_purpose_recall']-base['utility']['worst_purpose_recall']
        privacy_ok = all(s['attacks'][t]['mae_m'] >= .90*base['attacks'][t]['mae_m'] and
                         s['attacks'][t]['hit100'] <= base['attacks'][t]['hit100']+.05 for t in ('S9','S10'))
        cost_ok = s['bytes_per_input']<=1.5*base['bytes_per_input'] or gain>=.05
        s.update(gain_worst_purpose=gain,eligible=bool(utility_ok and gain>=.01 and privacy_ok and cost_ok),
                 gates=dict(all_purposes90=utility_ok,minimum_gain1pp=gain>=.01,
                            selection_endpoint_guard=privacy_ok,public_cost_guard=cost_ok))
        if s['eligible']:candidates.append(config['id'])
    selected = (min(candidates,key=lambda m:(summaries[m]['bytes_per_input'],
        -summaries[m]['utility']['worst_purpose_recall'],m)) if candidates else BASELINE)
    return {'selected':selected,'baseline':BASELINE,'promoted':selected!=BASELINE,
            'all_selection_summaries':summaries,'test_used_for_selection':False}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out',type=Path,required=True)
    parser.add_argument('--resources',type=Path,default=Path('/private/tmp/trajectory-research-20261005-public-map'))
    args = parser.parse_args();out=args.out
    if out.exists():raise FileExistsError('Use a fresh evidence directory')
    out.mkdir(parents=True)
    source_files = BACKBONE+['experiments/geoi_purpose_refinement.py','benchmark/geoi_lbs.py',
        'benchmark/public_purpose_belief.py','benchmark/query_purpose.py','evaluation/endpoint_noise_attacks.py',
        'experiments/public_research_resources.py','experiments/endpoint_noise_depth_loop.py',
        'evaluation/live_poi.py','evaluation/lane_travel.py']
    hashes = {name:sha(ROOT/name) for name in source_files}
    splits = {f'family-{i}':('fit' if i<707 else 'selection' if i<710 else 'test') for i in range(701,713)}
    protocol = {'schema':'geoi-public-purpose-refinement-v1','date':'2026-10-05',
        'source_sha256':hashes,'dataset_sha256':sha(DATA),'frozen_backbone_files':BACKBONE,
        'split_by_family':splits,'session_rule':'lexically first source session in each family',
        'configs':CONFIGS,'baseline':BASELINE,'repetitions':[0,1],
        'private_mechanism':'unchanged GeoI-Endpoint20 epsilon test/release .0025/m, cap .0575/m, H12 K5',
        'public_objective':'uniform four-purpose mixture, radius1000m and eight public farthest POI states',
        'selection_rule':'Worst-purpose Recall>=.90, improve baseline>=.01; each selected endpoint MAE>=.90baseline '
            'and Hit100<=baseline+.05; bytes<=1.5baseline unless worst-purpose gain>=.05; '
            'min bytes then max worst-purpose recall; fallback alpha0L20',
        'clock':'60s plus final, public close visible; not private demand-driven',
        'wire':'fixed all-category requests + POI ID replies JSON; no HTTP/TLS',
        'scope':'Previously used SUMO families and reconstructed public map: development only; '
                'postprocessing comparison, not new primitive or confirmation',
        'seeds':'independent per session/rep and common only across paired methods; reproducibility seeds not deployment RNG'}
    common.write(out/'protocol.json',protocol)
    rn,beliefs,ranking,metadata = resources(args.resources)
    public = build_public_purpose_weights(ranking,beliefs[.25,20].state_ids,
                                          public_destination_states=public_destinations(ranking))
    np.savez_compressed(out/'public_purpose_weights.npz',weights=public.weights.toarray())
    common.write(out/'public_purpose_weights.json',dict(public.metadata,sha256=public.sha256))
    common.write(out/'resources.json',metadata)
    data=common.read(DATA)
    inputs = list(common.sessions(data,splits))
    first = {f:min(s['session_id'] for s in inputs if s['family_id']==f) for f in splits}
    inputs = [s for s in inputs if s['session_id']==first[s['family_id']]]
    history = PublicHistory(rn,[[rn.nearest(p['lat'],p['lon'])[0] for p in s['points']]
        for s in inputs if splits[s['family_id']]=='fit'])
    world = AvailabilityWorld(ranking.n,2026100597,probability=.8,epoch_seconds=60.)
    groups,utilities,models,choices,summaries,attack_rows = {},[],{},{},{},[]
    configurations = CONFIGS+[dict(id='raw',alpha=0.,L=20)]
    for config in configurations:
        method=config['id'];groups[method]={'fit':[],'selection':[]}
        for session in inputs:
            split=splits[session['family_id']]
            if split=='test':continue
            for rep in (0,1):
                run,u = generate(session,config,rep,split,rn,beliefs,public,ranking,world)
                groups[method][split].append(run);utilities.extend(u)
        common.write(out/f'development-{method}.json.gz',groups[method])
        models[method],choices[method],scores,rows = fit_attack(groups[method],history,rn,method)
        attack_rows.extend(rows)
        summaries[method]={'utility':utility_summary(utilities,'selection',method),
            'bytes_per_input':family_mean(groups[method]['selection'],'bytes_per_input'),'attacks':scores}
        print('Purpose development',method,summaries[method]['utility'],flush=True)
    # Exact same private mechanism/clock/seed under every public objective.
    baseline_runs = {(r['session_id'],r['rep']):r for v in groups[BASELINE].values() for r in v}
    for method,parts in groups.items():
        if method=='raw':continue
        for rows in parts.values():
            for r in rows:
                base=baseline_runs[r['session_id'],r['rep']]
                assert r['protected_anchors_evaluator_only']==base['protected_anchors_evaluator_only']
                assert r['ledger_evaluator_only']==base['ledger_evaluator_only']
    selection = select_candidate(summaries)
    common.write(out/'selection.json',selection)
    selected=selection['selected']
    chosen_config=next(c for c in CONFIGS if c['id']==selected)
    sameL=f'alpha000_L{chosen_config["L"]}'
    test_methods=list(dict.fromkeys([selected,BASELINE,sameL,'raw']))
    test=[];test_attacks=[];test_readout={}
    for method in test_methods:
        config=next(c for c in configurations if c['id']==method)
        executions=[]
        for session in inputs:
            if splits[session['family_id']]!='test':continue
            for rep in (0,1):
                run,u=generate(session,config,rep,'test',rn,beliefs,public,ranking,world)
                executions.append(run);test.extend([run]);utilities.extend(u)
        scored=[]
        for scenario in ('S9','S10'):
            scored.extend(common.bank_rows(executions,scenario,models[method][scenario],rn,history))
        test_attacks.extend(scored)
        test_readout[method]={'utility':utility_summary(utilities,'test',method),
            'bytes_per_input':family_mean(executions,'bytes_per_input'),
            'attacks':common.score_summary(scored,choices[method])}
    baseline_test={(r['session_id'],r['rep']):r for r in test if r['method']==BASELINE}
    for r in test:
        if r['method']=='raw':continue
        base=baseline_test[r['session_id'],r['rep']]
        assert r['protected_anchors_evaluator_only']==base['protected_anchors_evaluator_only']
        assert r['ledger_evaluator_only']==base['ledger_evaluator_only']
    common.write(out/'test_runs.json.gz',test)
    common.write(out/'utility_rows.json.gz',utilities)
    common.write(out/'selection_attack_rows.json.gz',attack_rows)
    common.write(out/'test_attack_rows.json.gz',test_attacks)
    common.write(out/'readout.json',{'selection':selected,'alpha':chosen_config['alpha'],'L':chosen_config['L'],
        'promoted_in_development':selection['promoted'],'test':test_readout,
        'backbone_sources_unchanged':all(sha(ROOT/name)==hashes[name] for name in BACKBONE),
        'private_anchor_and_ledger_identity_checked':True,'private_intent_not_network_input':True,
        'all_constraint_violations':sum(r['invalid_returned_items'] for r in utilities),
        'limits':['three reused test families, two RNG draws; development only',
                  'public detour destination prior is a utility approximation, not private intent distribution',
                  'protected-source public objective changes can alter empirical attacks despite same Geo-I cap',
                  'endpoint bank chosen on selection; no untouched privacy confirmation of new Q objective']})
    assert all(sha(ROOT/name)==digest for name,digest in hashes.items())
    print(json.dumps({'selected':selected,'test':test_readout}),flush=True)


if __name__=='__main__':main()
