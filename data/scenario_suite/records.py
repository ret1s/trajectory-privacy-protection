"""Scenario gates and evaluator labels, derived AFTER SUMO simulation.

Stored indices refer to untouched 1 Hz FCD, never an interpolated trajectory.
The full bundle is evaluator-only. Consumers must use the bounded device view.
"""
from collections import Counter
import math

from .catalog import catalogue
from .mobility import successors, predecessors


def distance(a, b):
    # Local geographic distance is sufficient for declared 100/500 m gates.
    lat = math.radians((a['lat'] + b['lat']) / 2)
    return 6371000 * math.hypot(math.radians(a['lat']-b['lat']),
                               math.cos(lat)*math.radians(a['lon']-b['lon']))


def sample(trace, indices=None, interval=20):
    indices = range(len(trace)) if indices is None else indices
    selected = []
    for i in indices:
        if not selected or trace[i]['time_s']-trace[selected[-1]]['time_s'] >= interval:
            selected.append(i)
    return selected


def runs(trace, predicate):
    result, current = [], []
    for i, p in enumerate(trace):
        if predicate(p) and (not current or p['time_s']-trace[current[-1]]['time_s'] <= 1.01):
            current.append(i)
        else:
            if current:
                result.append(current)
            current = [i] if predicate(p) else []
    if current:
        result.append(current)
    return result


def build_records(design, traces, completed, network, split):
    roles = {s['role']: s for s in design['sessions']}
    out = []
    rejected = []

    def add(case, sessions, observed, labels, evidence):
        if not all(s['session_id'] in completed for s in sessions):
            rejected.append({'case_id': case, 'reason': 'requires_completed_sessions'})
            return
        if any(not indices for indices in observed):
            rejected.append({'case_id': case, 'reason': 'empty_observation'})
            return
        out.append({'record_id': f"r{design['seed']}-{len(out):03d}", 'case_id': case,
                    'scenario': case.split('.')[0], 'family_id': design['family_id'], 'split': split,
                    'session_ids': [s['session_id'] for s in sessions], 'observed_indices': observed,
                    'labels': labels, 'evidence': evidence,
                    'observation_policy': {'coordinates': 'device_only_until_protected',
                                           'clock': 'common_pair_epoch' if case.startswith('S8.') else 'relative_to_first_allowed_sample_per_session',
                                           'route_plan': 'evaluator_only', 'full_trip_duration': 'withheld'}})

    base = roles['base']
    b = traces[base['session_id']]
    external = [i for i, p in enumerate(b) if not p['edge_id'].startswith(':')]
    degree = {p['edge_id']: len(successors(network.getEdge(p['edge_id']))) for p in b if not p['edge_id'].startswith(':')}
    for case, pred in [('S1.A', lambda d: d >= 2), ('S1.B', lambda d: d == 1)]:
        indices = [i for i in external if pred(degree[b[i]['edge_id']])]
        if indices:
            i = indices[len(indices)//2]
            add(case, [base], [[i]], {'target_index': i}, {'legal_successors': degree[b[i]['edge_id']]})

    for role, case, minimum, interval in [('short_stop','S2.A',20,5), ('long_stop','S2.B',120,20),
                                          ('return_stop','S2.C',20,5)]:
        s = roles[role]
        trace = traces[s['session_id']]
        stop = s['stops'][0]
        stationary = runs(trace, lambda p: p['speed_m_s'] <= .05 and p['lane_id'] == stop['lane'] and
                           abs(p['lane_pos_m']-stop['endPos']) <= 1)
        stationary = [r for r in stationary if trace[r[-1]]['time_s']-trace[r[0]]['time_s'] >= minimum]
        count = 2 if case == 'S2.C' else 1
        if len(stationary) >= count:
            chosen = stationary[:count]
            between = count == 1 or any(p['speed_m_s'] > 1 for p in trace[chosen[0][-1]+1:chosen[1][0]])
            if between:
                add(case, [s], [sum((sample(trace, r, interval) for r in chosen), [])],
                    {'stop_intervals': [[r[0],r[-1]] for r in chosen], 'target_index': chosen[0][0],
                     'purpose': 'scheduled_synthetic_stop_not_inferred_home'},
                    {'actual_durations_s': [trace[r[-1]]['time_s']-trace[r[0]]['time_s'] for r in chosen],
                     'same_lane': stop['lane'], 'movement_between_visits': between})

    moving = sorted(runs(b, lambda p: p['speed_m_s'] > .05), key=len, reverse=True)
    eligible = [r for r in moving if b[r[-1]]['time_s']-b[r[0]]['time_s'] >= 60 and
                len({b[i]['edge_id'] for i in r if not b[i]['edge_id'].startswith(':')}) >= 3]
    if eligible:
        r = eligible[0]
        for case, interval in [('S3.A',20),('S3.C',60)]:
            obs = sample(b,r,interval)
            add(case,[base],[obs],{'target_indices': obs}, {'query_interval_s':interval,'moving_span_s':b[r[-1]]['time_s']-b[r[0]]['time_s']})
        for r in eligible:
            obs = sample(b,r)
            ids = [i for i in obs if b[i]['edge_id'] in degree]
            fraction = sum(degree[b[i]['edge_id']] == 1 for i in ids)/max(1,len(ids))
            if fraction >= .5:
                add('S3.B',[base],[obs],{'target_indices':obs},{'single_successor_fraction':fraction})
                break

    for case, role in [('S4.A','repeat'), ('S4.B','new_device'), ('S4.C','shared_device')]:
        s = roles[role]
        add(case,[base,s],[sample(b),sample(traces[s['session_id']])],
            {'same_person':base['person_id']==s['person_id'], 'same_device':base['device_id']==s['device_id'],
             'task':'pairwise_linkage_not_closed_set_identification'}, {'non_overlapping': b[-1]['time_s'] < traces[s['session_id']][0]['time_s']})

    # Next external-edge transition, not fixed-30-second location forecasting.
    transitions = [(a,c) for a,c in zip(external,external[1:]) if b[a]['edge_id'] != b[c]['edge_id'] and
                   b[c]['edge_id'] in {e.getID() for e in successors(network.getEdge(b[a]['edge_id']))}]
    for case, pred in [('S5.A',lambda n:n>=2),('S5.B',lambda n:n==1)]:
        choices = [(a,c) for a,c in transitions if a >= 20 and pred(degree[b[a]['edge_id']])]
        if choices:
            a,c = choices[0]
            obs = sample(b,range(max(0,a-120),a+1))
            if obs[-1] != a:
                obs.append(a)
            add(case,[base],[obs],{'next_edge':b[c]['edge_id'],'future_index':c},
                {'legal_successors':degree[b[a]['edge_id']], 'horizon_s':b[c]['time_s']-b[a]['time_s']})

    alternate = roles['partial']
    alt = traces[alternate['session_id']]
    fork = design['fork_edge']
    pre = [[i for i,p in enumerate(t) if p['edge_id']==fork] for t in [b,alt]]
    if all(pre):
        cuts = [p[-1] for p in pre]
        future = [next((i for i in range(c+1,len(t)) if not t[i]['edge_id'].startswith(':')), None) for t,c in zip([b,alt],cuts)]
        observed = [sample(t,range(c+1)) + ([] if sample(t,range(c+1))[-1]==c else [c]) for t,c in zip([b,alt],cuts)]
        if all(i is not None for i in future) and b[future[0]]['edge_id'] != alt[future[1]]['edge_id']:
            add('S5.C',[base,alternate],observed,{'future_indices':future,'next_edges':[b[future[0]]['edge_id'],alt[future[1]]['edge_id']]},
                {'shared_planned_prefix':True,'fork_edge':fork,'note':'speed/timing need not be identical'})
            for case in ['S6.A','S6.B','S10.B']:
                separation = distance(b[-1],alt[-1])
                if case == 'S6.B' and separation > 500:
                    continue
                add(case,[base,alternate],observed,{'target_indices':[len(b)-1,len(alt)-1]},
                    {'endpoint_separation_m':separation,'prefix_cut':'end_of_shared_fork_edge',
                     'interpretation':'online_destination' if case.startswith('S6') else 'offline_hidden_endpoint'})

    # Synthetic query intent layer; it is NOT a SUMO-generated behavior label.
    for case in ['S7.A','S7.B']:
        i = len(b)//2
        for intent, query in [('medical_visit','clinic'),('routine_shopping','cafe'),('road_trip','fuel')]:
            add(case,[base],[[i]],{'true_queries':[query],'intent':intent,'intent_source':'synthetic'},
                {'reference_query_policy':'plaintext' if case=='S7.A' else 'constant_six_category_cover'})
    for intent, sequence in [('medical_visit',['pharmacy','clinic','hospital']),
                             ('routine_shopping',['pharmacy','cafe','restaurant']),
                             ('road_trip',['pharmacy','fuel','restaurant'])]:
        obs = sample(b, interval=20)[:3]
        if len(obs)==3:
            add('S7.C',[base],[obs],{'true_queries':sequence,'intent':intent,'intent_source':'synthetic'},
                {'reference_query_policy':'plaintext_sequence', 'shared_first_query':'pharmacy',
                 'note':'handwritten diagnostic templates, not representative human intent'})

    for case, role, declared in [('S8.A','partial',True),('S8.B','companion',True),('S8.C','incidental',False)]:
        s=roles[role]
        t=traces[s['session_id']]
        by_time={p['time_s']:i for i,p in enumerate(t)}
        pairs=[(i,by_time[p['time_s']]) for i,p in enumerate(b) if p['time_s'] in by_time]
        close=[(i,j) for i,j in pairs if distance(b[i],t[j])<=100]
        fraction=len(close)/max(1,len(pairs))
        near_times={b[i]['time_s'] for i,j in close}
        longest=max((len(r)-1 for r in runs(b,lambda p:p['time_s'] in near_times)),default=0)
        # A requires observed separation as well as a distinct planned continuation.
        if longest>=10 and (case!='S8.B' or fraction>=.8) and (case!='S8.A' or len(close)<len(pairs)):
            add(case,[base,s],[sample(b),sample(t)],{'declared_companions':declared,
                'proximity_pairs':close,'target_definition':'synthetic_relation_separate_from_measured_proximity'},
                {'aligned_timestamps':len(pairs),'within_100m_samples':len(close),'proximity_fraction':fraction,
                 'max_consecutive_close_s': longest,
                 'companion_knowledge':'separate optional attacker side information, not automatically disclosed'})

    # Retain the WHOLE permitted trip (sampled 20 s), unlike v2's 220 s excerpt.
    first = next(i for i in external)
    access = roles['access_endpoint']
    access_trace = traces[access['session_id']]
    for case, degree_count in [('S9.A',degree[b[first]['edge_id']]),
                              ('S10.A',len(predecessors(network.getEdge(access_trace[-1]['edge_id']))))]:
        if degree_count == 1:
            start_case = case.startswith('S9')
            s,t = (base,b) if start_case else (access,access_trace)
            keep=[i for i,p in enumerate(t) if p['time_s'] >= t[0]['time_s']+60] if start_case else [i for i,p in enumerate(t) if p['time_s'] <= t[-1]['time_s']-60]
            add(case,[s],[sample(t,keep)],{'target_index':0 if start_case else len(t)-1},
                {'mask_s':60,'legal_access_count':degree_count,'endpoint_semantics':'first_or_last_FCD_not_home'})
    merged=roles['merged']
    mt=traces[merged['session_id']]
    merge=design['merge_edge']
    starts=[next((i for i,p in enumerate(t) if p['edge_id']==merge),None) for t in [b,mt]]
    if all(i is not None and i>0 for i in starts):
        add('S9.B',[base,merged],[sample(t,range(i,len(t))) for t,i in zip([b,mt],starts)],
            {'target_indices':[0,0]}, {'common_suffix_from':merge,'origin_separation_m':distance(b[0],mt[0])})
    repeat=roles['repeat']
    rt=traces[repeat['session_id']]
    for case in ['S9.C','S10.C']:
        start_case=case.startswith('S9')
        obs=[]
        for t in [b,rt]:
            keep=[i for i,p in enumerate(t) if p['time_s'] >= t[0]['time_s']+60] if start_case else [i for i,p in enumerate(t) if p['time_s'] <= t[-1]['time_s']-60]
            obs.append(sample(t,keep))
        add(case,[base,repeat],obs,{'target_indices':[0,0] if start_case else [len(b)-1,len(rt)-1]},
            {'mask_s':60,'same_planned_endpoint':True,'endpoint_semantics':'FCD_not_real_home'})

    counts=Counter(r['case_id'] for r in out)
    for c in catalogue():
        if not counts[c['case_id']]:
            rejected.append({'case_id':c['case_id'],'reason': 'not_implemented_requires_context_or_activity_model' if c['case_id'] in {'S1.C','S6.C'} else 'realized_data_did_not_meet_gate'})
    return out,rejected


def device_view(record, traces, slot=0):
    """Only pre-protection observations. Never use this as attacker output.

    No global clock, IDs, seed, labels, future or lane plan crosses this API.
    """
    trace=traces[record['session_ids'][slot]]
    allowed=record['observed_indices'][slot]
    epoch=trace[allowed[0]]['time_s']
    if record['observation_policy']['clock']=='common_pair_epoch':
        # Preserve inter-user timing for co-location. Epoch uses only first
        # allowed event times, not hidden departure time or future coordinates.
        epoch=min(traces[sid][idx[0]]['time_s'] for sid,idx in zip(record['session_ids'],record['observed_indices']))
    queries=record['labels'].get('true_queries', [])
    for j,i in enumerate(allowed):
        p=trace[i]
        yield {'time_s':p['time_s']-epoch,'lat':p['lat'],'lon':p['lon'],
               'query_category':queries[j] if j<len(queries) else 'cafe'}


def reference_queries(record, traces):
    """Query-only diagnostic controls, NOT output from the proposed mechanism."""
    policy=record['evidence']['reference_query_policy']
    return [{'time_s':p['time_s'],'categories': ['cafe','clinic','fuel','hospital','pharmacy','restaurant']
             if policy=='constant_six_category_cover' else [p['query_category']]}
            for p in device_view(record,traces)]
