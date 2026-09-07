"""Extend realized v1 records, without editing FCD or leaking future inputs."""
from collections import Counter

from data.scenario_suite.catalog import catalogue as base_catalogue
from data.scenario_suite.records import build_records as base_records, sample, distance


def catalogue():
    result = base_catalogue()
    for c in result:
        if c['case_id']=='S1.C':
            c['eligibility']='Within 150 m of an OSM POI whose category is <=5% of the pinned catalogue'
        elif c['case_id']=='S6.C':
            c['eligibility']='Six completed historical day trips (5 routine, 1 rare); current prefix precedes destination'
    return result


def build_records(design,traces,completed,network,split,context):
    records,rejected = base_records(design,traces,completed,network,split)
    # The next OBSERVED edge at 1 Hz need not be the immediate next edge.
    # Keep S5.C only when its target is the actual adjacent external edge.
    kept=[]
    for r in records:
        if r['case_id']=='S5.C':
            valid=all(target in {e.getID() for e,links in network.getEdge(traces[sid][idx[-1]]['edge_id']).getOutgoing().items()
                                if not e.getID().startswith(':') and e.allows('passenger') and
                                any(c.getFromLane().allows('passenger') and c.getToLane().allows('passenger') for c in links)}
                      for sid,idx,target in zip(r['session_ids'],r['observed_indices'],r['labels']['next_edges']))
            if not valid:
                rejected.append({'case_id':'S5.C','reason':'immediate_next_edge_not_observed_at_1Hz'})
                continue
        kept.append(r)
    records=kept
    # Replace the old nearby-destination attempt with a purpose-built pair.
    records = [r for r in records if r['case_id']!='S6.B']
    roles = {s['role']:s for s in design['sessions']}

    def add(case,sessions,observations,labels,evidence):
        if any(s['session_id'] not in completed for s in sessions) or any(not x for x in observations):
            rejected.append({'case_id':case,'reason':'required_completed_observations_missing'})
            return
        records.append({'case_id':case,'scenario':case.split('.')[0],
                        'family_id':design['family_id'],'split':split,
                        'session_ids':[s['session_id'] for s in sessions],
                        'observed_indices':observations,'labels':labels,'evidence':evidence,
                        'observation_policy':{'coordinates':'device_only_until_protected',
                            'clock':'relative_to_first_allowed_sample_per_session',
                            'route_plan':'evaluator_only','full_trip_duration':'withheld'}})

    visit = roles['rare_poi_visit']
    trace = traces[visit['session_id']]
    poi = next(p for p in context['pois'] if p['id']==visit['planned_poi_id'])
    idx = min(range(len(trace)),key=lambda i:distance(trace[i],poi))
    offset = distance(trace[idx],poi)
    if offset <= context['max_observed_distance_m']:
        add('S1.C',[visit],[[idx]],{'target_index':idx},
            {'poi_id':poi['id'],'category':poi['category'],'distance_m':offset,
             'category_share':context['category_counts'][poi['category']]/len(context['pois']),
             'rarity_definition':context['definition']})

    base,near = roles['base'],roles['near_destination']
    b,n = traces[base['session_id']],traces[near['session_id']]
    separation = distance(b[-1],n[-1])
    cuts = [next((i for i,p in enumerate(t) if p['edge_id']==design['fork_edge']),None) for t in (b,n)]
    if all(i is not None for i in cuts) and 0 < separation <= 500:
        add('S6.B',[base,near],[sample(t,range(c+1)) for t,c in zip((b,n),cuts)],
            {'target_indices':[len(b)-1,len(n)-1]},
            {'endpoint_separation_m':separation,'prefix_cut':'first_observation_on_shared_fork_edge',
             'interpretation':'online_destination'})

    history = [roles[f'activity_day_{d}'] for d in range(1,7)]
    frequencies = dict(Counter(s['planned_destination_class'] for s in history))
    for day in (7,8):
        current = roles[f'activity_day_{day}']
        t = traces[current['session_id']]
        cut = next((i for i,p in enumerate(t) if p['edge_id']==design['fork_edge']),None)
        if cut is None or cut>=len(t)-1:
            continue
        sessions = history+[current]
        obs = [sample(traces[s['session_id']]) for s in history]+[sample(t,range(cut+1))]
        add('S6.C',sessions,obs,
            {'target_index':len(t)-1,'target_slot':6,'destination_class':current['planned_destination_class'],
             'historical_destination_counts':frequencies,'label_source':'synthetic_diagnostic_schedule'},
            {'history_days':list(range(1,7)),'query_day':day,'prefix_cut':'first_observation_on_shared_fork_edge',
             'history_access':'historical device observations require protection before attacker access',
             'overnight_activity':'outside_observation_windows'})

    for i,r in enumerate(records):
        r['record_id']=f"v2-r{design['seed']}-{i:03d}"
    counts=Counter(r['case_id'] for r in records)
    rejected=[r for r in rejected if not counts[r['case_id']]]
    for c in catalogue():
        if not counts[c['case_id']] and not any(r['case_id']==c['case_id'] for r in rejected):
            rejected.append({'case_id':c['case_id'],'reason':'realized_data_did_not_meet_v2_gate'})
    return records,rejected
