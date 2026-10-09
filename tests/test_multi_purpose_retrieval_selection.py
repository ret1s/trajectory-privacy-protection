from experiments.multi_purpose_retrieval_20261009 import CHANNELS,CANDIDATES,select


def summary():
    purposes=['nearest_distance','fastest_travel','within_radius','minimum_detour','equal_purpose_macro']
    return {m:dict(utility={p:dict(family_mean=.9) for p in purposes},cost=dict(reply_bytes=100)) for m in CHANNELS}


def test_selection_allows_no_winner_and_does_not_reward_more_requests_alone():
    s=summary()
    for m in CANDIDATES:s[m]['cost']['reply_bytes']=150
    assert select(s)['selected_method'] is None


def test_macro_gain_cannot_hide_a_large_loss_in_one_private_purpose():
    s=summary();m=CANDIDATES[0]
    for v in s[m]['utility'].values():v['family_mean']=.92
    s[m]['utility']['within_radius']['family_mean']=.89
    assert select(s)['selected_method'] is None


def test_matched_response_ceiling_and_cost_choose_only_an_eligible_candidate():
    s=summary()
    for i,m in enumerate(CANDIDATES):
        for v in s[m]['utility'].values():v['family_mean']=.92
        s[m]['cost']['reply_bytes']=120-i*5
    assert select(s)['selected_method']==CANDIDATES[2]
    for v in s['nearest_l40']['utility'].values():v['family_mean']=.93
    assert select(s)['selected_method']==CANDIDATES[0]
