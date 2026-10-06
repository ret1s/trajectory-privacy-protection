"""Independent synthetic fixtures for unchanged second-round selection gates.

This module never opens recorded utility/attack scores or the fresh cohort.
"""
import copy

from experiments.qplanner_select_and_freeze_20261006 import RULE as FIRST_RULE
from experiments.qplanner_select_and_freeze_20261006_v2 import RULE, choose


METHODS = ('legacy_l10', 'aligned_nearest', 'normalized_mean',
           'normalized_tight', 'normalized_tail')


def fixture():
    utility, attacks = {'summary': {}}, {'results': {}}
    for method in METHODS:
        baseline = method == 'legacy_l10'
        utility['summary'][method] = {'selection': {'current': {'all': {
            'equal_purpose_macro': {'family_mean': .8 if baseline else .82,
                                   'family_lower_quartile_cvar': .70 if baseline else .75},
            'nearest_distance': {'family_mean': .9}}}},
            'test': object()}
        for scenario in ('S9', 'S10'):
            attacks['results'][f'{method}--{scenario}'] = {
                'selection': {'mae_m': 500., 'hit100': .1, 'hit500': .4},
                'test': object()}
        for scenario in ('S5', 'S6'):
            attacks['results'][f'{method}--turn_visible--{scenario}'] = {
                'selection': {'exact_candidate_edge_accuracy': .5},
                'test': object()}
    return utility, attacks


def test_original_utility_privacy_and_fresh_confirmation_gates_are_unchanged():
    fields = ('utility_min_gain', 'nearest_max_loss', 'endpoint_min_mae_ratio',
              'endpoint_max_hit_increase', 'future_max_accuracy_increase',
              'fresh_criterion', 'fresh_draws_by_split')
    assert {k: RULE[k] for k in fields} == {k: FIRST_RULE[k] for k in fields}
    assert RULE['fresh_criterion'] == {
        'minimum_absolute_mean_gain': .02, 'paired95_lower_bound_gt': 0.,
        'every_private_draw_gain_gt': 0., 'family_bootstrap_replicates': 10000,
        'public_analysis_seed': 2026100617}


def test_utility_failure_retains_every_candidate_and_does_not_force_a_winner():
    utility, attacks = fixture()
    for method in METHODS[1:]:
        utility['summary'][method]['selection']['current']['all'][
            'equal_purpose_macro']['family_mean'] = .8099
    selected, records = choose(utility, attacks)
    assert selected is None
    assert set(records) == set(METHODS[1:])
    assert all(not r['eligible'] and 'insufficient utility gain' in r['reasons']
               for r in records.values())


def test_better_lower_tail_cannot_override_endpoint_or_future_failures():
    utility, attacks = fixture()
    for method in ('normalized_tight', 'normalized_tail'):
        utility['summary'][method]['selection']['current']['all'][
            'equal_purpose_macro']['family_lower_quartile_cvar'] = .80
    attacks['results']['normalized_tight--S9']['selection']['mae_m'] = 449.
    attacks['results']['normalized_tail--turn_visible--S6']['selection'][
        'exact_candidate_edge_accuracy'] = .601
    selected, records = choose(utility, attacks)
    assert selected == 'aligned_nearest'
    assert records['normalized_tight']['reasons'] == ['S9 endpoint MAE guard failed']
    assert records['normalized_tail']['reasons'] == ['S6 future guard failed']


def test_nearest_floor_and_hit_guard_remain_separate_requirements():
    utility, attacks = fixture()
    utility['summary']['normalized_mean']['selection']['current']['all'][
        'nearest_distance']['family_mean'] = .894
    attacks['results']['normalized_tight--S10']['selection']['hit500'] = .501
    _, records = choose(utility, attacks)
    assert records['normalized_mean']['reasons'] == ['nearest utility floor failed']
    assert records['normalized_tight']['reasons'] == ['S10 endpoint Hit guard failed']


def test_public_risk_tie_break_and_poisoned_test_fields_have_no_score_input():
    utility, attacks = fixture()
    for method in ('normalized_mean', 'normalized_tail'):
        utility['summary'][method]['selection']['current']['all'][
            'equal_purpose_macro']['family_lower_quartile_cvar'] = .80
    selected, _ = choose(utility, attacks)
    assert selected == 'normalized_mean'
    changed = copy.deepcopy(utility)
    for method in METHODS:
        changed['summary'][method]['test'] = {'all': {'unseen': float('nan')}}
    assert choose(changed, attacks)[0] == selected
