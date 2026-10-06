"""Fixed development gates retain failures and use no fresh test field."""
import copy
import pytest

from experiments.qplanner_select_and_freeze_20261006 import choose


def inputs():
    methods = ('legacy_l10', 'aligned_nearest', 'multi_mean', 'tail25', 'tail50')
    utility = {'summary': {}}
    attacks = {'results': {}}
    for method in methods:
        mean = .90 if method == 'legacy_l10' else .95
        tail = {'legacy_l10': .70, 'aligned_nearest': .80, 'multi_mean': .82, 'tail25': .86, 'tail50': .86}[method]
        utility['summary'][method] = {'selection': {'current': {'all': {
            'equal_purpose_macro': {'family_mean': mean, 'family_lower_quartile_cvar': tail},
            'nearest_distance': {'family_mean': mean}}}}}
        for scenario in ('S9', 'S10'):
            attacks['results'][method+'--'+scenario] = {'selection': {'mae_m': 500., 'hit100': .1, 'hit500': .4}}
        for task in ('S5', 'S6'):
            attacks['results'][method+'--turn_visible--'+task] = {'selection': {'exact_candidate_edge_accuracy': .5}}
    return utility, attacks


def test_tie_prefers_lower_risk_and_never_consults_fake_test_scores():
    utility, attacks = inputs()
    utility['summary']['tail50']['test'] = 'arbitrary test value that must never be read'
    selected, records = choose(utility, attacks)
    assert selected == 'tail25' and all(v['eligible'] for v in records.values())


def test_endpoint_and_nearest_guards_reject_best_utility_without_relaxation():
    utility, attacks = inputs()
    attacks['results']['tail25--S9']['selection']['hit500'] = .9
    utility['summary']['tail50']['selection']['current']['all']['nearest_distance']['family_mean'] = .8
    selected, records = choose(utility, attacks)
    assert selected == 'multi_mean'
    assert not records['tail25']['eligible'] and not records['tail50']['eligible']


def test_no_forced_winner_when_all_utility_gates_fail():
    utility, attacks = inputs()
    for method in utility['summary']:
        utility['summary'][method]['selection']['current']['all']['equal_purpose_macro']['family_mean'] = .9
    selected, records = choose(utility, attacks)
    assert selected is None and all(not r['eligible'] for r in records.values())
