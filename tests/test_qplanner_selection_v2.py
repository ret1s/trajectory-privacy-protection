"""Fixed development gates retain failures and use no fresh test field."""
import copy
import pytest

from experiments.qplanner_select_and_freeze_20261006_v2 import choose, RULE


def inputs():
    methods = ('legacy_l10', 'aligned_nearest', 'normalized_mean', 'normalized_tight', 'normalized_tail')
    utility = {'summary': {}}
    attacks = {'results': {}}
    for method in methods:
        mean = .90 if method == 'legacy_l10' else .95
        tail = {'legacy_l10': .70, 'aligned_nearest': .80, 'normalized_mean': .82, 'normalized_tight': .86, 'normalized_tail': .86}[method]
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
    utility['summary']['normalized_tail']['test'] = 'arbitrary test value that must never be read'
    selected, records = choose(utility, attacks)
    assert selected == 'normalized_tight' and all(v['eligible'] for v in records.values())


def test_endpoint_and_nearest_guards_reject_best_utility_without_relaxation():
    utility, attacks = inputs()
    attacks['results']['normalized_tight--S9']['selection']['hit500'] = .9
    utility['summary']['normalized_tail']['selection']['current']['all']['nearest_distance']['family_mean'] = .8
    selected, records = choose(utility, attacks)
    assert selected == 'normalized_mean'
    assert not records['normalized_tight']['eligible'] and not records['normalized_tail']['eligible']


def test_no_forced_winner_when_all_utility_gates_fail():
    utility, attacks = inputs()
    for method in utility['summary']:
        utility['summary'][method]['selection']['current']['all']['equal_purpose_macro']['family_mean'] = .9
    selected, records = choose(utility, attacks)
    assert selected is None and all(not r['eligible'] for r in records.values())


def test_second_round_does_not_relax_original_thresholds():
    from experiments.qplanner_select_and_freeze_20261006 import RULE as original
    for key in ('utility_min_gain','nearest_max_loss','endpoint_min_mae_ratio',
                'endpoint_max_hit_increase','future_max_accuracy_increase','fresh_criterion'):
        assert RULE[key] == original[key]
