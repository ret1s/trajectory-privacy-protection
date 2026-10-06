"""Fixed development gate and freeze before new synthetic TEST evaluation.

The same thresholds as the first fixed development rule are retained. Candidate
weighting and slack follow failed complete old-development results; this is
exploratory defense selection, not a blind preregistration. The new
test contrasts/criterion are fixed before their first benchmark is generated.
No thresholds are relaxed when a candidate fails; all rejected arms remain.
"""
import argparse
from pathlib import Path

from experiments.build_future_sumo_cohort import ROOT, sha, save
from experiments.qplanner_study_20261006_v2 import declare, read, OUT, configuration

FRESH_DATA = ROOT/'artifacts/datasets/qplanner_fresh_native_20261006_v2/dataset.json.gz'
FRESH_OUT = ROOT/'artifacts/benchmarks/qplanner_generalization_20261006_v1'
RULE = {'development_status': 'second exploratory round after failed first objective; original thresholds unchanged; fresh scores unopened',
    'utility_min_gain': .01, 'nearest_max_loss': .005,
    'endpoint_min_mae_ratio': .90, 'endpoint_max_hit_increase': .10,
    'future_max_accuracy_increase': .10,
    'ranking': 'descending equal-purpose current family lower25%CVaR, descending mean, ascending risk weight, lexical name',
    'fresh_primary': 'selected candidate minus legacy_l10; current/all/equal-purpose/equal-family Recall@5',
    'fresh_criterion': {'minimum_absolute_mean_gain': .02, 'paired95_lower_bound_gt': 0.,
        'every_private_draw_gain_gt': 0., 'family_bootstrap_replicates': 10000, 'public_analysis_seed': 2026100617},
    'fresh_draws_by_split': {'train': 1, 'selection': 1, 'test': 3},
    'secondary': 'alignment control, individual purposes, cold, early, temporal tail, N/A coverage, finite attacker bank and costs',
    'no_equivalence_claim': 'privacy development guard is heuristic; no noninferiority power/equivalence test; same ideal cap does not imply same inference risk',
    'failure': 'if no candidate passes, retain none and do not freeze a purported winner'}


def choose(utility, attacks):
    summaries = utility['summary']; baseline = summaries['legacy_l10']['selection']['current']['all']
    records = {}
    for method in ('aligned_nearest', 'normalized_mean', 'normalized_tight', 'normalized_tail'):
        target = summaries[method]['selection']['current']['all']; reasons = []
        gain = target['equal_purpose_macro']['family_mean']-baseline['equal_purpose_macro']['family_mean']
        nearest_loss = baseline['nearest_distance']['family_mean']-target['nearest_distance']['family_mean']
        if gain < RULE['utility_min_gain']: reasons.append('insufficient utility gain')
        if nearest_loss > RULE['nearest_max_loss']: reasons.append('nearest utility floor failed')
        privacy = {}
        for scenario in ('S9', 'S10'):
            original = attacks['results'][f'legacy_l10--{scenario}']['selection']
            candidate = attacks['results'][f'{method}--{scenario}']['selection']
            privacy[scenario] = {'mae_ratio': candidate['mae_m']/original['mae_m'],
                'hit100_delta': candidate['hit100']-original['hit100'],
                'hit500_delta': candidate['hit500']-original['hit500']}
            if privacy[scenario]['mae_ratio'] < RULE['endpoint_min_mae_ratio']:
                reasons.append(scenario+' endpoint MAE guard failed')
            if any(privacy[scenario][h+'_delta'] > RULE['endpoint_max_hit_increase'] for h in ('hit100', 'hit500')):
                reasons.append(scenario+' endpoint Hit guard failed')
        for task in ('S5', 'S6'):
            original = attacks['results'][f'legacy_l10--turn_visible--{task}']['selection']['exact_candidate_edge_accuracy']
            candidate = attacks['results'][f'{method}--turn_visible--{task}']['selection']['exact_candidate_edge_accuracy']
            privacy[task] = {'accuracy_delta': candidate-original}
            if candidate-original > RULE['future_max_accuracy_increase']: reasons.append(task+' future guard failed')
        risk_weight = {'aligned_nearest': 0., 'normalized_mean': 0., 'normalized_tight': 0., 'normalized_tail': .25}[method]
        records[method] = {'eligible': not reasons, 'reasons': reasons, 'macro_gain': gain,
            'nearest_loss': nearest_loss, 'family_lower25_cvar': target['equal_purpose_macro']['family_lower_quartile_cvar'],
            'family_mean': target['equal_purpose_macro']['family_mean'], 'risk_weight': risk_weight,
            'privacy_guard': privacy}
    eligible = [m for m in records if records[m]['eligible']]
    selected = min(eligible, key=lambda m: (-records[m]['family_lower25_cvar'], -records[m]['family_mean'],
                   records[m]['risk_weight'], m)) if eligible else None
    return selected, records


def run(development, dataset, fresh):
    selection_path = development/'defense_selection.json'
    if selection_path.exists() or (fresh/'freeze.json').exists(): raise FileExistsError('Selection/freeze is write-once')
    inputs = {name: sha(development/name) for name in ('protocol.json', 'utility_readout.json', 'attack_selection.json', 'attack_readout.json')}
    selected, records = choose(read(development/'utility_readout.json'), read(development/'attack_readout.json'))
    receipt = {'schema': 'qplanner-development-selection-v1', 'rule': RULE,
        'source_sha256': sha(Path(__file__)), 'development_inputs_sha256': inputs,
        'records': records, 'selected': selected, 'fresh_test_scores_viewed': False}
    save(selection_path, receipt)
    if selected is None:
        print('No candidate passed the fixed development gate; no claimed winner frozen', flush=True)
        return receipt
    methods = ['legacy_l10', 'aligned_nearest']
    methods.append(selected if selected != 'aligned_nearest' else 'normalized_mean')
    declare(fresh, dataset, methods, 2, ['train', 'selection', 'test'],
        'New SAME-MAP SYNTHETIC junction-group generalization; defense selected only on old development; real mobility/cross-city/dynamic service remain unvalidated',
        RULE['fresh_draws_by_split'])
    save(fresh/'freeze.json', {'schema': 'qplanner-synthetic-generalization-freeze-v1',
        'selected': selected, 'baseline': 'legacy_l10', 'alignment_control': 'aligned_nearest',
        'primary_rule': RULE['fresh_primary'], 'criterion': RULE['fresh_criterion'],
        'selection_path': str(selection_path.relative_to(ROOT)), 'selection_sha256': sha(selection_path),
        'fresh_protocol_sha256': sha(fresh/'protocol.json'), 'configuration': configuration(methods),
        'freeze_source_sha256': sha(Path(__file__)), 'fresh_test_evaluated': False,
        'scope': 'new same-map/generator synthetic families, not real GPS confirmation or publication readiness'})
    print('Frozen candidate', selected, 'methods', methods, 'before fresh test benchmark', flush=True)
    return receipt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--development', type=Path, default=OUT)
    parser.add_argument('--dataset', type=Path, default=FRESH_DATA)
    parser.add_argument('--output', type=Path, default=FRESH_OUT)
    args = parser.parse_args(); run(args.development, args.dataset, args.output)


if __name__ == '__main__': main()
