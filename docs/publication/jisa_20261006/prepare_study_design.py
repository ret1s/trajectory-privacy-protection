"""Write a NEW draft study design; never acquire, score or declare fresh data.

Run from repository root after the matched development protocol is sealed.
For a later revision use --output NEW_PATH, review source/config changes, and
keep the preceding JSON/audit. This generator does not freeze confirmation.
"""
import argparse
import ast
import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[3]
OUT = Path(__file__).with_name('study_design.json')
sys.path.insert(0, str(ROOT))
from experiments.jisa_publication_preflight import content_digest, file_digest, read_json


def source_closure(names):
    """Conservative static local imports, including from-package submodules."""
    pending, found = list(names), set()
    while pending:
        name = pending.pop()
        if name in found: continue
        path = ROOT/name
        if not path.is_file(): raise FileNotFoundError(name)
        found.add(name)
        if path.suffix != '.py': continue
        package = list(Path(name).parent.parts)
        for node in ast.walk(ast.parse(path.read_text())):
            modules = []
            if isinstance(node, ast.Import):
                modules = [alias.name for alias in node.names]
            elif isinstance(node, ast.ImportFrom):
                prefix = package[:len(package)-node.level+1] if node.level else []
                base = '.'.join(prefix + (node.module.split('.') if node.module else []))
                modules = [base] + ['.'.join(filter(None, (base, alias.name))) for alias in node.names]
            for module in modules:
                stem = module.replace('.', '/')
                for candidate in (stem+'.py', stem+'/__init__.py'):
                    if (ROOT/candidate).is_file() and candidate not in found:
                        pending.append(candidate)
    return sorted(found)


def build():
    pilot = read_json(ROOT/'artifacts/benchmarks/jisa_native_anchor_ablation_20261006_v1/protocol.json')
    paths = set(pilot['source_sha256'])
    paths.update([
        'experiments/jisa_publication_preflight.py',
        'evaluation/robust_endpoint_selection.py',
        'evaluation/endpoint_noise_attacks.py',
        'evaluation/query_intent.py',
        'experiments/query_intent_sequence.py',
        'evaluation/ordered_endpoint_attacks.py',
        'evaluation/live_comparison_attacks.py',
        'evaluation/live_comparison_endpoint_attacks.py',
        'experiments/endpoint_robust_selection_20261006.py',
    ])
    paths = set(source_closure(paths))
    pins = [dict(path=p, sha256=file_digest(ROOT/p)) for p in sorted(paths)]
    by = {p['path']: p for p in pins}
    def pinlist(names):
        return [by[p] for p in source_closure(list(names) + ['requirements.txt', 'requirements-sumo.txt'])]
    common = {
        'N': 8, 'H_accounting_parameter': 12, 'max_units': 23,
        'epoch_cap_per_m': .23, 'unit_epsilon_per_m': .00125,
        'nominal_session_cap_per_m': .03, 'effective_session_cap_per_m': .02875,
        'read_interval_s': 60, 'K': 5, 'reply_depth_L': 20,
        'planner_depth_L': 10, 'reference_top_k': 5,
        'theta_m': 200, 'utility_slack': .03,
        'cache_policies': ['current_only', 'versioned_epoch8'],
        'public_epoch_s': [0, 12000],
        'GPS_read_count': 'H12 branch-dependent stopping count12–22; short traces may read fewer; native600s clock has11 scheduled reads',
        'status': 'current development controls; NOT the unimplemented risk-aware candidate',
    }
    methods = []
    for identity, fidelity, primitive, module in [
        ('rem_epoch8', 'proposed', 'road-supported exponential kernel', 'benchmark/engines/paced_slack.py'),
        ('planar_epoch8', 'control', 'full-plane Planar Laplace kernel', 'benchmark/engines/planar_paced.py'),
    ]:
        methods.append(dict(
            id=identity, fidelity=fidelity,
            output_contract='Q_only_coordinates_public_clock_static_POI',
            configuration=dict(common, primitive=primitive,
                emission='primitive-matched release/reuse belief',
                paper_fidelity='primitive control; not faithful full paper reproduction'),
            source_pins=pinlist([
                module, 'benchmark/engines/matched_filter.py',
                'benchmark/engines/filtered_cover.py', 'benchmark/anchor_belief.py',
                'benchmark/response_aware_belief.py', 'core/mechanisms.py',
                'core/session_budget.py', 'benchmark/versioned_static_poi_cache.py',
                'experiments/jisa_native_anchor_ablation_20261006.py',
            ]),
        ))
    attackers = [dict(
        id='candidate_future_geometry_bank',
        source_pins=pinlist(['evaluation/candidate_future_attack.py',
                            'evaluation/identity_future.py',
                            'experiments/jisa_native_anchor_ablation_20261006.py']),
        configuration=dict(
            tasks=['S5_next_edge', 'S6_future_destination'],
            knowledge='two public geometric candidates; causal prefix and six prior sessions',
            bank='CandidateFutureAttack geometry/ExtraTrees/decoder/prior bank',
            training='per mechanism/task; train/selection/confirmation source groups disjoint',
            selection='selection maximum balanced accuracy, then minimum logloss, lexical tie; frozen before confirmation',
        ),
    ), dict(
        id='robust_endpoint_bank',
        source_pins=pinlist(['evaluation/robust_endpoint_selection.py',
                            'evaluation/endpoint_noise_attacks.py',
                            'evaluation/ordered_endpoint_attacks.py',
                            'experiments/endpoint_robust_selection_20261006.py']),
        configuration=dict(
            tasks=['S9_hidden_start', 'S10_completed_endpoint'],
            bank='OLS,kNN,ExtraTrees,Viterbi,Hungarian geometry tracking; per-metric selector',
            status='available development bank; auxiliary and full-history extension pending',
            selection='only train/selection; heldout source-group exclusion; no test-selected orientation',
        ),
    )]
    contracts = ['Q_only_coordinates_public_clock_static_POI']
    supported = {m['id']: dict(status='supported') for m in methods}
    metrics = []
    for identity, kind, unit, direction, definition in [
        ('hit100', 'privacy', '1', 'lower', 'Estimate within100m; average within session then source group; event/empty denominators declared per scenario.'),
        ('hit500', 'privacy', '1', 'lower', 'Estimate within500m; same target and weighting as Hit100; estimator selected and frozen per metric/radius on selection.'),
        ('mae_m', 'privacy', 'm', 'higher', 'Attacker projected Euclidean coordinate error; task/estimator/weighting explicit; not trajectory utility MAE.'),
        ('recall@5', 'utility', '1', 'higher', 'Reference overlap/min(5,reference_count); conditional on nonempty reference; macro by group; completion/undefined count separately.'),
        ('candidate_accuracy', 'privacy', '1', 'lower', 'Exact candidate edge/destination selection; public candidate set and chance/class prior separately.'),
        ('payload_bytes', 'cost', 'B', 'lower', 'Compact UTF8 JSON Q request plus full static POI reply estimate including repeats; excludes HTTP/TLS; per public event.'),
        ('requests', 'cost', 'count', 'lower', 'Total per-Q requests within epoch; no free reset/refill or omitted failed events.'),
    ]:
        metrics.append(dict(id=identity, kind=kind, unit=unit, direction=direction,
                            definition=definition, supported_contracts=contracts,
                            methods=supported))
    metrics.append(dict(
        id='native_real_member_ASR', kind='privacy', unit='1', direction='higher',
        definition='Dummy anonymity score requiring a true member and paper-defined recognition rule; not our Q-only attacker success rate.',
        supported_contracts=['real_member_candidate_set'],
        methods={m['id']: dict(status='n/a', reason='Q-only does not designate a true location member; the paper real-member anonymity score is inapplicable.') for m in methods},
    ))
    history = []
    for path in sorted((ROOT/'artifacts/datasets').rglob('dataset.json*')):
        if path.suffix not in ('.json', '.gz'): continue
        history.append(dict(path=str(path.relative_to(ROOT)),
            sha256=file_digest(path), content_sha256=content_digest(read_json(path))))
    preflight = dict(
        schema_version=1, protocol_id='jisa-20261006-draft-current-controls-v3', stage='draft',
        primary_claim='Matched Geo-I controls prepare a constrained multi-purpose road-feasible Q study; novelty and confirmation pending.',
        analysis_unit='independent source group: real person/vehicle as defined by source; SUMO synthetic family; linked sessions together',
        split_unit=['source_group_id'], source_pins=pins,
        privacy_budget=dict(distance_unit='m', time_unit='s', epsilon_unit='m^-1',
            epoch_cap=.23, public_slots=8, public_horizon=12,
            epsilon_unit_value=.00125, nominal_session_cap=.03, effective_session_cap=.02875),
        methods=methods, attackers=attackers, metrics=metrics,
        confirmation=dict(dataset_registrations=[], historical_datasets=history,
                          frozen_methods={}, frozen_attackers={}),
    )
    return dict(
        prepared_on='2026-10-06', venue='Journal of Information Security and Applications',
        target_route='regular research article assumed pending fresh CFP/deadline',
        status='draft; NOT publication ready; pilot controls do not implement planned risk-aware Q',
        publication_preflight=preflight,
        planned_method=dict(
            implemented=False, component='risk-aware constrained Q selection; Geo-I/filter/epoch retained',
            private_inputs_allowed_for_network='protected extended history only; no raw GPS or true QuerySpec',
            objective='(1-lambda)*mean_coverage + lambda*lower_tail_CVaR_a(coverage), public feasibility and explicit coverage floor',
            parameter_status='a/lambda/public purpose prototypes selected only on development train/selection then frozen; no chosen values/results',
            proof_status='postprocessing conditional on ideal filter; optimizer guarantees not inherited automatically',
            prototype_evidence=dict(
                objective_implemented=True, engine_integrated=False,
                module='benchmark/risk_aware_cover.py',
                source_sha256=file_digest(ROOT/'benchmark/risk_aware_cover.py'),
                tests='tests/test_risk_aware_cover.py',
                publication_effect='not benchmarked or frozen as a candidate method',
            ),
        ),
        planned_evaluation=dict(
            sources=['GeoLife person mobility after license check', 'Porto taxi-source vehicle mobility', 'new SUMO routes/groups'],
            historical_inventory_scope='Canonical dataset.json/json.gz in artifacts/datasets; not a semantic overlap proof or all SQLite/raw traces. Inspected cohorts remain development.',
            confirmation_data_status='not acquired/registered; no false inspection declaration',
            data_split='source_group_id binds linked person/vehicle/family sessions; temporal/geographic stress cohorts separate; no dependent tick shuffle',
            planned_budget_grid_per_m=[.02, .05, .1, .23],
            candidate_selection='Pareto/utility-floor rule on selection preserving privacy/cost; finalize numeric floors using precision/power before new holdout',
            co_primary_endpoints=['cold-start/lower-tail Recall@5 at matched cap and traffic', 'strong-attacker privacy scores with predeclared noninferiority tolerance'],
            uncertainty='paired resampling by independent source group; replicate draws nested within group',
            pending=['full-history and mobility-correlated intent attack', 'real-trajectory loaders/provenance', 'native fidelity audits', 'runtime/mobile cost and dynamic POI freshness', 'numerical sampler guarantee or bounded scope', 'systematic novelty audit/full papers'],
            application_gate=dict(
                status='pending',
                controls=['public static full-catalogue prefetch/local-only',
                          'catalogue-size/cross-region sweep',
                          'dynamic provider availability/price/travel-time with explicit API contract'],
                planner_information='public proxy/model plus actually received replies; no counterfactual current provider state oracle',
                next_service_alignment='actual replyL20 signatures, reference top5 and multiple public-purpose prototypes',
            ),
        ),
    )


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=OUT)
    args = parser.parse_args()
    design = build()
    with args.output.open('x', encoding='utf-8') as stream:
        json.dump(design, stream, ensure_ascii=False, indent=2)
        stream.write('\n')
    p = design['publication_preflight']
    print('Created draft:', len(p['source_pins']), 'actual pins;',
          len(p['confirmation']['historical_datasets']), 'historical datasets; no fresh registration.')
