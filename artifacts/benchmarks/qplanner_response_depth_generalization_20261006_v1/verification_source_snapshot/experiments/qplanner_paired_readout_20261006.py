"""Independent paired-family uncertainty for frozen Q-planner evidence.

All sessions and secret draws stay inside the resampled family cluster. This
script never selects a defense and never treats ticks/draws as independent
subjects. Its protocol is saved before reading utility rows. Fresh-test access
requires an explicit already-durable method/source freeze record.
"""
from pathlib import Path
import argparse
from collections import defaultdict
import gzip
import hashlib
import json

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
DEFAULT = ROOT/'artifacts/benchmarks/qplanner_development_20261006_v2'
REPLICATES = 10000
SEED = 2026100617
PHASES = ('all', 'cold', 'early_0_180', 'temporal_tail_400_600')
CACHES = ('current', 'static_epoch_cache')
PUBLIC_CACHE_FILES = ('native.net.xml', 'reference5.npz', 'reply10.npz',
    'reply20.npz', 'belief-0.01.npz', 'belief-0.00125.npz')


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    path = Path(path)
    data = path.read_bytes()
    return json.loads(gzip.decompress(data) if path.suffix == '.gz' else data)


def save(path, value):
    with Path(path).open('x') as stream:
        stream.write(json.dumps(value, indent=2, allow_nan=False)+'\n')


def lower_tail(values, mass=.25, axis=-1):
    """Fractional empirical lower-tail mean; empty inputs stay undefined."""
    if not 0 < mass <= 1:
        raise ValueError('Tail mass must lie in (0,1]')
    values = np.asarray(values, dtype=float)
    n = values.shape[axis]
    if not n:
        return None
    ordered = np.sort(values, axis=axis)
    amount = n*mass
    whole, fraction = int(amount), amount-int(amount)
    total = np.take(ordered, np.arange(whole), axis=axis).sum(axis=axis)
    if fraction:
        total = total+fraction*np.take(ordered, whole, axis=axis)
    answer = total/amount
    return float(answer) if answer.ndim == 0 else answer


def paired(left, right, *, replicates=REPLICATES, seed=SEED, tail_mass=.25):
    """Resample matched subjects; CVaR is a difference of method tail means."""
    all_families = sorted(set(left) | set(right))
    families = [f for f in all_families if left.get(f) is not None and right.get(f) is not None]
    excluded = [f for f in all_families if f not in families]
    result = {'family_ids': families, 'independent_family_clusters': len(families),
        'excluded_missing_or_undefined_family_pairs': excluded,
        'replicates': replicates, 'statistical_seed': seed,
        'sign': 'left minus right; utility positive favors left'}
    if not families:
        return result | {'status': 'N/A: no common defined family pairs',
            'mean_difference': None, 'percentile95_family_bootstrap': None,
            'lower_tail_mean_difference': None, 'percentile95_lower_tail_difference': None}
    a = np.asarray([left[f] for f in families], float)
    b = np.asarray([right[f] for f in families], float)
    if not np.all(np.isfinite(a)) or not np.all(np.isfinite(b)):
        raise ValueError('Undefined scores must be None, not NaN/Inf')
    indices = np.random.default_rng(seed).integers(0, len(families), size=(replicates, len(families)))
    differences = a-b
    mean_bootstrap = differences[indices].mean(axis=1)
    tail_bootstrap = lower_tail(a[indices], tail_mass)-lower_tail(b[indices], tail_mass)
    return result | {'status': 'defined', 'family_differences': dict(zip(families, differences.tolist())),
        'mean_difference': float(differences.mean()),
        'percentile95_family_bootstrap': np.quantile(mean_bootstrap, [.025, .975]).tolist(),
        'lower_tail_mass': tail_mass,
        'left_lower_tail_mean': lower_tail(a, tail_mass),
        'right_lower_tail_mean': lower_tail(b, tail_mass),
        'lower_tail_mean_difference': lower_tail(a, tail_mass)-lower_tail(b, tail_mass),
        'percentile95_lower_tail_difference': np.quantile(tail_bootstrap, [.025, .975]).tolist()}


def draw_diagnostic(left, right, draws):
    """Descriptive MC stability on identical family pairs for every draw."""
    common = sorted(f for f in set(left) & set(right)
        if all(left[f].get(str(d)) is not None and right[f].get(str(d)) is not None for d in draws))
    rows = {}
    for draw in draws:
        delta = [left[f][str(draw)]-right[f][str(draw)] for f in common]
        rows[str(draw)] = {'family_clusters': len(common),
            'paired_mean_difference': float(np.mean(delta)) if delta else None}
    means = [v['paired_mean_difference'] for v in rows.values() if v['paired_mean_difference'] is not None]
    return {'scope': 'descriptive within-draw means; draws remain nested in family, not independent sample count',
        'complete_family_pairs': common, 'draws': rows,
        'sign_consistency': {'positive_draws': sum(v > 0 for v in means),
            'zero_draws': sum(v == 0 for v in means), 'negative_draws': sum(v < 0 for v in means),
            'defined_draws': len(means)},
        'draw_mean_min': min(means) if means else None, 'draw_mean_max': max(means) if means else None}


def phases(row):
    result = ['all']
    if row['slot'] == 0:
        result.append('cold')
    if row['t'] <= 180:
        result.append('early_0_180')
    if row['t'] >= 400:
        result.append('temporal_tail_400_600')
    return result


def draws_for_split(expected_draws, split):
    return expected_draws[split] if isinstance(expected_draws, dict) else expected_draws


def declared_clock_inventory(bundle, methods):
    public = bundle.get('public')
    if public is None:
        raise ValueError('Public fixed-clock streams required before statistical readout')
    if set(public['streams']) != set(methods):
        raise ValueError('Public method inventory differs from frozen methods')
    expected = set()
    for slot in range(8):
        clocks = set(range(0, 601, 20))
        if slot >= 6:
            clocks.update(public['public_clocks'][name] for name in ('shared_fork_t', 'turn_visible_t'))
        if any(t < 0 or t > 600 for t in clocks):
            raise ValueError('Public fixed-window clock outside declared0..600s')
        expected.update((slot, t) for t in clocks)
    for method in methods:
        sessions = public['streams'][method]
        if len(sessions) != 8:
            raise ValueError('All eight public sessions must be retained')
        inventory = []
        for slot, session in enumerate(sessions):
            events = session['events']
            times = [e['timestamp_s'] for e in events]
            if times != sorted(times) or len(times) != len(set(times)):
                raise ValueError('Public timestamps must be unique and in causal order')
            inventory.extend((slot, t) for t in times)
        if len(inventory) != len(expected) or set(inventory) != expected:
            raise ValueError('Public event inventory differs from declared fixed observation clocks')
    return expected


def summarize_bundles(bundles, purposes, expected_draws, *, expected_methods=None,
                      expected_caches=None, require_public_clock=False):
    """Independent reconstruction of conditional family/draw means and N/A."""
    stats = defaultdict(lambda: [0., 0, 0, 0, 0])
    seen_rows = set()
    seen_bundles = set()
    reference_contract = {}
    family_splits = {}
    family_inventory = {}
    methods, caches = expected_methods, expected_caches
    costs = defaultdict(lambda: {'requests': 0, 'request_bytes': 0, 'reply_bytes': 0})
    for bundle in bundles:
        truth = bundle['evaluator_only']
        family, split, draw = truth['family_id'], truth['split'], truth['draw']
        if (family, draw) in seen_bundles:
            raise ValueError('Duplicate family/draw bundle')
        seen_bundles.add((family, draw))
        if family in family_splits and family_splits[family] != split:
            raise ValueError('Family crosses source-group splits')
        family_splits[family] = split
        methods = tuple(methods) if methods is not None else tuple(sorted({r['method'] for r in bundle['utility']}))
        caches = tuple(caches) if caches is not None else tuple(sorted({r['cache'] for r in bundle['utility']}))
        inventories = defaultdict(set)
        wire_inventory = defaultdict(set)
        public_inventory = declared_clock_inventory(bundle, methods) if require_public_clock else None
        for row in bundle['utility']:
            if (row['family_id'], row['split'], row['draw']) != (family, split, draw):
                raise ValueError('Utility row provenance does not match family bundle')
            rid = (family, draw, row['method'], row['cache'], row['slot'], row['t'])
            if rid in seen_rows:
                raise ValueError('Duplicate utility event in a cluster')
            seen_rows.add(rid)
            if row['method'] not in methods or row['cache'] not in caches:
                raise ValueError('Unexpected method/cache row outside frozen inventory')
            inventories[(row['method'], row['cache'])].add((row['slot'], row['t']))
            if tuple(row['purposes']) != tuple(purposes):
                raise ValueError('Purpose identities/order differ from frozen protocol')
            for purpose in purposes:
                score = row['purposes'][purpose]
                value = score['recall5']
                defined = score['reference_category_count']
                total_categories = score['all_category_count']
                if (value is None) != (defined == 0):
                    raise ValueError('Empty reference must be N/A; nonempty reference must be defined')
                if value is not None and not 0 <= value <= 1:
                    raise ValueError('Recall outside [0,1]')
                contract = (family, row['slot'], row['t'], purpose)
                previous = reference_contract.setdefault(contract, (defined, total_categories, value is None))
                if previous != (defined, total_categories, value is None):
                    raise ValueError('Reference/unknown coverage changes across method/cache/draw')
                for phase in phases(row):
                    key = (row['method'], split, row['cache'], phase, purpose, family, draw)
                    cell = stats[key]
                    cell[0] += 0. if value is None else value
                    cell[1] += value is not None
                    cell[2] += 1
                    cell[3] += defined
                    cell[4] += total_categories
        for row in bundle.get('wire', []):
            if (row['family_id'], row['split'], row['draw']) != (family, split, draw):
                raise ValueError('Wire row provenance does not match family bundle')
            if row['method'] not in methods or (row['slot'], row['t']) in wire_inventory[row['method']]:
                raise ValueError('Unexpected or duplicate wire event outside frozen inventory')
            wire_inventory[row['method']].add((row['slot'], row['t']))
            cell = costs[(row['method'], split, family, draw)]
            for key in cell:
                cell[key] += row[key]
        actual_controls = {(m, c) for m in methods for c in caches}
        if set(inventories) != actual_controls:
            raise ValueError('Missing method/cache event inventory')
        expected_inventory = public_inventory if public_inventory is not None else next(iter(inventories.values()))
        if any(inventory != expected_inventory for inventory in inventories.values()):
            raise ValueError('Exact event inventories differ across method/cache')
        prior_inventory = family_inventory.setdefault(family, expected_inventory)
        if prior_inventory != expected_inventory:
            raise ValueError('Exact event inventories differ across secret draws')
        if require_public_clock and (set(wire_inventory) != set(methods) or any(
                inventory != expected_inventory for inventory in wire_inventory.values())):
            raise ValueError('Wire event inventory does not match public/utility clocks')
    for family in family_splits:
        actual = sorted(d for f, d in seen_bundles if f == family)
        if actual != list(draws_for_split(expected_draws, family_splits[family])):
            raise ValueError('Every family must retain all declared secret draws')
    result = {}
    prefixes = sorted({k[:4] for k in stats})
    for method, split, cache, phase in prefixes:
        output = {}
        split_draws = draws_for_split(expected_draws, split)
        families = sorted(f for f, s in family_splits.items() if s == split)
        for purpose in purposes:
            family_values, family_draw_values = {}, {}
            coverage = {'defined_windows': 0, 'total_windows': 0,
                'defined_categories': 0, 'total_categories': 0}
            for family in families:
                cells = [stats[(method, split, cache, phase, purpose, family, d)] for d in split_draws]
                if any(c[2] == 0 for c in cells):
                    raise ValueError('Missing method/cache/phase windows in declared family/draw')
                if len({(c[1], c[2], c[3], c[4]) for c in cells}) != 1:
                    raise ValueError('Draws have different public/reference coverage; do not pool silently')
                valid = sum(c[1] for c in cells)
                family_values[family] = sum(c[0] for c in cells)/valid if valid else None
                family_draw_values[family] = {str(d): c[0]/c[1] if c[1] else None
                    for d, c in zip(split_draws, cells)}
                for name, index in (('defined_windows', 1), ('total_windows', 2),
                    ('defined_categories', 3), ('total_categories', 4)):
                    coverage[name] += sum(c[index] for c in cells)
            output[purpose] = {'family_values': family_values,
                'family_draw_values': family_draw_values, 'coverage': coverage}
        macro, macro_draws, complete = {}, {}, []
        for family in families:
            values = [output[p]['family_values'][family] for p in purposes
                if output[p]['family_values'][family] is not None]
            macro[family] = float(np.mean(values)) if values else None
            if len(values) == len(purposes):
                complete.append(family)
            macro_draws[family] = {}
            for draw in split_draws:
                per_purpose = [output[p]['family_draw_values'][family][str(draw)] for p in purposes
                    if output[p]['family_draw_values'][family][str(draw)] is not None]
                macro_draws[family][str(draw)] = float(np.mean(per_purpose)) if per_purpose else None
        output['equal_purpose_macro'] = {'family_values': macro, 'family_draw_values': macro_draws,
            'complete_all_purpose_families': complete,
            'partial_or_undefined_purpose_families': [f for f in families if f not in complete],
            'coverage_rule': 'average defined purposes locally; separately retain complete-purpose paired sensitivity'}
        result[(method, split, cache, phase)] = output
    return result, dict(costs)


def validate_study_sources(output, protocol, generation=None, *, root=ROOT):
    """Read-only common/executor/source/public-input integrity; no scores read."""
    output, root = Path(output), Path(root).resolve()
    if sha(output/'protocol.json') != (output/'protocol.sha256').read_text().strip():
        raise ValueError('Frozen study protocol hash mismatch')
    counts = {'common_sources': 0, 'public_inputs': 0, 'execution_sources': 0,
        'public_execution_cache_files': 0}
    def local(name):
        path = Path(name)
        if path.is_absolute() or not (root/path).resolve().is_relative_to(root):
            raise ValueError('Pinned source/public input must remain root-relative')
        return root/path
    for name, digest in protocol['source_sha256'].items():
        if sha(local(name)) != digest or sha(output/'source_snapshot'/name) != digest:
            raise ValueError('Frozen common source/snapshot changed: '+name)
        counts['common_sources'] += 1
    if sha(local(protocol['dataset_path'])) != protocol['dataset_sha256']:
        raise ValueError('Frozen dataset source changed')
    for name, digest in protocol['public_inputs_sha256'].items():
        if sha(local(name)) != digest:
            raise ValueError('Frozen public input changed: '+name)
        counts['public_inputs'] += 1
    execution = output/'execution_protocol.json'
    if execution.exists():
        if sha(execution) != (output/'execution_protocol.sha256').read_text().strip():
            raise ValueError('Execution protocol hash changed')
        e = read(execution)
        if e['common_protocol_sha256'] != sha(output/'protocol.json'):
            raise ValueError('Execution protocol does not bind the same common protocol')
        for name, digest in e['source_sha256'].items():
            if sha(local(name)) != digest or sha(output/'execution_source_snapshot'/name) != digest:
                raise ValueError('Frozen parallel source/snapshot changed: '+name)
            counts['execution_sources'] += 1
        for name, digest in e.get('predeclared_files_sha256', {}).items():
            if sha(output/name) != digest:
                raise ValueError('Predeclared freeze changed: '+name)
        if set(e['public_cache_files_sha256']) != set(PUBLIC_CACHE_FILES):
            raise ValueError('Public cache manifest differs from fixed safe whitelist')
        cache = Path(e['public_cache_source'])
        for name, digest in e['public_cache_files_sha256'].items():
            path = cache/name
            if path.is_symlink() or not path.is_file() or sha(path) != digest:
                raise ValueError('Pinned public execution cache changed: '+name)
            counts['public_execution_cache_files'] += 1
        counts['execution_protocol_sha256'] = sha(execution)
        if generation is not None:
            if generation['execution_protocol_sha256'] != sha(execution):
                raise ValueError('Generation does not bind the frozen execution protocol')
            planned = {job['name'] for job in e['jobs']}
            if set(generation['family_files_sha256']) != planned:
                raise ValueError('Generation omitted or added declared family/draw jobs')
    if generation is not None and (generation['identical_private_transcript_asserted'] is not True
            or generation['private_keys_exported'] is not False):
        raise ValueError('Generation lacks required backbone/privacy-material assertions')
    return counts


def crosscheck_summary(cells, saved):
    count = 0
    for (method, split, cache, phase), purposes in cells.items():
        for purpose, values in purposes.items():
            expected = saved['summary'][method][split][cache][phase][purpose]
            observed = values['family_values']
            comparable = {f: v for f, v in observed.items() if v is not None} if purpose == 'equal_purpose_macro' else observed
            if set(comparable) != set(expected['family_values']):
                raise ValueError('Saved utility family denominator mismatch')
            for family, value in comparable.items():
                other = expected['family_values'][family]
                if (value is None) != (other is None) or (value is not None and not np.isclose(value, other, atol=1e-12, rtol=0)):
                    raise ValueError('Saved utility family mean not independently reproduced')
                count += 1
            valid = [v for v in observed.values() if v is not None]
            mean = float(np.mean(valid)) if valid else None
            if (mean is None) != (expected['family_mean'] is None) or (mean is not None and not np.isclose(mean, expected['family_mean'], atol=1e-12, rtol=0)):
                raise ValueError('Saved utility macro mean mismatch')
            tail = lower_tail(valid)
            other_tail = expected['family_lower_quartile_cvar']
            if (tail is None) != (other_tail is None) or (tail is not None and not np.isclose(tail, other_tail, atol=1e-12, rtol=0)):
                raise ValueError('Saved family lower-tail mean mismatch')
            if purpose != 'equal_purpose_macro':
                if any(values['coverage'][name] != expected[name] for name in values['coverage']):
                    raise ValueError('Saved reference coverage denominator mismatch')
    return count


def build_readout(cells, draws, methods, baseline, primary_left=None, primary_right=None,
                  tail_mass=.25, alignment_control=None):
    contrasts, raw = {}, {}
    for method, split, cache, phase in sorted(cells):
        if method == 'raw':
            values = cells[(method, split, cache, phase)]
            raw[f'{split}--{cache}--{phase}'] = {p: {
                'family_mean': float(np.mean([v for v in scores['family_values'].values() if v is not None]))
                    if any(v is not None for v in scores['family_values'].values()) else None,
                'minimum_defined_family_mean': min((v for v in scores['family_values'].values() if v is not None), default=None),
                **({'coverage': scores['coverage']} if 'coverage' in scores else {
                    'partial_or_undefined_purpose_families': scores['partial_or_undefined_purpose_families']})}
                for p, scores in values.items()}
            continue
        if method not in methods or method == baseline:
            continue
        comparisons = [baseline]
        if primary_right and method == primary_left and primary_right not in comparisons:
            comparisons.append(primary_right)
        if alignment_control and method == primary_left and method != alignment_control and alignment_control not in comparisons:
            comparisons.append(alignment_control)
        for right in comparisons:
            if (right, split, cache, phase) not in cells:
                raise ValueError('Declared baseline/primary control missing')
            for purpose, left_cell in cells[(method, split, cache, phase)].items():
                right_cell = cells[(right, split, cache, phase)][purpose]
                key = f'{method}--minus--{right}--{split}--{cache}--{phase}--{purpose}'
                record = paired(left_cell['family_values'], right_cell['family_values'], tail_mass=tail_mass)
                record['within_draw'] = draw_diagnostic(left_cell['family_draw_values'], right_cell['family_draw_values'], draws_for_split(draws, split))
                record['primary_contrast'] = (method, right, split, cache, phase, purpose) == (
                    primary_left, primary_right, 'test', 'current', 'all', 'equal_purpose_macro')
                if purpose == 'equal_purpose_macro':
                    complete = set(left_cell['complete_all_purpose_families']) & set(right_cell['complete_all_purpose_families'])
                    record['complete_all_purpose_sensitivity'] = paired(
                        {f: v for f, v in left_cell['family_values'].items() if f in complete},
                        {f: v for f, v in right_cell['family_values'].items() if f in complete}, tail_mass=tail_mass)
                    record['partial_or_undefined_purpose_families'] = sorted(set(
                        left_cell['partial_or_undefined_purpose_families']) | set(right_cell['partial_or_undefined_purpose_families']))
                else:
                    record['left_reference_coverage'] = left_cell['coverage']
                    record['right_reference_coverage'] = right_cell['coverage']
                contrasts[key] = record
    return contrasts, raw


def validate_freeze(freeze, study_protocol, protocol_sha256, left, right, alignment_control, *, root=ROOT):
    """Bind the adopted defense to the durable development selection chain."""
    if freeze['selected'] != left or freeze['baseline'] != right or freeze['alignment_control'] != alignment_control:
        raise ValueError('Primary/control identities differ from frozen defense selection')
    if left == right:
        raise ValueError('Primary selected-versus-baseline contrast must be distinct')
    if freeze['fresh_protocol_sha256'] != protocol_sha256 or freeze['configuration'] != study_protocol['configuration']:
        raise ValueError('Freeze does not bind the actual fresh protocol/configuration')
    relative = Path(freeze['selection_path'])
    if relative.is_absolute() or not (root/relative).resolve().is_relative_to(root.resolve()):
        raise ValueError('Development selection path must remain root-relative')
    selection_path = root/relative
    if sha(selection_path) != freeze['selection_sha256']:
        raise ValueError('Frozen development selection hash mismatch')
    selection = read(selection_path)
    if selection['selected'] != left or selection['fresh_test_scores_viewed'] is not False:
        raise ValueError('Freeze differs from pre-test development selection')
    if freeze.get('fresh_test_evaluated') is not False:
        raise ValueError('Freeze must retain its original before-test declaration')
    criterion = freeze['criterion']
    if criterion['family_bootstrap_replicates'] != REPLICATES or criterion['public_analysis_seed'] != SEED:
        raise ValueError('Statistical seed/replicates differ from frozen primary criterion')
    if selection['rule']['fresh_criterion'] != criterion:
        raise ValueError('Primary criterion differs from frozen development rule')
    return criterion


def primary_decision(contrasts, left, right, criterion):
    key = f'{left}--minus--{right}--test--current--all--equal_purpose_macro'
    primary = contrasts.get(key)
    if not primary or primary['mean_difference'] is None:
        return {'status': 'N/A: primary has no defined paired family score', 'passes': False}
    draw_values = [r['paired_mean_difference'] for r in primary['within_draw']['draws'].values()]
    gates = {'mean_gain': primary['mean_difference'] >= criterion['minimum_absolute_mean_gain'],
        'paired95_lower_bound': primary['percentile95_family_bootstrap'][0] > criterion['paired95_lower_bound_gt'],
        'every_private_draw_gain': bool(draw_values) and all(v is not None and v > criterion['every_private_draw_gain_gt'] for v in draw_values)}
    return {'status': 'defined', 'passes': all(gates.values()), 'gates': gates,
        'primary_key': key, 'independent_family_clusters': primary['independent_family_clusters'],
        'scope': 'predeclared conditional static-utility criterion only; not a privacy equivalence or real-data confirmation claim'}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=DEFAULT)
    parser.add_argument('--baseline', default='legacy_l10')
    parser.add_argument('--primary-left')
    parser.add_argument('--primary-right')
    parser.add_argument('--alignment-control', default='aligned_nearest')
    parser.add_argument('--freeze-record', type=Path)
    parser.add_argument('--scope', choices=('development', 'independent-synthetic-generalization'), default='development')
    parser.add_argument('--expected-draws', type=int, default=3)
    args = parser.parse_args()
    output = args.output.resolve()
    if (output/'paired_family_readout.json').exists() or (output/'paired_family_protocol.json').exists():
        raise FileExistsError('Preserve sealed readout; use a new output version for recalculation')
    if args.scope != 'development' and (not args.freeze_record or not args.freeze_record.is_file()
        or not args.primary_left or not args.primary_right):
        raise ValueError('Fresh test readout requires durable method/source freeze and explicit single primary contrast')
    source_protocol = read(output/'protocol.json')
    if sha(output/'protocol.json') != (output/'protocol.sha256').read_text().strip():
        raise ValueError('Frozen study protocol hash mismatch')
    if 'test' in source_protocol['splits'] and 'qplanner_fresh_native_' in source_protocol['dataset_path'] and args.scope == 'development':
        raise ValueError('Fresh test artifact requires explicit generalization scope and bound freeze record')
    methods = tuple(source_protocol['configuration']['methods'])
    if args.baseline not in methods or (args.primary_left and args.primary_left not in methods) or (
        args.primary_right and args.primary_right not in methods):
        raise ValueError('Contrasts must use declared methods')
    draw_count = source_protocol['draw_count']
    draw_counts = source_protocol.get('draws_by_split', {s: draw_count for s in source_protocol['splits']})
    evaluated_split = 'test' if 'test' in draw_counts else 'selection'
    if args.expected_draws is not None and draw_counts[evaluated_split] != args.expected_draws:
        raise ValueError('Declared draw count does not match predeclared readout count')
    if args.scope != 'development' and draw_counts.get('test', 0) < 3:
        raise ValueError('Fresh multi-draw stability readout requires at least three declared draws')
    frozen_criterion = None
    if args.scope != 'development':
        freeze = read(args.freeze_record)
        if args.baseline != freeze['baseline'] or args.alignment_control not in methods:
            raise ValueError('Comparison baseline/alignment control must match the frozen declared methods')
        frozen_criterion = validate_freeze(freeze, source_protocol, sha(output/'protocol.json'),
            args.primary_left, args.primary_right, args.alignment_control)
    generation = read(output/'generation.json')
    source_validation = validate_study_sources(output, source_protocol, generation)
    source_hashes = {str(Path(__file__).relative_to(ROOT)): sha(Path(__file__)),
        'tests/test_qplanner_paired_readout.py': sha(ROOT/'tests/test_qplanner_paired_readout.py')}
    declaration = {'schema': 'qplanner-paired-family-protocol-v1', 'scope': args.scope,
        'source_protocol_sha256': sha(output/'protocol.json'), 'generation_sha256': sha(output/'generation.json'),
        'readout_source_sha256': source_hashes, 'family_files_sha256': generation['family_files_sha256'],
        'study_source_validation': source_validation,
        'matched_event_inventory': 'exact (slot,t) equality across every frozen method,cache,draw; all8slots match declared public0..600/20s plus queryfork clocks; wire inventory also exact',
        'verification_scope': 'independent source integrity, event coverage and statistical arithmetic; separate full generation/service/ledger verifier required before scientific interpretation',
        'nominal_draw_count': draw_count, 'draws_by_split': draw_counts,
        'purposes': source_protocol['configuration']['utility_purposes'],
        'methods': methods, 'baseline': args.baseline,
        'alignment_control': args.alignment_control,
        'primary': {'left': args.primary_left, 'right': args.primary_right, 'split': 'test',
            'cache': 'current', 'phase': 'all', 'metric': 'equal_purpose_macro'},
        'freeze_record_path': str(args.freeze_record) if args.freeze_record else None,
        'freeze_record_sha256': sha(args.freeze_record) if args.freeze_record else None,
        'replicates': REPLICATES, 'statistical_seed': SEED, 'family_tail_mass': .25,
        'frozen_primary_criterion': frozen_criterion,
        'resampling': 'whole matched family clusters with all eight sessions and every declared secret draw; no tick or draw independence',
        'missing_reference': 'N/A preserved, common defined family pairs; complete-four-purpose macro sensitivity',
        'purpose_dependence': 'nearest_distance and fastest_travel are strongly correlated on this static native graph; four-purpose mean is a predeclared workload aggregate, not four independent replications',
        'multiple_comparisons': 'all development candidates/purposes/cache/phases retained; unadjusted exploratory intervals; one explicitly recorded primary fresh contrast only',
        'scope_limits': 'conditional on same native synthetic map/generator and finite observed draws; not realGPS/crosscity confirmation; no privacy theorem or universal winner'}
    save(output/'paired_family_protocol.json', declaration)
    try:
        def bundles():
            for name, expected in generation['family_files_sha256'].items():
                path = output/'families'/name
                if sha(path) != expected:
                    raise ValueError('Frozen family bundle changed')
                yield read(path)
        draws = {s: list(range(1, n+1)) for s, n in draw_counts.items()}
        cells, costs = summarize_bundles(bundles(), declaration['purposes'], draws,
            expected_methods=('raw', *methods), expected_caches=CACHES, require_public_clock=True)
        saved = read(output/'utility_readout.json')
        if saved['protocol_sha256'] != declaration['source_protocol_sha256'] or saved['generation_sha256'] != declaration['generation_sha256']:
            raise ValueError('Utility readout does not pin the same study protocol/generation')
        checked = crosscheck_summary(cells, saved)
        contrasts, raw = build_readout(cells, draws, methods, args.baseline,
            args.primary_left, args.primary_right, alignment_control=args.alignment_control)
        cost_rows = [{'method': m, 'split': s, 'family_id': f, 'draw': d, **v}
            for (m, s, f, d), v in sorted(costs.items())]
        record = {'schema': 'qplanner-paired-family-readout-v1',
            'protocol_sha256': sha(output/'paired_family_protocol.json'),
            'utility_readout_sha256': sha(output/'utility_readout.json'),
            'independently_checked_family_scores': checked, 'contrasts': contrasts,
            'exact_matched_event_inventory_asserted': True,
            'study_source_validation': source_validation,
            'verification_scope': declaration['verification_scope'],
            'raw_utility_control': raw, 'family_draw_costs': cost_rows,
            'primary_criterion_result': primary_decision(contrasts, args.primary_left, args.primary_right,
                frozen_criterion) if frozen_criterion else None,
            'raw_cost_scope': 'raw has one query per event; protected planners five; raw utility is a positive/context control, not cost-matched superiority evidence',
            'defense_selected_by_this_readout': False,
            'purpose_dependence': declaration['purpose_dependence'],
            'test_independent_unit': 'family', 'real_data_confirmation': False}
        save(output/'paired_family_readout.json', record)
        print('Saved family-cluster readout; contrasts', len(contrasts), 'independent family scores checked', checked)
    except Exception as error:
        save(output/'paired_family_failure.json', {'error': type(error).__name__, 'message': str(error),
            'protocol_sha256': sha(output/'paired_family_protocol.json')})
        raise


if __name__ == '__main__':
    main()
