"""Statistical and N/A contracts on small synthetic fixtures, not study tests."""
import copy
import hashlib
import json
import numpy as np
import pytest

from experiments.qplanner_paired_readout_20261006 import (
    lower_tail, paired, draw_diagnostic, summarize_bundles, crosscheck_summary, build_readout,
    validate_freeze, primary_decision, REPLICATES, SEED,
    declared_clock_inventory, validate_study_sources, sha, PUBLIC_CACHE_FILES,
)


def test_fractional_lower_tail_and_cvar_difference_are_method_specific():
    assert lower_tail([0., 1., 2.], .5) == pytest.approx(1/3)
    left, right = {'a': 0., 'b': .8}, {'a': .6, 'b': 0.}
    result = paired(left, right, replicates=100, tail_mass=.5)
    assert result['lower_tail_mean_difference'] == 0.
    assert lower_tail([left[f]-right[f] for f in left], .5) == -.6
    assert result['independent_family_clusters'] == 2


def test_undefined_pairs_and_missing_subjects_are_retained_as_coverage():
    result = paired({'a': .9, 'b': None, 'c': .2}, {'a': .7, 'b': .5, 'd': .8}, replicates=100)
    assert result['family_ids'] == ['a']
    assert result['excluded_missing_or_undefined_family_pairs'] == ['b', 'c', 'd']
    assert result['percentile95_family_bootstrap'] == pytest.approx([.2, .2])
    assert paired({'a': None}, {'a': None}, replicates=100)['mean_difference'] is None


def test_draw_diagnostic_preserves_subject_alignment_without_counting_draws_as_subjects():
    left = {'a': {'1': .9, '2': .8, '3': .6}, 'b': {'1': .9, '2': .8, '3': .6}}
    right = {'a': {'1': .8, '2': .8, '3': .8}, 'b': {'1': .8, '2': .8, '3': .8}}
    result = draw_diagnostic(left, right, [1, 2, 3])
    assert result['sign_consistency'] == {'positive_draws': 1, 'zero_draws': 1, 'negative_draws': 1, 'defined_draws': 3}
    assert all(r['family_clusters'] == 2 for r in result['draws'].values())


def fixture(draw=1):
    purposes = {'near': {'recall5': .8, 'reference_category_count': 1, 'all_category_count': 2},
        'empty': {'recall5': None, 'reference_category_count': 0, 'all_category_count': 2}}
    rows = [{'family_id': 'f', 'split': 'test', 'draw': draw, 'method': m, 'cache': 'current',
        'slot': 0, 't': t, 'purposes': copy.deepcopy(purposes)} for m in ('left', 'right') for t in (0, 400)]
    return {'evaluator_only': {'family_id': 'f', 'split': 'test', 'draw': draw}, 'utility': rows, 'wire': []}


def test_reference_na_keeps_conditional_macro_and_complete_purpose_coverage():
    cells, _ = summarize_bundles([fixture(1), fixture(2), fixture(3)], ['near', 'empty'], [1, 2, 3])
    group = cells[('left', 'test', 'current', 'all')]
    assert group['near']['coverage'] == {'defined_windows': 6, 'total_windows': 6,
        'defined_categories': 6, 'total_categories': 12}
    assert group['empty']['family_values'] == {'f': None}
    assert group['equal_purpose_macro']['family_values']['f'] == pytest.approx(.8)
    assert group['equal_purpose_macro']['complete_all_purpose_families'] == []
    assert group['equal_purpose_macro']['partial_or_undefined_purpose_families'] == ['f']


def test_missing_draw_and_duplicate_tick_cannot_be_counted_as_extra_samples():
    with pytest.raises(ValueError, match='all declared secret draws'):
        summarize_bundles([fixture(1), fixture(2)], ['near', 'empty'], [1, 2, 3])
    bundle = fixture()
    bundle['utility'].append(copy.deepcopy(bundle['utility'][0]))
    with pytest.raises(ValueError, match='Duplicate utility event'):
        summarize_bundles([bundle], ['near', 'empty'], [1])


def test_reference_contract_cannot_change_with_protected_method_or_draw():
    bundle = fixture()
    bundle['utility'][-1]['purposes']['empty'] = {'recall5': 0., 'reference_category_count': 1, 'all_category_count': 2}
    with pytest.raises(ValueError, match='coverage changes'):
        summarize_bundles([bundle], ['near', 'empty'], [1])


def test_disjoint_splits_can_have_different_predeclared_draw_counts():
    training = fixture(1)
    training['evaluator_only'].update(family_id='train-family', split='train')
    for row in training['utility']:
        row.update(family_id='train-family', split='train')
    cells, _ = summarize_bundles([training, fixture(1), fixture(2), fixture(3)],
        ['near', 'empty'], {'train': [1], 'test': [1, 2, 3]})
    assert len(cells[('left', 'train', 'current', 'all')]['near']['family_draw_values']['train-family']) == 1
    assert len(cells[('left', 'test', 'current', 'all')]['near']['family_draw_values']['f']) == 3


def test_saved_macro_may_omit_all_undefined_family_but_purpose_denominators_must_not():
    unknown = fixture()
    unknown['evaluator_only']['family_id'] = 'g'
    for row in unknown['utility']:
        row['family_id'] = 'g'
        row['purposes']['near'].update(recall5=None, reference_category_count=0)
    cells, _ = summarize_bundles([fixture(), unknown], ['near', 'empty'], [1])
    cell = cells[('left', 'test', 'current', 'all')]
    saved = {'summary': {'left': {'test': {'current': {'all': {
        'near': {'family_values': {'f': .8, 'g': None}, 'family_mean': .8,
            'family_lower_quartile_cvar': .8, 'defined_windows': 2, 'total_windows': 4,
            'defined_categories': 2, 'total_categories': 8},
        'empty': {'family_values': {'f': None, 'g': None}, 'family_mean': None,
            'family_lower_quartile_cvar': None, 'defined_windows': 0, 'total_windows': 4,
            'defined_categories': 0, 'total_categories': 8},
        'equal_purpose_macro': {'family_values': {'f': .8}, 'family_mean': .8,
            'family_lower_quartile_cvar': .8}}}}}}}
    assert crosscheck_summary({('left', 'test', 'current', 'all'): cell}, saved) == 5
    del saved['summary']['left']['test']['current']['all']['near']['family_values']['g']
    with pytest.raises(ValueError, match='family denominator'):
        crosscheck_summary({('left', 'test', 'current', 'all'): cell}, saved)


def test_only_one_fresh_primary_case_is_marked_all_secondary_purposes_retained():
    cells, _ = summarize_bundles([fixture(1), fixture(2), fixture(3)], ['near', 'empty'], [1, 2, 3])
    records, _ = build_readout(cells, {'test': [1, 2, 3]}, ['left', 'right'], 'right', 'left', 'right')
    assert sum(r['primary_contrast'] for r in records.values()) == 1
    primary = records['left--minus--right--test--current--all--equal_purpose_macro']
    assert primary['primary_contrast']
    assert primary['complete_all_purpose_sensitivity']['mean_difference'] is None
    assert 'left--minus--right--test--current--all--empty' in records


def test_alignment_secondary_is_explicit_and_not_a_second_primary():
    cells, _ = summarize_bundles([fixture(1), fixture(2), fixture(3)], ['near', 'empty'], [1, 2, 3])
    for key, value in list(cells.items()):
        if key[0] == 'right':
            cells[('aligned', *key[1:])] = copy.deepcopy(value)
    records, _ = build_readout(cells, {'test': [1, 2, 3]}, ['left', 'right', 'aligned'],
        'right', 'left', 'right', alignment_control='aligned')
    assert 'left--minus--aligned--test--current--all--equal_purpose_macro' in records
    assert sum(r['primary_contrast'] for r in records.values()) == 1


def test_fresh_freeze_binds_method_config_protocol_and_development_selection(tmp_path):
    criterion = {'family_bootstrap_replicates': REPLICATES, 'public_analysis_seed': SEED}
    selection = {'selected': 'candidate', 'fresh_test_scores_viewed': False, 'rule': {'fresh_criterion': criterion}}
    path = tmp_path/'selection.json'
    path.write_text(json.dumps(selection))
    freeze = {'selected': 'candidate', 'baseline': 'legacy', 'alignment_control': 'aligned',
        'fresh_protocol_sha256': 'publicprotocol', 'configuration': {'K': 5},
        'selection_path': 'selection.json', 'selection_sha256': hashlib.sha256(path.read_bytes()).hexdigest(),
        'criterion': criterion, 'fresh_test_evaluated': False}
    assert validate_freeze(freeze, {'configuration': {'K': 5}}, 'publicprotocol',
        'candidate', 'legacy', 'aligned', root=tmp_path) == criterion
    with pytest.raises(ValueError, match='identities'):
        validate_freeze(freeze, {'configuration': {'K': 5}}, 'publicprotocol',
            'other', 'legacy', 'aligned', root=tmp_path)
    with pytest.raises(ValueError, match='protocol/configuration'):
        validate_freeze(freeze, {'configuration': {'K': 6}}, 'publicprotocol',
            'candidate', 'legacy', 'aligned', root=tmp_path)
    path.write_text(json.dumps({**selection, 'selected': 'changed'}))
    with pytest.raises(ValueError, match='selection hash'):
        validate_freeze(freeze, {'configuration': {'K': 5}}, 'publicprotocol',
            'candidate', 'legacy', 'aligned', root=tmp_path)


def test_primary_success_needs_all_three_prespecified_conditions():
    criterion = {'minimum_absolute_mean_gain': .02, 'paired95_lower_bound_gt': 0., 'every_private_draw_gain_gt': 0.}
    key = 'candidate--minus--legacy--test--current--all--equal_purpose_macro'
    record = {'mean_difference': .03, 'percentile95_family_bootstrap': [.01, .05],
        'independent_family_clusters': 24,
        'within_draw': {'draws': {'1': {'paired_mean_difference': .02},
            '2': {'paired_mean_difference': .04}, '3': {'paired_mean_difference': .03}}}}
    assert primary_decision({key: record}, 'candidate', 'legacy', criterion)['passes']
    record['within_draw']['draws']['3']['paired_mean_difference'] = -.001
    result = primary_decision({key: record}, 'candidate', 'legacy', criterion)
    assert not result['passes'] and not result['gates']['every_private_draw_gain']


def test_equal_sized_shifted_method_windows_fail_exact_inventory_guard():
    bundle = fixture()
    bundle['utility'][-1]['t'] = 420
    with pytest.raises(ValueError, match='Exact event inventories differ across method/cache'):
        summarize_bundles([bundle], ['near', 'empty'], [1])


def test_equal_sized_shifted_draw_windows_fail_even_with_identical_reference_counts():
    second = fixture(2)
    for row in second['utility']:
        if row['t'] == 400:
            row['t'] = 420
    with pytest.raises(ValueError, match='Exact event inventories differ across secret draws'):
        summarize_bundles([fixture(1), second], ['near', 'empty'], [1, 2])


def test_shifted_static_cache_inventory_cannot_be_paired_with_current_reply_inventory():
    bundle = fixture()
    stale = copy.deepcopy(bundle['utility'])
    for row in stale:
        row['cache'] = 'static_epoch_cache'
        if row['t'] == 400:
            row['t'] = 420
    bundle['utility'] += stale
    with pytest.raises(ValueError, match='Exact event inventories differ across method/cache'):
        summarize_bundles([bundle], ['near', 'empty'], [1])


def test_shared_omission_from_all_public_methods_still_fails_declared_fixed_clock():
    sessions = []
    for slot in range(8):
        times = sorted(set(range(0, 601, 20)) | ({123, 133} if slot >= 6 else set()))
        sessions.append({'events': [{'timestamp_s': t} for t in times]})
    bundle = {'public': {'public_clocks': {'shared_fork_t': 123, 'turn_visible_t': 133},
        'streams': {'left': copy.deepcopy(sessions), 'right': copy.deepcopy(sessions)}}}
    assert len(declared_clock_inventory(bundle, ('left', 'right'))) == 252
    for method in ('left', 'right'):
        bundle['public']['streams'][method][0]['events'] = [e for e in
            bundle['public']['streams'][method][0]['events'] if e['timestamp_s'] != 20]
    with pytest.raises(ValueError, match='declared fixed observation clocks'):
        declared_clock_inventory(bundle, ('left', 'right'))


def source_fixture(tmp_path):
    root = tmp_path/'root'
    root.mkdir()
    out = root/'output'
    out.mkdir()
    source = root/'engine.py'
    source.write_text('PUBLIC_K = 5\n')
    dataset = root/'dataset.json'
    dataset.write_text('{"public metadata": 1}')
    public = root/'public.json'
    public.write_text('{"public catalogue": []}')
    protocol = {'source_sha256': {'engine.py': sha(source)},
        'dataset_path': 'dataset.json', 'dataset_sha256': sha(dataset),
        'public_inputs_sha256': {'public.json': sha(public)}}
    (out/'protocol.json').write_text(json.dumps(protocol))
    (out/'protocol.sha256').write_text(sha(out/'protocol.json')+'\n')
    snapshot = out/'source_snapshot'
    snapshot.mkdir()
    (snapshot/'engine.py').write_bytes(source.read_bytes())
    return root, out, protocol


def test_standalone_common_source_and_public_input_hash_guards_precede_scores(tmp_path):
    root, out, protocol = source_fixture(tmp_path)
    assert validate_study_sources(out, protocol, root=root)['common_sources'] == 1
    (root/'engine.py').write_text('PUBLIC_K = 9\n')
    with pytest.raises(ValueError, match='common source/snapshot changed'):
        validate_study_sources(out, protocol, root=root)
    (root/'engine.py').write_text('PUBLIC_K = 5\n')
    (root/'public.json').write_text('{"catalogue changed": true}')
    with pytest.raises(ValueError, match='public input changed'):
        validate_study_sources(out, protocol, root=root)


def test_parallel_entrypoint_cache_and_job_completion_are_bound_independently(tmp_path):
    root, out, protocol = source_fixture(tmp_path)
    (root/'executor.py').write_text('EXECUTOR_VERSION = 1\n')
    snapshot = out/'execution_source_snapshot'
    snapshot.mkdir()
    (snapshot/'executor.py').write_bytes((root/'executor.py').read_bytes())
    cache = tmp_path/'public-cache'
    cache.mkdir()
    for name in PUBLIC_CACHE_FILES:
        (cache/name).write_bytes(name.encode())
    execution = {'common_protocol_sha256': sha(out/'protocol.json'),
        'source_sha256': {'executor.py': sha(root/'executor.py')},
        'public_cache_source': str(cache),
        'public_cache_files_sha256': {name: sha(cache/name) for name in PUBLIC_CACHE_FILES},
        'jobs': [{'name': 'family--draw1.json.gz'}]}
    (out/'execution_protocol.json').write_text(json.dumps(execution))
    (out/'execution_protocol.sha256').write_text(sha(out/'execution_protocol.json')+'\n')
    generation = {'execution_protocol_sha256': sha(out/'execution_protocol.json'),
        'family_files_sha256': {'family--draw1.json.gz': 'recorded-hash'},
        'identical_private_transcript_asserted': True, 'private_keys_exported': False}
    assert validate_study_sources(out, protocol, generation, root=root)['execution_sources'] == 1
    omitted = {**generation, 'family_files_sha256': {}}
    with pytest.raises(ValueError, match='omitted or added declared'):
        validate_study_sources(out, protocol, omitted, root=root)
    (root/'executor.py').write_text('EXECUTOR_VERSION = 2\n')
    with pytest.raises(ValueError, match='parallel source/snapshot changed'):
        validate_study_sources(out, protocol, generation, root=root)
    (root/'executor.py').write_text('EXECUTOR_VERSION = 1\n')
    (cache/'reply20.npz').write_bytes(b'public cache changed')
    with pytest.raises(ValueError, match='public execution cache changed'):
        validate_study_sources(out, protocol, generation, root=root)
