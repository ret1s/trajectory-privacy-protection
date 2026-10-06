"""Adversarial source-only fixtures for the before-test freeze contract.

Fresh dataset bytes, family bundles and fresh metrics deliberately do not exist.
Historical metric files are non-JSON stubs: only their hashes may be inspected.
"""
from copy import deepcopy
import hashlib
import json
from pathlib import Path

import pytest

from experiments.qplanner_select_and_freeze_20261006_v2 import RULE
from experiments.qplanner_study_20261006_v2 import configuration, METHOD_CONFIGS
from experiments.verify_qplanner_generalization_20261006_v2 import (
    CRITERION, DRAW_SCHEDULE, EXECUTOR, PRIMARY, SELECTOR, fresh_contract,
)


CORE = 'experiments/qplanner_study_20261006_v2.py'
PROFILE = 'benchmark/public_service_profiles_v2.py'
PUBLIC = 'artifacts/public/toy.net.xml'
ERRORS = (AssertionError, FileNotFoundError, KeyError, ValueError)


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2)+'\n')


def load(path):
    return json.loads(Path(path).read_text())


def seal_protocol(out):
    (out/'protocol.sha256').write_text(digest(out/'protocol.json')+'\n')
    freeze = load(out/'freeze.json')
    freeze['fresh_protocol_sha256'] = digest(out/'protocol.json')
    write(out/'freeze.json', freeze)
    seal_execution(out)


def seal_execution(out):
    execution = load(out/'execution_protocol.json')
    execution['common_protocol_sha256'] = digest(out/'protocol.json')
    execution['predeclared_files_sha256']['freeze.json'] = digest(out/'freeze.json')
    write(out/'execution_protocol.json', execution)
    (out/'execution_protocol.sha256').write_text(digest(out/'execution_protocol.json')+'\n')


def seal_selection(root, out):
    freeze = load(out/'freeze.json')
    freeze['selection_sha256'] = digest(root/freeze['selection_path'])
    write(out/'freeze.json', freeze)
    seal_execution(out)


def fixture(root, selected='normalized_tight'):
    out = root/'artifacts/benchmarks/fresh'
    development = root/'artifacts/benchmarks/development'
    sources = {CORE: b'public common source fixture\n', PROFILE: b'public profile source fixture\n'}
    for name, content in {**sources, SELECTOR: b'public selector fixture\n',
                          EXECUTOR: b'public executor fixture\n',
                          PUBLIC: b'public graph fixture\n'}.items():
        path = root/name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(content)
    source_pins = {name: digest(root/name) for name in sources}
    public_pins = {PUBLIC: digest(root/PUBLIC)}
    for name, content in sources.items():
        for snapshot in ('source_snapshot', 'execution_source_snapshot'):
            path = out/snapshot/name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(content)
    executor_snapshot = out/'execution_source_snapshot'/EXECUTOR
    executor_snapshot.parent.mkdir(parents=True, exist_ok=True)
    executor_snapshot.write_bytes((root/EXECUTOR).read_bytes())
    methods = ['legacy_l10', 'aligned_nearest',
               selected if selected != 'aligned_nearest' else 'normalized_mean']
    protocol = {
        'configuration': configuration(methods), 'dataset_path': 'artifacts/datasets/fresh-absent.json.gz',
        'dataset_sha256': 'a'*64, 'source_sha256': source_pins,
        'public_inputs_sha256': public_pins, 'splits': ['train', 'selection', 'test'],
        'draws_by_split': deepcopy(DRAW_SCHEDULE)}
    prior = deepcopy(protocol)
    prior.update(configuration=configuration(list(METHOD_CONFIGS)),
                 dataset_path='artifacts/datasets/development-absent.json.gz',
                 dataset_sha256='b'*64, splits=['train', 'selection'])
    write(development/'protocol.json', prior)
    for name in ('utility_readout.json', 'attack_selection.json', 'attack_readout.json'):
        (development/name).write_bytes(b'This is deliberately not JSON metric evidence.\n')
    selector_pin = digest(root/SELECTOR)
    selection = {
        'schema': 'qplanner-development-selection-v1', 'selected': selected,
        'source_sha256': selector_pin, 'fresh_test_scores_viewed': False,
        'rule': deepcopy(RULE), 'records': {selected: {'eligible': True, 'reasons': []}},
        'development_inputs_sha256': {name: digest(development/name) for name in (
            'protocol.json', 'utility_readout.json', 'attack_selection.json', 'attack_readout.json')}}
    write(development/'defense_selection.json', selection)
    write(out/'protocol.json', protocol)
    (out/'protocol.sha256').write_text(digest(out/'protocol.json')+'\n')
    freeze = {
        'schema': 'qplanner-synthetic-generalization-freeze-v1', 'selected': selected,
        'baseline': 'legacy_l10', 'alignment_control': 'aligned_nearest',
        'primary_rule': PRIMARY, 'criterion': deepcopy(CRITERION),
        'selection_path': str((development/'defense_selection.json').relative_to(root)),
        'selection_sha256': digest(development/'defense_selection.json'),
        'fresh_protocol_sha256': digest(out/'protocol.json'),
        'configuration': deepcopy(protocol['configuration']),
        'freeze_source_sha256': selector_pin, 'fresh_test_evaluated': False}
    write(out/'freeze.json', freeze)
    write(out/'execution_protocol.json', {
        'common_protocol_sha256': digest(out/'protocol.json'), 'paired_development': None,
        'predeclared_files_sha256': {'freeze.json': digest(out/'freeze.json')},
        'source_sha256': {**source_pins, EXECUTOR: digest(root/EXECUTOR)}})
    (out/'execution_protocol.sha256').write_text(digest(out/'execution_protocol.json')+'\n')
    return out, development


@pytest.mark.parametrize('selected', ['aligned_nearest', 'normalized_mean', 'normalized_tight', 'normalized_tail'])
def test_valid_contract_checks_no_fresh_dataset_or_metrics_and_only_hashes_old_metrics(tmp_path, selected):
    out, _ = fixture(tmp_path, selected)
    assert not (tmp_path/load(out/'protocol.json')['dataset_path']).exists()
    assert not (out/'utility_readout.json').exists()
    assert not (out/'attack_readout.json').exists()
    receipt = fresh_contract(out, root=tmp_path)
    assert receipt['selected'] == selected
    assert receipt['fresh_dataset_opened_for_contract'] is False
    assert receipt['fresh_metrics_opened_for_contract'] is False
    assert receipt['historical_metric_inputs_hash_checked_only'] is True
    assert receipt['old_development_realization_reused'] is False


@pytest.mark.parametrize('fault', ['missing_freeze', 'protocol_hash', 'freeze_protocol', 'selection_hash',
                                  'execution_hash', 'execution_freeze_hash'])
def test_missing_or_hash_inconsistent_adoption_evidence_rejects(tmp_path, fault):
    out, _ = fixture(tmp_path)
    if fault == 'missing_freeze':
        (out/'freeze.json').unlink()
    elif fault == 'protocol_hash':
        (out/'protocol.sha256').write_text('0'*64+'\n')
    elif fault in ('freeze_protocol', 'selection_hash'):
        freeze = load(out/'freeze.json')
        freeze['fresh_protocol_sha256' if fault == 'freeze_protocol' else 'selection_sha256'] = '0'*64
        write(out/'freeze.json', freeze)
        seal_execution(out)
    elif fault == 'execution_hash':
        (out/'execution_protocol.sha256').write_text('0'*64+'\n')
    else:
        execution = load(out/'execution_protocol.json')
        execution['predeclared_files_sha256']['freeze.json'] = '0'*64
        write(out/'execution_protocol.json', execution)
        (out/'execution_protocol.sha256').write_text(digest(out/'execution_protocol.json')+'\n')
    with pytest.raises(ERRORS):
        fresh_contract(out, root=tmp_path)


@pytest.mark.parametrize('fault', ['selected_none', 'ineligible', 'rejection_reason', 'changed_rule',
                                  'scores_viewed', 'test_already_evaluated'])
def test_rehashed_invalid_selection_or_relaxed_gate_does_not_pass(tmp_path, fault):
    out, development = fixture(tmp_path)
    selection = load(development/'defense_selection.json')
    freeze = load(out/'freeze.json')
    selected = freeze['selected']
    if fault == 'selected_none':
        selection['selected'] = None
        freeze['selected'] = None
    elif fault == 'ineligible':
        selection['records'][selected]['eligible'] = False
    elif fault == 'rejection_reason':
        selection['records'][selected]['reasons'] = ['fixed utility gate failed']
    elif fault == 'changed_rule':
        selection['rule']['utility_min_gain'] = .001
    elif fault == 'scores_viewed':
        selection['fresh_test_scores_viewed'] = True
    else:
        freeze['fresh_test_evaluated'] = True
    write(development/'defense_selection.json', selection)
    write(out/'freeze.json', freeze)
    seal_selection(tmp_path, out)
    with pytest.raises(ERRORS):
        fresh_contract(out, root=tmp_path)


@pytest.mark.parametrize('fault', ['method_config', 'budget_config', 'clock_config', 'public_prototypes',
                                  'source_bytes', 'source_snapshot', 'public_bytes', 'public_pins',
                                  'selector_source', 'historical_metric_hash', 'same_dataset_path',
                                  'same_dataset_hash', 'old_key_pairing', 'extra_arm'])
def test_binding_faults_reject_even_with_rehashed_freeze_and_execution(tmp_path, fault):
    out, development = fixture(tmp_path)
    protocol = load(out/'protocol.json')
    prior = load(development/'protocol.json')
    freeze = load(out/'freeze.json')
    if fault == 'method_config':
        protocol['configuration']['methods']['normalized_tight']['utility_slack'] = .03
    elif fault == 'budget_config':
        protocol['configuration']['budget']['total_effective_epsilon_per_m'] = .46
    elif fault == 'clock_config':
        protocol['configuration']['budget']['read_interval_s'] = 30.
    elif fault == 'public_prototypes':
        protocol['configuration']['public_destination_grid_quantiles'] = [.1, .5, .9]
    elif fault == 'source_bytes':
        (tmp_path/PROFILE).write_bytes(b'changed public source\n')
    elif fault == 'source_snapshot':
        (out/'source_snapshot'/PROFILE).write_bytes(b'changed snapshot\n')
    elif fault == 'public_bytes':
        (tmp_path/PUBLIC).write_bytes(b'changed public graph\n')
    elif fault == 'public_pins':
        protocol['public_inputs_sha256'] = {}
    elif fault == 'selector_source':
        (tmp_path/SELECTOR).write_bytes(b'changed selector source\n')
    elif fault == 'historical_metric_hash':
        (development/'utility_readout.json').write_bytes(b'changed old utility bytes\n')
    elif fault == 'same_dataset_path':
        protocol['dataset_path'] = prior['dataset_path']
    elif fault == 'same_dataset_hash':
        protocol['dataset_sha256'] = prior['dataset_sha256']
    elif fault == 'extra_arm':
        protocol['configuration']['methods']['normalized_mean'] = deepcopy(METHOD_CONFIGS['normalized_mean'])
    write(out/'protocol.json', protocol)
    freeze['configuration'] = deepcopy(protocol['configuration'])
    write(out/'freeze.json', freeze)
    seal_protocol(out)
    if fault == 'old_key_pairing':
        execution = load(out/'execution_protocol.json')
        execution['paired_development'] = {'retained_key_blocks': ['old-study-private-key']}
        write(out/'execution_protocol.json', execution)
        (out/'execution_protocol.sha256').write_text(digest(out/'execution_protocol.json')+'\n')
    with pytest.raises(ERRORS):
        fresh_contract(out, root=tmp_path)
