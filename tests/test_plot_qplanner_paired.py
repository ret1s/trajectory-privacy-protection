"""Figure contract checks on fabricated unit fixtures, never research scores."""
from copy import deepcopy
import json

import pytest

from experiments.plot_qplanner_paired_20261006 import PURPOSES,prepare,render,coverage_text


def fixture():
    coverage=dict(defined_windows=12,total_windows=15,defined_categories=24,total_categories=30)
    row=dict(primary_contrast=False,family_ids=['f1','f2'],independent_family_clusters=2,
        family_differences={'f1':.01,'f2':.05},mean_difference=.03,
        percentile95_family_bootstrap=[.01,.05],lower_tail_mean_difference=.01,
        percentile95_lower_tail_difference=[-.01,.03],
        left_reference_coverage=coverage,right_reference_coverage=deepcopy(coverage))
    contrasts={}
    for right in ('legacy_l10','aligned_nearest'):
        for purpose in (*PURPOSES,'equal_purpose_macro'):
            key=f'multi_mean--minus--{right}--test--current--all--{purpose}'
            contrasts[key]=deepcopy(row)
            contrasts[key]['primary_contrast']=right=='legacy_l10' and purpose=='equal_purpose_macro'
    return dict(schema='qplanner-paired-family-readout-v1',test_independent_unit='family',
        defense_selected_by_this_readout=False,protocol_sha256='a'*64,contrasts=contrasts)


def test_plot_data_preserves_saved_primary_secondary_intervals_family_units_and_na_coverage():
    data=fixture();result=prepare(data)
    assert result['family_ids']==['f1','f2'] and result['methods']==['multi_mean','legacy_l10','aligned_nearest']
    assert result['groups'][0]['rows']['equal_purpose_macro']['percentile95_family_bootstrap']==[.01,.05]
    assert result['absolute_family_recall_available'] is False
    assert coverage_text(result['coverage'][('multi_mean','nearest_distance')])=='24/30 (80.0%)\nwindows 12/15'
    key='multi_mean--minus--legacy_l10--test--current--all--within_radius'
    data['contrasts'][key].update(family_ids=[],family_differences={},independent_family_clusters=0,
        mean_difference=None,percentile95_family_bootstrap=None,
        lower_tail_mean_difference=None,percentile95_lower_tail_difference=None)
    assert prepare(data)['groups'][0]['rows']['within_radius']['mean_difference'] is None


def test_figure_cannot_choose_missing_primary_or_hide_missing_alignment_comparison():
    data=fixture()
    for row in data['contrasts'].values():row['primary_contrast']=False
    with pytest.raises(ValueError,match='Exactly one'):
        prepare(data)
    data=fixture();del data['contrasts']['multi_mean--minus--aligned_nearest--test--current--all--equal_purpose_macro']
    with pytest.raises(ValueError,match='alignment contrast'):
        prepare(data)


def test_changed_denominators_and_invalid_interval_are_rejected():
    data=fixture();key='multi_mean--minus--legacy_l10--test--current--all--nearest_distance'
    data['contrasts'][key]['independent_family_clusters']=3
    with pytest.raises(ValueError,match='denominators'):
        prepare(data)
    data=fixture();data['contrasts'][key]['percentile95_family_bootstrap']=[.1,-.1]
    with pytest.raises(ValueError,match='ordered95%'):
        prepare(data)
    data=fixture();data['contrasts'][key]['left_reference_coverage']['defined_categories']=31
    with pytest.raises(ValueError,match='coverage cannot exceed'):
        prepare(data)


@pytest.mark.parametrize('method',['normalized_mean','normalized_tight','normalized_tail'])
def test_normalized_v2_names_preserve_single_primary_and_alignment_contrast(method):
    data=fixture()
    data['contrasts']={key.replace('multi_mean',method):value for key,value in data['contrasts'].items()}
    prepared=prepare(data)
    assert prepared['left']==method
    assert prepared['methods']==[method,'legacy_l10','aligned_nearest']
    assert len(prepared['groups'])==2


def test_actual_figure_artifact_is_source_backed_and_write_once(tmp_path):
    source=tmp_path/'fabricated_unit_fixture.json';source.write_text(json.dumps(fixture()))
    output=tmp_path/'unit_test_figure'
    provenance=render(source,output)
    assert (output/'paired_family_current_recall.pdf').read_bytes().startswith(b'%PDF')
    assert (output/'paired_family_current_recall.png').read_bytes().startswith(b'\x89PNG')
    assert set(provenance['output_sha256'])=={'paired_family_current_recall.pdf','paired_family_current_recall.png'}
    assert provenance['displayed_units'].startswith('percentage-point')
    assert provenance['private_data_or_rng_keys_read'] is False
    with pytest.raises(FileExistsError):render(source,output)
