"""Metric denominator, study sealing and independent four-purpose checks."""
import gzip
import json

import numpy as np
import pytest

from benchmark.public_poi_context import PublicPoiContext
from benchmark.query_purpose import MultiPurposeRoadRanking
from evaluation.lane_travel import LanePoiService
from experiments import qplanner_study_20261006_v2 as study
from tests.test_lane_comparison import road


def test_fractional_family_lower_tail_is_not_quantile_or_minimum():
    assert study.lower_tail([0., .5, 1.], .5) == pytest.approx(1/6)
    assert study.lower_tail([.2, .4, .8, 1.], .25) == .2
    assert study.lower_tail([]) is None


def test_family_macro_retains_empty_references_and_does_not_weight_by_ticks():
    def row(family, value):
        return {'family_id': family, 'purposes': {p: {'recall5': value,
            'reference_category_count': int(value is not None), 'all_category_count': 1} for p in study.PURPOSES}}
    result = study.summarize_utility([row('a', 0.), row('a', 1.), row('b', 1.), row('c', None)])
    for purpose in study.PURPOSES:
        assert result[purpose]['family_mean'] == .75
        assert result[purpose]['defined_windows'] == 3 and result[purpose]['total_windows'] == 4
        assert result[purpose]['family_values']['c'] is None
    assert result['equal_purpose_macro']['family_mean'] == .75


def test_four_purpose_masks_use_exact_local_reference_and_keep_radius_na():
    rn = road()
    pois = [{'id': f'p{i:02d}', 'lat': 0., 'lon': i*.0001, 'category': 'cafe'} for i in range(1, 28, 3)]
    reference = PublicPoiContext(LanePoiService(rn, pois, k=5))
    reply = PublicPoiContext(LanePoiService(rn, pois, k=20))
    evaluator = study.UtilityEvaluator(rn, reply, MultiPurposeRoadRanking(reference))
    full = evaluator.score(0, 20, set(range(len(pois))))
    empty = evaluator.score(0, 20, set())
    assert all(r['recall5'] == 1. and r['completion'] == 1. for r in full.values())
    assert all(r['recall5'] == 0. and r['completion'] == 0. for r in empty.values())
    evaluator.references(0, 20)
    assert len(evaluator.refs) == 1


def test_source_closure_includes_private_primitive_and_new_optimizer():
    files = study.source_closure()
    for path in ('core/mechanisms.py', 'core/session_budget.py',
                 'benchmark/public_service_profiles_v2.py', 'benchmark/engines/public_service_planner_v2.py',
                 'evaluation/endpoint_noise_attacks.py', 'evaluation/live_comparison_attacks.py'):
        assert path in files
    assert not any(path.startswith('archive/') for path in files)


def test_write_once_protocol_detects_source_and_dataset_change(tmp_path, monkeypatch):
    monkeypatch.setattr(study, 'ROOT', tmp_path)
    source = tmp_path/'runner.py'; source.write_text('immutable source\n')
    dataset = tmp_path/'dataset.json.gz'; dataset.write_bytes(gzip.compress(b'{}', mtime=0))
    monkeypatch.setattr(study, 'source_closure', lambda: ['runner.py'])
    monkeypatch.setattr(study, 'public_input_pins', lambda dataset: {})
    out = tmp_path/'output'
    study.declare(out, dataset, ['aligned_nearest'], 2, ['train', 'selection'], 'development')
    study.validate(out)
    with pytest.raises(ValueError, match='protocol changed'):
        study.declare(out, dataset, ['normalized_tail'], 2, ['train', 'selection'], 'development')
    source.write_text('new source\n')
    with pytest.raises(ValueError, match='Source changed'): study.validate(out)
    source.write_text('immutable source\n'); dataset.write_bytes(gzip.compress(b'{"new":true}', mtime=0))
    with pytest.raises(ValueError, match='Dataset changed'): study.validate(out)
