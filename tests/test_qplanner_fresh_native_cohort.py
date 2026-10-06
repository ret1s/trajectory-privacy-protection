"""Integrity tests for public grouped cohort planning, independent of scores."""
import collections
import pytest

pytest.importorskip('sumolib')

from experiments.build_qplanner_fresh_native_20261006 import (
    assignments, independent_route_pair, route_overlap, choice_pattern, build,
)


def test_public_assignments_are_disjoint_balanced_and_reproducible():
    rows = assignments()
    assert rows == assignments()
    assert len(rows) == len({r['family_id'] for r in rows}) == 60
    counts = collections.Counter((r['split'], r['public_speed_m_s']) for r in rows)
    assert counts == {('train', 6.): 12, ('train', 8.): 12,
        ('selection', 6.): 6, ('selection', 8.): 6, ('test', 6.): 12, ('test', 8.): 12}
    assert assignments(123) != rows
    assert {r['family_id'] for r in assignments(123)} == {r['family_id'] for r in rows}


def test_public_route_guard_rejects_near_duplicate_even_if_direction_context_changes():
    old = [['a', 'b', 'c', 'd', 'e']]
    assert route_overlap(old[0], ['e', 'd', 'c', 'b', 'a']) == 1.
    assert not independent_route_pair({'routes': [['a', 'b', 'c', 'd'], ['new']]}, old)
    assert independent_route_pair({'routes': [['a', 'new'], ['other']]}, old)


def test_hidden_choice_pattern_preserves_balanced_queries_without_geometry_inputs():
    key = bytes(range(32))
    for index in range(60):
        routine, pattern = choice_pattern(key, f'family-{index}')
        assert pattern[:6].count(routine) == 5
        assert sorted(pattern[6:]) == [0, 1]
        assert len(pattern) == 8
        assert choice_pattern(key, f'family-{index}') == (routine, pattern)


def test_cohort_refuses_in_repository_private_state_before_creating_files(tmp_path):
    from experiments.build_qplanner_fresh_native_20261006 import ROOT
    with pytest.raises(ValueError, match='private work directory'):
        build(ROOT/'artifacts/datasets/forbidden-cohort-test', ROOT/'tmp/private-state-test')
    assert not (ROOT/'artifacts/datasets/forbidden-cohort-test').exists()


def test_existing_output_is_never_overwritten(tmp_path):
    from experiments.build_qplanner_fresh_native_20261006 import ROOT
    with pytest.raises(FileExistsError, match='never resume or overwrite'):
        build(ROOT/'artifacts/datasets/future_controlled_20261005_v2', tmp_path/'private')
