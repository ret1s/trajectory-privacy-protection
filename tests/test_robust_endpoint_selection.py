import numpy as np
import pytest

from benchmark.paper_comparators import PublicHistory
from evaluation.ordered_endpoint_attacks import ordered_endpoint_features
from evaluation.robust_endpoint_selection import family_statistics, robust_select
from experiments.endpoint_robust_selection_20261006 import frozen_selection
from tests.test_belief_lane import fixture


def row(family, errors, rep=0):
    return dict(family_id=family, session_id='trip-'+family, seed=rep, errors=errors)


def test_robust_mae_penalizes_family_instability_instead_of_best_plain_mean():
    rows = [row('a', {'stable': 100., 'unstable': 0.}), row('b', {'stable': 100., 'unstable': 180.})]
    selected, stats = robust_select(rows, ['a', 'b'])
    assert stats['unstable']['mae']['mean'] == 90.
    assert stats['unstable']['mae']['se'] == 90.
    assert selected['mae'] == 'stable'
    assert stats['stable']['mae']['objective'] == 100.


def test_robust_hit_penalizes_success_only_in_one_family_and_uses_lexical_ties():
    rows = [row('a', {'unstable': 0., 'stable': 0.}, 0), row('a', {'unstable': 0., 'stable': 200.}, 1),
            row('b', {'unstable': 200., 'stable': 0.}, 0), row('b', {'unstable': 200., 'stable': 200.}, 1)]
    selected, stats = robust_select(rows, ['b', 'a'])
    assert stats['unstable']['hit100']['mean'] == stats['stable']['hit100']['mean'] == .5
    assert stats['unstable']['hit100']['se'] == .5
    assert selected['hit100'] == 'stable'
    ties = [row('a', {'z': 10., 'a': 10.}), row('b', {'z': 20., 'a': 20.})]
    assert set(robust_select(ties, ['a', 'b'])[0].values()) == {'a'}


def test_families_not_repetitions_determine_the_risk_and_se():
    sparse = [row('a', {'attack': 0.}), row('b', {'attack': 100.})]
    duplicated_family = [row('a', {'attack': 0.}, i) for i in range(20)]+[sparse[1]]
    _, a = robust_select(sparse, ['a', 'b'])
    _, b = robust_select(duplicated_family, ['a', 'b'])
    assert a == b
    assert a['attack']['mae']['mean'] == 50.
    assert a['attack']['mae']['se'] == 50.


@pytest.mark.parametrize('bad', [
    [row('a', {'attack': 1.})],
    [row('a', {'attack': 1.}), row('b', {'other': 1.})],
    [row('a', {'attack': 1.}), row('b', {'attack': np.nan})],
    [row('a', {'attack': 1.}), row('b', {'attack': -1.})],
    [row('a', {'attack': 1.}), row('a', {'attack': 1.}), row('b', {'attack': 1.})],
])
def test_missing_candidates_families_invalid_errors_and_duplicate_reps_are_rejected(bad):
    with pytest.raises(ValueError):
        robust_select(bad, ['a', 'b'])


def test_feature_cache_has_no_held_out_family_history_dependency():
    rn, _, _ = fixture()
    uniform = PublicHistory(rn, [])
    trained = PublicHistory(rn, [[15]*100, list(range(15))])
    events = [dict(timestamp_s=20.*i, coordinates=[rn.latlon(i+j) for j in range(3)]) for i in range(6)]
    for scenario in ('S9', 'S10'):
        a, _ = ordered_endpoint_features(events, scenario, rn, uniform, observable_close_s=100.)
        b, _ = ordered_endpoint_features(events, scenario, rn, trained, observable_close_s=100.)
        assert all(np.array_equal(a[channel], b[channel]) for channel in a)


def test_opening_test_requires_frozen_selector_and_se_is_sample_based(tmp_path):
    with pytest.raises(RuntimeError, match='Freeze selectors'):
        frozen_selection(tmp_path)
    stats = family_statistics([0., 2., 4.])
    assert stats['mean'] == 2.
    assert stats['se'] == pytest.approx(2./np.sqrt(3.))
