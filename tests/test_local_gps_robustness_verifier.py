"""Independent local-sensor fixtures; no historical or new scores are opened."""
import math

import pytest

from experiments.verify_local_gps_robustness_20261007 import (
    ARMS, PURPOSES, VARIANTS, aggregate, causal_estimate, causal_fix_history,
    percentile, rank_score, standardized_offset,
)


def test_sensor_hash_common_across_modes_depths_and_error_scales():
    z = standardized_offset("family", 1, 0, 60.)
    assert all(math.isfinite(v) for v in z)
    assert z == standardized_offset("family", 1, 0, 60)
    assert z != standardized_offset("family", 2, 0, 60.)
    assert z != standardized_offset("family", 1, 1, 60.)
    raw = {0: dict(lat=0., lon=0.), 60: dict(lat=1., lon=2.)}
    projected = lambda lat, lon: (lon, lat)
    zero = causal_fix_history(raw, "family", 1, 0, 80., 0., projected)
    five = causal_fix_history(raw, "family", 1, 0, 80., 5., projected)
    fifteen = causal_fix_history(raw, "family", 1, 0, 80., 15., projected)
    for (_, x), (_, a), (_, b) in zip(zero, five, fifteen):
        assert tuple(b[i] - x[i] for i in range(2)) == pytest.approx(tuple(3 * (a[i] - x[i]) for i in range(2)))


def test_hold_velocity_one_fix_fallback_and_speed_cap():
    assert causal_estimate([(0., (0., 0.))], 20., "velocity_twofix60") == ((0., 0.), (0., 0.), 20., 0.)
    fixes = [(0., (0., 0.)), (60., (600., 800.))]
    held = causal_estimate(fixes, 80., "hold_last60")
    extrapolated = causal_estimate(fixes, 80., "velocity_twofix60")
    assert held[0] == (600., 800.) and held[2:] == (20., 60.)
    assert math.hypot(*extrapolated[1]) == pytest.approx(8.)
    assert extrapolated[0] == pytest.approx((696., 928.))
    assert causal_estimate(fixes, 60., "velocity_twofix60")[0] == fixes[-1][1]


def test_future_fix_coordinates_never_read_and_poisoned_current_event_not_a_fix():
    class Poison:
        def __iter__(self):
            raise AssertionError("Future/current private coordinate was consumed")
    fixes = [(0., (0., 0.)), (60., (60., 0.)), (120., Poison())]
    assert causal_estimate(fixes, 80., "velocity_twofix60")[0] == (80., 0.)
    raw = {0: dict(lat=0., lon=0.), 60: dict(lat=0., lon=60.), 80: Poison(), 120: Poison()}
    history = causal_fix_history(raw, "family", 1, 0, 80., 0., lambda lat, lon: (lon, lat))
    assert history == [(0., (0., 0.)), (60., (60., 0.))]
    assert causal_estimate(history, 80., "velocity_twofix60")[0] == (80., 0.)


def test_causal_history_rejects_missing_declared_fix_and_time_rewind():
    with pytest.raises(KeyError):
        causal_fix_history({0: dict(lat=0., lon=0.)}, "family", 1, 0, 80., 0., lambda lat, lon: (lon, lat))
    with pytest.raises(AssertionError):
        causal_estimate([(60., (0., 0.)), (0., (0., 0.))], 80., "hold_last60")


def test_actual_current_reference_is_not_replaced_by_local_estimate_reference():
    actual = {"nearest_distance": [[0, 1, 2, 3, 4, 5]]}
    approximate = {"nearest_distance": [[5, 4, 3, 2, 1, 0]]}
    row = rank_score(actual, approximate, set(range(6)))["nearest_distance"]
    assert row["recall5"] == .8 and row["overlap_total"] == 4 and row["reference_poi_total"] == 5
    assert rank_score(approximate, approximate, set(range(6)))["nearest_distance"]["recall5"] == 1.


def test_empty_estimated_radius_is_zero_for_nonempty_true_reference():
    row = rank_score({"within_radius": [[0]]}, {"within_radius": [[]]}, {0})["within_radius"]
    assert row["recall5"] == row["completion"] == 0.
    assert row["zero_answer_reference_categories"] == 1
    inverse = rank_score({"within_radius": [[]]}, {"within_radius": [[0]]}, {0})["within_radius"]
    assert inverse["recall5"] is None and inverse["empty_reference_categories"] == 1
    assert inverse["returned_outside_true_domain"] == 1


def test_scope_completion_does_not_credit_outside_true_radius():
    row = rank_score({"within_radius": [[0, 1]]}, {"within_radius": [[2, 0, 3]]}, {0, 1, 2, 3})["within_radius"]
    assert row["recall5"] == row["completion"] == .5
    assert row["returned_items"] == 3 and row["returned_outside_true_domain"] == 2


def test_invalid_estimate_keeps_actual_reference_denominator_and_zero_score():
    row = rank_score({"within_radius": [[0, 1], []]}, {}, {0, 1}, invalid=True)["within_radius"]
    assert row["recall5"] == 0. and row["reference_poi_total"] == 2
    assert row["reference_category_count"] == row["empty_reference_categories"] == 1
    assert row["invalid_estimate_reference_categories"] == 1
    assert row["estimated_empty_categories"] == 2


def test_linear_quantiles_are_explicit_and_undefined_stays_none():
    assert percentile([30., 0., 20., 10.], .5) == 15.
    assert percentile([30., 0., 20., 10.], .95) == pytest.approx(28.5)
    assert percentile([], .5) is None
    assert percentile([2.], .95) == 2.


def test_family_macro_equal_weights_despite_different_tick_counts():
    blocks = []
    for family, count, good in [("a", 1, False), ("b", 10, True)]:
        for draw in (1, 2, 3):
            rows, estimates = [], []
            for index in range(count):
                scores = rank_score({p: [[0]] for p in PURPOSES}, {p: [[0]] for p in PURPOSES}, {0} if good else set())
                for arm in ARMS:
                    rows.append(dict(arm=arm, family_id=family, draw=draw, slot=0, t=400 + index, purposes=scores))
                for variant in VARIANTS:
                    estimates.append(dict(variant=variant, invalid_reason=None, velocity_clipped=False, first_fix_fallback=False,
                                          error_before_snap_m=float(good), error_after_snap_m=float(good), snap_distance_m=0.))
            wires = [dict(method=f"service_l{L}", requests=5 * count, request_bytes=10 * count, reply_bytes=L * count) for L in (20, 30)]
            blocks.append(dict(rows=rows, estimates=estimates, wire=wires))
    result = aggregate(blocks)
    row = result["summary"][ARMS[0]]["all"]
    assert row["equal_purpose_macro"]["family_mean"] == .5
    assert row["three_purpose_macro"]["family_mean"] == .5
    assert row["three_purpose_macro"]["within_draw_family_mean"] == {"1": .5, "2": .5, "3": .5}
    assert row["nearest_distance"]["explicit_denominators"]["total_events"] == 33
    assert result["cost"][ARMS[0]]["requests"] == 165
