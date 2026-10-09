"""Versioned serialization-only correction; original eight tests retained."""
import pytest

from tests.test_dynamic_provider_status_verifier import (
    test_directed_oracle_distinguishes_distance_time_radius_detour,
    test_empty_reference_stays_na_and_nonempty_with_no_answer_is_zero,
    test_epoch_boundary_cache_causal_and_malformed_call_does_not_mutate,
    test_hash_id_and_absolute_epoch_do_not_depend_on_order_or_gps,
    test_safe_partition_and_unknown_never_becomes_unavailable,
    test_stale_invalid_control_counts_false_availability_without_safe_credit,
    test_whole_family_nested_draw_mean_does_not_weight_events_as_subjects,
    test_wire_preserves_absolute_float_and_counts_only_status_overhead,
)
from experiments.verify_dynamic_provider_status_20261006_v2 import canonical_rows, same


def test_row_order_is_immaterial_only_with_unique_public_event_arm_keys():
    a = dict(slot=0, t=20, arm="raw", recall=.25)
    b = dict(slot=0, t=20, arm="protected", recall=.5)
    same([a, b], [b, a], "block/rows")
    assert canonical_rows([a, b]) == [b, a]
    with pytest.raises(AssertionError):
        same([a, b], [b, dict(a, recall=.75)], "block/rows")


def test_canonical_rows_reject_duplicate_missing_and_wrong_arm_inventory():
    a = dict(slot=0, t=20, arm="raw", recall=.25)
    b = dict(slot=0, t=20, arm="protected", recall=.5)
    with pytest.raises(AssertionError):
        canonical_rows([a, a])
    with pytest.raises(AssertionError):
        same([a, b], [b], "block/rows")
    with pytest.raises(AssertionError):
        same([a, b], [a, dict(b, arm="another")], "block/rows")
