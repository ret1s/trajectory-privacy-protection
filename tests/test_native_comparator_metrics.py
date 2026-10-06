import json
import math

import pytest

from evaluation.native_comparator_metrics import (
    anonymity_success_rate, comparator_native_metrics, dummy_effectiveness_rate,
    expected_travel_cost_distortion, inference_error_m, posterior_entropy_bits,
    posterior_expected_error_m, weight_entropy_bits,
)


def test_cost_absolute_value_is_inside_target_expectation():
    # Opposite target errors must not cancel: E|error|=100, |E error|=0.
    assert expected_travel_cost_distortion([[0, 100]], [[100, 0]], [.5, .5]) == 100
    assert expected_travel_cost_distortion([[10, 30], [40, 0]], [[0, 10], [30, 10]], [.25, .75]) == 13.75


def test_eq13_matches_existing_paper_core_on_directed_cost_table():
    from benchmark.engines.transprotect import expected_travel_cost_loss
    costs = [[0, 30, 80], [10, 0, 40], [60, 20, 0]]
    prior = [.2, .3, .5]
    losses = expected_travel_cost_loss(costs, prior, 0)
    assert expected_travel_cost_distortion([costs[0]], [costs[2]], prior) == pytest.approx(losses[2])


@pytest.mark.parametrize("bad", [[], [-1, 1], [math.nan], [math.inf], [True]])
def test_error_inputs_are_not_silently_filtered(bad):
    with pytest.raises(ValueError):
        inference_error_m(bad)


def test_posterior_risk_is_separate_from_point_estimator_error():
    # MAP picks the zero-error state, but the supplied posterior risk is 20m.
    assert inference_error_m([0]) == 0
    assert posterior_expected_error_m([[.8, .2]], [[0, 100]]) == 20
    with pytest.raises(ValueError, match="normalized"):
        posterior_expected_error_m([[8, 2]], [[0, 100]])
    with pytest.raises(ValueError):
        posterior_expected_error_m([[1]], [[0, 100]])


def test_entropy_requires_posterior_evidence_but_prior_weights_normalize_explicitly():
    assert weight_entropy_bits([2, 2, 2, 2]) == 2
    assert posterior_entropy_bits([1, 0]) == 0
    with pytest.raises(ValueError, match="normalized"):
        posterior_entropy_bits([2, 2])
    with pytest.raises(ValueError):
        weight_entropy_bits([0, 0])


def test_asr_boundary_and_effectiveness_need_actual_evidence():
    assert anonymity_success_rate([.2, .200001, .1], 5) == pytest.approx(200 / 3)
    assert dummy_effectiveness_rate([True, False, True]) == pytest.approx(2 / 3)
    for bad_k in (1, True, 5.5):
        with pytest.raises((TypeError, ValueError)):
            anonymity_success_rate([.2], bad_k)
    with pytest.raises(ValueError):
        dummy_effectiveness_rate([1, 0])


def test_dummy_only_is_not_awarded_perfect_asr_or_reduced_to_secret_nearest_member():
    result = comparator_native_metrics(
        "dummy_only", attack_errors_m=[100, 300], release_errors_m=[0, 0],
        real_target_costs=[[0, 100]], released_target_costs=[[0, 100]], target_prior=[.5, .5],
        candidate_prior_weights=[[1, 1]], candidate_posteriors=[[.5, .5]],
        real_member_indices=[0], dummy_effective_rows=[[True]],
    )
    assert result["eie_point_estimate_m"]["value"] == 200
    for name in ("semantic_asr_percent", "semantic_der", "dls_cell_entropy_bits",
                 "transprotect_delta_cost", "single_release_displacement_m"):
        assert result[name]["value"] is None
        assert result[name]["reason"]
        assert result[name]["status"] == "not_applicable"
    # N/A serializes as JSON null rather than a nonstandard NaN.
    json.dumps(result, allow_nan=False)


def test_native_metric_bundle_keeps_units_denominators_and_query_weighting():
    result = comparator_native_metrics(
        "real_plus_dummies", candidate_posteriors=[[.5, .5], [.7, .2, .1]],
        real_member_indices=[0, 2], dummy_effective_rows=[[True], [True, False]],
        candidate_prior_weights=[[1, 1], [1, 1, 1]],
        generation_ms=40., generation_event_count=2,
        indistinguishable_path_counts=[0, 4],
    )
    assert result["semantic_asr_percent"]["value"] == 100.
    assert result["semantic_asr_percent"]["unit"] == "%"
    assert result["semantic_der"]["value"] == .75  # equal query weighting, not 2/3
    assert result["semantic_der"]["denominator"] == 2
    assert result["generation_ms_per_event"]["value"] == 20
    assert result["fake_query_indistinguishable_paths"]["value"] == 2
    assert result["dls_cell_entropy_bits"]["category"] == "diagnostic"


def test_missing_and_incompatible_metrics_do_not_become_zero():
    result = comparator_native_metrics("replacement_trajectory")
    assert all(row["value"] is None for row in result.values())
    result = comparator_native_metrics("replacement_trajectory", real_target_costs=[[0, 1]],
                                       released_target_costs=[[2, 3]], target_prior=[.5, .5])
    assert result["transprotect_delta_cost"]["value"] == 2
    with pytest.raises(ValueError):
        comparator_native_metrics("unknown")


def test_invalid_reproduction_inputs_fail_closed():
    with pytest.raises(ValueError):
        expected_travel_cost_distortion([[0, math.inf]], [[1, 2]], [.5, .5])
    with pytest.raises(ValueError):
        comparator_native_metrics("real_plus_dummies", candidate_posteriors=[[.5, .5]],
                                   real_member_indices=[2])
    with pytest.raises(ValueError):
        comparator_native_metrics("dummy_only", generation_ms=1, generation_event_count=0)
    with pytest.raises(ValueError):
        comparator_native_metrics("dummy_only", indistinguishable_path_counts=[.5])


def test_readout_retains_failed_rows_and_weights_records_before_families():
    from experiments.native_metric_readout import summaries
    def row(family, record, rep, error):
        return {"cohort": "fixture", "method": "raw", "scenario": "S1", "k": 5,
                "case_id": "S1", "family_id": family, "record_id": record, "rep": rep,
                "status": "ok", "metrics": comparator_native_metrics("replacement_trajectory", attack_errors_m=[error])}
    values = [row("a", "r1", 0, 10), row("a", "r1", 1, 10), row("a", "r2", 0, 30),
              row("b", "r3", 0, 100)]
    values.append({"cohort": "fixture", "method": "raw", "scenario": "S1", "k": 5,
                   "case_id": "S1", "family_id": "b", "record_id": "failed", "rep": 0, "status": "failed"})
    result = summaries(values)[0]
    assert result["attempted_rows"] == 5 and result["completed_rows"] == 4
    metric = result["metrics"]["eie_point_estimate_m"]
    assert metric["value"] == 60  # mean(mean(10,30),100), not pooled rows
    assert metric["unavailable_rows"] == 1 and metric["family_count"] == 2


def test_readout_independently_checks_frozen_attack_coordinates():
    from experiments.native_metric_readout import paper_rows, projected
    truth = [40, 116]
    xy = projected(truth, 40)
    frozen = {"manifests": [{"seed": 81, "projection_lat0": 40}], "rows": [
        {"k": 5, "method": "unprotected", "seed": 81, "scenario": "S1", "record_id": "r1", "status": "ok",
         "public": {"output_kind": "replacement_trajectory", "events": [{"candidates": [{"lat": 40, "lon": 116}]}]},
         "truth": [truth], "attack_xy": [[xy[0] + 30, xy[1] + 40]], "attacker_selected": "fixture",
         "metrics": {"per_event_error_m": [50], "location_mae_m": 50, "amortized_generation_ms": 1.}}]}
    result = paper_rows(frozen, 5)[0]
    assert result["metrics"]["eie_point_estimate_m"]["value"] == 50
    assert result["metrics"]["single_release_displacement_m"]["value"] == 0
    frozen["rows"][0]["metrics"]["per_event_error_m"] = [49]
    with pytest.raises(ValueError, match="do not match"):
        paper_rows(frozen, 5)
