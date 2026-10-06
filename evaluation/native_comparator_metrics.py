"""Paper metric formulas with explicit output-contract and evidence gates.

These functions consume evaluator-side measurements. They never infer a real
candidate from coordinates, invent an attacker posterior, or select a secret-
nearest member of a dummy set. ``None`` means unavailable, never perfect privacy.
Primary definitions and reproduction limits are in
``docs/research/2026-10-05_native_metrics.md``.
"""
from dataclasses import asdict, dataclass
import math
from numbers import Integral


KINDS = {"replacement_trajectory", "real_plus_dummies", "dummy_only"}


@dataclass(frozen=True)
class MetricResult:
    value: object
    unit: str
    direction: str
    category: str
    denominator: int
    status: str = "computed"
    reason: str = ""

    def to_dict(self):
        return asdict(self)


def _values(values, name, *, nonnegative=True):
    result = list(values)
    if not result:
        raise ValueError(f"{name} must be nonempty")
    if any(isinstance(x, bool) for x in result):
        raise ValueError(f"{name} must contain numbers, not booleans")
    result = [float(x) for x in result]
    if any(not math.isfinite(x) or (nonnegative and x < 0) for x in result):
        raise ValueError(f"{name} must contain finite nonnegative numbers")
    return result


def _probabilities(values, name="posterior"):
    result = _values(values, name)
    if not math.isclose(math.fsum(result), 1., rel_tol=1e-8, abs_tol=1e-10):
        raise ValueError(f"{name} must be normalized; do not fabricate a posterior")
    return result


def _rows(values, name):
    result = [_values(row, name) for row in values]
    if not result or any(len(row) != len(result[0]) for row in result):
        raise ValueError(f"{name} must be a nonempty rectangular table")
    return result


def inference_error_m(errors_m):
    """Empirical EIE for a fixed point estimator, not a posterior Bayes risk."""
    values = _values(errors_m, "attack error")
    return math.fsum(values) / len(values)


def posterior_expected_error_m(posterior_rows, distance_rows_m):
    """Mean of sum_j p_t(j) d(x_t, j), on an explicitly supplied state domain."""
    probabilities = [_probabilities(row) for row in posterior_rows]
    distances = _rows(distance_rows_m, "distance to true position")
    if len(probabilities) != len(distances) or any(
        len(p) != len(d) for p, d in zip(probabilities, distances)
    ):
        raise ValueError("one aligned posterior/distance row is required per target")
    return math.fsum(math.fsum(p * d for p, d in zip(pr, ds))
                     for pr, ds in zip(probabilities, distances)) / len(distances)


def posterior_entropy_bits(probabilities):
    """Shannon entropy of a normalized distribution; zero terms contribute zero."""
    p = _probabilities(probabilities)
    return -math.fsum(x * math.log2(x) for x in p if x > 0.)


def weight_entropy_bits(weights):
    """DLS cell entropy: normalize explicit historical query weights over a set.

    This is a prior-weight objective, not automatically an attacker posterior.
    For RDG max-product objectives, callers must compute the stated path weights
    separately; do not substitute the forward posterior for those weights.
    """
    values = _values(weights, "query weight")
    total = math.fsum(values)
    if total <= 0:
        raise ValueError("query weights need positive mass")
    return posterior_entropy_bits([x / total for x in values])


def expected_travel_cost_distortion(real_cost_rows, released_cost_rows, target_prior):
    """TransProtect Eq. 13: mean_t sum_l q_l |c(x_t,l)-c(z_t,l)|.

    A row represents one released location. Unreachable costs must be handled by
    a prespecified public convention upstream, not silently dropped here. The
    caller must declare the cost unit and target prior in the experiment.
    """
    real = _rows(real_cost_rows, "true target cost")
    released = _rows(released_cost_rows, "released target cost")
    prior = _probabilities(target_prior, "target prior")
    if len(real) != len(released) or any(
        len(a) != len(prior) or len(b) != len(prior)
        for a, b in zip(real, released)
    ):
        raise ValueError("aligned target-cost rows and a shared target prior required")
    return math.fsum(math.fsum(q * abs(a - b) for q, a, b in zip(prior, x, z))
                     for x, z in zip(real, released)) / len(real)


def anonymity_success_rate(real_posteriors, k_values):
    """Semantic paper §6.3 ASR, in percent: P(real member) <= 1/K."""
    p = _values(real_posteriors, "real-member posterior")
    if any(x > 1 for x in p):
        raise ValueError("real-member posteriors must lie in [0,1]")
    ks = [k_values] * len(p) if isinstance(k_values, Integral) else list(k_values)
    if len(ks) != len(p) or any(isinstance(k, bool) or not isinstance(k, Integral)
                               or k < 2 for k in ks):
        raise ValueError("one integer K >= 2 per query is required")
    return 100. * sum(x <= 1. / k for x, k in zip(p, ks)) / len(p)


def dummy_effectiveness_rate(effective):
    """DER k'/k for explicit effectiveness decisions; no invented Sim function."""
    flags = list(effective)
    if not flags or any(type(x) is not bool for x in flags):
        raise ValueError("nonempty boolean effectiveness decisions required")
    return sum(flags) / len(flags)


def comparator_native_metrics(
    output_kind, *, attack_errors_m=None, release_errors_m=None,
    real_target_costs=None, released_target_costs=None, target_prior=None,
    cost_unit="m", candidate_prior_weights=None, candidate_posteriors=None,
    real_member_indices=None, dummy_effective_rows=None,
    indistinguishable_path_counts=None, generation_ms=None,
    generation_event_count=None,
):
    """Return JSON-ready native metric results, including explicit N/A reasons.

    ``candidate_posteriors`` must come from an attacker; ``real_member_indices``
    are evaluator-only membership labels. Counts/semantic decisions are supplied
    only when the experiment has an explicit source-compatible test. Event
    rows represent the same eligible cohort, not a hand-picked successful subset.
    """
    if output_kind not in KINDS:
        raise ValueError("unsupported output kind")
    result = {}

    def put(key, value, unit, direction, category, denominator=0, reason="", status=None):
        result[key] = MetricResult(value, unit, direction, category, denominator,
                                   status or ("computed" if value is not None else "not_available"),
                                   reason).to_dict()

    put("eie_point_estimate_m", None, "m", "higher", "privacy",
        reason="No fixed attacker errors supplied; release displacement is not EIE.")
    if attack_errors_m is not None:
        errors = list(attack_errors_m)
        put("eie_point_estimate_m", inference_error_m(errors), "m", "higher", "privacy",
            len(errors), "Empirical error of the supplied point estimator; not VehiTrack parity.")

    single = output_kind == "replacement_trajectory"
    put("single_release_displacement_m", None, "m", "lower", "utility",
        reason="Need one aligned replacement point per input, not a dummy-set reduction.",
        status=None if single else "not_applicable")
    if single and release_errors_m is not None:
        errors = list(release_errors_m)
        put("single_release_displacement_m", inference_error_m(errors), "m", "lower", "utility",
            len(errors), "Release-to-truth distance; no privacy interpretation.")

    put("transprotect_delta_cost", None, cost_unit, "lower", "utility",
        reason=("Native metric requires one replacement point per input; no secret-nearest/centroid reduction."
                if not single else "Target-cost table and prior not archived or supplied."),
        status=None if single else "not_applicable")
    costs = (real_target_costs, released_target_costs, target_prior)
    if single and all(x is not None for x in costs):
        real = list(real_target_costs)
        score = expected_travel_cost_distortion(real, released_target_costs, target_prior)
        put("transprotect_delta_cost", score, cost_unit, "lower", "utility", len(real),
            "Eq. 13 formula; public target/cost convention remains an experiment assumption.")

    real_plus = output_kind == "real_plus_dummies"
    put("dls_cell_entropy_bits", None, "bit", "higher", "diagnostic",
        reason=("DLS candidate-prior meaning assumes a set containing the real member."
                if not real_plus else "Historical query weights were not archived or supplied."),
        status=None if real_plus else "not_applicable")
    if real_plus and candidate_prior_weights is not None:
        rows = list(candidate_prior_weights)
        if not rows:
            raise ValueError("nonempty candidate weight rows required")
        value = math.fsum(weight_entropy_bits(row) for row in rows) / len(rows)
        put("dls_cell_entropy_bits", value, "bit", "higher", "diagnostic", len(rows),
            "Historical-weight entropy; not proof against temporal inference.")

    put("semantic_asr_percent", None, "%", "higher", "privacy",
        reason=("No real-plus-dummy candidate contract; candidate ASR is not defined."
                if not real_plus else "Need an attacker posterior and evaluator real-member labels."),
        status=None if real_plus else "not_applicable")
    if real_plus and candidate_posteriors is not None and real_member_indices is not None:
        rows = [_probabilities(row) for row in candidate_posteriors]
        indices = list(real_member_indices)
        if not rows or len(rows) != len(indices) or any(
            isinstance(i, bool) or not isinstance(i, Integral) or not 0 <= i < len(row)
            for row, i in zip(rows, indices)
        ):
            raise ValueError("one valid evaluator real-member index per posterior required")
        put("semantic_asr_percent", anonymity_success_rate(
            [row[i] for row, i in zip(rows, indices)], [len(row) for row in rows]
        ), "%", "higher", "privacy", len(rows),
            "ASR from supplied attacker; paper does not ship a calibrated LSP posterior.")

    put("semantic_der", None, "ratio", "higher", "diagnostic",
        reason=("Native DER expects K-1 dummies compared with a published real member."
                if not real_plus else "Need semantic effectiveness decisions; paper Sim is unspecified."),
        status=None if real_plus else "not_applicable")
    if real_plus and dummy_effective_rows is not None:
        rows = list(dummy_effective_rows)
        if not rows:
            raise ValueError("nonempty per-query effectiveness rows required")
        value = math.fsum(dummy_effectiveness_rate(row) for row in rows) / len(rows)
        put("semantic_der", value, "ratio", "higher", "diagnostic", len(rows),
            "Equal query weighting; caller must archive its semantic decision rule.")

    put("fake_query_indistinguishable_paths", None, "paths/pair", "higher", "diagnostic",
        reason="Need source-compatible arrival-time/direction similarity and adjacent-query counts.")
    if indistinguishable_path_counts is not None:
        counts = list(indistinguishable_path_counts)
        if not counts or any(isinstance(x, bool) or not isinstance(x, Integral) or x < 0 for x in counts):
            raise ValueError("nonnegative integer path counts required")
        put("fake_query_indistinguishable_paths", math.fsum(counts) / len(counts),
            "paths/pair", "higher", "diagnostic", len(counts),
            "Supplied path counts; reachability alone is not the full paper similarity test.")

    put("generation_ms_per_event", None, "ms/event", "lower", "cost",
        reason="Need total measured generation time and exact generated-event denominator.")
    if generation_ms is not None and generation_event_count is not None:
        if isinstance(generation_event_count, bool) or not isinstance(generation_event_count, Integral) or generation_event_count < 1:
            raise ValueError("positive integer generation-event count required")
        duration = _values([generation_ms], "generation time")[0]
        put("generation_ms_per_event", duration / generation_event_count, "ms/event", "lower", "cost",
            generation_event_count, "Measured scope must state setup/service/RTT exclusions.")
    return result
