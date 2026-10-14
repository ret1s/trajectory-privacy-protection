"""Conditional transfer of a public reference floor to actual service utility.

This supplemental theorem does not alter the frozen GeoI-BQ planner. Supplied
TV/proxy bounds MUST hold conditional on every relevant sampled-history/cache
prefix. Matching only the unconditional GPS/request distribution is insufficient.
Unknown bounds remain unknown; a benchmark mean is not a certified floor.
"""
from fractions import Fraction


def _unit(value, name):
    if isinstance(value, bool):
        raise ValueError(f'{name} must be a rational value in [0,1]')
    value = Fraction(value)
    if not 0 <= value <= 1:
        raise ValueError(f'{name} must be in [0,1]')
    return value


def reference_transfer(floors, *, conditional_tv_bounds=None,
                       proxy_error_bounds=None, time_weights=None,
                       recall_threshold=None):
    """Exact conditional lower bound on expected weighted session Recall.

    `floors[t]` bounds the public-reference mean at EVERY retained history at t.
    TV compares actual service cases conditional on that history to the public
    reference law; proxy error uniformly bounds their [0,1] utility discrepancy.
    Time weights are fixed/public, not selected after seeing private utility.
    The optional failure bound concerns weighted session Recall, not all ticks.
    """
    floors = tuple(_unit(v, 'reference floor') for v in floors)
    if not floors:
        raise ValueError('At least one public evaluation step required')
    count = len(floors)

    def errors(values, name):
        if values is None:
            return (None,) * count
        values = tuple(values)
        if len(values) != count:
            raise ValueError('Matched public evaluation steps required')
        return tuple(None if v is None else _unit(v, name) for v in values)

    tv = errors(conditional_tv_bounds, 'history-conditional TV bound')
    proxy = errors(proxy_error_bounds, 'uniform proxy error bound')
    if time_weights is None:
        weights = (Fraction(1, count),) * count
    else:
        weights = tuple(_unit(v, 'public time weight') for v in time_weights)
        if len(weights) != count or sum(weights) != 1:
            raise ValueError('Matched public time weights must sum to one')
    threshold = None if recall_threshold is None else _unit(recall_threshold, 'Recall threshold')
    if threshold == 1:
        raise ValueError('Failure bound requires a Recall threshold below one')
    lower = tuple(None if t is None or e is None else max(Fraction(0), c-t-e)
                  for c, t, e in zip(floors, tv, proxy))
    # Unknown bounds at zero-weight steps do not prevent the requested claim.
    complete = all(v is not None or w == 0 for v, w in zip(lower, weights))
    mean = sum(w*v for v, w in zip(lower, weights) if w and v is not None) if complete else None
    failure = None if mean is None or threshold is None else min(Fraction(1), (1-mean)/(1-threshold))
    return dict(
        per_step_expected_recall_lower=[None if v is None else str(v) for v in lower],
        session_expected_recall_lower=None if mean is None else str(mean),
        session_below_threshold_probability_upper=None if failure is None else str(failure),
        assumptions_verified_by_this_function=False,
        status=('conditional on supplied EVERY-history TV and uniform proxy bounds'
                if complete else 'unknown conditional transfer bounds; no session guarantee'),
        scope='fixed/public weighted session Recall; no IID, calibration, or per-user floor assumed',
    )
