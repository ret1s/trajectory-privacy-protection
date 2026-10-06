"""Isolated lower-tail utility prototype over public/protected POI profiles.

No GPS, private QuerySpec, clock, network, RNG or Geo-I engine is accepted here.
Callers must establish the public/protected provenance of profiles/signatures.
The objective and bounded one-track exchanges confer no new privacy guarantee,
global optimum, submodular approximation or whole-trajectory utility claim.
"""
from dataclasses import dataclass
import math
from numbers import Integral, Real
from collections.abc import Mapping


def _finite(value, name):
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError(f'Finite numeric {name} required')
    try:
        number = float(value)
    except (OverflowError, ValueError):
        raise ValueError(f'Finite numeric {name} required') from None
    if not math.isfinite(number):
        raise ValueError(f'Finite numeric {name} required')
    return number


def _probabilities(weights):
    values = tuple(_finite(w, 'profile weight') for w in weights)
    if not values or any(w < 0 for w in values) or max(values) <= 0:
        raise ValueError('Nonnegative weights with positive total mass required')
    # Scale before summing so large, individually finite weights cannot overflow.
    largest = max(values)
    scaled = tuple(w / largest for w in values)
    total = math.fsum(scaled)
    return tuple(w / total for w in scaled)


def _index(value, name):
    if isinstance(value, bool) or not isinstance(value, Integral) or value < 0:
        raise ValueError(f'Nonnegative integer {name} required')
    return int(value)


def weighted_lower_tail_cvar(values, weights, tail_mass):
    """Mean utility in the lowest tail_mass probability, including fractional mass.

    Weights are normalized; zero-weight observations have no influence. This is
    lower-tail CVaR of utility, not upper-tail CVaR of a loss. For a=1 it equals
    the weighted mean. The implementation evaluates the discrete quantile form.
    """
    a = _finite(tail_mass, 'tail mass')
    if not 0 < a <= 1:
        raise ValueError('Tail mass must lie in (0,1]')
    values = tuple(_finite(v, 'utility') for v in values)
    probabilities = _probabilities(weights)
    if len(values) != len(probabilities):
        raise ValueError('Matched utility and weight lengths required')
    if a == 1:
        return math.fsum(v * p for v, p in zip(values, probabilities))
    remaining, terms = a, []
    for value, probability in sorted(zip(values, probabilities)):
        take = min(probability, remaining)
        terms.append(value * take)
        remaining = max(0., remaining - take)
        if remaining == 0:
            break
    return math.fsum(terms) / a


@dataclass(frozen=True)
class TailCoverScore:
    mean: float
    lower_tail_cvar: float
    objective: float


class TailAwareCoverObjective:
    """Weighted reference-set recall, with empty references explicitly N/A.

    reference_profiles[i] is a set/sequence of reference POI indices; its mass
    is profile_weights[i]. Duplicate POI IDs count once. Empty references are
    excluded BEFORE renormalizing the remaining profile probabilities, rather
    than represented as zero recall. Zero-weight profiles contribute no mass.
    All-empty/all-zero usable profiles are undefined and raise ValueError.
    response_signatures maps public candidate state IDs to returned POI IDs;
    callers must remove absent-slot sentinels such as -1 before constructing it.
    """
    def __init__(self, reference_profiles, profile_weights, response_signatures,
                 *, tail_mass=.25, risk_weight=.5):
        self.tail_mass = _finite(tail_mass, 'tail mass')
        self.risk_weight = _finite(risk_weight, 'risk weight')
        if not 0 < self.tail_mass <= 1 or not 0 <= self.risk_weight <= 1:
            raise ValueError('Require 0<a<=1 and 0<=lambda<=1')
        profiles = tuple(frozenset(_index(p, 'POI ID') for p in ids)
                         for ids in reference_profiles)
        raw_weights = tuple(_finite(w, 'profile weight') for w in profile_weights)
        if not profiles or len(profiles) != len(raw_weights) or any(w < 0 for w in raw_weights):
            raise ValueError('Matched nonempty profiles and nonnegative weights required')
        full_weights = _probabilities(raw_weights)
        used = [(ids, w) for ids, w in zip(profiles, raw_weights) if ids and w > 0]
        if not used:
            raise ValueError('At least one nonempty positive-weight reference profile required')
        self.profiles = tuple(ids for ids, _ in used)
        self.probabilities = _probabilities(w for _, w in used)
        self.empty_profile_count = sum(not ids for ids in profiles)
        self.empty_profile_mass = math.fsum(w for ids, w in zip(profiles, full_weights) if not ids)
        self.zero_weight_profile_count = sum(w == 0 for w in raw_weights)
        self.input_profile_count = len(profiles)
        if not isinstance(response_signatures, Mapping) or not response_signatures:
            raise ValueError('Nonempty public candidate-to-POI signature mapping required')
        self.signatures = {_index(state, 'candidate state'): frozenset(
            _index(p, 'POI ID') for p in ids) for state, ids in response_signatures.items()}

    def action(self, action):
        result = tuple(_index(state, 'candidate state') for state in action)
        if not result or any(state not in self.signatures for state in result):
            raise ValueError('Nonempty action of known candidate states required')
        return result

    def utilities(self, action):
        action = self.action(action)
        covered = frozenset().union(*(self.signatures[state] for state in action))
        return tuple(len(ids & covered) / len(ids) for ids in self.profiles)

    def score(self, action):
        utilities = self.utilities(action)
        mean = math.fsum(v * p for v, p in zip(utilities, self.probabilities))
        tail = weighted_lower_tail_cvar(utilities, self.probabilities, self.tail_mass)
        value = (1. - self.risk_weight) * mean + self.risk_weight * tail
        return TailCoverScore(mean, tail, value)


@dataclass(frozen=True)
class TailCoverStep:
    action: tuple
    score: TailCoverScore


@dataclass(frozen=True)
class TailCoverResult:
    action: tuple
    score: TailCoverScore
    baseline_score: TailCoverScore
    mean_floor: float
    iterations: int
    candidate_evaluations: int
    termination: str
    history: tuple


def improve_tail_cover(objective, candidate_buckets, baseline_action, *,
                       mean_slack=0., max_iterations=20, tolerance=1e-12):
    """Bounded deterministic best one-track exchanges, starting at baseline.

    The baseline is always retained as a feasible fallback. Every accepted
    action must increase J by more than tolerance AND satisfy the fixed mean
    floor baseline_mean-slack. Candidate tuples break exact objective ties
    lexically. Repeated states across different tracks are permitted, as in the
    existing Q planner. No multi-track/global/trajectory guarantee is asserted.
    """
    if not isinstance(objective, TailAwareCoverObjective):
        raise ValueError('TailAwareCoverObjective required')
    slack, tolerance = _finite(mean_slack, 'mean slack'), _finite(tolerance, 'tolerance')
    if not 0 <= slack <= 1 or tolerance < 0:
        raise ValueError('Slack in [0,1] and nonnegative tolerance required')
    if (isinstance(max_iterations, bool) or not isinstance(max_iterations, Integral)
            or max_iterations < 0):
        raise ValueError('Nonnegative integer iteration limit required')
    buckets = tuple(tuple(sorted({_index(v, 'candidate state') for v in bucket}))
                    for bucket in candidate_buckets)
    if (not buckets or any(not bucket for bucket in buckets)
            or any(v not in objective.signatures for bucket in buckets for v in bucket)):
        raise ValueError('Nonempty feasible buckets of known candidate states required')
    action = objective.action(baseline_action)
    if len(action) != len(buckets) or any(v not in bucket for v, bucket in zip(action, buckets)):
        raise ValueError('Baseline must choose one feasible candidate per track')
    baseline = score = objective.score(action)
    floor = baseline.mean - slack
    history = [TailCoverStep(action, score)]
    evaluations, termination = 0, 'iteration_limit'
    for _ in range(int(max_iterations)):
        best = None
        for j, bucket in enumerate(buckets):
            for state in bucket:
                if state == action[j]:
                    continue
                proposal = action[:j] + (state,) + action[j+1:]
                candidate = objective.score(proposal)
                evaluations += 1
                if candidate.mean < floor or candidate.objective <= score.objective + tolerance:
                    continue
                key = (-candidate.objective, proposal)
                if best is None or key < best[0]:
                    best = key, proposal, candidate
        if best is None:
            termination = 'no_improving_exchange'
            break
        _, action, score = best
        # Enforce the full accepted action, rather than an incremental proxy.
        if score.mean < floor or any(v not in b for v, b in zip(action, buckets)):
            raise RuntimeError('Accepted action violated mean floor or track feasibility')
        history.append(TailCoverStep(action, score))
    return TailCoverResult(action, score, baseline, floor, len(history)-1,
                           evaluations, termination, tuple(history))
