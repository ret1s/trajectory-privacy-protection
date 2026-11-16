"""Exact finite oracles for thesis corollaries; no defense or benchmark changes.

These checks illustrate proved formulas on finite channels. They do not certify
an implementation's upstream sampler, entropy, prior or real-data calibration.
Run with: python3 -m evaluation.argument_bounds_20261116
"""
from fractions import Fraction as Q
from itertools import product
import json


def rational(value):
    if isinstance(value, bool):
        raise ValueError('Boolean is not a probability')
    return value if isinstance(value, Q) else Q(str(value))


def pmf(values):
    values = tuple(map(rational, values))
    if not values or min(values) < 0 or sum(values) != 1:
        raise ValueError('A nonnegative, normalized finite PMF is required')
    return values


def subsets(n):
    return tuple(tuple(i for i in range(n) if mask & (1 << i))
                 for mask in range(1 << n))


def posterior_upper(prior, ratio):
    prior, ratio = rational(prior), rational(ratio)
    if not 0 <= prior <= 1 or ratio < 1:
        raise ValueError('Prior in [0,1] and ratio >=1 required')
    return ratio * prior / (1 + (ratio - 1) * prior)


def validate_channel(channel, ratio, *, enumerate_events=True):
    rows = tuple(pmf(row) for row in channel)
    ratio = rational(ratio)
    if not rows or ratio < 1 or any(len(row) != len(rows[0]) for row in rows):
        raise ValueError('Rectangular channel and ratio >=1 required')
    events = subsets(len(rows[0])) if enumerate_events else tuple((i,) for i in range(len(rows[0])))
    for first, second, event in product(rows, rows, events):
        if sum(first[i] for i in event) > ratio * sum(second[i] for i in event):
            raise ValueError('Declared likelihood-ratio bound is violated')
    return rows


def posterior_event_audit(prior, channel, ratio):
    """Enumerate all private events at all reachable outputs using exact Bayes."""
    prior = pmf(prior)
    channel = validate_channel(channel, ratio)
    if len(prior) != len(channel):
        raise ValueError('Prior and private channel domain must match')
    checked = 0
    for output in range(len(channel[0])):
        evidence = sum(p * row[output] for p, row in zip(prior, channel))
        if not evidence:
            continue
        for event in subsets(len(prior)):
            pi = sum(prior[x] for x in event)
            posterior = sum(prior[x] * channel[x][output] for x in event) / evidence
            assert posterior <= posterior_upper(pi, ratio)
            assert posterior >= 1 - posterior_upper(1-pi, ratio)
            checked += 1
    return checked


def estimator_audit(prior, channel, success_sets, ratio):
    """Each output selects its own private success event (e.g. a radius ball)."""
    prior = pmf(prior)
    channel = validate_channel(channel, ratio)
    if len(prior) != len(channel) or len(success_sets) != len(channel[0]):
        raise ValueError('Matched prior, channel and estimator events required')
    events = tuple(tuple(sorted(set(event))) for event in success_sets)
    if any(x < 0 or x >= len(prior) for event in events for x in event):
        raise ValueError('Success event outside private domain')
    concentration = max(sum(prior[x] for x in event) for event in events)
    success = sum(prior[x] * channel[x][o] for o, event in enumerate(events) for x in event)
    bound = posterior_upper(concentration, ratio)
    for o, event in enumerate(events):
        evidence = sum(prior[x] * channel[x][o] for x in range(len(prior)))
        if evidence:
            posterior = sum(prior[x] * channel[x][o] for x in event) / evidence
            assert posterior <= posterior_upper(sum(prior[x] for x in event), ratio) <= bound
    assert success <= bound
    return dict(success=success, concentration=concentration, upper=bound)


def conditional_product_channel(first, second_given_first, first_ratio, second_ratio):
    """Correlated private inputs are allowed; second channel adapts to public o1.

    Output rows are indexed by (x1,x2), then columns by (o1,o2). All conditional
    second-channel rows must obey its bound at every first-output context.
    """
    first = validate_channel(first, first_ratio)
    second = tuple(validate_channel(c, second_ratio) for c in second_given_first)
    if len(second) != len(first[0]) or any(len(c) != len(second[0]) for c in second):
        raise ValueError('One matched conditional channel per first output required')
    if any(len(row) != len(second[0][0]) for c in second for row in c):
        raise ValueError('Same finite second-output alphabet required')
    joint = tuple(tuple(first[x1][o1] * second[o1][x2][o2]
                        for o1 in range(len(first[0])) for o2 in range(len(second[0][0])))
                  for x1 in range(len(first)) for x2 in range(len(second[0])))
    validate_channel(joint, rational(first_ratio) * rational(second_ratio))
    return joint


def centered_kernel(weights, scores, epsilon):
    weights = pmf(weights)
    scores = tuple(map(rational, scores))
    epsilon = rational(epsilon)
    if len(scores) != len(weights) or min(weights) <= 0 or epsilon < 0 or any(not 0 <= z <= 1 for z in scores):
        raise ValueError('Positive public weights and bounded matched scores required')
    alpha = epsilon / (2 + epsilon)
    mu = sum(w*z for w,z in zip(weights,scores))
    probabilities = tuple(w*(1+alpha*(z-mu)) for w,z in zip(weights,scores))
    pmf(probabilities)
    return probabilities


def gain_audit(weights, scores, rewards, epsilon, *, proxy=None, eta=None):
    weights = pmf(weights)
    scores, rewards = tuple(map(rational,scores)), tuple(map(rational,rewards))
    if len(rewards) != len(weights) or any(not 0 <= r <= 1 for r in rewards):
        raise ValueError('Matched bounded rewards required')
    probabilities = centered_kernel(weights,scores,epsilon)
    alpha = rational(epsilon)/(2+rational(epsilon))
    mu = sum(w*z for w,z in zip(weights,scores))
    variance = sum(w*(z-mu)**2 for w,z in zip(weights,scores))
    score_gain = sum((p-w)*z for p,w,z in zip(probabilities,weights,scores))
    true_gain = sum((p-w)*r for p,w,r in zip(probabilities,weights,rewards))
    covariance = sum(w*(z-mu)*r for w,z,r in zip(weights,scores,rewards))
    assert score_gain == alpha*variance >= 0
    assert true_gain == alpha*covariance
    result = dict(probabilities=probabilities,score_gain=score_gain,true_gain=true_gain)
    if proxy is not None:
        proxy, eta = tuple(map(rational,proxy)), rational(eta)
        if len(proxy) != len(rewards) or eta < 0 or any(abs(r-v)>eta for r,v in zip(rewards,proxy)):
            raise ValueError('The declared uniform proxy error is not valid')
        lower = alpha*(sum(w*(z-mu)*v for w,z,v in zip(weights,scores,proxy))
                       - eta*sum(w*abs(z-mu) for w,z in zip(weights,scores)))
        assert true_gain >= lower
        result['true_gain_lower'] = lower
    return result


def floor_lower(c, tau, nu):
    c, tau, nu = map(rational,(c,tau,nu))
    if any(not 0 <= x <= 1 for x in (c,tau,nu)):
        raise ValueError('Unit-interval conditional bounds required')
    return max(Q(0),c-tau-nu)


def worked_examples():
    gain = gain_audit([Q(1,2)]*2,[1,0],[Q(9,10),Q(3,5)],Q(1,2),
                      proxy=[Q(43,50),Q(31,50)],eta=Q(1,25))
    reference = (Q(1,2),Q(3,10),Q(1,5))
    actual = (Q(47,100),Q(3,10),Q(23,100))
    proxy_rewards = (Q(91,100),Q(9,10),Q(89,100))
    actual_rewards = tuple(r-Q(1,50) for r in proxy_rewards)
    reference_mean = sum(p*r for p,r in zip(reference,proxy_rewards))
    actual_mean = sum(p*r for p,r in zip(actual,actual_rewards))
    tv = sum(abs(p-q) for p,q in zip(reference,actual))/2
    proxy_error = sum(p*abs(r-v) for p,r,v in zip(actual,actual_rewards,proxy_rewards))
    assert reference_mean >= Q(9,10) and tv == Q(3,100) and proxy_error == Q(1,50)
    lower = floor_lower(Q(9,10),tv,proxy_error)
    assert lower == Q(17,20) and actual_mean >= lower
    ratio = Q(2437372,1000000)  # Existing outward upper bound, not a new bound proof.
    return dict(gain=gain, floor=dict(reference_mean=reference_mean,actual_mean=actual_mean,
                tv=tv,proxy_error=proxy_error,lower=lower),
                one_session_balanced_upper=posterior_upper(Q(1,2),ratio),
                two_sessions_balanced_upper=posterior_upper(Q(1,2),ratio**2))


def main():
    binary = ((Q(2,3),Q(1,3)),(Q(1,3),Q(2,3)))
    joint = conditional_product_channel(binary,(binary,tuple(reversed(binary))),2,2)
    prior = (Q(2,5),Q(1,10),Q(1,10),Q(2,5))  # Correlated, not product marginals.
    checks = posterior_event_audit(prior,joint,4)
    estimator_audit(prior,joint,((0,),(1,),(2,),(3,)),4)
    def encode(value):
        if isinstance(value,Q):return str(value)
        if isinstance(value,dict):return {k:encode(v) for k,v in value.items()}
        if isinstance(value,(tuple,list)):return [encode(x) for x in value]
        return value
    print(json.dumps(dict(status='pass',scope='finite exact illustrations, not a deployment certificate',
                         posterior_event_checks=checks,examples=encode(worked_examples())),indent=2))


if __name__=='__main__':main()
