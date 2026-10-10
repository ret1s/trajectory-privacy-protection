"""Method-adapted endpoint attackers with permutation-invariant public inputs.

Features use only public timestamps/coordinates and public map/history. Fitting
labels are supplied separately from disjoint shadow families. Neither endpoint
truth nor internal Z, spent budget, source timestamps or candidate tags enter
the public feature API. The sequence learner adds resampled whole-window set
statistics to the existing aggregate attacker, instead of assuming persistent
candidate identities.
"""
import numpy as np

from evaluation.live_comparison_attacks import (
    LearnedAttack, descriptors, features_and_geometry, public_arrays,
    viterbi,
)
from evaluation.live_comparison_endpoint_attacks import extrapolations


def endpoint_features(events, scenario, rn, history, *, observable_close_s=None):
    if scenario not in ('S9', 'S10'):
        raise ValueError('S9 or S10 endpoint target required')
    x, geometry = features_and_geometry(events, scenario, rn, history)
    geometry.update(extrapolations(events, scenario, rn, history))
    xy, times = public_arrays(events, rn)
    first_time = float(events[0]['timestamp_s'])
    close = float(events[-1]['timestamp_s'] if observable_close_s is None else observable_close_s)
    if not np.isfinite(close) or close < float(events[-1]['timestamp_s']):
        raise ValueError('Public close must follow last release')
    x = np.c_[x, [[first_time/60., close/60., (close-float(events[-1]['timestamp_s']))/60.]]]
    # Unlike blind 30/60/120 s guesses, this bank uses the permitted observable
    # session clock, including original delay/warmup's startup and final gap.
    first = scenario == 'S9'
    streams = {'centroid': np.asarray([p.mean(axis=0) for p in xy]),
               'median': np.asarray([np.median(p, axis=0) for p in xy]),
               'viterbi': viterbi(xy, times, rn, history, True)}
    target_time = -first_time if first else close-first_time
    for name, points in streams.items():
        for count in (2, 3, 6):
            ids = np.arange(min(count, len(times))) if first else np.arange(max(0, len(times)-count), len(times))
            design = np.c_[np.ones(len(ids)), times[ids]]
            fit = np.linalg.lstsq(design, points[ids], rcond=None)[0]
            geometry[f'public_boundary_{name}_ols{count}'] = (np.array([1., target_time])@fit)[None, :]
    sets, _ = descriptors(xy, times)
    # Use fixed fractions of the visible sequence; no hidden original index.
    samples = np.rint(np.linspace(0, len(times)-1, 8)).astype(int)
    sequence = np.r_[x.ravel(), sets[samples].ravel(), times[samples]/60.][None, :]
    return x, sequence, geometry


class EndpointShadowBank:
    def __init__(self, aggregate, sequence, targets):
        self.aggregate = LearnedAttack(np.asarray(aggregate), np.asarray(targets))
        self.sequence = LearnedAttack(np.asarray(sequence), np.asarray(targets))

    def predictions(self, events, scenario, rn, history, *, observable_close_s=None):
        x, sequence, geometry = endpoint_features(events, scenario, rn, history,
                                                 observable_close_s=observable_close_s)
        return dict(geometry, **{
            'shadow_'+name: value for name, value in self.aggregate.predict(x).items()
        }, **{
            'sequence_'+name: value for name, value in self.sequence.predict(sequence).items()
        })


def family_mean(rows, key):
    families = sorted({r['family_id'] for r in rows})
    return float(np.mean([np.mean([r[key] for r in rows if r['family_id']==f])
                          for f in families])) if families else None


def select_attackers(rows):
    """Selection-only bank; independent decoder per declared loss."""
    names = sorted(set.intersection(*(set(r['errors']) for r in rows)))
    maes = {name: family_mean([dict(r, value=r['errors'][name]) for r in rows], 'value')
            for name in names}
    selected = {'mae': min(names, key=lambda n: (maes[n], n))}
    for radius in (50, 100, 200, 500):
        scores = {name: family_mean([dict(r, value=float(r['errors'][name] <= radius))
                                    for r in rows], 'value') for name in names}
        selected['hit'+str(radius)] = min(names, key=lambda n: (-scores[n], maes[n], n))
    return selected
