"""Endpoint challenge using public list positions and geometric association.

No internal candidate ID, real/fake label, private reference or target GPS is
accepted by the feature API. List positions are available in the ordered view;
shuffled publication removes their persistence, while geometric association
remains available in both views. This finite bank is not an unlinkability test.
"""
import numpy as np
from scipy.optimize import linear_sum_assignment

from evaluation.endpoint_noise_attacks import endpoint_features
from evaluation.live_comparison_attacks import LearnedAttack


def ordered_arrays(events, rn):
    """Preserve the public coordinate multiset and its observed list order."""
    times = np.asarray([e['timestamp_s'] for e in events], dtype=float)
    if not len(times) or not np.isfinite(times).all() or np.any(np.diff(times) <= 0):
        raise ValueError('Strict finite public clock required')
    arrays = [np.asarray([rn.point_xy(*p) for p in e['coordinates']], dtype=float) for e in events]
    if any(p.ndim != 2 or p.shape[1] != 2 or not len(p) or not np.isfinite(p).all() for p in arrays):
        raise ValueError('Nonempty finite public coordinates required')
    if len({len(p) for p in arrays}) != 1:
        raise ValueError('This declared challenge requires a fixed public K within each trip')
    return np.asarray(arrays), times-times[0]


def canonical_tracks(points, times, *, velocity=False):
    """Public geometry-only tracks, invariant to event-local list permutations.

    Initial tracks use spatial lexicographic order. Each subsequent event is
    spatially sorted before Hungarian matching to the preceding tracks (or
    constant-velocity prediction). Sorting fixes input-order ties. Duplicate
    coordinates are retained: their geometric labels are interchangeable.
    """
    points, times = np.asarray(points, float), np.asarray(times, float)
    if points.ndim != 3 or points.shape[2] != 2 or len(points) != len(times):
        raise ValueError('Expected event × fixed K × XY and matching public times')
    if not len(points) or points.shape[1] == 0 or np.any(np.diff(times) <= 0):
        raise ValueError('Nonempty tracks and strict clock required')
    tracks = np.empty_like(points)
    tracks[0] = points[0][np.lexsort((points[0, :, 1], points[0, :, 0]))]
    for j in range(1, len(points)):
        candidates = points[j][np.lexsort((points[j, :, 1], points[j, :, 0]))]
        predicted = tracks[j-1]
        if velocity and j >= 2:
            predicted = predicted+(tracks[j-1]-tracks[j-2])*(times[j]-times[j-1])/(times[j-1]-times[j-2])
        cost = np.linalg.norm(predicted[:, None, :]-candidates[None, :, :], axis=2)
        rows, columns = linear_sum_assignment(cost)
        tracks[j, rows] = candidates[columns]
    return tracks


def track_features(points, times, aggregate):
    """Fixed-size learned features include every public track, not just a mean."""
    samples = np.rint(np.linspace(0, len(times)-1, 8)).astype(int)
    slopes = np.linalg.lstsq(np.c_[np.ones(len(times)), times], points.reshape(len(times), -1), rcond=None)[0][1]
    lengths = np.linalg.norm(np.diff(points, axis=0), axis=2).sum(axis=0)
    spread = points.std(axis=0).ravel()
    return np.r_[aggregate.ravel(), points[samples].ravel()/1000., slopes/8.,
                 lengths/1000., spread/1000., times[samples]/60.][None, :]


def ordered_endpoint_features(events, scenario, rn, history, *, observable_close_s=None):
    if scenario not in ('S9', 'S10'):
        raise ValueError('S9 or S10 endpoint target required')
    aggregate, invariant_sequence, predictions = endpoint_features(events, scenario, rn, history,
        observable_close_s=observable_close_s)
    points, times = ordered_arrays(events, rn)
    close = float(events[-1]['timestamp_s'] if observable_close_s is None else observable_close_s)
    boundary = -float(events[0]['timestamp_s']) if scenario == 'S9' else close-float(events[0]['timestamp_s'])
    first = scenario == 'S9'
    tracks = {'observed_slots': points,
              'geometry_nearest': canonical_tracks(points, times),
              'geometry_velocity': canonical_tracks(points, times, velocity=True)}
    learned = {'invariant_aggregate': aggregate, 'invariant_sequence': invariant_sequence}
    for channel, sequence in tracks.items():
        learned[channel] = track_features(sequence, times, aggregate)
        for slot in range(sequence.shape[1]):
            predictions[f'{channel}_slot{slot}_endpoint'] = sequence[[0 if first else -1], slot]
        for count in (2, 3, 6):
            ids = np.arange(min(count, len(times))) if first else np.arange(max(0, len(times)-count), len(times))
            fit = np.linalg.lstsq(np.c_[np.ones(len(ids)), times[ids]],
                                 sequence[ids].reshape(len(ids), -1), rcond=None)[0]
            # Observable session boundary and four predeclared blind horizons.
            targets = {'boundary': boundary}
            end = times[0] if first else times[-1]
            targets.update({f'horizon{h}': end+(-h if first else h) for h in (0, 30, 60, 120)})
            for horizon, at in targets.items():
                estimated = (np.asarray([1., at])@fit).reshape(-1, 2)
                for slot, xy in enumerate(estimated):
                    predictions[f'{channel}_slot{slot}_ols{count}_{horizon}'] = xy[None, :]
    return learned, predictions


class OrderedEndpointBank:
    """Method- and publication-view-adapted shadow and track sequence learners."""
    def __init__(self, features, targets):
        self.learners = {channel: LearnedAttack(np.asarray(values), np.asarray(targets))
                         for channel, values in features.items()}

    def predictions(self, events, scenario, rn, history, *, observable_close_s=None):
        features, predictions = ordered_endpoint_features(events, scenario, rn, history,
            observable_close_s=observable_close_s)
        for channel, learner in self.learners.items():
            predictions.update({f'{channel}_{name}': value for name, value in learner.predict(features[channel]).items()})
        return predictions
