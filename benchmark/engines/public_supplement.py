"""Append a deployment-public service cover without changing adaptive state.

This buys service at an explicit extra query cost. Fixed tracks are public and
removable, so the coordinate experiment is information-equivalent to the parent
conditional on the public configuration/clock. No additional privacy claim.
"""
from copy import deepcopy
from dataclasses import replace
import numpy as np
from core.demo_protocol import OutputKind, PublicCandidate
from evaluation.public_cover import fit_public_cover


def fixed_candidates(coordinates):
    return [PublicCandidate(f'public_fixed_{i:04d}', *point).to_dict()
            for i, point in enumerate(coordinates)]


def append_public_view(view, coordinates):
    """A deterministic embedding; preserve IDs, ordering, timestamps and windows."""
    result = deepcopy(view); fixed = fixed_candidates(coordinates)
    ids = {c['candidate_id'] for c in fixed}
    for event in result['events']:
        if ids.intersection(c['candidate_id'] for c in event['candidates']):
            raise ValueError('Fixed public IDs collide with parent IDs')
        event['candidates'].extend(deepcopy(fixed))
    return result


def remove_public_view(view, coordinates):
    """Attacker-known inverse. Reject a changed suffix instead of hiding it."""
    result = deepcopy(view); fixed = fixed_candidates(coordinates)
    if not fixed:
        return result
    for event in result['events']:
        if len(event['candidates']) <= len(fixed) or event['candidates'][-len(fixed):] != fixed:
            raise ValueError('Expected identifiable constant public suffix')
        event['candidates'] = event['candidates'][:-len(fixed)]
    return result


class PublicServiceSupplement:
    name = 'public_service_supplement'

    def __init__(self, parent, *, public_queries=2):
        if isinstance(public_queries, bool) or int(public_queries) != public_queries or public_queries < 0:
            raise ValueError('Nonnegative integer public query count required')
        self.parent, self.public_queries = parent, int(public_queries)
        self.rn = parent.rn
        self.plan = (fit_public_cover(self.rn, parent.belief_model,
                                     parent.belief_model.context, self.public_queries) if public_queries else None)
        self.coordinates = tuple(tuple(c) for c in self.plan['coordinates']) if self.plan else ()
        self.k = parent.k+self.public_queries

    def reset(self):
        self.parent.reset()

    @property
    def spent_bound(self):
        return self.parent.spent_bound

    def protect_step(self, lat, lon, timestamp_s):
        return tuple(self.parent.protect_step(lat, lon, timestamp_s))+self.coordinates

    def protect_run(self, points):
        run = self.parent.protect_run(points)
        if not self.public_queries:
            return run
        if run.transcript.output_kind != OutputKind.DUMMY_ONLY:
            raise ValueError('Wrapper currently requires a dummy-only parent')
        fixed = tuple(PublicCandidate(f'public_fixed_{i:04d}', *c) for i, c in enumerate(self.coordinates))
        events = tuple(replace(event, candidates=event.candidates+fixed) for event in run.transcript.events)
        params = dict(run.transcript.public_parameters)
        params.update(parent_mechanism=run.transcript.mechanism, parent_k=self.parent.k,
                      k=self.k, public_supplement_queries=self.public_queries,
                      extra_privacy_reads=0, public_suffix_secret=False,
                      adaptive_state_unchanged=True,
                      privacy_scope='coordinate_information_equivalent_to_parent_given_public_configuration_and_clock')
        return replace(run, transcript=replace(run.transcript, mechanism=self.name,
                                              events=events, public_parameters=params))
