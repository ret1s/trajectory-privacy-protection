"""Exact mean Recall reference masses on a DECLARED finite public domain.

Equal states, equal defined purposes, equal defined destination prototypes,
equal defined categories and equal reference items. N/A cases are excluded.
These are not the distribution of actual private GPS or private requests.
"""
from fractions import Fraction
import numpy as np

from benchmark.public_pool_portfolio import PublicPoolPortfolio
from benchmark.public_segment_scores import local_sorted_pois
from benchmark.query_purpose import MultiPurposeRoadRanking,QueryPurpose,QuerySpec


def exact_reference_weights(reference, states, destinations, *, radius_m=1000.):
    ranking=MultiPurposeRoadRanking(reference,cache_limit=512)
    states=tuple(states);destinations=tuple(destinations)
    if not states or not destinations:raise ValueError('Nonempty declared PUBLIC reference domain required')
    all_items=np.ones(len(reference.pois),bool)
    weights=[Fraction(0) for _ in reference.pois]
    for state in states:
        purposes=[]
        for purpose in QueryPurpose:
            rows=[]
            for destination in destinations if purpose==QueryPurpose.MIN_DETOUR else (None,):
                categories=[]
                for category in ranking.categories:
                    spec=QuerySpec(purpose,category,k=5,
                        radius_m=radius_m if purpose==QueryPurpose.WITHIN_RADIUS else None,
                        destination_state=destination)
                    ids=local_sorted_pois(ranking,state,all_items,spec)[:5]
                    if ids:categories.append(ids)
                if categories:rows.append(categories)
            if rows:purposes.append(rows)
        if not purposes:raise ValueError('Public reference state has no defined service')
        for rows in purposes:
            for categories in rows:
                for ids in categories:
                    mass=Fraction(1,len(states)*len(purposes)*len(rows)*len(categories)*len(ids))
                    for item in ids:weights[item]+=mass
    assert sum(weights)==1
    return tuple(weights)


class ExactReferencePortfolio(PublicPoolPortfolio):
    def __init__(self,reply,reference,table):
        if reply.pois!=reference.pois:raise ValueError('Matched catalogue required')
        exact=exact_reference_weights(reference,table.states,table.destinations)
        super().__init__(reply,public_poi_weights=[float(w) for w in exact])
        self.exact_weights=exact


def certified_goals(portfolio, *, floor=Fraction(99,100), width=24):
    floor=Fraction(floor)
    if not 0<=floor<=1:raise ValueError('Public reference floor in [0,1] required')
    goals=tuple(g for g in portfolio.goals(width=width)
                if portfolio.quality_exact(portfolio.pool(g))>=floor)
    if not goals:raise ValueError('Public response depth cannot certify the declared mean-reference floor')
    return goals
