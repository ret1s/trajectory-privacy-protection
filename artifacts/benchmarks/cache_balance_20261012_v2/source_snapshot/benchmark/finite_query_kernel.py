"""Exact finite centered tilt after Geo-I; no exp/float probabilities.

p_a=w_a[1+alpha(z_a-E_w z)], alpha=epsilon/(2+epsilon).
For ANY bounded private integer scores, p_a/p'_a <= 1+epsilon <= exp(epsilon).
The integer total is PUBLIC and independent of scores. Default randomness uses
secrets.randbelow; its uniform/fresh-bit assumption remains explicit. Seeded
simulator streams do not constitute an information-theoretic RNG certificate.
"""
from fractions import Fraction
import math
import secrets

import numpy as np


def _positive_int(value, name):
    if isinstance(value, (bool,np.bool_)) or not isinstance(value,(int,np.integer)) or value < 1:
        raise ValueError(f'Positive integer {name} required')
    return int(value)


def quantize_scores(values, resolution=65536):
    """Total bounded private transformation; relative scores preserve ordering.

    Invalid numerical scores map to bounded neutral values, never a private
    refetch/stop. The privacy proof needs ONLY the integer range, not floating
    accuracy or calibration. Utility claims refer to the resulting exact z.
    """
    resolution=_positive_int(resolution,'resolution')
    values=np.asarray(values,dtype=float)
    if values.ndim!=1 or not values.size:
        raise ValueError('Nonempty private score vector required')
    values=np.nan_to_num(values,nan=0.,posinf=1.,neginf=0.)
    low,high=float(values.min()),float(values.max())
    if high==low:return tuple(0 for _ in values)
    # Subtracting extreme API values can overflow; bounded inputs are the
    # normal path. Scaling first keeps even malformed finite scores total.
    scale=max(abs(low),abs(high),1.)
    v=values/scale;lo,hi=v.min(),v.max()
    if hi==lo:return tuple(0 for _ in values)
    return tuple(int(x) for x in np.clip(np.floor((v-lo)/(hi-lo)*resolution),0,resolution))


def simulator_randbelow(rng):
    """Unbiased rejection IF rng.bytes is uniform; PCG remains simulator-only."""
    def draw(n):
        n=_positive_int(n,'public draw bound')
        bits=n.bit_length();size=(bits+7)//8;mask=(1<<bits)-1
        while True:
            value=int.from_bytes(rng.bytes(size),'big')&mask
            if value<n:return value
    return draw


class FiniteCenteredKernel:
    def __init__(self, masses, epsilon, *, resolution=65536):
        self.masses=tuple(_positive_int(m,'public mass') for m in masses)
        if not self.masses:raise ValueError('Common public support required')
        self.resolution=_positive_int(resolution,'resolution')
        if isinstance(epsilon,(bool,np.bool_)):raise ValueError('Nonnegative rational allowance required')
        self.epsilon=Fraction(epsilon)
        if self.epsilon<0:raise ValueError('Nonnegative rational allowance required')
        self.alpha=self.epsilon/(2+self.epsilon)
        self.M=sum(self.masses)
        self.public_total=self.alpha.denominator*self.M*self.M*self.resolution

    def integer_weights(self, scores):
        f=tuple(scores)
        if (len(f)!=len(self.masses) or any(isinstance(v,(bool,np.bool_)) or
                not isinstance(v,(int,np.integer)) or not 0<=v<=self.resolution for v in f)):
            raise ValueError('Matched bounded private integer scores required')
        weighted=sum(m*int(v) for m,v in zip(self.masses,f))
        n,d=self.alpha.numerator,self.alpha.denominator
        weights=tuple(m*(d*self.M*self.resolution+n*(self.M*int(v)-weighted)) for m,v in zip(self.masses,f))
        assert min(weights)>0 and sum(weights)==self.public_total
        return weights

    def probabilities(self,scores):
        return tuple(Fraction(n,self.public_total) for n in self.integer_weights(scores))

    def sample(self,scores,*,randbelow=None):
        draw=(secrets.randbelow if randbelow is None else randbelow)(self.public_total)
        if isinstance(draw,bool) or not isinstance(draw,int) or not 0<=draw<self.public_total:
            raise ValueError('Uniform integer source returned out-of-range draw')
        # Scan all actions: number of Python iterations is public. This is not
        # a hardware constant-time claim; logical packet timing stays public.
        cumulative=0;selected=0
        for i,weight in enumerate(self.integer_weights(scores)):
            if draw>=cumulative:selected=i
            cumulative+=weight
        return selected

    def utility_certificate(self,scores,*,planning_scores=None):
        f=tuple(scores);self.integer_weights(f)
        z=[Fraction(int(v),self.resolution) for v in f]
        w=[Fraction(m,self.M) for m in self.masses]
        mean=sum(a*b for a,b in zip(w,z))
        variance=sum(a*(b-mean)**2 for a,b in zip(w,z))
        gain=self.alpha*variance
        result=dict(public_base_score=float(mean),quantized_score_variance=float(variance),
                    exact_expected_quantized_score=float(mean+gain),
                    guaranteed_quantized_score_gain=float(gain),
                    score_status='exact rational quantized PLANNING score; not actual Recall')
        if planning_scores is not None:
            values=tuple(planning_scores)
            if len(values)!=len(f) or not all(math.isfinite(v) for v in values):
                raise ValueError('Matched finite planning scores required')
            v=[Fraction.from_float(float(x)) for x in values]
            base=sum(a*b for a,b in zip(w,v))
            expected=sum(a*b for a,b in zip(self.probabilities(f),v))
            result.update(public_base_planning_score=float(base),expected_planning_score=float(expected),
                exact_planning_score_gain=str(expected-base),planning_score_gain=float(expected-base),
                monotone_planning_gain_certified=expected>=base)
        return result
