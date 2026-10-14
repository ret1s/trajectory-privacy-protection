from fractions import Fraction
import numpy as np
import pytest

from benchmark.exact_public_reference import exact_reference_weights,ExactReferencePortfolio,certified_goals
from tests.test_public_pool_portfolio import objects


def test_exact_semantic_reference_mass_normalizes_defined_cases_and_full_pool():
    rn,reply,table,_,_=objects()
    exact=exact_reference_weights(reply,table.states,table.destinations)
    assert sum(exact)==1 and min(exact)>=0
    assert np.allclose([float(w) for w in exact],table.public_poi_weights,atol=1e-14,rtol=0)
    portfolio=ExactReferencePortfolio(reply,reply,table)
    assert portfolio.quality_exact(portfolio.attainable)==1
    goals=certified_goals(portfolio)
    assert all(portfolio.quality_exact(portfolio.pool(g))>=Fraction(99,100) for g in goals)
    with pytest.raises(ValueError):certified_goals(portfolio,floor=Fraction(101,100))
