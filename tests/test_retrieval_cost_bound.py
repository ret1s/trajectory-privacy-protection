from fractions import Fraction
import itertools

from benchmark.retrieval_cost_bound import necessary_records


def test_public_cache_coverage_bound_for_every_mask_on_disjoint_references():
    refs=({0,1},{2,3},{4,5})
    for bits in itertools.product((False,True),repeat=6):
        available={i for i,b in enumerate(bits) if b}
        recall=sum(Fraction(len(r&available),2) for r in refs)/3
        assert recall<=Fraction(len(available),6)
    result=necessary_records(1000,5,Fraction(95,100))
    assert 4473<=result['integer_record_lower']<=4474
    assert necessary_records(20,5,Fraction(95,100),gamma=0.)['integer_record_lower']==95
