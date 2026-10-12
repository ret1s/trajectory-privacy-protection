"""Fixed-trajectory late-change bounds; not an unconditional future-behavior claim."""
from fractions import Fraction
from benchmark.finite_session_bounds import prefix_ratio


def late_change_ratio(gamma, first_affected_block, final_block):
    if (isinstance(first_affected_block,bool) or not isinstance(first_affected_block,int) or first_affected_block<1
            or isinstance(final_block,bool) or not isinstance(final_block,int) or final_block<0):raise ValueError('Public positive start block and nonnegative end block required')
    if first_affected_block>final_block:
        prefix_ratio(gamma,0)  # validate Gamma even for no affected outputs
        return Fraction(1)
    return prefix_ratio(gamma,final_block)/prefix_ratio(gamma,first_affected_block-1)


def late_change_allowance(gamma, first_affected_block, final_block):
    late_change_ratio(gamma,first_affected_block,final_block)
    if first_affected_block>final_block:return Fraction(0)
    return Fraction(gamma)*(Fraction(1,first_affected_block)-Fraction(1,final_block+1))
