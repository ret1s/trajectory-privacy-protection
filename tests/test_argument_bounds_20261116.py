"""Exact, dependency-free tests for additional thesis argument corollaries."""
from fractions import Fraction as Q
from itertools import product
import unittest

from evaluation.argument_bounds_20261116 import (
    centered_kernel, conditional_product_channel, estimator_audit, floor_lower,
    gain_audit, posterior_event_audit, posterior_upper, validate_channel,
    worked_examples,
)


class ArgumentBoundsTests(unittest.TestCase):
    def test_all_private_and_output_events(self):
        channel=((Q(1,2),Q(1,3),Q(1,6)),(Q(1,3),Q(1,3),Q(1,3)),
                 (Q(1,4),Q(1,2),Q(1,4)))
        validate_channel(channel,2)
        self.assertEqual(posterior_event_audit((Q(1,2),Q(1,3),Q(1,6)),channel,2),24)

    def test_estimator_selected_sets_exhaustively(self):
        channel=((Q(2,3),Q(1,3)),(Q(1,3),Q(2,3)))
        sets=((),(0,),(1,),(0,1))
        for first,second in product(sets,repeat=2):
            result=estimator_audit((Q(2,5),Q(3,5)),channel,(first,second),2)
            self.assertLessEqual(result['success'],result['upper'])
        perfect=estimator_audit((Q(1,2),Q(1,2)),channel,((0,),(1,)),2)
        self.assertEqual(perfect['success'],perfect['upper'])

    def test_conditional_two_channel_correlated_inputs(self):
        channel=((Q(2,3),Q(1,3)),(Q(1,3),Q(2,3)))
        joint=conditional_product_channel(channel,(channel,tuple(reversed(channel))),2,2)
        prior=(Q(2,5),Q(1,10),Q(1,10),Q(2,5))
        self.assertEqual(posterior_event_audit(prior,joint,4),64)
        self.assertGreater(posterior_upper(Q(1,2),4),posterior_upper(Q(1,2),2))

    def test_raw_channel_is_not_covered_by_finite_bound(self):
        with self.assertRaises(ValueError):validate_channel(((1,0),(0,1)),1000)
        with self.assertRaises(ValueError):
            conditional_product_channel(((Q(2,3),Q(1,3)),(Q(1,3),Q(2,3))),
                                        (((1,0),(0,1)),)*2,2,2)

    def test_gamma_zero_ignores_all_bounded_score_vectors(self):
        w=(Q(1,6),Q(1,3),Q(1,2))
        for z in product((Q(0),Q(1,2),Q(1)),repeat=3):
            self.assertEqual(centered_kernel(w,z,0),w)

    def test_positive_score_gain_can_have_negative_true_gain(self):
        r=gain_audit((Q(1,2),)*2,(1,0),(Q(2,5),Q(9,10)),Q(1,2))
        self.assertGreater(r['score_gain'],0)
        self.assertEqual(r['true_gain'],-Q(1,40))

    def test_nonvacuous_worked_gain_and_floor(self):
        x=worked_examples()
        self.assertEqual(x['gain']['probabilities'],(Q(11,20),Q(9,20)))
        self.assertEqual(x['gain']['true_gain'],Q(3,200))
        self.assertEqual(x['gain']['true_gain_lower'],Q(1,125))
        self.assertEqual(x['floor']['reference_mean'],Q(903,1000))
        self.assertEqual(x['floor']['actual_mean'],Q(1103,1250))
        self.assertEqual(x['floor']['lower'],Q(17,20))
        self.assertLess(x['two_sessions_balanced_upper'],Q(855925,1000000))
        self.assertGreater(x['two_sessions_balanced_upper'],Q(8559,10000))

    def test_boundaries_and_invalid_calibration(self):
        self.assertEqual(posterior_upper(0,2),0)
        self.assertEqual(posterior_upper(1,2),1)
        self.assertEqual(floor_lower(Q(1,10),Q(1,5),Q(1,5)),0)
        for pi,ratio in [(-1,2),(2,2),(Q(1,2),Q(1,2))]:
            with self.assertRaises(ValueError):posterior_upper(pi,ratio)
        with self.assertRaises(ValueError):
            gain_audit((Q(1,2),)*2,(1,0),(1,0),1,proxy=(0,1),eta=Q(1,10))


if __name__=='__main__':unittest.main()
