"""Checks for the new diagnostic, independent of the large historical artifacts."""
import unittest
from copy import deepcopy
from experiments.report_boundary_audit import cut_public, predictions, utility, xy

class BoundaryAuditTests(unittest.TestCase):
    def setUp(self):
        self.public={'query_categories':['cafe'],'events':[
          {'timestamp_s':20.*i,'candidates':[{'lat':40.,'lon':116.+.001*i}]} for i in range(12)]}
    def test_cuts_and_input_immutability(self):
        original=deepcopy(self.public)
        a,ids=cut_public(self.public,'S9',2)
        self.assertEqual(ids,list(range(2,12)))
        b,ids=cut_public(self.public,'S10',4)
        self.assertEqual(ids,list(range(8)))
        a['events'][0]['candidates'][0]['lat']=0
        self.assertEqual(self.public,original)
        self.assertEqual(cut_public(self.public,'S9',0)[0],original)
    def test_linear_extrapolation_uses_public_time(self):
        pred=predictions(self.public,'S9',40.,60.)
        expected=xy(40.,115.997,40.)
        for a,b in zip(pred['linear_boundary'],expected):self.assertAlmostEqual(a,b,places=6)
        pred=predictions(self.public,'S10',40.,60.)
        for a,b in zip(pred['linear_boundary'],xy(40.,116.014,40.)):self.assertAlmostEqual(a,b,places=6)
    def test_removed_queries_remain_in_denominator(self):
        r={'public':self.public,'utility':{'poi_rows':[
            {'category':'cafe','reference':['a','b'],'returned':['a'],'recall':.5} for _ in range(12)]}}
        result=utility(r,list(range(8)))
        self.assertEqual(result['recall_retained'],.5)
        self.assertAlmostEqual(result['recall_all_original_queries'],1/3)
        self.assertEqual(result['original_evaluable_queries'],12)
    def test_prediction_has_no_label_dependency(self):
        p=deepcopy(self.public);p['evaluator_label']=[40.,116.]
        a=predictions(p,'S9',40.,60.)
        p['evaluator_label']=[0.,0.]
        self.assertEqual(a,predictions(p,'S9',40.,60.))
    def test_rejects_unregistered_cut(self):
        with self.assertRaises(ValueError):cut_public(self.public,'S9',3)
        with self.assertRaises(ValueError):cut_public(self.public,'S3',2)
if __name__=='__main__':unittest.main()
