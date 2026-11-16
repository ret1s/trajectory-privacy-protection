"""Boundary and integrity tests for the read-only thesis argument audit."""
from copy import deepcopy
from fractions import Fraction

import unittest

from experiments.audit_thesis_arguments_20261116 import (
    OUT, CACHES, PURPOSES, check, family_values, fraction_summary,
    mask_indices, validate_cohort, render_latex,
)


def valid_family():
    cell = {'recall5': 0.8, 'defined_categories': 1}
    row = {c: {p: dict(cell) for p in PURPOSES} for c in CACHES}
    return {'trips': [{'truth': [{'t': 0}], 'status': {'geoi_bq': 'ok'},
                       'utility': {'geoi_bq': [row]}}]}


def complete_cohort():
    families = []
    for i in range(48):
        split = 'train' if i < 20 else 'selection' if i < 28 else 'test'
        trips = [{'session_id': f'f{i}-s{s}-d{d}', 'native_session_id': f'f{i}-s{s}', 'draw': d}
                 for s in (7, 8) for d in (1, 2, 3)]
        families.append({'family_id': f'f{i}', 'split': split, 'trips': trips})
    return families



class ThesisArgumentAuditTests(unittest.TestCase):
    def test_latex_values_follow_audited_payload(self):
        data={'coverage':{'retained_test_draws':2},'catalogue_size':10,
              'saturation':{'first_packet':{'mean_exact':'15/2','min':7,'max':8},
                            'final_session_cache':{'mean_exact':'19/2','min':9,'max':10}},
              'cap':{'maximal_reads_in_600s':11,'maximal_spend_in_600s':21,'cap_units':23}}
        text=render_latex(data)
        self.assertIn('toàn bộ 2 TEST',text)
        self.assertIn('7,5 / 10 & 7--8',text)
        self.assertIn('9,5 / 10 & 9--10',text)
        self.assertNotIn('380,2',text)
        self.assertNotIn('@FIRST@',text)

    def test_packbits_bit_order_and_padding(self):
        assert mask_indices('8100', 9) == {0, 7}
        assert mask_indices('0080', 9) == {8}
        assert mask_indices('0000', 9) == set()
        with self.assertRaisesRegex(ValueError, 'padding'): mask_indices('0001', 9)
        with self.assertRaisesRegex(ValueError, 'byte'): mask_indices('ff', 9)
        with self.assertRaises(ValueError): mask_indices(None, 9)


    def test_exact_summary_and_empty_is_not_success(self):
        result = fraction_summary([379, 383], 418)
        assert result['mean_exact'] == '381'
        assert result['catalogue_fraction_exact'] == str(Fraction(381, 418))
        with self.assertRaisesRegex(ValueError, 'Empty'): fraction_summary([])


    def test_missing_or_failure_never_success(self):
        family = valid_family()
        assert family_values(family, 'geoi_bq', 'session')['macro'] == 0.8
        family['trips'][0]['status']['geoi_bq'] = 'failed'
        with self.assertRaisesRegex(ValueError, 'failed'): family_values(family, 'geoi_bq', 'session')
        family = valid_family(); family['trips'][0]['utility'] = {}
        with self.assertRaisesRegex(ValueError, 'Missing utility'): family_values(family, 'geoi_bq', 'session')


    def test_defined_and_undefined_reference_separated(self):
        family = valid_family()
        cell = family['trips'][0]['utility']['geoi_bq'][0]['session'][PURPOSES[0]]
        cell.update(recall5=None, defined_categories=0)
        assert family_values(family, 'geoi_bq', 'session')[PURPOSES[0]] is None
        cell['defined_categories'] = 1
        with self.assertRaisesRegex(ValueError, 'N/A'): family_values(family, 'geoi_bq', 'session')


    def test_cohort_requires_every_retained_draw_and_family(self):
        families = complete_cohort()
        assert len(validate_cohort(families)) == 20
        broken = deepcopy(families); broken[-1]['trips'].pop()
        with self.assertRaisesRegex(ValueError, 'all three'): validate_cohort(broken)
        broken = deepcopy(families); broken[-1]['trips'][-1]['draw'] = 2
        with self.assertRaisesRegex(ValueError, 'inventory'): validate_cohort(broken)
        with self.assertRaisesRegex(ValueError, '48-family'): validate_cohort(families[:-1])


    def test_saved_audit_check_is_read_only_and_scoped(self):
        paths = list(OUT.glob('*'))
        before = {p: (p.read_bytes(), p.stat().st_mtime_ns) for p in paths}
        result = check()
        assert result['coverage']['retained_test_draws'] == 120
        assert result['saturation']['first_packet']['mean_exact'] == '1901/5'
        assert result['saturation']['final_session_cache']['mean_exact'] == '3327/8'
        assert result['cap']['runs_reaching_cap'] == 0
        assert result['cap']['maximal_spend_in_600s'] == 21
        control = result['offline_information_control']
        assert 'NOT a measured' in control['status']
        assert control['defined_reference_recall_exact'] == '1'
        assert control['setup_and_catalogue_acquisition_cost'] is None
        assert result['gamma0_comparison']['CI_recomputed'] is False
        assert before == {p: (p.read_bytes(), p.stat().st_mtime_ns) for p in paths}

if __name__ == '__main__':
    unittest.main()
