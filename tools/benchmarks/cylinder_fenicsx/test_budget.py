"""Engineering margins derive only from synthetic external references, never Eqiora."""
import copy
import unittest
import numpy as np
from budget import compare, freeze_budget, residual_ceiling, check_falsifiers
from topology import ComparisonError
from test_topology import fixture


def reference():
    return {'fields': {'velocity_vertex': np.tile([.2, 0.], (6, 1)),
                       'velocity_cell_bubble': np.tile([.01, -.02], (4, 1)),
                       'pressure_vertex': np.full(6, .005)},
            'force_on_domain': [-.01, -.0001], 'force_on_selection': [.01, .0001], 'scaled_residual_inf': 0.}


class BudgetTests(unittest.TestCase):
    def test_predeclared_floor_and_falsifiers(self):
        references = {key: reference() for key in ('A', 'B', 'C')}
        control = residual_ceiling(40, 10., 2., 1.)
        frozen = freeze_budget(fixture(), references, control)
        self.assertAlmostEqual(frozen['metrics']['velocity']['tolerance'], np.sqrt(np.finfo(float).eps)*.3)
        compare(fixture(), reference(), reference(), frozen)
        for mutation in ('bubble', 'pressure_shift', 'force_sign', 'residual'):
            candidate = reference()
            if mutation == 'bubble': candidate['fields']['velocity_cell_bubble'][:] = 0.
            if mutation == 'pressure_shift': candidate['fields']['pressure_vertex'] += .001*.3/.41
            if mutation == 'force_sign': candidate['force_on_selection'][0] *= -1
            if mutation == 'residual': candidate['scaled_residual_inf'] = control['absolute_scaled_residual_ceiling']*2
            with self.subTest(mutation=mutation):
                with self.assertRaises(ComparisonError):
                    compare(fixture(), candidate, reference(), frozen)

    def test_force_pair_and_both_resolved_lift_falsifiers(self):
        references = {key: reference() for key in ('A', 'B', 'C')}
        for row in references.values():
            row['force_on_selection'][1] = .001
            row['force_on_domain'][1] = -.001
        frozen = freeze_budget(fixture(), references, residual_ceiling(40, 10., 2., 1.))
        rejected = check_falsifiers(fixture(), references['A'], frozen)
        self.assertIn('zero_lift', rejected); self.assertIn('reverse_lift', rejected)
        candidate = reference(); candidate['force_on_domain'][0] *= -1
        with self.assertRaises(ComparisonError):
            compare(fixture(), candidate, references['A'], frozen)

    def test_inconclusive_is_not_a_larger_allowance(self):
        references = {key: reference() for key in ('A', 'B', 'C')}
        references['B']['fields']['pressure_vertex'] *= 2
        with self.assertRaises(ComparisonError):
            freeze_budget(fixture(), references, residual_ceiling(40, 10., 2., 1.))
        for key in references: references[key]['fields']['velocity_cell_bubble'][:] = 0
        with self.assertRaises(ComparisonError):
            freeze_budget(fixture(), references, residual_ceiling(40, 10., 2., 1.))

    def test_candidate_does_not_modify_frozen_budget(self):
        references = {key: reference() for key in ('A', 'B', 'C')}
        frozen = freeze_budget(fixture(), references, residual_ceiling(40, 10., 2., 1.))
        before = copy.deepcopy(frozen)
        candidate = reference(); candidate['force_on_selection'][0] = 100.
        with self.assertRaises(ComparisonError):
            compare(fixture(), candidate, reference(), frozen)
        self.assertEqual(before, frozen)


if __name__ == '__main__':
    unittest.main()
