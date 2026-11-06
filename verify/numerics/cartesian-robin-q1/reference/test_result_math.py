"""Pure rational shared-result mathematics; no public observation schema assumed."""
from fractions import Fraction as F
import unittest

import robin as r


class ResultMathTests(unittest.TestCase):
    def test_coefficient_checks_need_no_private_reactions(self):
        for case in r.CASES:
            ref = r.reference(case)
            values = r._coefficient_measurements(ref, ref['exact'])
            self.assertTrue(all(r._coefficient_checks(values, ref['bounds']).values()))
            self.assertEqual(values['reference_free_residual_inf'], 0)
            self.assertEqual(values['coefficient_inf_error'], 0)
            bad = list(ref['exact'])
            bad[4] += 1
            values = r._coefficient_measurements(ref, bad)
            self.assertFalse(any(r._coefficient_checks(values, ref['bounds']).values()))

    def test_exact_affine_functional_and_fixed_ceiling(self):
        ref = r.reference('changed_geometry_material_cooling')
        weights = tuple(F(i-4, 3) for i in range(9))
        constant = F(-7, 5)
        expected = sum((t*v for t,v in zip(weights, ref['exact'])), constant)
        result = r._affine_functional(ref, ref['exact'], expected, weights, constant)
        l1 = sum(abs(t) for t in weights)
        self.assertEqual(result['value_from_coefficients'], expected)
        self.assertEqual(result['analytic_target'], expected)
        self.assertEqual(result['arithmetic_ceiling'], r.ETA*(l1*(ref['bounds']['U']+ref['bounds']['E'])+abs(constant)))
        self.assertEqual(result['target_allowance'], l1*ref['bounds']['E']+result['arithmetic_ceiling'])
        self.assertTrue(all(result['checks'].values()))
        bad = r._affine_functional(ref, ref['exact'], expected+10**100, weights, constant)
        self.assertFalse(any(bad['checks'].values()))
        self.assertEqual(result['arithmetic_ceiling'], bad['arithmetic_ceiling'])
        self.assertEqual(result['target_allowance'], bad['target_allowance'])

    def test_functional_recovery_and_target_gates_are_independent(self):
        ref = r.reference('robin')
        weights = (F(1),)+(F(0),)*8
        actual = ref['exact'][0]
        base = r._affine_functional(ref, ref['exact'], actual, weights)
        offset = (base['arithmetic_ceiling']+base['target_allowance'])/2
        bad_recovery = r._affine_functional(ref, ref['exact'], actual+offset, weights)
        self.assertTrue(bad_recovery['checks']['analytic_target'])
        self.assertFalse(bad_recovery['checks']['coefficient_functional'])
        shifted = list(ref['exact'])
        shifted[0] += 1
        bad_target = r._affine_functional(ref, shifted, actual+1, weights)
        self.assertTrue(bad_target['checks']['coefficient_functional'])
        self.assertFalse(bad_target['checks']['analytic_target'])

    def test_zero_and_constant_functionals_and_nonrational_weights(self):
        ref = r.reference('insulated')
        constant = F(17, 3)
        result = r._affine_functional(ref, ref['exact'], constant, (F(0),)*9, constant)
        self.assertEqual(result['weights_l1'], 0)
        self.assertEqual(result['arithmetic_ceiling'], r.ETA*constant)
        self.assertEqual(result['target_allowance'], result['arithmetic_ceiling'])
        with self.assertRaises(r.ReferenceError):
            r._affine_functional(ref, ref['exact'], constant, (0.0,)*9)
        with self.assertRaises(r.ReferenceError):
            r._affine_functional(ref, ref['exact'], constant, (F(0),)*8)


if __name__ == '__main__':
    unittest.main()
