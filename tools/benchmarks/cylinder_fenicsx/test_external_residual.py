"""Portable reference-operator checks use a known small matrix, not candidate data."""
import unittest
import numpy as np
from external_residual import csr_product, residual_inf, validate_operator
from topology import ComparisonError


class ExternalResidualTests(unittest.TestCase):
    def test_csr_product_including_empty_rows(self):
        # [[2,0,-1], [0,0,0], [0,3,4]] times [1,2,3].
        np.testing.assert_array_equal(csr_product([0, 2, 2, 4], [0, 2, 1, 2], [2., -1., 3., 4.], [1., 2., 3.]),
                                      [-1., 0., 18.])
        with self.assertRaises(ComparisonError):
            csr_product([0, 2], [0], [1.], [1.])
        with self.assertRaises(ComparisonError):
            csr_product(np.array(0), [0], [1.], [1.])

    def test_bad_free_basis_scaling_and_dimensions_fail(self):
        operator = {'indptr': np.array([0, 1, 2]), 'indices': np.array([0, 1]), 'values': np.array([2., 4.]),
                    'free': np.array([2, 0]), 'scales': np.array([2., .5]), 'rhs': np.array([3., 16.])}
        validate_operator(operator, canonical_size=3, expected_free=[2, 0], expected_scales=[2., .5])
        for key, value in (('indptr', np.array(0)), ('free', np.array(0)), ('free', np.array([0, 0])), ('scales', np.array([2., 1.])),
                           ('rhs', np.array([1.])), ('indices', np.array([0, 2]))):
            corrupt = dict(operator); corrupt[key] = value
            with self.assertRaises(ComparisonError):
                validate_operator(corrupt, canonical_size=3, expected_free=[2, 0], expected_scales=[2., .5])

    def test_scaling_and_free_basis_are_applied_before_residual(self):
        operator = {'indptr': np.array([0, 1, 2]), 'indices': np.array([0, 1]), 'values': np.array([2., 4.]),
                    'free': np.array([2, 0]), 'scales': np.array([2., .5]), 'rhs': np.array([3., 16.])}
        self.assertEqual(residual_inf(operator, [2., 99., 3.]), 0.)
        self.assertEqual(residual_inf(operator, [2., 99., 4.]), 1.)


if __name__ == '__main__':
    unittest.main()
