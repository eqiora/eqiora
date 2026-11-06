"""Protocol falsifiers use analytic fields, never candidate expected values."""
import unittest
import numpy as np
from accuracy import (A, K, LEVELS, ComparisonError, check_accuracy, check_convergence,
                      check_equilibrium, exact, force, stress)


def fixture(cells):
    coordinates = np.array([(i/cells, j/cells) for i in range(cells+1) for j in range(cells+1)])
    return coordinates, exact(coordinates)[0]


class AccuracyTests(unittest.TestCase):
    def test_manufactured_equilibrium(self):
        points = np.array(((.13, .27), (.31, .73), (.64, .41)))
        divergence = np.zeros_like(points)
        step = 1e-5
        for axis in range(2):
            delta = np.eye(2)[axis]*step
            divergence += (stress(points+delta)[..., :, axis]-stress(points-delta)[..., :, axis])/(2*step)
        np.testing.assert_allclose(-divergence, force(points), rtol=1e-8, atol=1e-8)

    def test_independent_integrated_norms(self):
        nodes, weights = np.polynomial.legendre.leggauss(40)
        coordinates = np.array([((x+1)/2, (y+1)/2) for x in nodes for y in nodes])
        values, gradient = exact(coordinates)
        weights = np.outer(weights, weights).reshape(-1)/4
        self.assertAlmostEqual(float(np.sum(np.sum(values**2, axis=-1)*weights)), 3*A*A/8, places=16)
        self.assertAlmostEqual(float(np.sum(np.sum(gradient**2, axis=(-1, -2))*weights)), 2*K*K*A*A, places=15)

    def test_analytic_interpolation_convergence(self):
        rows = {n: check_accuracy(*fixture(n), n) for n in LEVELS}
        check_convergence(rows)

    def test_field_falsifiers(self):
        for cells in LEVELS:
            coordinates, values = fixture(cells)
            for label, corrupt in (('zero', np.zeros_like(values)), ('negated', -values), ('swapped', values[:, ::-1])):
                with self.subTest(cells=cells, corruption=label):
                    with self.assertRaises(ComparisonError):
                        check_accuracy(coordinates, corrupt, cells)
        coordinates, values = fixture(8)
        broken = values.copy(); broken[0, 0] = 1e-5
        with self.assertRaises(ComparisonError):
            check_accuracy(coordinates, broken, 8)
        with self.assertRaises(ComparisonError):
            check_accuracy(coordinates[:-1], values[:-1], 8)
        duplicated = coordinates.copy(); duplicated[0] = duplicated[1]
        with self.assertRaises(ComparisonError):
            check_accuracy(duplicated, values, 8)

    def test_quadrature_independent_of_export_order(self):
        coordinates, values = fixture(8)
        self.assertEqual(check_accuracy(coordinates, values, 8), check_accuracy(coordinates[::-1], values[::-1], 8))

    def test_equilibrium_falsifiers(self):
        check_equilibrium((-1, 0), (1, 0))
        for reaction, body_force in (((0, 0), (0, 0)), ((1, 0), (1, 0)), ((0, -1), (1, 0))):
            with self.assertRaises(ComparisonError):
                check_equilibrium(reaction, body_force)


if __name__ == '__main__':
    unittest.main()
