"""Synthetic full-vector comparator tests; no production solver or output."""
import copy
import itertools
import json
import unittest
from fractions import Fraction as F

import compare as c
import generate as g


def records(lengths, n, value):
    result = []
    for index in itertools.product(range(2*n+1), repeat=len(lengths)):
        coordinates = [g.D(length)*k/(2*n) for length, k in zip(lengths, index)]
        result.append({'coordinates_m': [str(x) for x in coordinates],
                       'value_binary64_hex': float(value(index, coordinates)).hex()})
    return {'schema': 1, 'lengths_m': list(lengths), 'N': n, 'gauss_points': 4, 'records': result}


def synthetic_galerkin(lengths, n):
    # Independent tiny amplitude mathematics supplies synthetic test data only.
    # The comparator always reads and integrates every physical coefficient.
    p = g.pi()
    lam = p*p*sum((g.I(1)/(l*l) for l in lengths), g.I(0))
    axes = [g.one_dimensional(l, n, p/n, 4) for l in lengths]
    terms = [g.kron([axis[1 if i == j else 0] for j, axis in enumerate(axes)])
             for i in range(len(lengths))]
    matrix = [[sum((term[i][j] for term in terms), g.I(0)) for j in range(2**len(lengths))]
              for i in range(2**len(lengths))]
    q = [g.product(v) for v in itertools.product(*(axis[3] for axis in axes))]
    amplitude = g.solve(matrix, [lam*v for v in q])
    def value(index, _):
        if any(k in (0, 2*n) for k in index):
            return 0.0
        mode = sum((k % 2)*2**(len(lengths)-j-1) for j, k in enumerate(index))
        result = amplitude[mode]*g.product(c.sine_position(g.I(k)/2, n, p) for k in index)
        return float(g.NEAR.divide(g.NEAR.add(result.lo, result.hi), g.D(2)))
    return records(lengths, n, value)


class CompareTests(unittest.TestCase):
    def setUp(self):
        g.precision(80)

    def overlaps(self, a, b):
        self.assertLessEqual(a.lo, b.hi)
        self.assertGreaterEqual(a.hi, b.lo)

    def test_zero_vector_norms_and_budget_failure(self):
        result = c.compare(records((2, 3), 2, lambda *_: 0))
        self.overlaps(g.I(*result['continuous_error_intervals'][0]), (g.I(6)/4).sqrt())
        self.assertFalse(result['accepted'])
        self.assertFalse(all(b['budget_certified'] for b in result['budgets']))
        self.assertEqual(result['physical_coefficient_count'], 25)
        self.assertEqual(result['interior_coefficient_count'], 9)

    def test_arbitrary_full_vector_exact_polynomial_norms(self):
        lengths = (2, 3, 4)
        def bubble(_, coordinates):
            result = g.D(1)
            for length, x in zip(lengths, coordinates):
                result *= x*(length-x)
            return result
        data = records(lengths, 2, bubble)
        result = c.compare(data)
        p = g.pi()
        lam = p*p*sum((g.I(1)/(l*l) for l in lengths), g.I(0))
        exact_mass = g.product(g.I(l)**5/30 for l in lengths)
        exact_energy = sum((g.I(lengths[j])**3/3 * g.product(g.I(l)**5/30
                            for i, l in enumerate(lengths) if i != j) for j in range(3)), g.I(0))
        cross = g.product(4*g.I(l)**3/p**3 for l in lengths)
        exact_u = g.product(lengths)/8
        expected = [(exact_u-2*cross+exact_mass).sqrt(),
                    (lam*exact_u-2*lam*cross+exact_energy).sqrt()]
        for actual, oracle in zip(result['continuous_error_intervals'], expected):
            self.overlaps(g.I(*actual), oracle)
        self.assertFalse(result['accepted'])

    def test_full_operator_against_fraction_tensor_assembly(self):
        # Arbitrary coefficients excite every nodal family, not just a sine mode.
        lengths, n = (2, 3), 2
        vector = [g.I(k-4) for k in range(9)]
        actual_m, actual_a = c.operators(lengths, n, vector)
        def dense(length, stiffness):
            local = [[7,-8,1],[-8,16,-8],[1,-8,7]] if stiffness else [[4,2,-1],[2,16,2],[-1,2,4]]
            scale = F(n, 3*length) if stiffness else F(length, 30*n)
            a = [[F(0) for _ in range(3)] for _ in range(3)]
            for cell in range(n):
                for i, j in itertools.product(range(3), repeat=2):
                    row, col = 2*cell+i-1, 2*cell+j-1
                    if 0 <= row < 3 and 0 <= col < 3:
                        a[row][col] += scale*local[i][j]
            return a
        mx, my, kx, ky = dense(2, False), dense(3, False), dense(2, True), dense(3, True)
        for i, j in itertools.product(range(3), repeat=2):
            em = sum((mx[i][k]*my[j][l]*(3*k+l-4) for k,l in itertools.product(range(3), repeat=2)), F(0))
            ea = sum(((kx[i][k]*my[j][l]+mx[i][k]*ky[j][l])*(3*k+l-4)
                      for k,l in itertools.product(range(3), repeat=2)), F(0))
            self.overlaps(actual_m[3*i+j], g.I(em.numerator)/em.denominator)
            self.overlaps(actual_a[3*i+j], g.I(ea.numerator)/ea.denominator)

    def test_synthetic_solution_and_controlled_full_vector_perturbation(self):
        data = synthetic_galerkin((2, 3), 2)
        good = c.compare(data)
        self.assertTrue(good['accepted'])
        perturbed = copy.deepcopy(data)
        target = next(r for r in perturbed['records'] if r['coordinates_m'] == ['0.5', '0.75'])
        target['value_binary64_hex'] = (float.fromhex(target['value_binary64_hex'])+2**-10).hex()
        bad = c.compare(perturbed)
        self.assertFalse(bad['accepted'])
        delta = g.I(g.D.from_float(float.fromhex(target['value_binary64_hex'])))-c.binary64(
            next(r for r in data['records'] if r['coordinates_m'] == ['0.5', '0.75'])['value_binary64_hex'])
        # Single midpoint perturbation: exact diagonal tensor mass/energy.
        mdiag = g.I(16)/30 * (g.I(16)*3/2/30)
        adiag = (g.I(16)/3)*(g.I(16)*3/2/30) + (g.I(16)/30)*(g.I(16)*2/9)
        for diagonal, name in ((mdiag, 'B0'), (adiag, 'B1')):
            physical_delta = (delta.square()*diagonal).sqrt()
            self.assertLessEqual(physical_delta.hi, g.UP.add(g.D(bad[name][1]), g.D(good[name][1])))
        # Triangle inequality verifies the residual bound contains the norm change.
        for j, name in enumerate(('B0', 'B1')):
            a, b = g.I(*good['continuous_error_intervals'][j]), g.I(*bad['continuous_error_intervals'][j])
            self.assertLessEqual(g.abs_upper(a-b), g.D(bad[name][1])+g.D(good[name][1]))

    def test_three_dimensional_synthetic_solution(self):
        result = c.compare(synthetic_galerkin((2, 3, 4), 2))
        self.assertTrue(result['accepted'])
        self.assertEqual(result['physical_coefficient_count'], 125)
        self.assertEqual(result['interior_coefficient_count'], 27)

    def test_complete_rate_propagation(self):
        output = c.compare_levels([synthetic_galerkin((2, 3), n) for n in (8, 2, 4)])
        self.assertTrue(output['accepted'])
        self.assertEqual([r['N_pair'] for r in output['rates']], [[2,4],[4,8]])
        self.assertLess(g.D(output['predeclared_two_level_rate_uncertainty_bound'][1]), g.D('0.002886'))
        for pair in output['rates']:
            for rate in pair['norm_rates']:
                outer, inner = rate['uncertainty_enlarged_interval'], rate['measured_interval']
                self.assertLessEqual(g.D(outer[0]), g.D(inner[0]))
                self.assertGreaterEqual(g.D(outer[1]), g.D(inner[1]))
        with self.assertRaises(c.ComparisonError):
            c.compare_levels([synthetic_galerkin((2, 3), 2)])

    def test_malformed_grid_and_values_fail_closed(self):
        base = records((2, 3), 2, lambda *_: 0)
        mutations = [lambda x: x['records'].pop(),
                     lambda x: x['records'].append(copy.deepcopy(x['records'][0])),
                     lambda x: x['records'][0].update(coordinates_m=['0.00000000000001','0']),
                     lambda x: x['records'][0].update(value_binary64_hex='nan'),
                     lambda x: x['records'][0].update(value_binary64_hex='inf'),
                     lambda x: x['records'][0].update(value_binary64_hex='0x1.0p+0'),
                     lambda x: x['records'][0].update(value='0'),
                     lambda x: x.update(gauss_points=3),
                     lambda x: x.update(N=True),
                     lambda x: x.update(extra=0)]
        for mutation in mutations:
            data = copy.deepcopy(base)
            mutation(data)
            with self.subTest(mutation=mutation):
                with self.assertRaises(c.ComparisonError):
                    c.decode(data)
        vertices_only = copy.deepcopy(base)
        vertices_only['records'] = [r for r in vertices_only['records'] if all(g.D(x) in (0, 2, 3) for x in r['coordinates_m'])]
        with self.assertRaises(c.ComparisonError):
            c.decode(vertices_only)

    def test_duplicate_json_keys_rejected(self):
        with self.assertRaises(c.ComparisonError):
            json.loads('{"N":2,"N":4}', object_pairs_hook=c.unique_object)

    def test_binary64_capture_is_exact(self):
        value = c.binary64((0.1).hex())
        self.assertEqual(value.lo, g.D.from_float(0.1))
        self.assertNotEqual(value.lo, g.D('0.1'))
        tiny = c.binary64(float.fromhex('0x0.0000000000001p-1022').hex())
        self.assertGreater(tiny.lo, 0)


if __name__ == '__main__':
    unittest.main()
