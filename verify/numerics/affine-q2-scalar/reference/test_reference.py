"""Focused mathematical tests, independent of Eqiora and its numerical output."""
import json
from fractions import Fraction as F
from pathlib import Path
import unittest
import generate as g


class ReferenceTests(unittest.TestCase):
    def setUp(self):
        g.precision(80)

    def contains(self, value, expected):
        expected = g.interval(expected)
        self.assertLessEqual(value.lo, expected.hi)
        self.assertGreaterEqual(value.hi, expected.lo)

    def test_interval_primitives(self):
        self.contains(g.I(1)/3*3, 1)
        self.assertEqual(g.I(-2, 3).square().json(), ['0', '9'])
        self.contains((g.I(2).sqrt()).square(), 2)
        self.contains(g.I(1).ln(), 0)
        with self.assertRaises(ValueError):
            g.I(1)/g.I(-1, 1)
        with self.assertRaises(ValueError):
            g.I('NaN')

    def test_pi_trig(self):
        # Published decimal prefixes are broad independent sanity bounds, not
        # the proof: the alternating/Taylor remainders provide that enclosure.
        p = g.pi()
        self.assertGreater(p.lo, g.D('3.14159265358979323846264338327950288419716939937510'))
        self.assertLess(p.hi, g.D('3.14159265358979323846264338327950288419716939937511'))
        self.contains(g.trig(p/2), 1)
        self.contains(g.trig(p/2, True), 0)
        self.contains(g.trig(p/6), g.I(1)/2)
        with self.assertRaises(ValueError):
            g.trig(g.I(3))

    def test_gauss_rule_moments(self):
        for points in (3, 4):
            nodes, weights = g.gauss_rule(points)
            for power in range(2*points):
                moment = sum((w*t**power for t, w in zip(nodes, weights)), g.I(0))
                self.contains(moment-g.I(1)/(power+1), 0)
        with self.assertRaises(ValueError):
            g.gauss_rule(5)

    def test_exact_local_polynomial_integrals(self):
        basis = [[F(1), F(-3), F(2)], [F(0), F(4), F(-4)], [F(0), F(-1), F(2)]]
        def integral_product(a, b):
            return sum((x*y/F(i+j+1) for i, x in enumerate(a) for j, y in enumerate(b)), F(0))
        mass = [[4, 2, -1], [2, 16, 2], [-1, 2, 4]]
        stiff = [[7, -8, 1], [-8, 16, -8], [1, -8, 7]]
        deriv = [[F(i)*x for i, x in enumerate(p)][1:] for p in basis]
        for i in range(3):
            for j in range(3):
                self.assertEqual(integral_product(basis[i], basis[j]), F(mass[i][j], 30))
                self.assertEqual(integral_product(deriv[i], deriv[j]), F(stiff[i][j], 3))

    def test_sine_mode_closure_against_global_assembly(self):
        for n in (2, 4):
            theta = g.pi()/n
            reduced = g.one_dimensional(1, n, theta)
            for local, matrix in (([[4, 2, -1], [2, 16, 2], [-1, 2, 4]], reduced[0]),
                                  ([[7, -8, 1], [-8, 16, -8], [1, -8, 7]], reduced[1])):
                mass = local[0][0] == 4
                scale = g.I(1)/(30*n) if mass else g.I(n)/3
                assembled = [[g.I(0) for _ in range(2*n+1)] for _ in range(2*n+1)]
                for cell in range(n):
                    for i in range(3):
                        for j in range(3):
                            assembled[2*cell+i][2*cell+j] += scale*local[i][j]
                def sine(k):
                    if k in (0, 2*n):
                        return g.I(0)
                    return g.trig(theta*min(k, 2*n-k)/2)
                for kind in (0, 1):
                    vector = [sine(k) if k % 2 == kind else g.I(0) for k in range(2*n+1)]
                    applied = g.apply(assembled, vector)
                    for k in range(1, 2*n):
                        self.contains(applied[k]-matrix[k % 2][kind]*sine(k), 0)

    def test_analytic_load_against_integrated_taylor_polynomials(self):
        # cos(theta*t) integrated against 2t²-3t+1 term by term.
        theta = g.pi()/2
        q = g.one_dimensional(1, 2, theta)[2]
        total = g.I(0)
        coefficient = g.I(1)
        for k in range(160):
            integral = g.I(2)/(2*k+3)-g.I(3)/(2*k+2)+g.I(1)/(2*k+1)
            total += coefficient*integral
            coefficient *= -theta*theta/((2*k+1)*(2*k+2))
        # Endpoint factor 2h=1; Taylor integral remainder <= theta^320/320!
        bound = g.I(1)
        for k in range(1, 321):
            bound = bound*theta/k
        self.contains(q[0]-total+g.I(bound.hi.copy_negate(), bound.hi), 0)
        # Centered coordinate s=t-1/2: bm=1-4s². Odd moments vanish.
        midpoint = g.I(0)
        coefficient = g.I(1)
        for k in range(160):
            moment = (g.I(1)/2)**(2*k) * (g.I(1)/(2*k+1)-g.I(1)/(2*k+3))
            midpoint += coefficient*moment/2  # h=1/2
            coefficient *= -theta*theta/((2*k+1)*(2*k+2))
        self.contains(q[1]-midpoint+g.I(bound.hi.copy_negate(), bound.hi), 0)

    def test_frozen_table_and_budget(self):
        table = json.loads(Path(__file__).with_name('intervals.json').read_text())
        self.assertEqual(g.generate(), table)
        failed = [(b['lengths_m'], l['N']) for b in table['boxes'] for l in b['levels']
                  if not l['quadrature_budget_certified']]
        self.assertEqual(failed, [([2, 3, 4], 2)])
        # Audit greater precision separately; outward rounding is the proof.
        audit = g.generate(110)
        for box, high in zip(table['boxes'], audit['boxes']):
            for lowlevel, highlevel in zip(box['levels'], high['levels']):
                for key in ('exact_load_errors', 'gauss3_load_errors',
                            'load_perturbation_relative_to_exact_error'):
                    for low, refined in zip(lowlevel[key], highlevel[key]):
                        self.assertLessEqual(g.D(low[0]), g.D(refined[0]))
                        self.assertGreaterEqual(g.D(low[1]), g.D(refined[1]))
            for load in box['rates']:
                for rates in box['rates'][load]:
                    for rate in rates:
                        self.assertLess(g.D(rate[0]), g.D(rate[1]))
                        self.assertLess(g.D(rate[1])-g.D(rate[0]), g.D('1e-60'))

    def test_four_point_reference_audit(self):
        frozen = json.loads(Path(__file__).with_name('intervals-gauss4-audit.json').read_text())
        self.assertEqual(g.generate(points=4), frozen)
        self.assertTrue(all(level['quadrature_budget_certified']
                            for box in frozen['boxes'] for level in box['levels']))

    def test_controls_fail_closed(self):
        for bad in (10, 201, 80.5):
            with self.assertRaises(ValueError):
                g.precision(bad)
        for box, n in (((2, 4), 2), ((2, 3), 3)):
            with self.assertRaises(ValueError):
                g.level(box, n)
        with self.assertRaises(ValueError):
            g.solve([[g.I(-1, 1)]], [g.I(1)])


if __name__ == '__main__':
    unittest.main()
