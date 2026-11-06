"""Pure mathematical controls; no native candidate imports."""
import unittest
from fractions import Fraction as Q
import reference as r


class ReferenceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.K, cls.b = r.assemble()

    def test_patch_and_energy(self):
        self.assertEqual(r.residual(self.K,self.b), [0]*54)
        self.assertEqual(len(r.D),27)
        self.assertEqual(len(r.U),81)
        self.assertEqual(self.K, list(map(list,zip(*self.K))))
        energy = sum(r.U[i]*self.K[i][j]*r.U[j] for i in range(81) for j in range(81))/2
        self.assertEqual(energy,Q(261,32))
        self.assertEqual([-12*r.stress()[i][0] for i in range(3)], [Q(-27,2),Q(-9,2),Q(-9,4)])

    def test_independent_wrong_forms_and_loads(self):
        for label, options in [('omit_lambda',dict(lam=Q(0))),
                               ('full_gradient',dict(full=True)),
                               ('reverse_traction',dict(traction_sign=-1)),
                               ('unit_jacobian',dict(unit_jacobian=True))]:
            with self.subTest(label=label):
                K,b = r.assemble(**options)
                self.assertGreater(max(map(abs,r.residual(K,b))),Q(1,1000))

    def test_wrong_coordinate_and_component_mappings(self):
        variants = [list(r.U), list(r.U), list(r.U)]
        variants[0][3*13+1] = 0
        for n in range(27):
            variants[1][3*n],variants[1][3*n+1] = variants[1][3*n+1],variants[1][3*n]
        # A wrong corner map reflects physical x while retaining the declared coordinates.
        for n in range(27):
            other = n//3*3 + 2-n%3
            variants[2][3*n:3*n+3] = r.U[3*other:3*other+3]
        for u in variants:
            self.assertGreater(max(map(abs,r.residual(self.K,self.b,u))),Q(1,1000))

    def test_tensor_integral_partition(self):
        for p in range(3):
            for q in range(3):
                for b in r.CORNERS:
                    self.assertEqual(sum(r.gradient_product(a,b,p,q) for a in r.CORNERS),0)
        self.assertEqual(sum(self.b[3*n] for n in range(27)),Q(27,2))


class ProtocolTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.limits = r.budgets()

    def fixture(self):
        import protocol
        return dict(identity=dict(dimension=3,family='Hex',order='Q1',geometry='affine-cartesian',
                                  cells=[2,2,2],quadrature=[2,2,2],units=['m','J','N']),
                    lineage=dict(mesh='analytic-mesh',model='analytic-model',result='analytic-control'),
                    coefficients=[dict(coordinate_m=[float(v).hex() for v in x],component=c,
                                       value_m=float(r.U[3*n+i]).hex())
                                  for n,x in enumerate(r.NODES) for i,c in enumerate(('x','y','z'))],
                    energy_J=float(Q(261,32)).hex(),
                    left_traction_N=[float(-12*r.stress()[i][0]).hex() for i in range(3)])

    def test_positive_and_strict_rejections(self):
        import json
        import protocol
        protocol.check(json.dumps(self.fixture()), self.limits)
        for mutation in ('missing','duplicate','nonfinite','negative_zero','unknown','quadrature','wrong_value'):
            obj = self.fixture()
            if mutation == 'missing': obj['coefficients'].pop()
            if mutation == 'duplicate': obj['coefficients'][0] = obj['coefficients'][1]
            if mutation == 'nonfinite': obj['energy_J'] = 'inf'
            if mutation == 'negative_zero': obj['coefficients'][0]['value_m'] = '-0x0.0p+0'
            if mutation == 'unknown': obj['extra'] = 1
            if mutation == 'quadrature': obj['identity']['quadrature'] = [1,1,1]
            if mutation == 'wrong_value': obj['coefficients'][40]['value_m'] = float(1).hex()
            with self.subTest(mutation=mutation), self.assertRaises(ValueError):
                protocol.check(json.dumps(obj), self.limits)
        with self.assertRaises(ValueError): protocol.decode('{"identity":1,"identity":2}')

    def test_inverse_certificate_and_functional_rows(self):
        q = self.limits
        self.assertEqual(q['traction_norms'], [220,96,90])
        self.assertEqual([r.dot(row,r.U) for row in r.traction_rows()],
                         [Q(-27,2),Q(-9,2),Q(-9,4)])
        # Recompute and verify both matrix products; neither assumes symmetry.
        K = [[q['K'][i][j] for j in r.F] for i in r.F]
        inv = r.inverse(K)
        for i in range(54):
            for j in range(54):
                self.assertEqual(r.dot(K[i],[inv[k][j] for k in range(54)]),Q(i==j))
                self.assertEqual(r.dot(inv[i],[K[k][j] for k in range(54)]),Q(i==j))

    def test_analytic_target_alone_is_insufficient(self):
        import json
        import protocol
        q = self.limits
        for functional in ('energy','traction'):
            obj = self.fixture()
            if functional == 'energy':
                delta = (q['energy_native']+q['energy_analytic'])/2
                observed = float(Q(261,32)+delta)
                self.assertLess(abs(Q(observed)-Q(261,32)),q['energy_analytic'])
                obj['energy_J'] = observed.hex()
            else:
                delta = (q['traction_native'][0]+q['traction_analytic'][0])/2
                observed = float(Q(-27,2)+delta)
                self.assertLess(abs(Q(observed)+Q(27,2)),q['traction_analytic'][0])
                obj['left_traction_N'][0] = observed.hex()
            with self.subTest(functional=functional), self.assertRaisesRegex(ValueError,'exported-field'):
                protocol.check(json.dumps(obj),q)

    def test_essential_values_are_exact(self):
        import json
        import math
        import protocol
        obj = self.fixture()
        index = next(i for i in r.D if r.U[i])
        obj['coefficients'][index]['value_m'] = math.nextafter(float(r.U[index]),math.inf).hex()
        with self.assertRaisesRegex(ValueError,'essential'):
            protocol.check(json.dumps(obj),self.limits)

if __name__ == '__main__':
    unittest.main()
