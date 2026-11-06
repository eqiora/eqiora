"""Analytic identities and deliberately corrupted synthetic Robin exports only."""
import copy
from fractions import Fraction as F
import hashlib
import itertools
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import robin as r


def hex_value(value):
    return float(value).hex()


def decimal(value):
    # All fixed grid coordinates are dyadic and exactly representable binary64.
    return str(float(value))


def document(case):
    ref = r.reference(case)
    def record(point, value):
        return {'coordinates_m': [decimal(x) for x in point], 'value_binary64_hex': hex_value(value)}
    return {'schema': 1, 'case': case, 'order': [[decimal(x) for x in point] for point in ref['coordinates']],
            'units': dict(r.UNITS), 'heat_convention': r.HEAT_CONVENTION,
            'A_binary64_hex': [[hex_value(x) for x in row] for row in ref['A']],
            'b_binary64_hex': [hex_value(x) for x in ref['b']],
            'coefficients': [record(point, value) for point, value in zip(ref['coordinates'], ref['exact'])],
            'left_reactions': [record(ref['coordinates'][i], value) for i, value in zip(ref['left'], ref['targets'])]}


def replace_operator(data, matrix):
    data['A_binary64_hex'] = [[hex_value(x) for x in row] for row in matrix]


def decode_fraction(value):
    return F(int(value['numerator']), int(value['denominator']))


class RobinTests(unittest.TestCase):
    def test_affine_solution_inverse_reactions_and_heat(self):
        slopes = (F(0), F(80,7), F(200,13), F(-60,7))
        for case, slope in zip(r.CASES, slopes):
            ref = r.reference(case)
            self.assertEqual(ref['slope'], slope)
            a = [[ref['A'][i][j] for j in ref['free']] for i in ref['free']]
            for i, j in itertools.product(range(6), repeat=2):
                self.assertEqual(sum(a[i][k]*ref['inverse'][k][j] for k in range(6)), int(i == j))
            # Independent continuum balance: k*a+h*(Tc+aL)=h*Tinf.
            length,height,k,h,tc,ambient = ref['parameters']
            self.assertEqual(k*slope+h*(tc+slope*length),h*ambient)
            self.assertEqual(sum(ref['targets']),-k*slope*height)
            self.assertEqual(ref['physical_heat']['left'],k*slope*height)
            self.assertEqual(ref['physical_heat']['right'],h*height*(tc+slope*length-ambient))
            self.assertEqual(sum(ref['physical_heat'].values()),0)
            result = r.compare(document(case))
            self.assertTrue(result['accepted'],(case,result['checks']))

    def test_direct_q1_polynomial_integration(self):
        basis = ((F(1),F(-1)),(F(0),F(1)))
        def integrate(a,b):
            return sum((x*y/F(i+j+1) for i,x in enumerate(a) for j,y in enumerate(b)), F(0))
        local = [[integrate(a,b) for b in basis] for a in basis]
        self.assertEqual(local,[[F(1,3),F(1,6)],[F(1,6),F(1,3)]])
        m,k,q=r.axis(2)
        self.assertEqual(m,[[F(1,3),F(1,6),0],[F(1,6),F(2,3),F(1,6)],[0,F(1,6),F(1,3)]])
        self.assertEqual(k,[[1,-1,0],[-1,2,-1],[0,-1,1]])
        self.assertEqual(q,[F(1,2),1,F(1,2)])
        self.assertEqual(sum(q),2)
        for ref in (r.reference(case) for case in r.CASES):
            self.assertEqual(r.matrix_norm(ref['inverse']),ref['bounds']['C'])
            self.assertLess(ref['bounds']['C']*ref['bounds']['deltaA'],1)
            self.assertEqual(ref['bounds']['eta'],8192*F(1,2**53))

    def test_omitted_robin_and_reversed_boundary_sign(self):
        for case in list(r.CASES)[1:]:
            ref=r.reference(case)
            for multiplier in (0,-1):
                data=document(case)
                matrix=[[a+multiplier*b for a,b in zip(row,rob)] for row,rob in zip(ref['diffusion'],ref['robin'])]
                replace_operator(data,matrix)
                if multiplier == -1:
                    data['b_binary64_hex']=[hex_value(-value) for value in ref['b']]
                result=r.compare(data)
                self.assertFalse(result['accepted'])
                self.assertFalse(result['checks']['full_operator'])

    def test_changed_h_only_matrix_or_only_load(self):
        ordinary,changed=r.reference('robin'),r.reference('changed_h')
        for part in ('matrix','load'):
            data=document('changed_h')
            if part=='matrix':
                replace_operator(data,ordinary['A'])
            else:
                data['b_binary64_hex']=[hex_value(x) for x in ordinary['b']]
            result=r.compare(data)
            self.assertFalse(result['accepted'])
            self.assertFalse(result['checks']['full_operator' if part=='matrix' else 'full_load'])
        self.assertNotEqual(ordinary['slope'],changed['slope'])

    def test_wrong_face_eliminated_right_and_missing_endpoint_weights(self):
        for case in list(r.CASES)[1:]:
            ref=r.reference(case)
            _,height,_,h,_,ambient=ref['parameters']
            my,_,qy=r.axis(height)
            nodes=list(itertools.product(range(3),repeat=2))
            for defect in ('wrong_face','eliminated_trace','endpoint_weights'):
                data=document(case)
                matrix=copy.deepcopy(ref['A'])
                load=list(ref['b'])
                if defect=='wrong_face':
                    matrix=[[ref['diffusion'][i][j]+(h*my[p[1]][q[1]] if p[0]==q[0]==0 else 0)
                             for j,q in enumerate(nodes)] for i,p in enumerate(nodes)]
                    load=[h*ambient*qy[j] if i==0 else F(0) for i,j in nodes]
                elif defect=='eliminated_trace':
                    for i in range(6,9):
                        for j in range(9):
                            matrix[i][j]=F(int(i==j))
                            matrix[j][i]=F(int(i==j))
                        load[i]=ref['exact'][i]
                else:
                    for i in (6,8):
                        load[i]=F(0)
                        for j in range(9):
                            matrix[i][j]-=ref['robin'][i][j]
                replace_operator(data,matrix)
                data['b_binary64_hex']=[hex_value(x) for x in load]
                result=r.compare(data)
                self.assertFalse(result['accepted'],(case,defect))
                self.assertFalse(result['checks']['full_operator'])

    def test_coefficient_residual_and_reaction_falsifiers(self):
        for defect in ('coefficient','reaction','reaction_sign'):
            data=document('robin')
            if defect=='coefficient':
                field=data['coefficients'][4]
                field['value_binary64_hex']=(float.fromhex(field['value_binary64_hex'])+2**-10).hex()
            elif defect=='reaction':
                field=data['left_reactions'][1]
                field['value_binary64_hex']=(float.fromhex(field['value_binary64_hex'])+2**-10).hex()
            else:
                for field in data['left_reactions']:
                    field['value_binary64_hex']=(-float.fromhex(field['value_binary64_hex'])).hex()
            result=r.compare(data)
            self.assertFalse(result['accepted'])
        data=document('robin')
        data['heat_convention']='physical_outward_heat=weak_reaction'
        with self.assertRaises(r.ReferenceError) as caught:
            r.compare(data)
        self.assertEqual(caught.exception.code,'heat_sign')

    def test_recovery_falsifier_inside_looser_reference_allowance(self):
        data=document('robin')
        ref=r.reference('robin')
        a,b,v,_=r.decode(data,ref)
        actual=[x-y for x,y in zip(r.apply(a,v),b)]
        offset=(ref['bounds']['R']+ref['bounds']['rho'])/2
        data['left_reactions'][0]['value_binary64_hex']=hex_value(actual[0]+offset)
        result=r.compare(data)
        self.assertTrue(result['checks']['every_reaction_vs_reference_full_vector'])
        self.assertFalse(result['checks']['every_reaction_vs_exported_physical_operator'])
        self.assertFalse(result['accepted'])

    def test_all_budgets_fixed_before_candidate(self):
        original=r.compare(document('robin'))['bounds']
        data=document('robin')
        data['A_binary64_hex'][4][4]=(1e100).hex()
        rejected=r.compare(data)
        self.assertFalse(rejected['accepted'])
        self.assertEqual(rejected['bounds'],original)
        ref=r.reference('robin')
        b=ref['bounds']
        self.assertEqual(b['E'],b['C']*(b['tau']+b['deltaA']*b['U']+b['deltab'])/(1-b['C']*b['deltaA']))
        self.assertEqual(b['rho'],b['deltaA']*(b['U']+b['E'])+b['deltab']+b['R'])

    def test_strict_shape_order_units_finite_and_identity(self):
        changes=[lambda x:x['A_binary64_hex'].pop(),lambda x:x['A_binary64_hex'][0].pop(),
                 lambda x:x['b_binary64_hex'].pop(),lambda x:x['coefficients'].pop(),
                 lambda x:x['left_reactions'].append(copy.deepcopy(x['left_reactions'][0])),
                 lambda x:x['coefficients'].reverse(),lambda x:x['order'].reverse(),
                 lambda x:x['left_reactions'][0].update(coordinates_m=['2','0']),
                 lambda x:x['left_reactions'][0].update(value_binary64_hex='nan'),
                 lambda x:x['A_binary64_hex'][0].__setitem__(0,'inf'),
                 lambda x:x['coefficients'][0].update(value_binary64_hex=(280+2**-30).hex()),
                 lambda x:x['order'][0].__setitem__(0,'0.0000000001'),
                 lambda x:x['units'].update(b='W'),lambda x:x.update(extra=1),
                 lambda x:x.update(case='unsupported'),lambda x:x.update(schema=True)]
        for change in changes:
            data=document('robin')
            change(data)
            with self.subTest(change=change):
                with self.assertRaises(r.ReferenceError):
                    r.compare(data)
        with self.assertRaises(r.ReferenceError):
            json.loads('{"case":"robin","case":"insulated"}',object_pairs_hook=r.unique_object)
        self.assertEqual(r.binary64((0.1).hex()),F.from_float(0.1))

    def test_cli_raw_input_sha_and_reference_generation(self):
        with tempfile.TemporaryDirectory(dir=Path.home()) as folder:
            source=Path(folder)/'input.json'
            output=Path(folder)/'output.json'
            raw=(json.dumps(document('robin'),indent=2)+'\n').encode()
            source.write_bytes(raw)
            command=[sys.executable,str(Path(r.__file__)),'compare',str(source),'--output',str(output)]
            completed=subprocess.run(command,capture_output=True,text=True)
            self.assertEqual(completed.returncode,0,completed.stderr)
            self.assertEqual(json.loads(output.read_text())['input_sha256'],hashlib.sha256(raw).hexdigest())
            command=[sys.executable,str(Path(r.__file__)),'reference','changed_h','--output',str(output)]
            completed=subprocess.run(command,capture_output=True,text=True)
            self.assertEqual(completed.returncode,0,completed.stderr)
            generated=json.loads(output.read_text())
            self.assertTrue(generated['preparation_only'])
            self.assertEqual(decode_fraction(generated['slope']),F(200,13))


if __name__=='__main__':
    unittest.main()
