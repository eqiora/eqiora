"""Independent rational polynomial-patch and full-identity reaction tests."""
import copy
from fractions import Fraction as F
import json
import hashlib
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import polynomial as p


def document(lengths):
    reference = p.build_reference(lengths)
    exact = reference['exact']
    reaction = p.residual(reference, exact)
    def coordinates(index):
        return [str(float(F(length*k, 4))) for length,k in zip(lengths,index)]
    records = [{'coordinates_m': coordinates(index), 'value_binary64_hex': float(value).hex()}
               for index,value in zip(reference['indices'],exact)]
    reactions = [{'coordinates_m': coordinates(reference['indices'][i]),
                  'value_binary64_hex': float(reaction[i]).hex()} for i in reference['upper']]
    return {'schema':1, 'lengths_m':list(lengths), 'N':2, 'gauss_points':4, 'records':records,
            'upper_reactions':reactions}


class PolynomialTests(unittest.TestCase):
    def test_exact_patch_free_residual_and_analytic_reaction_sum(self):
        for lengths,target in (((2,3),F(4)), ((2,3,4),F(64,3))):
            ref = p.build_reference(lengths)
            residual = p.residual(ref,ref['exact'])
            self.assertTrue(all(residual[i] == 0 for i in ref['free']))
            self.assertEqual(sum(residual[i] for i in ref['upper']),target)
            self.assertTrue(p.compare(document(lengths))['accepted'])
            # Partition of unity gives global volume+natural load balance.
            self.assertEqual(-sum(ref['rhs']),target)

    def test_local_polynomial_integration_and_partition(self):
        mass,stiff,constant,poly = p.one_axis(3)
        self.assertEqual(sum(constant),3)
        self.assertEqual(sum(poly),4)
        self.assertTrue(all(sum(row) == 0 for row in stiff))
        self.assertEqual(sum(sum(row) for row in mass),3)
        self.assertEqual(mass[1][1],F(16,30)*F(3,2))
        self.assertEqual(stiff[1][1],F(16,3)/F(3,2))
        # Exact integration of p''=2/L² against each basis plus endpoint flux.
        nodal = [1+F(i*i,16) for i in range(5)]
        applied = [sum(a*b for a,b in zip(row,nodal)) for row in stiff]
        self.assertEqual(applied,[-F(2,9)*w+(F(2,3) if i == 4 else 0) for i,w in enumerate(constant)])

    def test_fixed_budgets_and_declared_rounding_multiplier(self):
        self.assertEqual(p.RESIDUAL_BUDGET,p.ROUNDING_MULTIPLIER*p.UNIT_ROUNDOFF)
        for lengths,inverse in (((2,3),F(3200,3)),((2,3,4),F(16000))):
            ref=p.build_reference(lengths)
            mu=p.product(F(length,60) for length in lengths)
            self.assertEqual(1/(F(9,16)*mu),inverse)
            self.assertEqual(ref['Bu'],inverse*F(1,2**40))
            self.assertEqual(ref['rho'],F(1,2**40)*max(1,ref['Kinf']*2**len(lengths)+ref['Binf']))
            actual_kinf=max(sum(abs(x) for x in row) for row in ref['K'])
            self.assertLessEqual(actual_kinf,ref['Kinf'])

    def test_coefficient_and_residual_perturbation_rejected(self):
        data=document((2,3))
        point=next(r for r in data['records'] if r['coordinates_m']==['0.5','0.75'])
        point['value_binary64_hex']=(float.fromhex(point['value_binary64_hex'])+2**-20).hex()
        result=p.compare(data)
        self.assertFalse(result['accepted'])
        self.assertFalse(result['checks']['free_residual_squared'])
        self.assertFalse(result['checks']['every_coefficient'])
        self.assertEqual(result['free_residual_squared_budget'],p.rational_json(F(1,2**80)))

    def test_native_reaction_component_sign_and_identity_falsifiers(self):
        base=document((2,3))
        # A single corruption and a sign reversal both violate fixed component budget.
        for operation in ('offset','sign','swap'):
            data=copy.deepcopy(base)
            reactions=data['upper_reactions']
            if operation=='offset':
                reactions[0]['value_binary64_hex']=(float.fromhex(reactions[0]['value_binary64_hex'])+2**-15).hex()
            elif operation=='sign':
                for record in reactions:
                    record['value_binary64_hex']=(-float.fromhex(record['value_binary64_hex'])).hex()
            else:
                reactions[0]['value_binary64_hex'],reactions[1]['value_binary64_hex']=reactions[1]['value_binary64_hex'],reactions[0]['value_binary64_hex']
            result=p.compare(data)
            self.assertFalse(result['accepted'])
            self.assertFalse(result['checks']['every_native_reaction'])

    def test_natural_flux_sign_and_missing_flux_falsifiers(self):
        ref=p.build_reference((2,3,4))
        for multiplier in (0,-1):
            wrong=dict(ref)
            wrong['rhs']=tuple(f+multiplier*g for f,g in zip(ref['F'],ref['G']))
            r=p.residual(wrong,ref['exact'])
            squared=sum(r[i]**2 for i in ref['free'])
            self.assertGreater(squared,p.RESIDUAL_BUDGET**2)
            self.assertNotEqual(sum(r[i] for i in ref['upper']),ref['target_sum'])

    def test_malformed_input_rejects(self):
        base=document((2,3))
        changes=[lambda x:x['records'].pop(),
                 lambda x:x['records'].append(copy.deepcopy(x['records'][0])),
                 lambda x:x['upper_reactions'].pop(),
                 lambda x:x['upper_reactions'].append(copy.deepcopy(x['upper_reactions'][0])),
                 lambda x:x['upper_reactions'][0].update(coordinates_m=['0','0']),
                 lambda x:x['upper_reactions'][0].update(value_binary64_hex='nan'),
                 lambda x:x['records'][0].update(value_binary64_hex='inf'),
                 lambda x:x['records'][0].update(coordinates_m=['0.00000000000001','0']),
                 lambda x:x['records'][-1].update(value_binary64_hex=(4.0+2**-40).hex()),
                 lambda x:x['records'][0].update(value='1'),
                 lambda x:x.update(gauss_points=3),lambda x:x.update(N=4),lambda x:x.update(N=True),lambda x:x.update(extra=0)]
        for change in changes:
            data=copy.deepcopy(base)
            change(data)
            with self.subTest(change=change):
                with self.assertRaises(p.PolynomialError):
                    p.compare(data)
        with self.assertRaises(p.PolynomialError):
            json.loads('{"N":2,"N":4}',object_pairs_hook=p.unique_object)

    def test_cli_raw_input_digest_and_typed_rejection(self):
        with tempfile.TemporaryDirectory(dir=Path.home()) as folder:
            source=Path(folder)/'input.json'
            output=Path(folder)/'output.json'
            raw=(json.dumps(document((2,3)),indent=2)+'\n').encode()
            source.write_bytes(raw)
            command=[sys.executable,str(Path(p.__file__)),str(source),'--output',str(output)]
            result=subprocess.run(command,capture_output=True,text=True)
            self.assertEqual(result.returncode,0,result.stderr)
            self.assertEqual(json.loads(output.read_text())['input_sha256'],hashlib.sha256(raw).hexdigest())
            source.write_text('{"N":2,"N":4}')
            result=subprocess.run(command,capture_output=True,text=True)
            self.assertEqual(result.returncode,2)
            self.assertEqual(json.loads(output.read_text())['error'],'schema')

    def test_arbitrary_record_order_preserves_identity(self):
        data=document((2,3,4))
        data['records'].reverse()
        data['upper_reactions'].reverse()
        self.assertTrue(p.compare(data)['accepted'])
        self.assertEqual(p.binary64((0.1).hex()),F.from_float(0.1))
        self.assertNotEqual(p.binary64((0.1).hex()),F(1,10))


if __name__=='__main__':
    unittest.main()
