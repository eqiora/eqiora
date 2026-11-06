"""Distinct flux-sign falsifiers and positive saved-vector level orchestration."""
from fractions import Fraction as Q
import hashlib
import json
from pathlib import Path
import unittest
from unittest.mock import patch
from compare import compare_levels
from exact import mesh
from patch_gate import check
from recheck_patch import load_packet

class PacketTests(unittest.TestCase):
    def test_wrong_natural_flux_sign_both_dimensions(self):
        for p in load_packet()['patches']:
            K=[[Q(a) for a in row] for row in p['stiffness']]
            F=list(map(Q,p['volume_load']));G=list(map(Q,p['natural_load']))
            values=list(map(Q,p['exact_coefficients']));free=p['free']
            wrong=list(values)
            for i,row in zip(free,p['inverse_free']):
                wrong[i]-=2*sum(Q(a)*G[j] for a,j in zip(row,free))
            self.assertTrue(all(wrong[i]==values[i] for i in p['essential']))
            # Independently verify the perturbed vector solves precisely the
            # wrong problem K v = F-G on every free row, not just a new field.
            wrong_residual=[sum(a*v for a,v in zip(K[i],wrong))-F[i]+G[i] for i in free]
            self.assertTrue(all(r==0 for r in wrong_residual))
            self.assertTrue(any(wrong[i]!=values[i] for i in free))
            with self.assertRaises(ValueError):check(p,wrong)
    def test_positive_compare_levels_from_bound_saved_certificates(self):
        root=Path(__file__).with_name('data')
        manifest=json.loads((root/'manifest.json').read_text())
        for d in (2,3):
            references=[];records={}
            for n in (2,4,8):
                path=root/'refinement'/f'd{d}-n{n}.json'
                self.assertEqual(hashlib.sha256(path.read_bytes()).hexdigest(),manifest['files'][str(path.relative_to(root))])
                r=json.loads(path.read_text());references.append(r)
                records[n]=[{'coordinates_m':list(map(str,x)),'value_binary64_hex':value} for x,value in zip(mesh(d,n)[3],r['coefficients_binary64_hex'])]
            # This guard forbids six fresh assemblies. It does not replace an
            # outcome: compare validates the full vectors and uses exact delta0.
            with patch('compare.assemble',side_effect=AssertionError('unexpected reassembly')):
                result=compare_levels(references,records)
            self.assertEqual(sorted(result['levels']),[2,4,8])
            self.assertEqual(sorted(result['rates']),[2,4])
            self.assertTrue(all(set(row)=={'L2','H1_seminorm'} for row in result['rates'].values()))

if __name__=='__main__':unittest.main()
