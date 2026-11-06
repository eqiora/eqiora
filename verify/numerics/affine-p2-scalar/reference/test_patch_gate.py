"""The same rational gate rejects numerical coefficient/action mutants."""
from fractions import Fraction as Q
import unittest
from generate_patch import generate
from patch_gate import check

class GateTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls): cls.packet=generate(2)
    def test_exact_patch_passes(self):
        p=self.packet
        self.assertTrue(check(p,p['exact_coefficients'],p['exact_constrained_actions']))
    def test_missing_or_zero_or_negated_coefficients(self):
        p=self.packet; exact=p['exact_coefficients']; reaction=p['exact_constrained_actions']
        for bad in [exact[:-1],[Q(0)]*len(exact),[-x for x in exact]]:
            with self.assertRaises(ValueError): check(p,bad,reaction)
    def test_boundary_midpoint_and_shared_edge_mutants(self):
        p=self.packet; exact=p['exact_coefficients']; reaction=p['exact_constrained_actions']
        # N2 has 9 vertices; every remaining entry is a genuine global edge.
        boundary=next(i for i in p['essential'] if i>=9)
        free_edge=next(i for i in p['free'] if i>=9)
        for i in (boundary,free_edge):
            bad=list(exact);bad[i]+=Q(1,1024)
            with self.assertRaises(ValueError): check(p,bad,reaction)
    def test_zero_and_sign_reaction_mutants(self):
        p=self.packet
        for bad in [[Q(0)]*len(p['essential']),[-x for x in p['exact_constrained_actions']]]:
            with self.assertRaises(ValueError):check(p,p['exact_coefficients'],bad)
    def test_skipped_natural_midpoint_load(self):
        from copy import deepcopy
        p=deepcopy(self.packet)
        i=next(i for i in p['free'] if i>=9 and p['natural_load'][i])
        p['natural_load'][i]=Q(0)
        with self.assertRaisesRegex(ValueError,'residual'):check(p,p['exact_coefficients'],p['exact_constrained_actions'])

if __name__=='__main__':unittest.main()
