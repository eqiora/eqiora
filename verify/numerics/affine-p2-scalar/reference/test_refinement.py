"""Small-level certification and exact square-root enclosure falsifiers."""
from fractions import Fraction as Q
import unittest
from refinement import certify,enclosure_sqrt

class RefinementTests(unittest.TestCase):
    def test_square_root_bounds_are_exactly_checkable(self):
        for q in (Q(0),Q(1),Q(2),Q(1,10**100),Q(10**100,3)):
            lo,hi=enclosure_sqrt(q)
            self.assertLessEqual(lo*lo,q);self.assertGreaterEqual(hi*hi,q)
        with self.assertRaises(ArithmeticError):enclosure_sqrt(Q(-1))
    def test_both_dimensions_small_level(self):
        for d in (2,3):
            result=certify(d,2)
            self.assertEqual(result['coefficient_count'],25 if d==2 else 125)
            for name in ('L2','H1_seminorm'):
                self.assertGreater(result[name]['lower'],0)
                self.assertLess(result[name]['relative_uncertainty_upper'],Q(1,10000))


class ComparisonTests(unittest.TestCase):
    def test_full_vector_and_mutants_both_dimensions(self):
        import json
        from compare import compare
        from exact import mesh
        for d in (2,3):
            result=certify(d,2)
            reference=json.loads(json.dumps(result,default=str))
            nodes=mesh(d,2)[3]
            records=[{'coordinates_m':list(map(str,x)),'value_binary64_hex':v} for x,v in zip(nodes,result['coefficients_binary64_hex'])]
            self.assertEqual(set(compare(reference,records)),{'L2','H1_seminorm'})
            with self.assertRaises(ValueError):compare(reference,records[:-1])
            duplicated=records[:-1]+[records[0]]
            with self.assertRaises(ValueError):compare(reference,duplicated)
            for factor in (0,-1):
                mutant=[dict(r,value_binary64_hex=(factor*float.fromhex(r['value_binary64_hex'])).hex()) for r in records]
                with self.assertRaises(ValueError):compare(reference,mutant)
            edge_index=next(i for i in range(3**d,len(nodes)) if float.fromhex(records[i]['value_binary64_hex']))
            mutant=[dict(r) for r in records];mutant[edge_index]['value_binary64_hex']='0x0.0p+0'
            with self.assertRaises(ValueError):compare(reference,mutant)

if __name__=='__main__':unittest.main()
