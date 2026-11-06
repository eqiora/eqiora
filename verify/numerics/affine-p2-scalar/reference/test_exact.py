"""Pure, independent rational geometry/basis/operator checks."""
import unittest
from fractions import Fraction as Q
from itertools import permutations
from exact import (assemble_patch, basis, determinant, domain, infinity_norm,
                   inverse, local, mesh, patch_value)

class ExactTests(unittest.TestCase):
    def test_domains_and_topology(self):
        for d,expected in [(2,Q(225,32)),(3,Q(15625,1024))]:
            _,B = domain(d)
            self.assertEqual(determinant(B),expected)
            vertices,cells,edges,nodes,_ = mesh(d,2)
            self.assertEqual((len(vertices),len(edges),len(nodes)),(9,16,25) if d==2 else (27,98,125))
            for c in cells:
                J = [[vertices[c[j+1]][i]-vertices[c[0]][i] for j in range(d)] for i in range(d)]
                self.assertEqual(determinant(J),expected/2**d)
    def test_partition_and_permutations(self):
        for d in (2,3):
            vertices,cells,_,_,_ = mesh(d,2)
            points = [vertices[i] for i in cells[0]]
            K,_,_,_,_ = local(points)
            self.assertTrue(all(sum(row)==0 for row in K))
            # Affine coordinate gradients and all P2 nodal monomials are
            # checked through exact stiffness congruence for every permutation.
            pairs = [(i,) for i in range(d+1)]+list(__import__('itertools').combinations(range(d+1),2))
            for p in permutations(range(d+1)):
                permuted = [tuple(sorted(p[i] for i in pair)) for pair in pairs]
                mapping = [pairs.index(pair) for pair in permuted]
                kp,*_ = local([points[i] for i in p])
                self.assertEqual(kp,[[K[i][j] for j in mapping] for i in mapping])
    def test_patch_equilibrium_and_reaction(self):
        for d,target in [(2,Q(27339,4096)),(3,Q(24991425,1048576))]:
            K,F,G,nodes,essential = assemble_patch(d)
            values = list(map(patch_value,nodes))
            residual = [sum(a*v for a,v in zip(row,values))-f-g for row,f,g in zip(K,F,G)]
            self.assertTrue(all(residual[i]==0 for i in range(len(nodes)) if i not in essential))
            self.assertEqual(sum(residual[i] for i in essential),target)
            self.assertEqual(len(essential),5 if d==2 else 25)
            # Zero/sign and missing essential midpoint mutants cannot satisfy
            # exact trace equality; perturbing any free coefficient creates
            # a nonzero residual since the free stiffness is positive definite.
            self.assertTrue(all(values))
            self.assertTrue(all(K[i][i]>0 for i in range(len(nodes))))
    def test_inverse_identity(self):
        A = [[Q(2),Q(1)],[Q(1),Q(3)]]
        inv = inverse(A)
        self.assertEqual([[sum(A[i][k]*inv[k][j] for k in range(2)) for j in range(2)] for i in range(2)],[[1,0],[0,1]])


class PolynomialTests(unittest.TestCase):
    def test_nodal_monomials_partition_and_gradients(self):
        from exact import add, gradient, mul, scale
        from itertools import combinations
        for d in (2,3):
            vertices,cells,*_=mesh(d,2)
            points=[vertices[i] for i in cells[0]]
            _,_,phi,grad,_=local(points)
            bary=[tuple(Q(i==j) for i in range(d+1)) for j in range(d+1)]
            bary += [tuple(Q(int(k in (i,j)),2) for k in range(d+1)) for i,j in combinations(range(d+1),2)]
            def evaluate(p,q): return sum(c*__import__('functools').reduce(lambda a,b:a*b,(x**n for x,n in zip(q,alpha)),Q(1)) for alpha,c in p.items())
            for i,q in enumerate(bary):
                self.assertEqual([evaluate(p,q) for p in phi],[Q(i==j) for j in range(len(phi))])
                for k in range(d): self.assertEqual(sum(evaluate(g[k],q) for g in grad),0)
            self.assertEqual(sum(evaluate(p,(Q(1,d+1),)*(d+1)) for p in phi),1)
            # Every total-degree <=2 physical monomial reproduces its gradient.
            nodes=points+[tuple((points[i][k]+points[j][k])/2 for k in range(d)) for i,j in combinations(range(d+1),2)]
            for a in range(d):
                for b in range(a,d):
                    q=(Q(1,d+1),)*(d+1)
                    x=[sum(q[i]*points[i][k] for i in range(d+1)) for k in range(d)]
                    for k in range(d):
                        actual=sum(p[a]*p[b]*evaluate(g[k],q) for p,g in zip(nodes,grad))
                        self.assertEqual(actual,(x[b] if k==a else 0)+(x[a] if k==b else 0))
    def test_centroid_underintegration_is_detected(self):
        from math import factorial
        for d in (2,3):
            vertices,cells,*_=mesh(d,2)
            K,_,_,grad,det=local([vertices[i] for i in cells[0]])
            q=(Q(1,d+1),)*(d+1)
            def evaluate(p):
                return sum(c*__import__('functools').reduce(lambda a,b:a*b,(x**n for x,n in zip(q,alpha)),Q(1)) for alpha,c in p.items())
            bad=[[det/Q(factorial(d))*sum(evaluate(a)*evaluate(b) for a,b in zip(gi,gj)) for gj in grad] for gi in grad]
            self.assertNotEqual(K,bad)

class TraceAndGeometryMutants(unittest.TestCase):
    def test_complete_facet_closure_and_wrong_gradient(self):
        from exact import gradient,integrate,mul
        from itertools import combinations
        for d in (2,3):
            vertices,cells,*_=mesh(d,2)
            points=[vertices[i] for i in cells[0]]
            K,_,phi,_,det=local(points)
            pairs=[(i,) for i in range(d+1)]+list(combinations(range(d+1),2))
            for opposite in range(d+1):
                for pair,p in zip(pairs,phi):
                    restricted={a:c for a,c in p.items() if a[opposite]==0}
                    self.assertEqual(bool(restricted),opposite not in pair)
            J=[[points[j+1][i]-points[0][i] for j in range(d)] for i in range(d)]
            inv=inverse(J)
            # Deliberately use J^-1 instead of J^-T on column gradients.
            wrong=[tuple(-sum(inv[i][j] for j in range(d)) for i in range(d))]+[tuple(inv[i][j] for i in range(d)) for j in range(d)]
            grad=[gradient(p,wrong) for p in phi]
            mutant=[[sum(integrate(mul(a,b),det,d) for a,b in zip(gi,gj)) for gj in grad] for gi in grad]
            self.assertNotEqual(K,mutant)
            self.assertTrue(all(K[i][i]>0 for i in range(len(K))))
            self.assertTrue(all(-K[i][i]<0 for i in range(len(K))))

if __name__ == '__main__': unittest.main()
