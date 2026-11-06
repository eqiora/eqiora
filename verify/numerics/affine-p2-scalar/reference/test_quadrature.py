"""Rational Duffy underintegration mutants; no numerical root approximations."""
from fractions import Fraction as Q
from math import comb
import unittest
from exact import (assemble_patch, gradient, inverse, local, mesh, mul)
from patch_gate import check
from generate_patch import generate


def duffy_two(p,det,d):
    # Two-point Gauss moments on [0,1], after eliminating irrational nodes.
    # Stiffness polynomials here require at most the fourth moment.
    moment=[Q(1),Q(1,2),Q(1,3),Q(1,4),Q(7,36)]
    total=Q(0)
    for alpha,c in p.items():
        value=c*det
        for j in range(d):
            power=d-1-j+sum(alpha[j+1:])
            value*=sum((-1)**k*comb(alpha[j],k)*moment[power+k] for k in range(alpha[j]+1))
        total+=value
    return total

class QuadratureTests(unittest.TestCase):
    def test_two_point_tensor_rule_is_exact_2d_but_not_3d_stiffness(self):
        for d in (2,3):
            vertices,cells,*_=mesh(d,2)
            K,_,_,grad,det=local([vertices[i] for i in cells[0]])
            bad=[[sum(duffy_two(mul(a,b),det,d) for a,b in zip(gi,gj)) for gj in grad] for gi in grad]
            if d==2:self.assertEqual(K,bad)
            else:self.assertNotEqual(K,bad)
    def test_3d_underintegrated_solution_fails_same_patch_gate(self):
        p=generate(3)
        vertices,cells,_,nodes,dofs=mesh(3,2)
        bad=[[Q(0)]*len(nodes) for _ in nodes]
        for ci,c in enumerate(cells):
            _,_,_,grad,det=local([vertices[i] for i in c])
            for a,i in enumerate(dofs[ci]):
                for b,j in enumerate(dofs[ci]):
                    bad[i][j]+=sum(duffy_two(mul(x,y),det,3) for x,y in zip(grad[a],grad[b]))
        free=p['free'];ess=p['essential'];v=list(p['exact_coefficients'])
        inv=inverse([[bad[i][j] for j in free] for i in free])
        rhs=[p['volume_load'][i]+p['natural_load'][i]-sum(bad[i][j]*v[j] for j in ess) for i in free]
        for i,row in zip(free,inv):v[i]=sum(a*b for a,b in zip(row,rhs))
        # Only the stiffness rule is corrupted; all data and fixed budgets stay.
        with self.assertRaises(ValueError):check(p,v,p['exact_constrained_actions'])

if __name__=='__main__':unittest.main()
