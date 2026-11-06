"""Authored manufactured inputs, with explicit physical units and no solver.

The reference's approved physical b,B determine logical coordinates and the
metric for the smooth witness. This does not evaluate any candidate solution.
"""
from fractions import Fraction
from itertools import combinations
from simplex_production import inputs, require


def number(q):
    q=Fraction(q)
    return f'({q.numerator}.0/{q.denominator}.0)'


def quantity(q,unit):return f'({number(q)}*1[{unit}])'


def source(dimension,profile):
    require(dimension in (2,3) and profile in ('patch','refinement'),'undeclared source profile')
    names=[f'q{i}_{side}' for i in range(dimension) for side in ('lower','upper')]
    supports=[f'support body:volume(ambient_dimension={dimension})']+[f'support {name}:boundary(parent=body)' for name in names]
    if profile=='patch':
        x=[f'(coordinate({i})/{2**(i+1)}[m])' for i in range(dimension)]
        exact='('+'+'.join(['1']+x+[f'{q}*{q}' for q in x]+[f'{a}*{b}' for a,b in combinations(x,2)])+')'
        forcing=quantity(-2*sum(Fraction(1,2**(2*(i+1))) for i in range(dimension)),'1/m^2')
        gradients=[f'((1+2*{x[i]}'+''.join(f'+{x[j]}' for j in range(dimension) if j!=i)+f')/{2**(i+1)}[m])' for i in range(dimension)]
        normals = [[Fraction(24,25),Fraction(7,25)],[Fraction(-4,5),Fraction(3,5)]] if dimension==2 else [
            [Fraction(624,625),Fraction(32,625),Fraction(-15,625)],
            [Fraction(-96,125),Fraction(72,125),Fraction(35,125)],
            [Fraction(16,25),Fraction(-12,25),Fraction(15,25)]]
        laws=[]
        for i in range(dimension):
            for side,sign in (('lower',-1),('upper',1)):
                name=f'q{i}_{side}'
                if name=='q0_upper':law=f'trace(u)={exact}'
                else:law='normal(grad(u))='+'+'.join(f'({number(sign*a)}*{g})' for a,g in zip(normals[i],gradients))
                laws.append(f'relation bc_{name} on {name} {{{law};}}')
    else:
        b,B=inputs.reference.domain(dimension);inv=inputs.reference.inverse(B)
        q=['('+'+'.join(f'({quantity(inv[i][j],"1/m")}*(coordinate({j})-{quantity(b[j],"m")}))' for j in range(dimension))+')' for i in range(dimension)]
        p=[f'({x}*(1-{x}))' for x in q]
        metric=[[sum(inv[i][k]*inv[j][k] for k in range(dimension)) for j in range(dimension)] for i in range(dimension)]
        terms=[]
        for i in range(dimension):
            terms.append('('+quantity(2*metric[i][i],'1/m^2')+'*'+'*'.join(p[j] for j in range(dimension) if j!=i)+')')
            for j in range(i+1,dimension):
                factors=[quantity(-2*metric[i][j],'1/m^2'),f'(1-2*{q[i]})',f'(1-2*{q[j]})']+[p[k] for k in range(dimension) if k not in (i,j)]
                terms.append('('+'*'.join(factors)+')')
        forcing='('+'+'.join(terms)+')'
        laws=[f'relation bc_{name} on {name} {{trace(u)=0;}}' for name in names]
    return 'public component AffineP2(\n'+',\n'.join(supports)+') {\nvariable u:1 on body in smooth;\nrelation balance on body {-div(grad(u))='+forcing+';}\n'+'\n'.join(laws)+'\n}\n'
