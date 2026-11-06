"""Independent sparse approximation, certified by exact rational residuals.

SciPy only proposes coefficients. All assembled entries, residuals, polynomial
norms and certification bounds are subsequently evaluated as exact rationals.
"""
from fractions import Fraction as Q
from itertools import product
from decimal import Decimal, localcontext
from math import factorial
from exact import (add, basis, determinant, domain, integrate, inverse, local,
                   mesh, mul, scale, infinity_norm)


def enclosure_sqrt(q):
    if q < 0: raise ArithmeticError('negative exact squared quantity')
    with localcontext() as context:
        context.prec = 70
        x = (Decimal(q.numerator)/Decimal(q.denominator)).sqrt()
        # One final decimal ulp on either side encloses rounded division+sqrt.
        pad = Decimal(10)**(x.adjusted()-context.prec+2) if x else Decimal(0)
        lower,upper = Q(max(Decimal(0),x-pad)), Q(x+pad)
        if not lower*lower <= q <= upper*upper:
            raise ArithmeticError("square-root enclosure failed exact squared check")
        return lower,upper


def assemble(d,n):
    vertices,cells,edges,nodes,dofs = mesh(d,n)
    count = len(nodes)
    K = [dict() for _ in nodes]; M = [dict() for _ in nodes]
    F = [Q(0)]*count; U = [Q(0)]*count
    b,B = domain(d); Bi = inverse(B)
    metric = [[sum(Bi[i][k]*Bi[j][k] for k in range(d)) for j in range(d)] for i in range(d)]
    logical = [tuple(sum(Bi[i][j]*(x[j]-b[j]) for j in range(d)) for i in range(d)) for x in vertices]
    node_logical = [tuple(sum(Bi[i][j]*(x[j]-b[j]) for j in range(d)) for i in range(d)) for x in nodes]
    essential = {i for i,q in enumerate(node_logical) if any(x in (0,1) for x in q)}
    phi = basis(d); constant = {(0,)*(d+1):Q(1)}
    cache = {}
    def times(polys):
        out = constant
        for p in polys: out = mul(out,p)
        return out
    for ci,c in enumerate(cells):
        points = [vertices[i] for i in c]
        shape = tuple(tuple(x-y for x,y in zip(p,points[0])) for p in points[1:])
        if shape not in cache: cache[shape] = local(points)
        k,m,_,_,det = cache[shape]
        ps=[]; ds=[]
        for axis in range(d):
            qpoly = {}
            for j,vi in enumerate(c):
                alpha=[0]*(d+1);alpha[j]=1
                qpoly[tuple(alpha)] = logical[vi][axis]
            ps.append(mul(qpoly,add(constant,scale(qpoly,-1))))
            ds.append(add(constant,scale(qpoly,-2)))
        u = times(ps); f = {}
        for i in range(d):
            f = add(f,scale(times([ps[j] for j in range(d) if j!=i]),2*metric[i][i]))
            for j in range(i+1,d):
                f = add(f,scale(times([ds[i],ds[j]]+[ps[k] for k in range(d) if k not in (i,j)]),-2*metric[i][j]))
        for a,i in enumerate(dofs[ci]):
            F[i] += integrate(mul(phi[a],f),det,d)
            U[i] += integrate(mul(phi[a],u),det,d)
            for z,j in enumerate(dofs[ci]):
                K[i][j] = K[i].get(j,Q(0))+k[a][z]
                M[i][j] = M[i].get(j,Q(0))+m[a][z]
    return K,M,F,U,essential,determinant(B),metric,cache


def certify(d,n, coefficient_hex=None):
    K,M,F,U,essential,detB,metric,cache = assemble(d,n)
    free = [i for i in range(len(K)) if i not in essential]
    if coefficient_hex is None:
        import numpy as np
        from scipy.sparse import csr_matrix
        from scipy.sparse.linalg import spsolve
        fi = {v:i for i,v in enumerate(free)}
        rows=[];cols=[];data=[]
        for i in free:
            for j,a in K[i].items():
                if j in fi and a:
                    rows.append(fi[i]);cols.append(fi[j]);data.append(float(a))
        A = csr_matrix((data,(rows,cols)),shape=(len(free),len(free)))
        approx = spsolve(A,np.array([float(F[i]) for i in free]))
        if not np.isfinite(approx).all(): raise ArithmeticError('nonfinite independent approximation')
        v = [Q(0)]*len(K)
        for i,x in zip(free,approx): v[i]=Q.from_float(float(x))
    else:
        if len(coefficient_hex)!=len(K): raise ValueError('incomplete coefficient inventory')
        values=[float.fromhex(x) for x in coefficient_hex]
        if any(x.hex()!=text for x,text in zip(values,coefficient_hex)):
            raise ValueError('noncanonical binary64 coefficient')
        v=list(map(Q.from_float,values))
        if any(v[i] for i in essential): raise ValueError('nonzero essential trace')
    residual = [sum(a*v[j] for j,a in K[i].items())-F[i] for i in free]
    eta2 = sum(r*r for r in residual)
    eta = enclosure_sqrt(eta2)[1]
    _,B=domain(d)
    lam = Q(9*d)/sum(x*x for row in B for x in row)
    # Exact reference-simplex mass, independent of physical shape.
    reference_vertices = [tuple(Q(0) for _ in range(d))]+[tuple(Q(i==j) for i in range(d)) for j in range(d)]
    _,mrefmatrix,*_=local(reference_vertices)
    mref = 1/infinity_norm(inverse(mrefmatrix))
    mu = mref*min(item[4] for item in cache.values())
    uncertainty_l2 = eta/(lam*enclosure_sqrt(mu)[0])
    uncertainty_h1 = eta/enclosure_sqrt(lam*mu)[0]
    def quadratic(a): return sum(v[i]*sum(c*v[j] for j,c in row.items()) for i,row in enumerate(a))
    exact_l2 = detB/Q(30**d)
    exact_h1 = detB*sum(metric[i][i] for i in range(d))/Q(3*30**(d-1))
    l2sq = exact_l2+quadratic(M)-2*sum(a*b for a,b in zip(v,U))
    h1sq = exact_h1+quadratic(K)-2*sum(a*b for a,b in zip(v,F))
    result={'dimension':d,'N':n,'coefficient_count':len(v),'free_count':len(free),'eta_squared':eta2,'lambda':lam,'mu':mu,'coefficients_binary64_hex':[float(x).hex() for x in v]}
    for name,squared,unc in [('L2',l2sq,uncertainty_l2),('H1_seminorm',h1sq,uncertainty_h1)]:
        lo,hi=enclosure_sqrt(squared)
        lower=lo-unc; upper=hi+unc
        if lower<=0 or unc>Q(1,10000)*lower: raise ArithmeticError(f'{d}D N{n} {name}: reference allocation inconclusive')
        result[name]={'lower':lower,'upper':upper,'approximation_uncertainty_upper':unc,'relative_uncertainty_upper':unc/lower}
    return result
