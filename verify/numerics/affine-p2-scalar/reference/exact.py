"""Independent rational affine P2 geometry and polynomial integration.

Only Python's standard library is used. Coordinates are physical metres;
barred numerical scales equal the fixed unit values in README.md.
"""
from fractions import Fraction as Q
from itertools import combinations, permutations, product
from math import factorial


def determinant(a):
    if len(a) == 1:
        return a[0][0]
    return sum((-1)**j * a[0][j] * determinant([r[:j]+r[j+1:] for r in a[1:]]) for j in range(len(a)))


def inverse(a):
    n = len(a)
    rows = [[Q(x) for x in row]+[Q(i == j) for j in range(n)] for i, row in enumerate(a)]
    for j in range(n):
        pivot = next(i for i in range(j, n) if rows[i][j])
        rows[j], rows[pivot] = rows[pivot], rows[j]
        scale = rows[j][j]
        rows[j] = [x/scale for x in rows[j]]
        for i in range(n):
            if i != j and rows[i][j]:
                scale = rows[i][j]
                rows[i] = [x-scale*y for x, y in zip(rows[i], rows[j])]
    return [r[n:] for r in rows]


def domain(d):
    if d == 2:
        return tuple(map(Q, ('1/4', '-1/2'))), [[Q(x) for x in row] for row in [('9/8','-21/16'),('3/2','9/2')]]
    if d == 3:
        return tuple(map(Q, ('1/4','-1/2','3/4'))), [[Q(x) for x in row] for row in [('15/16','-3/32','11/64'),('5/4','3','-51/32'),('0','5/2','15/4')]]
    raise ValueError('dimension must be two or three')


def mesh(d, n):
    b, B = domain(d)
    logical = list(product(range(n+1), repeat=d))
    index = {q: i for i, q in enumerate(logical)}
    vertices = [tuple(b[i]+sum(B[i][j]*Q(q[j], n) for j in range(d)) for i in range(d)) for q in logical]
    cells = []
    for a in product(range(n), repeat=d):
        for p in permutations(range(d)):
            q = list(a)
            ids = [index[tuple(q)]]
            for axis in p:
                q[axis] += 1
                ids.append(index[tuple(q)])
            J = [[vertices[ids[j+1]][i]-vertices[ids[0]][i] for j in range(d)] for i in range(d)]
            if determinant(J) < 0:
                ids[-1], ids[-2] = ids[-2], ids[-1]
            cells.append(tuple(ids))
    edges = sorted({tuple(sorted(e)) for c in cells for e in combinations(c, 2)})
    nodes = vertices+[tuple((vertices[i][k]+vertices[j][k])/2 for k in range(d)) for i,j in edges]
    edge_index = {e: len(vertices)+i for i,e in enumerate(edges)}
    dofs = [list(c)+[edge_index[tuple(sorted((c[i],c[j]))) ] for i,j in combinations(range(d+1),2)] for c in cells]
    return vertices, cells, edges, nodes, dofs


def add(a, b):
    out = dict(a)
    for k,v in b.items():
        out[k] = out.get(k,Q(0))+v
        if not out[k]: del out[k]
    return out


def scale(a, c):
    return {k:v*c for k,v in a.items() if v*c}


def mul(a, b):
    out = {}
    for i,x in a.items():
        for j,y in b.items():
            k = tuple(s+t for s,t in zip(i,j))
            out[k] = out.get(k,Q(0))+x*y
    return {k:v for k,v in out.items() if v}


def basis(d):
    zero = (0,)*(d+1)
    lam = []
    for i in range(d+1):
        e = list(zero); e[i] = 1
        lam.append({tuple(e):Q(1)})
    return [add(scale(mul(l,l),2),scale(l,-1)) for l in lam]+[scale(mul(lam[i],lam[j]),4) for i,j in combinations(range(d+1),2)]


def gradient(p, gradients):
    d = len(gradients[0]); out = [{} for _ in range(d)]
    for alpha,c in p.items():
        for i,power in enumerate(alpha):
            if power:
                beta = list(alpha); beta[i] -= 1
                for k in range(d):
                    out[k] = add(out[k],{tuple(beta):c*power*gradients[i][k]})
    return out


def integrate(p, det, d):
    return det*sum(c*Q(product_factorial(alpha), factorial(d+sum(alpha))) for alpha,c in p.items())


def product_factorial(alpha):
    out = 1
    for a in alpha: out *= factorial(a)
    return out


def local(vertices):
    d = len(vertices)-1
    J = [[vertices[j+1][i]-vertices[0][i] for j in range(d)] for i in range(d)]
    det = abs(determinant(J)); inv = inverse(J)
    gradients = [tuple(-sum(inv[j][i] for j in range(d)) for i in range(d))]+[tuple(r) for r in inv]
    phi = basis(d); grad = [gradient(p, gradients) for p in phi]
    K = [[sum(integrate(mul(a,b),det,d) for a,b in zip(gi,gj)) for gj in grad] for gi in grad]
    M = [[integrate(mul(a,b),det,d) for b in phi] for a in phi]
    return K,M,phi,grad,det


def patch_value(x):
    X = [a/Q(2**(i+1)) for i,a in enumerate(x)]
    return 1+sum(X)+sum(a*a for a in X)+sum(a*b for a,b in combinations(X,2))


def patch_gradient(x):
    X = [a/Q(2**(i+1)) for i,a in enumerate(x)]
    return [(1+2*a+sum(X)-a)/Q(2**(i+1)) for i,a in enumerate(X)]


def assemble_patch(d, n=2):
    vertices,cells,edges,nodes,dofs = mesh(d,n)
    count = len(nodes); K = [[Q(0)]*count for _ in nodes]
    F = [Q(0)]*count; G = [Q(0)]*count
    facets = {}
    for ci,c in enumerate(cells):
        for opposite in range(d+1):
            key = tuple(sorted(c[:opposite]+c[opposite+1:]))
            facets.setdefault(key,[]).append((ci,opposite))
    b,B = domain(d); Binv = inverse(B)
    def logical(x):
        return tuple(sum(Binv[i][j]*(x[j]-b[j]) for j in range(d)) for i in range(d))
    essential = [i for i,x in enumerate(nodes) if logical(x)[0] == 1]
    forcing = -2*sum(Q(1,2**(2*(i+1))) for i in range(d))
    for ci,c in enumerate(cells):
        points = [vertices[i] for i in c]
        k,_,phi,_,det = local(points)
        for a,i in enumerate(dofs[ci]):
            F[i] += forcing*integrate(phi[a],det,d)
            for z,j in enumerate(dofs[ci]): K[i][j] += k[a][z]
        J = [[points[j+1][i]-points[0][i] for j in range(d)] for i in range(d)]
        inv = inverse(J)
        lg = [tuple(-sum(inv[j][i] for j in range(d)) for i in range(d))]+[tuple(r) for r in inv]
        for opposite in range(d+1):
            key = tuple(sorted(c[:opposite]+c[opposite+1:]))
            if len(facets[key]) != 1 or all(logical(vertices[i])[0] == 1 for i in key): continue
            # Oriented surface Jacobian: minus det(J) grad(lambda_opposite).
            flux = {}
            for j,x in enumerate(points):
                alpha = [0]*(d+1); alpha[j] = 1
                flux[tuple(alpha)] = -det*sum(a*z for a,z in zip(lg[opposite],patch_gradient(x)))
            for a,i in enumerate(dofs[ci]):
                restricted = {alpha[:opposite]+alpha[opposite+1:]:v for alpha,v in mul(phi[a],flux).items() if alpha[opposite] == 0}
                G[i] += integrate(restricted,Q(1),d-1)
    return K,F,G,nodes,essential


def infinity_norm(a):
    return max(sum(abs(x) for x in row) for row in a)
