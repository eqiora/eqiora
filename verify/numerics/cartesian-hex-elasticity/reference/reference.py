"""Independent exact Cartesian Q1 elasticity; Python standard library only."""
from fractions import Fraction as Q
from itertools import product
import json

A = [[Q(n, 16) for n in row] for row in ((1, 2, 0), (0, 3, 1), (1, 0, 2))]
H = (Q(1), Q(3, 2), Q(2))
CORNERS = [(i, j, k) for k in range(2) for j in range(2) for i in range(2)]
NODES = [(Q(i), Q(3*j, 2), Q(2*k)) for k in range(3) for j in range(3) for i in range(3)]
U = [sum(A[c][d]*x[d] for d in range(3)) for x in NODES for c in range(3)]
D = [3*n+c for n, x in enumerate(NODES) if x[0] == 0 for c in range(3)]
F = [n for n in range(81) if n not in D]


def stress(mu=Q(3), lam=Q(2), full=False):
    tr = sum(A[i][i] for i in range(3))
    return [[mu*(2*A[i][j] if full else A[i][j]+A[j][i]) + (lam*tr if i == j else 0)
             for j in range(3)] for i in range(3)]


def factor(a, b, da, db, h):
    """Integral of two 1D shapes or physical derivatives over [0,h]."""
    if da and db:
        return Q((2*a-1)*(2*b-1), 1)/h
    if da:
        return Q(2*a-1, 2)
    if db:
        return Q(2*b-1, 2)
    return h * (Q(1, 3) if a == b else Q(1, 6))


def gradient_product(a, b, p, q):
    v = Q(1)
    for d in range(3):
        v *= factor(a[d], b[d], d == p, d == q, H[d])
    return v


def assemble(lam=Q(2), full=False, traction_sign=1, unit_jacobian=False):
    K = [[Q(0) for _ in U] for _ in U]
    b = [Q(0) for _ in U]
    sigma = stress()
    for cell in product(range(2), repeat=3):
        ids = [((cell[2]+a[2])*9+(cell[1]+a[1])*3+cell[0]+a[0]) for a in CORNERS]
        for aa, a in enumerate(CORNERS):
            for bb, z in enumerate(CORNERS):
                for i in range(3):
                    for j in range(3):
                        v = (3*sum(gradient_product(a,z,d,d) for d in range(3)) if i == j else Q(0))
                        v += (0 if full else 3*gradient_product(a,z,j,i))
                        if full and i == j:
                            v += 3*sum(gradient_product(a,z,d,d) for d in range(3))
                        v += lam*gradient_product(a,z,i,j)
                        K[3*ids[aa]+i][3*ids[bb]+j] += v/(3 if unit_jacobian else 1)
            for axis in range(3):
                for side in range(2):
                    if cell[axis] != side or a[axis] != side or (axis == 0 and side == 0):
                        continue
                    area = Q(1)
                    for d in range(3):
                        if d != axis:
                            area *= H[d]/2
                    for i in range(3):
                        b[3*ids[aa]+i] += traction_sign*(2*side-1)*sigma[i][axis]*area
    return K, b


def residual(K, b, u=U):
    return [sum(v*x for v,x in zip(K[i],u))-b[i] for i in F]


def inverse(matrix):
    n = len(matrix)
    rows = [list(row)+[Q(i == j) for j in range(n)] for i,row in enumerate(matrix)]
    for j in range(n):
        pivot = next(i for i in range(j,n) if rows[i][j])
        rows[j], rows[pivot] = rows[pivot], rows[j]
        scale = rows[j][j]
        rows[j] = [x/scale for x in rows[j]]
        for i in range(n):
            if i != j:
                scale = rows[i][j]
                rows[i] = [x-scale*y for x,y in zip(rows[i],rows[j])]
    return [row[n:] for row in rows]


def traction_rows():
    """Exact rows for integral of sigma(v)(-ex) on x=0."""
    rows = [[Q(0) for _ in U] for _ in range(3)]
    for cy,cz in product(range(2),repeat=2):
        for a in CORNERS:
            n = (cz+a[2])*9+(cy+a[1])*3+a[0]
            gradients = [Q(2*a[0]-1)/H[0]*H[1]*H[2]/4,
                         Q(2*a[1]-1)*H[2]/2 if a[0] == 0 else Q(0),
                         Q(2*a[2]-1)*H[1]/2 if a[0] == 0 else Q(0)]
            for i in range(3):
                rows[i][3*n+i] -= 3*gradients[0]
                rows[i][3*n] -= 3*gradients[i]
                if i == 0:
                    for c in range(3):
                        rows[i][3*n+c] -= 2*gradients[c]
    return rows


def dot(a,b):
    return sum(x*y for x,y in zip(a,b))


def energy(K,u):
    return sum(u[i]*dot(row,u) for i,row in enumerate(K))/2


def budgets():
    K,b = assemble()
    free = [[K[i][j] for j in F] for i in F]
    inv = inverse(free)
    # Verify every entry before using this inverse as an error certificate.
    if not all(dot(row,[inv[k][j] for k in range(len(F))]) == Q(i == j)
               for i,row in enumerate(free) for j in range(len(F))):
        raise ArithmeticError("exact inverse identity failed")
    c = max(sum(abs(v) for v in row) for row in inv)
    u0 = max(Q(1),max(map(abs,U)))
    scale = max(sum(map(abs,row)) for row in K)*u0+max(map(abs,b))
    eta,sigma = Q(1,2**40),Q(1,2**46)
    tau = (eta+sigma)*scale
    error = c*tau
    v = u0+error
    B = sum(sum(map(abs,row)) for row in K)
    norms = [sum(map(abs,row)) for row in traction_rows()]
    energy_native = eta*B*v*v/2
    return dict(K=K,b=b,C=c,U0=u0,S=scale,eta=eta,sigma=sigma,tau=tau,E=error,
                V=v,B=B,traction_norms=norms,
                traction_native=[eta*n*v for n in norms],
                traction_analytic=[n*error+eta*n*v for n in norms],
                energy_native=energy_native,
                energy_analytic=sum(abs(dot(row,U)) for row in K)*error+B*error*error/2+energy_native)


def report():
    values = budgets()
    values.pop('K')
    values.pop('b')
    values.update(coefficients=U,stress=stress(),energy_J=Q(261,32),
                  left_traction_N=[dot(row,U) for row in traction_rows()])
    def rational(value):
        if isinstance(value,list):
            return [rational(v) for v in value]
        return str(value)
    return {k:rational(v) for k,v in values.items()}

if __name__ == '__main__':
    print(json.dumps(report(), indent=2, sort_keys=True))
