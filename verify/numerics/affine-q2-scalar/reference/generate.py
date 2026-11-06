#!/usr/bin/env python3
"""Independent, outward-rounded sine-mode Q2 reference. No Eqiora dependency."""
import argparse
import itertools
import json
from decimal import Decimal as D, Context, ROUND_FLOOR, ROUND_CEILING, ROUND_HALF_EVEN
from pathlib import Path

PRECISION = 80
DOWN = UP = NEAR = None


def precision(digits):
    global PRECISION, DOWN, UP, NEAR
    if not isinstance(digits, int) or digits < 60 or digits > 200:
        raise ValueError("precision must be an integer in [60, 200]")
    PRECISION = digits
    DOWN = Context(prec=digits, rounding=ROUND_FLOOR)
    UP = Context(prec=digits, rounding=ROUND_CEILING)
    NEAR = Context(prec=digits, rounding=ROUND_HALF_EVEN)


precision(80)


class I:
    def __init__(self, lo, hi=None):
        self.lo = D(lo)
        self.hi = D(lo if hi is None else hi)
        if not self.lo.is_finite() or not self.hi.is_finite() or self.lo > self.hi:
            raise ValueError("invalid finite interval")

    def __add__(self, other):
        b = interval(other)
        return I(DOWN.add(self.lo, b.lo), UP.add(self.hi, b.hi))
    __radd__ = __add__

    def __neg__(self):
        return I(self.hi.copy_negate(), self.lo.copy_negate())

    def __sub__(self, other):
        return self + -interval(other)

    def __rsub__(self, other):
        return interval(other) + -self

    def __mul__(self, other):
        b = interval(other)
        pairs = list(itertools.product((self.lo, self.hi), (b.lo, b.hi)))
        return I(min(DOWN.multiply(a, c) for a, c in pairs),
                 max(UP.multiply(a, c) for a, c in pairs))
    __rmul__ = __mul__

    def __truediv__(self, other):
        b = interval(other)
        if b.lo <= 0 <= b.hi:
            raise ValueError("interval denominator contains zero")
        return self * I(DOWN.divide(D(1), b.hi), UP.divide(D(1), b.lo))

    def __rtruediv__(self, other):
        return interval(other) / self

    def __pow__(self, exponent):
        if not isinstance(exponent, int) or exponent < 0:
            raise ValueError("only nonnegative integer powers")
        result = I(1)
        for _ in range(exponent):
            result = result * self
        return result

    def square(self):
        if self.lo <= 0 <= self.hi:
            return I(0, max(UP.multiply(self.lo, self.lo), UP.multiply(self.hi, self.hi)))
        return I(min(DOWN.multiply(self.lo, self.lo), DOWN.multiply(self.hi, self.hi)),
                 max(UP.multiply(self.lo, self.lo), UP.multiply(self.hi, self.hi)))

    def sqrt(self):
        if self.lo < 0:
            raise ValueError("negative square-root interval")
        # Decimal sqrt and ln are correctly rounded HALF_EVEN, irrespective of
        # rounding mode. One representable neighbour on each side encloses them.
        lo, hi = NEAR.sqrt(self.lo), NEAR.sqrt(self.hi)
        return I(0 if self.lo == 0 else NEAR.next_minus(lo), NEAR.next_plus(hi))

    def ln(self):
        if self.lo <= 0:
            raise ValueError("nonpositive logarithm interval")
        return I(NEAR.next_minus(NEAR.ln(self.lo)), NEAR.next_plus(NEAR.ln(self.hi)))

    def json(self):
        return [str(self.lo), str(self.hi)]


def interval(x):
    return x if isinstance(x, I) else I(x)


def abs_upper(x):
    return max(x.lo.copy_abs(), x.hi.copy_abs())


def atan_inverse(q):
    # Alternating series: two consecutive partial sums bracket atan(1/q).
    x = I(1) / q
    total = I(0)
    term = x
    for k in range(2 * PRECISION):
        total += ((-1) ** k) * term / (2*k+1)
        term *= x*x
    next_term = term / (4*PRECISION+1)
    return total + I(0, next_term.hi)  # even term count; next term positive


def pi():
    return 16*atan_inverse(5) - 4*atan_inverse(239)


def trig(x, cosine=False):
    x = interval(x)
    if abs_upper(x) > 2:
        raise ValueError("Taylor argument outside certified range [-2,2]")
    # Taylor theorem |R_m| <= |x|^(m+1)/(m+1)! for sine/cosine.
    count = 2*PRECISION
    term = I(1) if cosine else x
    total = term
    for k in range(1, count):
        a = 2*k-1 if cosine else 2*k
        term = -term*x*x/(a*(a+1))
        total += term
    degree = 2*(count-1) + (0 if cosine else 1)
    bound = I(1)
    for k in range(1, degree+2):
        bound = bound * abs_upper(x) / k
    return total + I(bound.hi.copy_negate(), bound.hi)


def tensor(a, b):
    return [[x*y for x in rowa for y in rowb] for rowa in a for rowb in b]


def product(items):
    result = I(1)
    for item in items:
        result *= item
    return result


def kron(matrices):
    result = [[I(1)]]
    for matrix in matrices:
        result = tensor(result, matrix)
    return result


def dot(a, b):
    return sum((x*y for x, y in zip(a, b)), I(0))


def apply(a, x):
    return [dot(row, x) for row in a]


def solve(a, b):
    a = [list(row) + [rhs] for row, rhs in zip(a, b)]
    n = len(b)
    for k in range(n):
        if a[k][k].lo <= 0 <= a[k][k].hi:
            raise ValueError("uncertified elimination pivot")
        for j in range(k+1, n):
            factor = a[j][k]/a[k][k]
            for l in range(k+1, n+1):
                a[j][l] -= factor*a[k][l]
    result = [I(0)]*n
    for k in reversed(range(n)):
        result[k] = (a[k][n]-dot(a[k][k+1:n], result[k+1:]))/a[k][k]
    return result


def gauss_rule(points):
    if points == 3:
        root = (I(3)/5).sqrt()
        return [(1-root)/2, I(1)/2, (1+root)/2], [I(5)/18, I(4)/9, I(5)/18]
    if points == 4:
        inner = ((3-2*(I(6)/5).sqrt())/7).sqrt()
        outer = ((3+2*(I(6)/5).sqrt())/7).sqrt()
        small, large = (18-I(30).sqrt())/72, (18+I(30).sqrt())/72
        return [(1-outer)/2, (1-inner)/2, (1+inner)/2, (1+outer)/2], [small, large, large, small]
    raise ValueError("only independently declared 3- or 4-point Gauss rules")


def one_dimensional(length, n, theta, points=3):
    h = I(length)/n
    ct, st = trig(theta, True), trig(theta)
    ch, sh = trig(theta/2, True), trig(theta/2)
    mass = [[h*(8-2*ct)/30, h*4*ch/30], [h*4*ch/30, h*16/30]]
    stiff = [[(14+2*ct)/(3*h), -16*ch/(3*h)], [-16*ch/(3*h), 16/(3*h)]]
    # Direct integration of the local endpoint and midpoint polynomials.
    q = [2*h*((ct+3)/theta**2-4*st/theta**3),
         h*(16*sh/theta**3-8*ch/theta**2)]
    nodes, weights = gauss_rule(points)
    qg = [2*h*sum((w*(2*t*t-3*t+1)*trig(theta*t, True)
                      for t, w in zip(nodes, weights)), I(0)),
          h*sum((w*4*t*(1-t)*trig(theta*(t-I(1)/2), True)
                 for t, w in zip(nodes, weights)), I(0))]
    return mass, stiff, q, qg


def level(lengths, n, points=3):
    if tuple(lengths) not in ((2, 3), (2, 3, 4)) or n not in (2, 4, 8):
        raise ValueError("unsupported box or mesh")
    d = len(lengths)
    p = pi()
    lam = p*p*sum((I(1)/(l*l) for l in lengths), I(0))
    axes = [one_dimensional(l, n, p/n, points) for l in lengths]
    mass = kron([axis[0] for axis in axes])
    terms = [kron([axis[1 if j == k else 0] for j, axis in enumerate(axes)]) for k in range(d)]
    stiff = [[sum((term[i][j] for term in terms), I(0))
              for j in range(2**d)] for i in range(2**d)]
    q = [product(v) for v in itertools.product(*(axis[2] for axis in axes))]
    qg = [product(v) for v in itertools.product(*(axis[3] for axis in axes))]
    exact, gauss = solve(stiff, [lam*x for x in q]), solve(stiff, [lam*x for x in qg])
    w = I(n//2)**d
    c = product(lengths)/(2**d)
    def errors(a):
        cross = w*dot(q, a)
        return [(c-2*cross+w*dot(a, apply(mass, a))).sqrt(),
                (lam*c-2*lam*cross+w*dot(a, apply(stiff, a))).sqrt()]
    e, eg = errors(exact), errors(gauss)
    delta = [g-x for g, x in zip(gauss, exact)]
    distances = []
    for matrix in (mass, stiff):
        squared = w*dot(delta, apply(matrix, delta))
        # A positive quadratic form may have an interval lower bound below zero.
        distances.append(I(max(D(0), squared.lo), squared.hi).sqrt())
    ratios = [distances[j]/e[j] for j in range(2)]
    budget = all(r.hi <= D('0.01') for r in ratios)
    energy_defect = dot(exact, apply(stiff, exact))-lam*dot(q, exact)
    if not energy_defect.lo <= 0 <= energy_defect.hi:
        raise ValueError("Galerkin energy identity failed")
    return {"N": n, "exact_load_errors": [x.json() for x in e],
            f"gauss{points}_load_errors": [x.json() for x in eg],
            "load_perturbation_norms": [x.json() for x in distances],
            "load_perturbation_relative_to_exact_error": [x.json() for x in ratios],
            "quadrature_budget_certified": budget,
            "galerkin_energy_defect": energy_defect.json()}


def generate(digits=80, points=3):
    precision(digits)
    boxes = []
    for lengths in ((2, 3), (2, 3, 4)):
        levels = [level(lengths, n, points) for n in (2, 4, 8)]
        rates = {}
        for load in ("exact_load_errors", f"gauss{points}_load_errors"):
            rates[load] = [[(I(*a[load][j])/I(*b[load][j])).ln().__truediv__(I(2).ln()).json()
                            for j in range(2)] for a, b in zip(levels, levels[1:])]
        boxes.append({"lengths_m": list(lengths), "norm_units": [f"m^({len(lengths)}/2)",
                      f"m^({len(lengths)-2}/2)"], "levels": levels, "rates": rates})
    return {"schema": 1, "method": "directed Decimal interval arithmetic; Machin pi; Taylor remainder; interval elimination",
            "decimal_precision": digits, "load_quadrature": f"{points}-point Gauss-Legendre per cell axis",
            "quadrature_status": "reference-only; candidate production selection requires review",
            "error_order": ["continuous physical L2", "continuous physical H1 seminorm"],
            "quadrature_budget": "D_j <= 0.01 E_j,exact",
            "candidate_comparator_implemented": False, "boxes": boxes}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--precision', type=int, default=80)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--gauss-points', type=int, choices=(3, 4), default=3)
    args = parser.parse_args()
    table = generate(args.precision, args.gauss_points)
    args.output.write_text(json.dumps(table, indent=2)+'\n')
    failed = [(b['lengths_m'], l['N']) for b in table['boxes'] for l in b['levels']
              if not l['quadrature_budget_certified']]
    if failed:
        raise SystemExit(f"{args.gauss_points}-point Gauss load fails predeclared budget: {failed}")


if __name__ == '__main__':
    main()
