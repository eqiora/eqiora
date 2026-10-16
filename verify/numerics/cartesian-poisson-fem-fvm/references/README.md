# Independent sine-mode reference

The continuum solution is `u = product_j sin(pi x_j)` on the unit box with
homogeneous Dirichlet data and source `d pi^2 u`; see
[`models/problem.md`](../models/problem.md). The same derivation serves the
[three-dimensional case](../../cartesian-poisson-3d-fem-fvm/README.md).

`generate.py` computes both convergence tables at 80 decimal digits using only
scalar analytic formulas. It imports no Eqiora code and reads no solver output.
Run from the repository root:

```bash
uv run --no-project verify/numerics/cartesian-poisson-fem-fvm/references/generate.py --check
```

Omit `--check` to regenerate the tables. They contain continuous L2 errors of the
exact discrete solutions, not snapshots of floating-point execution. Observed
orders and balance residuals are no longer stored as reference values: the tests
check decreasing errors, order greater than 1.9, and relative global balance below
`2e-11` directly.

## Discrete solutions

Write `h = 1/n`, `theta = pi h`. On the interior nodal sine vector the 1D Q1
stiffness and consistent mass matrices have eigenvalues

```text
k = 4 sin(theta/2)^2 / h
m = h (2 + cos(theta)) / 3.
```

The prescribed two-point Gauss load rule has nodal coefficient

```text
b = h sum_t (1-t) cos(theta t),  t = (1 ± 1/sqrt(3))/2.
```

Indeed, the two cells adjacent to a node contribute opposite sine phases;
their sum is `2 sin(theta i) cos(theta t)`, cancelling the quadrature weight
`1/2`. The tensor stiffness eigenvalue is `d k m^(d-1)` and the source
coefficient is `d pi^2 b^d`. Thus the FEM nodal amplitude is

```text
A_FEM = pi^2 b^d / (k m^(d-1)).
```

For centroid-source TPFA, the cell-centered sine vector is an eigenvector with
eigenvalue `4 d h^(d-2) sin(theta/2)^2`. The half-cell boundary distances give
the same eigenrelation in boundary rows. Its amplitude is

```text
A_FVM = pi^2 h^2 / (4 sin(theta/2)^2).
```

FEM reconstructs on nodes `0,h,...,1`. FVM reconstructs on dual-grid nodes
`0,h/2,3h/2,...,1-h/2,1`. In either case let `p` be the piecewise linear
interpolant of `sin(pi x)` at these nodes. The reconstructed field is
`A product_j p(x_j)`. For each segment `[x,y]` with endpoint values `a,b`,

```text
integral p^2 = (y-x) (a^2 + a b + b^2)/3
integral p sin(pi t) dt
  = (a cos(pi x) - b cos(pi y))/pi
    + (b-a) (sin(pi y) - sin(pi x))/((y-x) pi^2).
```

Summing these gives `M = integral p^2` and `B = integral p sin(pi x)`, hence
`E^2 = A^2 M^d - 2 A B^d + 2^(-d)`. No linear solve or multidimensional
numerical integration enters the reference.

## Comparison allowance

The shared Rust fixture uses the requested solver residual target (`rtol=1e-12`,
`atol=1e-14`) and an a priori forward-error allowance. For these uniform meshes,
`lambda_min(K) >= 4 d h^(d-2) sin(pi h/2)^2 / 3^(d-1)` for FEM: each 1D
mass eigenvalue is at least `h/3`. TPFA uses the same bound without the divisor.
The reconstruction mass norm is at most `h^d`, including the dual-grid boundary
segments. Therefore a Euclidean residual `r` changes the continuous L2 norm by
at most `h^(d/2) ||r|| / lambda_min(K)`.

There are at most `n^d` coordinates, each load entry is bounded by
`d pi^2 h^d`, and each absolute stiffness row sum by `4 d h^(d-2)`.
Both analytic amplitudes are below 2 on the declared meshes. In addition to the
requested residual target, the fixture reserves a local floating-point
backward-error allowance

```text
gamma(4096) n^(d/2) (d pi^2 h^d + 2 * 4 d h^(d-2)),
gamma(k) = k eps / (1-k eps), eps = f64::EPSILON.
```

4096 is a conservative arithmetic allowance for these bounded Q1/TPFA stencils
(at most eight incident cells, eight assembly quadrature points, eight basis
functions and three derivative components), including elementary-function and
coordinate evaluation. This is a numerical regression allowance, not a certified
interval bound for arbitrary backends or meshes. It is fixed independently of
observed solver errors. Summation of the positive squared-error samples reserves
`gamma(((n+1)*8)^d) E`, plus `128 eps` for field/norm evaluation.

Error measurement uses eight Gauss points per axis, independently of the
assembly rule. Its analytic truncation is below the roundoff allowance: on each
segment, `|p| <= 1`, `|p'| <= pi` and `A < 2`. The 16th derivative of the squared
error in any one coordinate is bounded by
`D = 68 pi^16 + (2 pi)^16/2`. The tensor quadrature squared-norm error is at most
`d (8!)^4 / (17 (16!)^3) h^16 D`. Dividing by the reference norm bounds the
norm error; this is below `2e-18` for every declared row. The original four-point
measurement did not resolve the continuous norm at the former snapshot tolerance.

These cases verify the declared uniform, scalar, real-valued, host-CPU profiles.
They make no bitwise reproducibility, general-mesh, alternate-backend, or
arbitrary nonpolynomial quadrature-exactness claim.
