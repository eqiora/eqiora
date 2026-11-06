# Rational Q2 polynomial patch and physical reactions

`polynomial.py` is a standalone rational reference. It imports no Eqiora code,
reads no Eqiora output as an oracle, and runs no native solve. All mathematical
quantities, comparisons and thresholds below use exact Python fractions. The
protocol and budgets are declared before any candidate comparison.

## Fixed problem

Use boxes L=(2,3) m and L=(2,3,4) m, with two cells in every axis and full tensor
Q2 nodes. The dimensionless solution is

```
p_i(x_i) = 1 + (x_i/L_i)^2
u = product_i p_i
f = -sum_i (2/L_i^2) product_(j != i) p_j.
```

Impose the exact value of u on the **upper x face only**. On every other face,
impose the analytic outward flux. Lower-face fluxes vanish. Upper faces in axes
i>0 have flux (2/L_i) product_(j != i) p_j. The normal is outward; f=-Delta u.

The basis on each [0,1] cell is (2t²-3t+1, 4t-4t², 2t²-t).
The implementation integrates these polynomials and their derivatives directly
with rational coefficients. With h_i=L_i/2, form the complete five-by-five axis
mass and stiffness matrices M_i and K_i, and axis vectors

```
a_i[k] = integral phi_k dx_i
b_i[k] = integral phi_k p_i dx_i.
```

Retain all 5^d physical nodes, including every boundary, edge, face and cell
interior node. Exact tensor assembly gives

```
K = sum_i K_i tensor product with M_j in all other axes
F[row] = -sum_i (2/L_i^2) a_i[row_i] product_(j != i) b_j[row_j]
G[row] = sum_(i>0, row_i=upper) (2/L_i) product_(j != i) b_j[row_j].
r = K*v - F - G.
```

Natural-face contributions at intersections with the essential face remain in G.
They must not be dropped merely because that coefficient is constrained. The
reaction convention is the **unscaled physical** product-minus-load residual r.
For the exact nodal coefficients, every free residual is exactly zero. The sum
of constrained reactions is the integral of the upper-x flux:

```
(2/L_x) product_(j>0) integral_0^L_j p_j dx_j
= (2/L_x) product_(j>0) (4 L_j/3)
= +4       in 2D
= +64/3    in 3D.
```

Reactions have units m^(d-2), since u is dimensionless. They are dual nodal loads,
not a pointwise normal derivative. Four-point tensor Gauss quadrature integrates
all volume and face polynomial terms here exactly: the largest per-axis degree
is four. The reference itself performs polynomial integration, not numerical
quadrature.

## Fixed coefficient and residual budgets

Require every essential coefficient to equal its exact dyadic value, with no
boundary projection or tolerance. Let epsilon=2^-40. On the free physical vector,
compare **sum r_i² <= epsilon²** as rational numbers, avoiding an approximate
square root. Also check every supplied coefficient against its exact nodal u.
The coefficient error bounds are fixed:

```
Bu_2D = (3200/3) epsilon
Bu_3D = 16000 epsilon.
```

To derive these constants, the one-essential-face Poincare lower bound is
lambda>=9/16: the exact mixed box eigenvalue is pi²/(4 L_x²)=pi²/16>9/16.
The full one-dimensional Q2 mass matrix has Gershgorin lower bound h_i/30;
therefore mu=product_i(h_i/30) bounds the tensor mass eigenvalue below, including
its free principal restriction. The free stiffness eigenvalue is at least
(9/16)mu. Hence ||v-u||_infinity <= ||v-u||_2 <= epsilon/((9/16)mu), giving the
stated Bu values. Essential coefficients are exact, so they contribute no error.
A plausible coefficient error cannot compensate for a failed residual check, or
vice versa: both must pass.

## Fixed reaction budgets

The declared full tensor row-sum bound is

```
Kinf = sum_i [32/(3h_i)] product_(j != i) [2h_j/3]
Ustar = 2^d
Binf = ||F+G||_infinity
rho = 2^-40 max(1, Kinf Ustar + Binf).
```

Here 32/(3h_i) bounds the stiffness row-sum norm and 2h_i/3 bounds the mass
row-sum norm, including absolute values of negative mass entries. Ustar bounds
the exact patch coefficients. Binf comes from the independent rational load.
The factor 2^-40 equals 8192 times binary64 unit roundoff 2^-53. **This is a
declared 8192-unit-roundoff acceptance ceiling, not a universal proved bound on
floating-point assembly or solving.** Observed residuals and reactions never
increase this ceiling.

For each upper-x node, compare the native physical reaction with the independent
r computed from the **actual complete exported coefficient vector**, requiring
absolute difference <=rho. Match by complete coordinate identity, not by record
position, a projected vertex set, or summed reactions alone. With m=5 in 2D and
m=25 in 3D, also require

```
|sum native - sum actual r| <= m rho
|sum native - analytic target| <= m Kinf Bu + m rho.
```

The latter uses the coefficient bound and the full K row-sum bound. Every check is
reported independently, and a failure rejects the candidate without fitting or
inflating any threshold.

## Input and producer responsibilities

The strict JSON schema has exactly these keys:

```json
{
  "schema": 1,
  "lengths_m": [2, 3],
  "N": 2,
  "gauss_points": 4,
  "records": [
    {"coordinates_m": ["0", "0"], "value_binary64_hex": "0x1.0000000000000p+0"}
  ],
  "upper_reactions": [
    {"coordinates_m": ["2", "0"], "value_binary64_hex": "0x0.0p+0"}
  ]
}
```

The arrays above illustrate record shape only. A valid document includes all 5^d
coefficient records and all 5^(d-1) upper-x reaction records. Both arrays may use
any order. Coordinates must be exact decimal strings equal to k L_i/4. Values
must be finite canonical `float.hex()` strings; `as_integer_ratio` captures their
exact binary64 rational value. Decimal approximations of binary64 coefficients
are not substituted. Missing, duplicate, off-grid or wrong-dimensional nodes,
wrong-face reactions, extra/missing keys, duplicate JSON keys, nonfinite values,
unsupported controls, and incorrect essential values are typed rejections.

The `gauss_points:4` metadata is a required input assertion, **not proof of the
production assembly rule**. A separate producer receipt must verify actual Plan,
mesh, selected field, essential/natural policies and quadrature controls, bind the
input and artifacts by SHA-256, and establish that reactions were converted from
native dual storage to unscaled physical values with dimension m^(d-2). This
mathematical oracle does not parse Eqiora Plans or infer those facts from numbers.

```sh
python3 polynomial.py patch-input.json --output patch-comparison.json
python3 -m unittest discover -s . -p 'test_polynomial.py'
```

The CLI output includes SHA-256 of the exact raw input bytes, counts, exact rational
observations and fixed budgets, each check, and overall acceptance. Fractions are
serialized as decimal numerator/denominator strings, without floating-point
summaries. Exit status is zero for acceptance, one for a valid comparison that
fails a fixed check, and two for malformed or unsupported input. No reference
table or candidate-fitted expectation is generated.

Tests check exact zero free residual and analytic reaction sums in both boxes,
partition of unity and direct polynomial integration, the independent spectral
budget constants and full K row-sum bound, and exact identity under record
reordering. Falsifiers perturb a free coefficient, reaction component, reaction
sign or coordinate association, and remove or reverse the natural flux in the
reference equations. Malformed grids, wrong face, incorrect essential values and
unsupported controls fail explicitly. All test inputs are synthetic mathematics;
no production solver or native candidate output supplies an expectation.
