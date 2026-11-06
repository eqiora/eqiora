# Independent affine Q2 sine reference

This directory computes a mathematical reference without importing Eqiora, reading its
source or output, or running its solvers. It solves only the 4-by-4 and 8-by-8 sine
amplitude systems derived below. The interval endpoints are generated expectations,
subject to scientific review before candidate comparisons.

**The declared three-point load rule fails the fixed 1% perturbation budget in 3D at
N=2.** The generator writes that failure and exits unsuccessfully. The separate
four-point table is a reference-only quadrature-selection audit. Neither table
approves a candidate production rule. Do not change the budget to accept a rule.

Run with the Python standard library:

```sh
python3 generate.py --output intervals.json
# Expected nonzero exit: three-point budget failure, table retained.
python3 generate.py --gauss-points 4 --output intervals-gauss4-audit.json
python3 -m unittest discover -s . -p 'test_*.py'
```

The default precision is 80 decimal digits; `--precision 110` provides a separate
precision audit. Missing output, unsupported rules, boxes, mesh sizes or precision,
zero-containing pivots, and uncertified positive norm squares fail explicitly.
Precision audits support debugging; they do not establish interval validity.

## Model and local integration

On boxes of lengths (2,3) m and (2,3,4) m, use N=2,4,8 cells in every axis,
u=product_i sin(pi x_i/L_i), and f=lambda u, with
lambda=pi² sum_i L_i^-2. All boundary values vanish. The function u is dimensionless.
The reported norms are the continuous physical L2 error E0 and H1 **seminorm**
error E1, with units m^(d/2) and m^(d/2-1). Rates are dimensionless.

For t in [0,1], the nodal polynomials at left, midpoint and right are
b0=2t²-3t+1, bm=4t(1-t), and b1=2t²-t. Exact rational integration gives

```
M_h = h/30 [[4,2,-1], [2,16,2], [-1,2,4]]
K_h = 1/(3h) [[7,-8,1], [-8,16,-8], [1,-8,7]].
```

Tensor products give the full Q2 space, including cross terms in each coordinate.
This is the tensor polynomial space illustrated by
[DefElement's degree-two quadrilateral Lagrange example](https://defelement.org/elements/examples/quadrilateral-lagrange-equispaced-2.html).
The focused tests integrate these polynomials using exact fractions, independently
of the generator's matrix expressions.

## Sine closure and amplitude equations

Put theta=pi/N and c=cos(theta/2). Endpoint coefficients are aV sin(j theta),
while midpoint coefficients are aM sin((j+1/2)theta). Assembly and the addition
formulas for sine show closure on this two-amplitude space:

```
Mbar_i = h_i/30 [[8-2cos(theta),4c], [4c,16]]
Kbar_i = 1/(3h_i) [[14+2cos(theta),-16c], [-16c,16]].
```

The two coefficient families have the same sum of squared sine values N/2.
They occupy disjoint nodes. Thus the physical quadratic forms obtain a common
factor w=(N/2)^d when reduced to amplitude space. Tests assemble independent
1D global matrices for N=2 and N=4 and apply both sine families to verify closure.

For the exact load in one dimension, symmetry about a vertex or cell midpoint
cancels odd sine terms, giving

```
qV = 2h integral_0^1 b0(t) cos(theta t) dt
   = 2h [(cos(theta)+3)/theta² - 4sin(theta)/theta³]
qM = h integral_0^1 bm(t) cos(theta(t-1/2)) dt
   = h [16sin(theta/2)/theta³ - 8cos(theta/2)/theta²].
```

These formulas follow by integrating the polynomial times cosine twice, retaining
the remaining constant-times-cosine integral. Tests separately integrate the
Taylor polynomials, including remainder bounds, to check both formulas.
With tensor ordering V before M in each axis, define

```
M = tensor_i Mbar_i
A = sum_i (Kbar_i tensor product with Mbar in every other axis)
q = tensor_i q_i
A a_exact = lambda q.
```

The tensor ordering is explicit in `itertools.product`; no Eqiora ordering is
assumed. All eigenmode systems are solved with interval Gaussian elimination,
checking every pivot excludes zero. Inclusion arithmetic encloses the exact
elimination applied to the fixed real matrix and load. No matrix inversion or
floating point solve supplies expected values.

## Continuous errors and quadrature perturbation

Let C=product_i L_i/2^d, so integral u²=C and integral |grad u|²=lambda C.
For any amplitude vector a, exact continuous integration gives

```
E0(a)² = C - 2w q^T a + w a^T M a
E1(a)² = lambda C - 2lambda w q^T a + w a^T A a.
```

The gradient cross term follows by integration by parts with zero boundary trace.
The exact-load Galerkin energy identity a^T A a=lambda q^T a is checked independently
of the error evaluation. Errors are evaluated from the general quadratic formulas.

For each declared Gauss rule, replace the two one-dimensional integrals defining
q by weighted sums to obtain q_g, and solve A a_g=lambda tensor_i q_gi.
The exact q remains in the continuous error formulas for a_g. The load perturbation
norms are D0²=w delta^T M delta and D1²=w delta^T A delta, delta=a_g-a_exact.
Since these quadratic forms are nonnegative, their interval lower endpoint may be
intersected with zero before taking a square root. This intersection does not turn
a negative upper bound into a valid norm.

The fixed budget is D_j <= 0.01 E_j(a_exact) at every level and in both norms.
A budget passes only when the upper endpoint of the ratio interval is <=0.01.
Both exact-load and Gauss-load finite rates are enclosed as
log(E_j(N)/E_j(2N))/log(2). The finite tables do not assert that N=2,4,8 are
asymptotic. The theoretical limiting Q2 orders (L2 three, H1 seminorm two) are
separate statements from these finite rates.

## Declared Gauss rules

Both rules map standard abscissas z on [-1,1] to t=(1+z)/2 and halve the weights.
For three points, z=0, +/-sqrt(3/5), with mapped weights 4/9, 5/18, 5/18.
For four points,

```
z_inner = +/-sqrt((3-2sqrt(6/5))/7), mapped weight (18+sqrt(30))/72
z_outer = +/-sqrt((3+2sqrt(6/5))/7), mapped weight (18-sqrt(30))/72.
```

The nodes and weights are evaluated as intervals. They can be derived from roots
of P3(z)=(5z³-3z)/2 and P4(z)=(35z⁴-30z²+3)/8 with weights
2/[(1-z²)P'_n(z)²]. The tests verify mapped polynomial moments through degree 2n-1.
The Gauss construction and exactness statement are described by
[NIST DLMF §3.5(v)](https://dlmf.nist.gov/3.5#v).
Both rules integrate Q2 mass and stiffness exactly on affine constant-coefficient
cells. Neither integrates the sine forcing exactly.

## Why the endpoints enclose the real values

Every basic operation computes lower and upper endpoints using separate Decimal
contexts with FLOOR and CEILING rounding. The precision is never inherited from
Python's default context. Input constants are integers or exact decimal strings.
Reciprocal intervals must exclude zero. Products inspect all endpoint pairs.

Pi uses Machin's identity 16 atan(1/5)-4 atan(1/239). For each arctangent, an even
number of alternating terms gives a lower partial sum; the next positive term
bounds the remainder above. Sine and cosine use Taylor polynomials on the certified
range [-2,2], plus the Taylor remainder |x|^(m+1)/(m+1)!, because every derivative
has magnitude at most one on the real axis. All polynomial arithmetic and bounds
are themselves outward rounded. No library float transcendental supplies a bound.

[Python's Decimal documentation](https://docs.python.org/3/library/decimal.html)
specifies correctly rounded sqrt and ln with HALF_EVEN rounding, including when a
context requests a different rounding mode. Accordingly, the implementation uses
a HALF_EVEN context and moves to the immediately adjacent representable value
below and above the rounded result. Monotonicity then encloses sqrt or ln of the
whole input interval. This does not assume these functions honor FLOOR/CEILING.

## Candidate comparison contract

The frozen table generation records `candidate_comparator_implemented` as false.
The full-vector comparator described below now implements the mathematical checks;
production assembly controls still require independent confirmation. It must apply the complete rational tensor operator and declared Gauss
load to the complete physical coefficient vector, enclosing rounding in the
residual r. With mu=product_i(h_i/15), mass Gershgorin and box Poincare imply
lambda_min(A_global)>=lambda mu. Hence B1=||r||2/sqrt(lambda mu) bounds the energy
norm solve error and B0=B1/sqrt(lambda) bounds the L2 solve error.

Solve error, norm integration uncertainty and reference enclosure uncertainty
together must be <=1e-3 E_j(a_g) at every level. This gives the two-level rate
uncertainty bound log2(1.001/0.999). No absolute endpoint tolerance is inferred
from these reference values. A higher-order quadrature norm audit must compare to
the exact-integral quadratic norm for the full coefficient vector; agreement of
two floating point quadrature orders alone is not an integration bound. The
production quadrature choice and general selection path remain to be reviewed and
frozen before any candidate run.

## Full physical coefficient comparator

`compare.py` now implements the full-vector residual and exact norm comparison
specified above. The frozen tables retain their original
`candidate_comparator_implemented: false` generation metadata; no table endpoints
or generation records were changed when adding the comparator. Production rule
selection and candidate use still require the stated review.

Each input file is a JSON object with **exactly** these keys:

```json
{
  "schema": 1,
  "lengths_m": [2, 3],
  "N": 2,
  "gauss_points": 4,
  "records": [
    {"coordinates_m": ["0", "0"], "value_binary64_hex": "0x0.0p+0"}
  ]
}
```

The example shows only one record; a valid file must supply all (2N+1)^d physical
nodes, including edges, faces, cell-interior Q2 nodes, and boundaries. Ordering is
arbitrary. Coordinates are exact decimal strings for x_i=k L_i/(2N). These boxes
and meshes have dyadic coordinates, so decimal coordinate validation needs no
rounding tolerance. Every node must appear exactly once. Vertex-only data, extra
or off-grid coordinates, missing nodes, duplicate coordinates, duplicate JSON
keys, unexpected keys, nonfinite numbers, wrong dimensions and unsupported controls
are rejected.

Coefficient values must be canonical Python `float.hex()` strings. The comparator
parses the binary64 and converts it with `Decimal.from_float`, preserving its
exact binary rational value, including subnormals. A decimal approximation of that
value is not accepted as a substitute. Boundary coefficients must be exactly zero
(signed zero is also zero); a nonzero boundary value is rejected with error code
`boundary`. The comparator does not silently impose boundary conditions or discard
any supplied nonzero boundary values.

For the interior vector, assemble the one-dimensional mass and stiffness matrices
from the exact local rational formulas, then apply tensor products by sparse axis
contractions. No amplitude projection is used. Assemble the four-point Gauss load
independently cell by cell on the physical grid, evaluating the sine with the
certified identity sin(pi s)=cos(pi(s-1/2)). The exact continuous cross integral uses
the analytic one-dimensional nodal load integrals, including endpoint/midpoint sine
factors. Consequently, both norm quadratic formulas above apply to arbitrary
complete coefficient vectors, not only vectors in the sine-mode space. Outward
interval rounding encloses matrix application, forcing, residual, norms and bounds.
The norm integration uncertainty is the complete resulting interval width; it
includes arithmetic and transcendental uncertainty. No quadrature approximation is
used to integrate the candidate's norms.

```sh
python3 compare.py coefficients.json --output comparison.json
# A JSON array containing one complete input for each N=2,4,8:
python3 compare.py levels.json --levels --output comparison-levels.json
```

Exit status is zero only when every budget and reference consistency check passes,
one for a valid comparison that fails its budget, and two for malformed or
unsupported input. Output includes the frozen reference file's SHA-256, complete,
interior and boundary coefficient counts, physical norm units, continuous error
intervals, Euclidean residual interval, lambda, mu, B0/B1, and separate reference,
norm integration, solve and total uncertainty bounds. Reference and norm widths
are counted in full, conservatively; the solve bound is counted by its upper
endpoint. The allowed bound is exactly 1e-3 times the reference Gauss-load norm,
with no candidate-derived threshold. Acceptance requires the total upper bound
not exceed the allowed lower bound, and the candidate norm interval intersect the
reference interval enlarged by the solve bound.

Rates are emitted only for a complete, same-box N=2,4,8 set. Each rate has a measured
continuous norm interval and an interval enlarged by the two independently
computed total uncertainty bounds. The predeclared log2(1.001/0.999) bound is also
reported. A failed comparison remains failed even when its rate looks plausible.
Synthetic tests check arbitrary 3D polynomial coefficients against analytic full
integrals, rational dense tensor application, exact single-node perturbation norms,
complete-level rate propagation, budgets and malformed input. These tests use no
production solver or exported candidate.
