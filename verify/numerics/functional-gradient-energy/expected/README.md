# Independent Q1 reference

Each cell has side h=1/2. In reference coordinates (s,t) in [0,1]², the central
corner hat on any cell is one of the reflections of N=s*t. Hence

```text
integral_cell |grad(N)|² dA = integral_0^1 integral_0^1 (s²+t²) ds dt = 2/3,
integral_cell N dA = h²/4 = 1/16.
```

Summing four cells gives K=8/3 and b=1/4. The eight boundary coefficients are zero,
the central coefficient is 3/32, and the energy is -3/256. Two-point tensor Gauss
quadrature integrates the Q1 energy exactly in each reference coordinate.

For the physical profile, u and w have dimension m, k has J/m², and f has J/m³.
Thus both the directional variation and F have dimension J. This is a two-
dimensional functional, not a three-dimensional elastic material or a stability
claim. The tested physical value type must be exactly real J.

The solver requests relative tolerance 1e-10 and absolute tolerance 1e-12.
With a single free coefficient, the load-scaled residual bound divided by K=8/3
is below 1e-10. The coefficient and energy assertions use absolute tolerance
1e-10 in their stated SI coordinates. At the stationary coefficient, the energy
perturbation is K*(delta u)²/2; the same tolerance also covers the few binary64
quadrature operations. These bounds come from this fixed discrete problem, not
from measured output. No continuum value is substituted for the Q1 energy.

## Essential left side and three zero-flux sides

On the same four cells, the exact discrete solution is independent of y and
interpolates u=x−x²/2. Its values are 0, 3/8 and 1/2 at the three x coordinates;
its piecewise x slopes are 3/4 and 1/4. Direct integration gives internal energy
( (3/4)² + (1/4)² )/4 = 5/32 and load pairing
(0+2*(3/8)+1/2)/4 = 5/16, hence F=−5/32. This is the discrete Q1 energy,
not the continuum energy −1/6.

For local vertex order (00,10,01,11), the unit-diffusion element stiffness has
diagonal 2/3, edge-neighbor entries −1/6 and opposite-corner entries −1/3.
Assemble the four cells and delete the three x=0 nodes. Exact rational elimination
of this six-by-six matrix gives inverse infinity norm 234/49. The free load
vector, in y-major order, is (1/8,1/16,1/4,1/8,1/8,1/16), with squared two-norm
15/128. Thus the requested relative residual 1e−10 gives coefficient error at
most (234/49)*sqrt(15/128)*1e−10 < 1.7e−10. The coefficient assertions use 1e−9
to cover binary64 assembly and solve rounding. At stationarity energy error is
one half of delta-u transposed times K times delta-u; the 1e−9 energy tolerance
also covers the fixed quadrature arithmetic. These are bounds from the independent
Q1 matrix and load, not measured solution errors.

## Essential left side and prescribed right flux

Keep the four-cell stiffness matrix and impose normal(grad(u))=x on x=1,
with zero flux on the horizontal sides. The added right-edge load is
(1/4,1/2,1/4) at y=0,1/2,1. In the same free-node order, the complete load is
(1/8,5/16,1/4,5/8,1/8,5/16), whose squared two-norm is 87/128.
The Q1 solution interpolates 2*x−x²/2, with values 0,7/8,3/2 and slopes
7/4,5/4. Direct element integration gives internal energy 37/32 and volume
load work 13/16, hence F=11/32. The boundary trace equals 3/2 along x=1,
so the separate retained surface functional gives S=−3/2 and F+S=−37/32.

The same exact inverse norm 234/49 gives the coefficient-error bound
(234/49)*sqrt(87/128)*1e−10 < 4e−10. Assertions use 1e−9 for coefficients
and both functional values. The surface load has one-norm 1, so its energy
error is bounded by the coefficient infinity error. The volume functional
has that same first-order bound at the loaded solution, plus the quadratic
stiffness remainder. These values and bounds follow from the independent
Q1 matrix, load and integrals; no solver output sets an expectation.

With the volume load set to zero and the same prescribed right flux, the Q1
solution is exactly u=x. Volume energy is 1/2 and surface energy is −1.
The free load squared norm decreases to 3/8, so the same 1e−9 bounds cover
this zero-volume-source path. Its solve and Plan replay are also exercised.

The same volume and surface definitions are referenced by one reduced Observable,
`total = energy + surface`. Its value is independently `11/32 - 3/2 = -37/32`.
The two first-order energy error bounds sum to less than `8e-10`; the total uses
`1e-9`. For the admissible direction `eta=x`, the bulk gradient pairing is `3/2`,
the volume load pairing is `1/2`, and the boundary work pairing is `1`, so the
State variation is zero. Only the gradient pairing depends on the solved
coefficients; its error is bounded by the endpoint coefficient error, below
`4e-10`. The test uses `1e-9`. Missing, reversed and misplaced surface terms
are rejected through exact weak-law comparison before numerical execution.

The ordered second State variation of the same composite is
`D²F[eta,zeta] = integral grad(eta).grad(zeta) dA`: the body and surface loads
are linear and contribute zero. With `eta=x`, the choices `zeta=x` and `zeta=x*y`
give `1` and `1/2` respectively on the unit square. Both directions are exactly
Q1-representable and vanish on the essential side. Two-point Gauss is exact;
these quadratic-energy products do not depend on the solved coefficients.
The `1e-12` tolerance covers the bounded basis, quadrature and arithmetic
operations. Exact Result replay retains the value and both Domain-specific rules.
This is an explicit State product, not a Hessian through the implicit solve.
