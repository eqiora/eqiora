# Proposed independent affine-simplex P2 scalar reference

This is a mathematical proposal awaiting independent review. It contains no
reference implementation, native execution, candidate output or capability
claim. Both triangles and tetrahedra are required. An exact quadratic patch is
not a convergence study; the separate refinement witness below remains required.

The reference will use rational arithmetic and polynomial integration. It will
import no Eqiora code, consume no Eqiora output as an expectation, and share no
product assembly implementation. Inspection of the existing kernel establishes
only the intended producer interface. Implementation starts after math review;
all reference constants and finite-level intervals must be frozen before any
product candidate is examined.

## Space and exact integration

Use continuous scalar P2: vertex values and one value at every edge midpoint,
with local functions

```
N_i = lambda_i (2 lambda_i - 1)
N_ij = 4 lambda_i lambda_j, i < j.
```

There are six local coefficients on a triangle and ten on a tetrahedron. There
are no face, cell-center or hierarchical bubble coefficients. The element is the
[degree-two simplex Lagrange element](https://defelement.org/elements/lagrange.html).
With one midpoint per edge, reversing an edge introduces neither a sign nor a
within-edge permutation; compare the
[Basix degree-two orientation example](https://docs.fenicsproject.org/basix/v0.10.0.post0/python/demo/demo_dof_transformations.py.html#degree-2-lagrange-element).
A cell permutation still relabels its vertices and edge pairs.

For an affine physical simplex x=v_0+J xi, use grad_x N=J^(-T) grad_xi N.
Derive every integral by expanding barycentric monomials and using

```
integral_T product_(i=0..d) lambda_i^alpha_i dx
  = abs(det J) product_i(alpha_i!) / (d + sum_i alpha_i)!.
```

This identity follows by iterated integer beta integrals. The reference has no
numerical volume or facet quadrature. Affine constant-diffusion stiffness entries
have polynomial degree two; mass entries have degree four. Trace functions are
the parent functions restricted to lambda_opposite=0, including all facet-closure
vertices and edges and excluding every off-facet coefficient.

## Fixed physical domains and multicell topology

Let q lie in [0,1]^d, and x=b+B q. Entries of b and B below are physical metres.
They are dyadic rationals, so the declared vertices and edge midpoints at the
proposed levels are exactly representable in binary64. Both B matrices are
skew, nonsingular and non-axis-aligned, not rotations of axis-aligned cells.

```
2D:
  b = (1/4, -1/2) m
  B = [[9/8, -21/16],
       [3/2,    9/2]] m
  det B = 225/32 m^2

3D:
  b = (1/4, -1/2, 3/4) m
  B = [[15/16, -3/32,  11/64],
       [  5/4,     3, -51/32],
       [    0,   5/2,   15/4]] m
  det B = 15625/1024 m^3.
```

Divide each logical axis into N equal intervals. For every grid anchor a and
permutation pi of the d coordinate axes, create the Freudenthal simplex with
logical vertices

```
a/N,
(a+e_pi[0])/N,
(a+e_pi[0]+e_pi[1])/N,
...,
(a+ones)/N.
```

There are 2 N^2 triangles or 6 N^3 tetrahedra. Before orientation normalization,
the signed determinant is sign(pi) det(B)/N^d. For an odd permutation, exchange
the last two local vertices to make the canonical cell determinant positive.
Cell volume is det(B)/(d! N^d). Global edges are unique unordered vertex pairs
from the actual simplex connectivity; each owns one midpoint and one nodal slot.
They are not inferred from coordinate strata of a tensor-product element.

The patch uses N=2: eight triangles with 9 vertex + 16 edge coefficients, or
48 tetrahedra with 27 vertex + 98 edge coefficients. On these meshes the complete
P2 node coordinates happen to form the same half-grid as Q2, but all nonvertex
coefficients belong to edges. A tetrahedral body-diagonal midpoint is an edge
coefficient, never a cell coefficient. Identical coordinate counts do not make
P2 and Q2 interchangeable.

The later producer must authenticate this exact physical mesh, complete edge
inventory and effective integration policy through its public Model/Plan/Result
interfaces. A missing public route is a product gap to report, not permission to
substitute an axis-aligned mesh or a vertex-only field.

## Manufactured quadratic patch and units

Set kappa=1 dimensionless and solve

```
-div(kappa grad u) = f.
```

The solution u is dimensionless. Coordinates and fixed reference lengths
L=(2,4) m or (2,4,8) m carry length units. Define X_i=x_i/L_i and

```
u = 1 + sum_i X_i + sum_i X_i^2 + sum_(i<j) X_i X_j
f = -2 sum_i 1/L_i^2.
```

Thus f=-5/8 m^(-2) in 2D and f=-21/32 m^(-2) in 3D. The mixed terms exercise
physical gradient transformations; this is a total-degree-two P2 polynomial,
not the tensor polynomial used by the Q2 reference. Its Hessian in X is
I+ones, which is positive definite. Its global minimum is (d+2)/(2(d+1))>0,
so every vertex and midpoint coefficient is nonzero. Its nodal values are dyadic
rationals; the eventual source must use an arithmetic expression whose bounded
dyadic intermediate values preserve these essential data exactly.

Apply u itself on the entire logical upper face q_0=1, including every boundary
midpoint. Apply g=kappa grad(u).n on all other boundary facets. The physical
outward flux g has units m^(-1); lower-face fluxes are generally nonzero here.
The constant outward unit normals on logical upper faces are

```
2D: n_0=(24,7)/25, n_1=(-4,3)/5
3D: n_0=(624,32,-15)/625,
    n_1=(-96,72,35)/125, n_2=(16,-12,15)/25.
```

Lower-face normals have the opposite sign. The reference will independently
verify the unit lengths, outward signs, face incidence and determinants.
Integrate all natural loads, including contributions to constrained coefficients
at essential/natural intersections. Do not remove those contributions before
forming the uneliminated physical residual.

Using the global P2 functions, assemble exactly

```
K_ij = integral_Omega kappa grad(N_i).grad(N_j) dx
F_i  = integral_Omega f N_i dx
G_i  = integral_GammaN g N_i ds
r(v) = K v - F - G.
```

Facet loads can use oriented area vectors, so no square-root approximation enters
the rational reference. The specified boundary face normals also have rational
components. For analytic nodal coefficients, every free residual is exactly zero.
K and weak dual loads have units m^(d-2). These are not mechanical forces.

The sum of essential reactions has the independent surface-integral expression

```
x_D = b + B e_0 + (1/2) sum_(j>0) B e_j
A_D = det(B) B^(-T) e_0
R_D = grad(u)(x_D).A_D.
```

The integrand is affine, so its face average is its centroid value. This gives
R_D=27339/4096 in 2D and R_D=24991425/1048576 m in 3D. There are 5 or 25
constrained nodal actions. This optional native-reaction witness must use an
existing physical weak-action interface; the reference does not define a new
public scalar force/flux API.

## Proposed fixed patch accuracy allocation

Use fixed reference scales ell_star=1 m, u_star=1 and kappa_star=1. In particular,

```
Kbar = K / (kappa_star ell_star^(d-2))
Fbar = F / (kappa_star u_star ell_star^(d-2))
Gbar = G / (kappa_star u_star ell_star^(d-2))
vbar = v/u_star
rbar = Kbar vbar - Fbar - Gbar.
```

Every barred quantity is dimensionless. These scales do not depend on candidate
values, mesh observations, solver convergence or the measured residual.

Propose epsilon=2^(-40), with squared comparison sum_F rbar_i^2<=epsilon^2,
and exact equality of every essential coefficient to its analytic dyadic value.
This epsilon is a declared accuracy allocation, not a theorem bounding floating
point assembly/solving errors. Proposed solver controls are rtol=atol=1e-14;
those settings do not replace the independent residual test.

Before any candidate, compute and freeze the exact rational constant
Cinf=||Kbar_FF^(-1)||_infinity for each of the two N=2 meshes. The free operator
is positive definite: a continuous P2 function with zero gradient is constant,
and its zero trace on a nonempty face makes that constant zero. Verify the exact
factorization/inverse identities. Require every coefficient error to be at most
Bu=Cinf epsilon. The independent implication is

```
||vbar-u_exact||_infinity
 <= ||Kbar_FF^(-1)||_infinity ||rbar_F||_infinity
 <= Cinf epsilon.
```

Both coefficient and residual gates must pass. Essential equality removes any
unbudgeted boundary lift error. No Cinf or budget is estimated from Eqiora data.
For native reactions, propose the separate dimensionless allowance

```
Umax = max of analytic u at domain vertices
rho = epsilon max(1, ||Kbar||_infinity Umax + ||Fbar+Gbar||_infinity).
```

This is a fixed accuracy ceiling (8192 binary64 unit roundoffs), not a universal
roundoff theorem. Check each native constrained action against rational rbar of
the actual complete coefficient vector within rho. Also check its sum against
the analytic R_D with allowance m ||Kbar||_infinity Bu + m rho. Here m=5 or 25.
The full reaction vector cannot be replaced by a summed quantity alone.

## Predeclared falsifiers and integration checks

The independent tests must reject missing/duplicated shared edges, an edge value
mislabelled as a face/cell/bubble, skipped essential midpoints, skipped natural
midpoint loads, wrong flux sign, using J^(-1) instead of J^(-T), and incorrect
signed-volume integration. Test all 3! triangle and 4! tetrahedron vertex
permutations with corresponding vertex/edge relabelling. Correct scalar P2
permutations preserve the operator and introduce no nodal signs; inconsistent
labels or spurious edge signs are mutants, not legitimate alternative methods.

For the patch, declare three-point-per-axis simplex Duffy Gauss in both dimensions
as the eventual effective production control. The mathematical volume minimum is
two points in 2D and three in 3D for stiffness/constant load; natural-facet minima
are two for segments and three for triangles. A polynomial of total degree p
under a d-dimensional Duffy map has first-axis degree at most p+d-1, so
n>=ceil((p+d)/2) is sufficient. Mass degree four needs three points per axis in 2D and four in 3D. The reference remains an exact polynomial integral.

Underintegration tests compare exact local operator entries with the deficient
rule, not only a patch solution that might accidentally survive an alias. For
example, two-point Gauss on [0,1] gives 7/36 for integral t^4 instead of 1/5.
An actual candidate must separately authenticate the persisted effective rule;
a field file declaring a rule is not proof of the Plan's rule.

## Separate genuine convergence reference: proposal

Use the same two affine domains and nested simplex meshes N=2,4,8, now with
homogeneous essential data on every face and

```
p_i=q_i(1-q_i), u=product_i p_i
Gmetric=B^(-1) B^(-T)
f = 2 sum_i Gmetric_ii product_(j!=i) p_j
    -2 sum_(i<j) Gmetric_ij p_i' p_j' product_(k!=i,j) p_k.
```

Here q=B^(-1)(x-b), u is dimensionless, and f has units m^(-2). Degrees are
four in 2D and six in 3D, so P2 does not reproduce this field. Off-diagonal metric
terms must not be discarded. Source/test degrees require at least three Duffy
points per axis in 2D and five in 3D for exact polynomial loads. Exact continuous
norms provide additional audits:

```
||u||_L2^2 = det(B)/30^d
|u|_H1^2 = det(B) trace(Gmetric)/(3 * 30^(d-1)).
```

Subsequent reference implementation must freeze finite-level L2/H1-seminorm
error intervals before any candidate. It can use independently computed sparse
approximate Galerkin solutions, checked by exact rational residuals, rather than
inverting a large rational matrix. With barred coordinates, a conservative
physical Poincare lower bound is lambda=9d/||Bbar||_F^2. An exact local P2 mass
matrix gives mref=1/||Mref^(-1)||_infinity>0; then
mu=mref min_T abs(det Jbar_T) bounds the global nodal mass matrix below. Hence a
reference coefficient residual eta bounds reference solution uncertainty by

```
L2: eta/(lambda sqrt(mu)); H1 seminorm: eta/sqrt(lambda mu).
```

Evaluate full candidate/reference field error squared by exact polynomial
integration, then outward-enclose square roots. Propose reference uncertainty
<=1e-4 of each reference error lower bound and total reference/candidate/norm
uncertainty <=1e-3. If these cannot be certified, the level is inconclusive; no
candidate is launched and no allowance is enlarged from its output.

Expected asymptotic orders are H1 seminorm two and L2 three for this smooth field
on shape-regular affine refinements. They are not finite-level thresholds.
Finite-level acceptance and rates must use the independently frozen intervals,
with all actual vertex and edge coefficients, not a vertex subset, tensor
amplitude projection or an exact-patch rate argument. This construction and its
numeric allocation need the separate math review before implementation.
