# Bounded coordinate-factor integrals

This case proves real scalar observations of a bounded coordinate density through ordinary
Model, Plan, State and Result owners. The finite-amplitude profile uses a solved finite Field; coordinates
have explicit finite bounds. Exact input support, selected measure and remaining output
support survive Model replay. Quadrature belongs to the numerical observation, keyed by
its exact measure Domain. This does not realize arbitrary phase-space Field data, solve a
product-domain PDE or provide State-direction products.
Bounded polynomial integral constraints, spherical coordinate-density observations and
prescribed polynomial cell Fields and static radial diffusion are covered below.

For `x` in `[0,2] m` and `v` in `[-2,4] m/s`, use
`f=A*(1+x/(2 m))*(1+(v/(4 m/s))²)` with solved `A=3 s/m²`.
At `x=1 m`, this is `9/2 + (9/32)*v²` in coherent SI coordinates.
Apply the independent antiderivative `v^(k+1)/(k+1)` at the endpoints to each monomial.
The zeroth, first and second velocity moments are respectively `135/4 1/m`,
`351/8 1/s` and `837/5 m/s²`; their explicit first/zeroth ratio is `13/10 m/s`.
Integrating the density over position gives `135/2`, as does the full product integral.
Three-point Gauss integrates the velocity polynomials (degree at most four) exactly;
two-point Gauss suffices for the remaining position factor. The absolute tolerance
`1e-11` covers the small finite linear solve and binary64 accumulation, not convergence.
A declared constant density gives `18 1/m` and proves that output support is not inferred
from numerical variation alone.

For `f=3 exp(-v²/2)` on `[-3,3]`, independent antiderivatives give
`I0=sqrt(2*pi)*erf(3/sqrt(2))=2.4998608894830947`, `I1=0`, and
`I2=I0-6*exp(-9/2)=2.4332069102536407`, before multiplying by three.
The fourth derivatives of `g`, `v*g`, and `v²*g` are `g` times respectively
`v⁴-6v²+3`, `v⁵-10v³+15v`, and `v⁶-14v⁴+39v²-12`.
On this interval the triangle inequality gives bounds `138`, `558`, `2226`.
Composite two-point Gauss with 1024 cells and `h=6/1024` therefore has error at most
`3*6*h⁴*M4/4320`; an additional `1e-12` covers rounding. These bounds are derived
before observing numerical output. Infinite Gaussian moments are not claimed:
the zeroth tail is less than `2*exp(-a²/2)/a`, and integration by parts gives the
second tail as `2*a*exp(-a²/2)` plus the zeroth tail, with `a=3`.

Falsifiers cover wrong factors, units and remaining support, implicit normalization,
zero denominators, wrong or unused quadrature, out-of-bounds output points, alpha-renaming,
and tampered input support. A pole missed by quadrature samples still rejects through
conservative regular-density admission. Spatial unknowns and equation support reject in
the finite Plan, and unsupported derived Observable partials reject instead of silently
returning zero. Installed Python ingress is additionally covered by its focused product
tests; the registered numerical evidence owner is the Rust target below.

Run `cargo run -p eqiora-verify -- run --case language.factor-integrals`.

## Fixed polynomial integral partials

A first partial of a named polynomial integral over fixed bounded coordinate intervals
reuses its exact measure and output support. A coordinate selector belongs to the remaining
output support and identifies the same exact factor/axis in the integrand; integrated
coordinates are bound and reject as free selectors. Independent Parameter selectors are
also admitted. The ordinary polynomial calculus differentiates the density before the
ordinary Result integral evaluates it. Nonzero signed real literal divisors retain inverse
physical units; a shared literal used elsewhere as a multiplier keeps its original meaning.

Independently, `n(x,A)=A*(15/2)*(1+x/2)` gives `n_x=15*A/4 1/m²` and
`n_A=(15/2)*(1+x/2) m/s` with coherent SI values; at `A=3`, the first is `45/4`.
The Parameter derivative of the full integral is `45/2 m²/s`. The existing `1e-11`
absolute bound covers the same finite arithmetic and exact polynomial quadrature.
A shared scale of `2 m` in `x/scale+scale*x/(1 m²)` gives `225/4` for the density
coordinate derivative; scale `-2 m` gives `-225/4`. Zero divisors, varying poles,
Parameter-dependent support geometry and bound-coordinate selectors reject.
Declaration reordering and alpha-renaming preserve the structural fingerprint; Model replay
preserves the derived observation. This profile is a declaration-root first partial of one
named integral. Higher/composite integral derivatives, nonpolynomial densities, Cartesian
Geometry measures and shape derivatives remain outside this evidence.

Spatial Field declarations alone do not establish differentiation-under-integral regularity.
The fixed polynomial partial profile rejects spatial Field densities until that regularity
is admitted; a valid undifferentiated spatial density reaches this specific rejection.

## Spherical radial volume measure

`spherical_measure(radius)` declares `4*pi*r^2 dr` on one exact length-valued
coordinate factor with bounds `[0,R]`, `R > 0`. The coordinate has length units
and the measure has volume units. This differs from `measure(radius)`, which
retains the ordinary line measure. The current executable path uses real scalar
coordinate densities and finite Result amplitudes, not a radial diffusion Field.
No smoothness or center boundary condition for an unknown Field is inferred from
this measure declaration; admitting that Field realization remains separate work.

Independently, for `c(r)=2+3r^2` in coherent SI units,
`total=4*pi*(2R^3/3+3R^5/5)` and `volume=4*pi*R^3/3`. At `R=2`, the total is
`1472*pi/15`, the explicit average is `46/5`, and the ordinary line integral is
`12`. A constant density of `2` has total `64*pi/3` and average `2`. Three-point
Gauss integrates the weighted degree-four polynomial exactly in exact arithmetic;
the absolute `1e-11` comparison allowance covers binary64 arithmetic at these
fixed scales. The nonconstant average catches a missing radial weight, while the
total catches a missing angular factor or affine Jacobian.

The Rust test evaluates after Model replay; installed Python also replays Plan
and Result. Shared typing rejects non-length factors, multiple radial axes,
foreign factors and erased remaining support. Whole-Model admission accepts the
positive `[0,2]` premise and rejects `[-1,2]` and `[0.5,2]` at the radial-bounds gate.
Spherical Field-state products, functional variations and under-integral source
partials remain unsupported until their regularity and realization are admitted.

## Finite integral constraints

A continuous noninitial condition Relation can retain a lumped Observable reference.
The finite real scalar Plan expands polynomial coordinate integrals using bounded
Gauss quadrature, sharing exact factor mapping and measure weights with Result
observation. The Model itself retains the original reference and integral. Canonical
original-operand evaluation accepts only exact typed lumped Observable candidates
computed at the same numerical Field/Parameter point; it does not perform quadrature.

For the distribution above, the independent total is `(45/2)A`. Totals 45 and 90
therefore determine amplitudes 2 and 4, and the independently observed density at
`x=1` is `(45/4)A`. A nested velocity-then-position constraint produces the same
amplitude and survives Model, Plan and Result replay and re-execution. For spherical
`a*(2+3r²)` on `[0,2]`, the average target `46/5` determines `a=1`; this catches
an omitted radial Jacobian even when a normalized ratio cancels the angular factor.
The total still checks that angular factor. These affine comparisons retain `1e-11`.

A nonlinear density with amplitude coefficient `A²` has residual `(45/2)A²-90`.
The ordinary strict-positive Newton lifecycle accepts `A=2`, rechecks the original
integral and replays its Result. The `1e-12` residual bound implies less than `1e-12`
root error near this regular root, where the residual derivative is 90.
Focused numerical-owner tests additionally derive `R=2kw²-2p`, `R_w=4kw`,
`R_p=-2`, and `R_k=2w²` at `(p,k,w)=(4,1,2)` and `(9,4,1.5)` and return to the
first point; they check that candidate acceptance and partial AD share changing
Parameter values. Those focused checks are additional product tests, not a separate
registered derivative claim.

Negative probes reach numerical admission after a valid Model: Gaussian coordinate
densities, coordinate-dependent denominators and polynomial degrees exceeding the
one-to-seven-point Gauss profile reject. A shared Observable dependency chain that
would expand exponentially rejects at the 65536-operation work bound. Foreign,
duplicate, wrong-type and spatial-output Observable candidates reject at the
canonical evaluation boundary. General spatial Fields, nonpolynomial integral
constraints and arbitrary nonlocal solver kernels are not claimed.


## Prescribed polynomial coordinate Field

A separate positive path declares `f` as a spatial variable on the exact position–velocity
product and retains the equality `f=3*(1+x/2)*(1+v²/16)` in the Model. A dimensioned-factor
Mesh binds only the exact factor IDs, order, units and bounds. It carries no physical
Geometry, Geometry correspondence or physical mesh-provider receipt. The common scalar
Plan uses a cell-constant trial space and two-point tensor Gauss quadrature to integrate
each cell equality, normalized by its positive cell measure. The ordinary assembly and
linear-solver owners solve the resulting diagonal system. This is a weak cell balance;
the represented Field need not satisfy the authored pointwise equality away from its
cell average. Model, Mesh, Plan and Result replay all precede observation in the test.

Admission bounds this profile to one invariant real scalar variable, one continuous
noninitial equality with the Field alone on one side, and a prescribed polynomial of
degree at most three per factor. Numerical execution supports one to three bounded
interval factors; this registered claim verifies the 1x1v example only. Density admission
is limited to 4096 expression nodes, and cell projection to 1048576 scalar expression
operations. Generic nonlocal Field constraints, Gaussian Field projection, transport,
product PDEs, time evolution and Field differentiation are outside this profile.

For two uniform position cells and `N` uniform velocity cells, let `h=6/N`, and let `x_i`
and `c_j` denote cell centers. Independent antiderivatives give the cell coefficient
`3*(1+x_i/2)*(1+(c_j²+h²/12)/16)`. The `h²/12` term distinguishes integration from a
midpoint sample. Integrating the represented constant Field against `1`, `v`, and `v²`
gives, after dividing by `3*(1+x_i/2)`, respectively:

- `15/2`;
- `39/4 - 9/(4N²)`;
- `186/5 - 18/N² + 54/(5N⁴)`.

These are projection errors relative to the continuous density, not quadrature errors.
Within one velocity cell, the missing first-moment covariance is `h³*c_j/96`.
For the second moment, the missing term is
`h*(c_j²*h²/3+h⁴/180)/16`. Summing with `sum(c_j)=N` and
`sum(c_j²)=N*(4-3/N²)` yields the expressions above. The test uses `N=1,3,6`.
Full and nested integrals both preserve the exact mass `135/2`; explicit moment ratios
use the represented density. A fixed absolute `1e-10` allowance covers binary64 solver
and quadrature arithmetic at these scales, separately from the stated projection error.

Result quadrature splits at every retained cell face. Interior faces belong to the
upper cell and the final endpoint belongs to the last cell; unit-checked point observations
exercise those choices. A spherical integral over the position factor additionally has
value `54*pi*(1+(c_j²+h²/12)/16)`: its two radial cell weights are `4*pi/3` and `28*pi/3`,
paired with position factors `5/4` and `7/4`. This proves weighted observation of the
represented Field, not a radial diffusion solve or center-regularity condition.

Negative probes use valid Models and canonical Mesh artifacts before reaching the intended
gate: Q1 cannot substitute for the admitted cell method; a velocity-only grid cannot erase
the Field's phase support; changing velocity units while retaining exactly the same mesh
coordinates changes the source identity and fails exact Model-factor admission. Installed
Python product tests exercise the same factory, solve, replay and moment refinement path.


## Static radial diffusion and particle average

The ordinary coordinate-grid Plan admits two fixed-coefficient equalities
`partial(r*r*j,wrt=r)=q*r*r` and `j=-D*partial(c,wrt=r)` on an exact length interval
`[0,R]`, with positive `D`, explicit zero inward center flux and an inward surface
concentration. The shared exact polynomial classifier checks coefficients and exact
Field/Coordinate identities; source names never select physics. Fixed scalar Parameters
retain their binary64 values, including the tested `D=1e-14 m²/s` scale. This bounded
profile does not admit time evolution, variable coefficients, nonlinear laws or shells
with a positive inner radius.

Cell-centered finite volumes retain face area `r²` and cell measure `(r_hi³-r_lo³)/3`.
The common angular `4*pi` cancels from the balance but remains in spherical observations.
The center has its declared zero face flux; the surface uses the center-to-boundary
half-cell distance. Accepted cell concentration coefficients and flux projections use
ordinary native assembly, linear acceptance and Result storage. These cell-constant
outputs do not acquire a unique pointwise value or an admitted coordinate derivative.
In particular, the zero center face flux is not a claim that the first stored cell flux
is zero. Point evaluation of this space remains unsupported.

Independently integrate `(r²*j)'=6r²` with regular `j(0)=0`: `j=2r`. With `D=1` and
`c(1)=0`, the second equality gives `c=1-r²`. Its spherical mean is `2/5`, whereas its
line mean is `2/3`. The ordinary Result test uses `N=1,2,4,8,16` uniform cells and runs
after Model, Mesh, Plan and Result replay. With `h=1/N` and cell midpoint `r_i`, the
specified finite-volume surface closure gives `c_i=1-r_i²+h²/4` and projected `j_i=2r_i`.
Exact spherical cell weights then give
`mean_h=2/5+2h²/3-h⁴/15`. Two-point Gauss is exact for the represented constant density
times `r²`; the displayed error is discretization/reconstruction error, not quadrature.
The fixed absolute tolerance `1e-10` covers binary64 assembly, solve and observation at
these scales. Total amount and volume are checked separately, so normalization cannot
hide an omitted angular factor.

A constant surface concentration with zero production has exactly the same average.
Renamed Fields and small physical coefficients exercise structural admission. Valid
compiled negative Models remove `r²`, introduce nonzero center flux or an outward side,
move the surface condition inside the interval, make `D` nonpositive, or add a nonlinear
constitutive term. They reject at numerical admission. Installed Python binds the same
checked-in `.eqi` model and verifies replay, refinement and Q1 substitution rejection.

The radial execution also requires a positive finite representable cell measure.
A radius of `1e-150 m` falsifies silent volume underflow: execution rejects the
zero binary64 cell volume instead of accepting a zero-source result. A radius
of `2 m` independently checks the expected quadratic scaling of concentration.
Conductance underflow that empties an assembled row is rejected by the existing
assembly nonzero-row gate.

## Independent physical surface measure check

The focused product test
`eqiora-numerics::constant_surface_density_uses_each_exact_face_area_and_mass_unit`
in `numerical_admission/tests/plans/observables.rs` separately checks the existing
physical boundary observation path. On a `2 × 3 × 5 m` Cartesian box, a constant
`7 kg/m²` density integrates to `105 kg` on either x face, `70 kg` on a y face,
and `42 kg` on a z face. Two-dimensional quadrature, exact boundary identity,
mass units and Result replay are checked. This ordinary product test complements
the coordinate-factor registered case; it makes no boundary-Field solve claim.

## Finite moving integral endpoints

The same integral accepts `lower` and `upper` on one complete Cartesian coordinate
interval. Endpoints are lumped real scalars in the coordinate unit and stay within the
fixed declared support. Their order defines orientation; equal bounds give zero.
A declaration-root first `partial` of a named integral with respect to an independent
Parameter expands to the under-integral partial plus upper density times upper velocity,
minus lower density times lower velocity. Existing polynomial calculus and exact point
evaluation own those contributions. The density and endpoint derivatives must satisfy
that polynomial admission; this does not differentiate a quadrature algorithm.

Independently, the antiderivative of x² gives I(a)=a³/3 for limits 0,a and I'=a².
For density a*x and limits a,2a, J(a)=3a³/2 and J'=9a²/2; the latter includes
3a²/2 from the density partial, 4a² at the upper end and -a² at the lower end.
For density 1 and limits a,2a, K=a and K'=1, including K(0)=0 and K'(0)=1.
The Rust target checks a=-1,0,1,2 with two-point Gauss and an absolute 1e-12
binary64 tolerance (the integrands have degree at most two), plus units, Model
replay, alpha-renaming, declaration reordering and orientation fingerprint changes.
Installed Python product tests separately replay Model, Plan and Result for all three.
Captured coordinates, endpoint units, out-of-support limits and a pole at x=a reject.

This profile excludes unknown Field densities, implicit-solve derivatives, moving-limit
solver constraints, partial product reductions, spherical weights, moving physical
Geometry, nonpolynomial/singular differentiation and higher integral partials.
No adaptive-quadrature derivative or universal symbolic antiderivative is claimed.
