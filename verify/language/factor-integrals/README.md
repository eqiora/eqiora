# Bounded coordinate-factor integrals

This case proves real scalar observations of a bounded coordinate density through ordinary
Model, Plan, State and Result owners. The amplitude is a solved finite Field; coordinates
have explicit finite bounds. Exact input support, selected measure and remaining output
support survive Model replay. Quadrature belongs to the numerical observation, keyed by
its exact measure Domain. This does not realize arbitrary phase-space Field data, solve a
product-domain PDE or radial diffusion Field, couple an integral into a solver equation,
or provide State-direction products. Spherical coordinate-density observations are covered below.

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
Parameter-dependent integration bounds and bound-coordinate selectors reject.
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
