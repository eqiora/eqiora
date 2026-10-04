# Bounded coordinate-factor integrals

This case proves real scalar observations of a bounded coordinate density through ordinary
Model, Plan, State and Result owners. The amplitude is a solved finite Field; coordinates
have explicit finite bounds. Exact input support, selected measure and remaining output
support survive Model replay. Quadrature belongs to the numerical observation, keyed by
its exact measure Domain. This does not realize arbitrary phase-space Field data, solve a
product-domain PDE, couple an integral into a solver equation, differentiate under an
integral, or provide spherical measures or State-direction products.

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
the finite Plan, and differentiating a derived Observable rejects instead of silently
returning zero. Installed Python ingress is additionally covered by its focused product
tests; the registered numerical evidence owner is the Rust target below.

Run `cargo run -p eqiora-verify -- run --case language.factor-integrals`.
