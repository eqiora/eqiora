# Exact point evaluation

This case binds all coordinates of one exact support through the ordinary Model and static
Result path. It distinguishes analytic expressions from retained Q1 coefficients. Source
bindings carry mathematical point/side meaning; the Result and Plan carry the admitted mesh,
Geometry and reconstruction. There is no renderer or output-frame input to observation.

For -T''=0 with T(0)=300 K and T(1 m)=302 K, T=300+2*x in coherent SI units.
At 1/8 m and 5/8 m the exact values are 300.25 K and 301.25 K; their sum is
601.5 K and the slope is 2 K/m. A supported Observable is sampled independently at each
point. Named assertions distinguish the two probes and run after Result replay.

For -T''=12 K/m² with both endpoints 300 K, T=300+6*x*(1-x). Uniform four-cell
Q1 reproduces the nodal values, so its reconstructed cell slopes are 4.5, 1.5, -1.5,
and -4.5 K/m. At 1/2 m the lower and upper approaches give 1.5 and -1.5 K/m;
no-side evaluation rejects. These are derivatives of the admitted reconstruction, not
claims that the quadratic continuum solution has a derivative jump.

The harmonic bilinear field u=x*(3+2*y) on the unit square belongs to Q1 exactly.
At (1/8,5/8) m it gives u=17/32, u_x=17/4 per metre and u_y=1/4 per metre.
Reversing the authored coordinate-binding order preserves their exact axis associations.
The absolute tolerances, fixed before execution, are 1e-11 K for affine values,
1e-10 K/m for four-cell slopes, and 1e-9 for the 12-by-12 reference-solve bilinear
values/derivatives in their stated units. They cover binary64 solve and reconstruction
error; there is no discretization error for the affine/bilinear fields.

The analytic ramp 2*x+1 m gives 1.5 m at 1/4 m. The sine value uses the alternating
Taylor series through degree 11, whose next term at 1/4 is below 3e-18; comparison
allows 1e-14 absolute error. Polynomial coordinate and independent Parameter partials
reuse the existing calculus. Two exact points in one expression and a dormant invalid
branch distinguish scoped caches and demand-driven evaluation.

Wrong units, foreign support, a Parameter used as a coordinate selector, duplicate bindings,
wrong output type, unsupported time/side syntax, outside coordinates and outward endpoint
approaches reject. A solved TPFA Result reaches and fails reconstruction admission.
Requested one-sided conditional or pure-operator branch limits reject explicitly.

Partial bindings, enclosing evaluation/moving-point derivatives, State products, transient
time selection, arbitrary spaces, piecewise-constant point values and mesh transfer are
outside the claim. Installed source/native authoring and wheel replay additionally run in
`bindings/python/tests/test_point_evaluation.py`.

Run `cargo run -p eqiora-verify -- run --case language.exact-point-evaluation`.
