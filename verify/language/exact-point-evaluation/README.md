# Exact point evaluation

This case binds all coordinates of one exact support through the ordinary Model and static
Result path. It distinguishes analytic expressions from retained Q1 and planar P1 coefficients. Source
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

For planar -div(grad(T))=0 with affine boundary data T=300+2*x-3*y K, the
same affine field solves the interior equation because its gradient is constant. It
belongs to continuous P1 on each triangle. On the unit square split into four triangles
around the center, the center coefficient is free; it is not prescribed as the expected
answer. Probes at (1/5,3/10), (1/2,1/2), (0,0), and (3/4,1/8) m give
299.5, 299.5, 300, and 301.125 K. The absolute tolerance is fixed at 1e-10 K
for the binary64 reference solve and reconstruction, with no affine discretization error.
The test cyclically permutes cell vertices, replays the exact Result, compares both an
authored point binding and `observe_at`, checks type and occurrence identity, rejects wrong
units and outside/missing coordinates, and proves that queries leave Result bytes unchanged.
The same Field restricted by `pullback` to the line `(x,y)=(s,1/2)` has
T(s)=298.5+2*s K. At s=0, 1/4, 3/4 and 1 m the expected values are
298.5, 299, 300 and 300.5 K, with the same independently fixed 1e-10 K
tolerance. The existing dimensioned `observe_at` path samples that authored Observable
before and after replay, preserving Result identity and rejecting s outside [0,1] m.
This is a scalar restriction, not an implicit arc-length measure or a path batch artifact.
Provider bytes are a bounded authenticated Mesh fixture; no external Gmsh execution is claimed.
Planar P1 coordinate derivatives still reject at reconstruction admission. Vector P1 samples
have ordinary product coverage but are outside this registered scalar claim.

The analytic ramp 2*x+1 m gives 1.5 m at 1/4 m. The sine value uses the alternating
Taylor series through degree 11, whose next term at 1/4 is below 3e-18; comparison
allows 1e-14 absolute error. Polynomial coordinate and independent Parameter partials
reuse the existing calculus. Two exact points in one expression and a dormant invalid
branch distinguish scoped caches and demand-driven evaluation.

Wrong units, foreign support, a Parameter used as a coordinate selector, duplicate bindings,
wrong output type, unsupported time/side syntax, outside coordinates and outward endpoint
approaches reject. A solved TPFA Result retains its cell-constant reconstruction: for the affine temperature
T=300+2*x K, the four cell averages are 300.25, 300.75, 301.25 and 301.75 K.
Probes at 0.06 and 0.18 m both return 300.25 K; probes at 0.31 and 0.94 m
return 300.75 and 301.75 K. At 0.5 m the lower and upper values are 300.75
and 301.25 K, while an unspecified side rejects. The represented Field has zero
interior derivative. These are P0 values and derivatives, not the affine continuum
Field evaluated at the query point. Tests repeat after Result replay with a fixed
1e-10 absolute tolerance in the stated units.
Requested one-sided conditional or pure-operator branch limits reject explicitly.

Partial bindings, enclosing evaluation/moving-point derivatives, State products, transient
time selection, arbitrary spaces and mesh transfer are
outside the claim. Installed source/native authoring and wheel replay additionally run in
`bindings/python/tests/test_point_evaluation.py`.

Run `cargo run -p eqiora-verify -- run --case language.exact-point-evaluation`.
