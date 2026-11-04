# Authored-face reference mesh

This case freezes the first face-scoped realization on the common immutable CAD
authoring graph. A caller supplies one graph-bound rectangular face handle, a
finite positive target edge length, a cell-quality gate, and a triangle budget.
The result keeps the exact source geometry and correspondence beside one
intrinsic two-dimensional triangular mesh and its exact affine lift into the
selected three-dimensional face.

The case was frozen as `specified` before implementation. Its two oracle routes
were derived independently without reading production mesh code and agreed
before the implementation lane began. The complete frozen results are in
[`expected/independent-oracles.md`](expected/independent-oracles.md). A
post-review mutant temporarily returned the case to `specified`; the registered
facade evidence now implements the amended realized-coordinate predicate and
the case is again `verified`.

## Contract boundary

This is a surface realization, not a body-volume mesh or a durable mesh-policy
schema. It admits rectangular faces only. All six faces of the rectangle
extrusion are eligible; after a circular through-cut, only the four unchanged
lateral rectangular faces remain eligible. Annular caps and the cylindrical cut
wall reject because they do not expose a rectangular face cycle.

The geometry-classification tolerance is fixed independently at `5e-10 m`. It
is not reused as a sizing tolerance. Per-axis subdivision uses the least
positive integer `n` for which the generated binary64 coordinates satisfy the
actual edge predicate. With `s = L/n`, coordinates are `x_i = i*s` for `i<n`
and `x_n = L`; if `D` is the maximum adjacent realized gap, admission is
`D.hypot(D) <= h`. This retains the exact endpoint without certifying only a
nominal spacing that rounding did not generate. The caller's triangle budget
and all scalar, handle, face-classification, and checked-arithmetic validation
occur before mesh allocation.

## Primary witness

The rectangle spans `x = [-2, 3] m`, `y = [-1, 2] m`, starts at `z = 0.5 m`,
and has depth `4 m`. Selecting the end cap with target `h = 2 m`, minimum
quality `0.95`, and maximum 24 triangles gives a 4-by-3 interval grid with 20
vertices, 24 triangles, 43 edges, 14 boundary edges, 29 interior edges, Euler
characteristic 1, area 15 m², and boundary perimeter 16 m. Every triangle has
area `5/8 m²`; the minimum quality is exactly `40/41`.

The start cap is a second orientation witness. It swaps the intrinsic axes,
reverses the outward normal, and therefore must not reuse the end-cap coordinate
array, connectivity, or handle.

## Mutation witnesses

Two adjacent binary64 targets at the length-5 subdivision boundary freeze the
actual comparator: `0x1.2db2eaabf5c80p+1` accepts three intervals and its
predecessor `0x1.2db2eaabf5c7fp+1` requires four.

A post-review witness freezes endpoint rounding itself. For a 3 m square at
`h = 0x1.3651a0eb63341p-1`, nominal n=7 appears to meet equality, but snapping
the endpoint to 3 m makes the maximum gap five ulps wider and its diagonal four
ulps larger than `h`. The repaired rule therefore selects n=8 and produces
81 vertices, 128 triangles, 208 edges, 32 boundary edges, and 176 interior
edges.

A separate `L = 4.875`, `h = 0x1.f844a57e8134bp-1` witness kills
implementations that return either `ceil(sqrt(2)*L/h)` or
`ceil(L/(h/sqrt(2)))`: both give seven, while realized n=7 has a gap six ulps
above nominal and rejects; exact n=8 spacing accepts, and the realized maximum
mesh edge stays at or below the target. The earlier 8.375 witness is retained
by name as `retained_regression_witness`: the realized-coordinate rule
correctly changes its answer from seven to eight, the same count both ceiling
estimates give, so it guards endpoint-aware correction instead of falsifying
estimates.

A non-square `17.5 m` by `10.5 m` witness exercises endpoint snapping on both
axes at once. Both nominal spacings round to the same binary64 `0.7 m`, whose
diagonal is exactly the target `0x1.fadaa8f7eed51p-1`; snapping widens the u
gap by 26 ulps and the v gap by 10 ulps, so the nominal 25-by-15 grid rejects
on both axes with unequal maximum realized gaps and the least accepted grid is
26 by 16, with 459 vertices, 832 triangles, 1290 edges, and its widest cell
diagonal at or below the target. Every actual triangle edge is checked against
the frozen accepted gap on each axis, and at least one edge must attain both
maxima together. This geometric check does not require a particular last bit
from the mixed-axis `hypot` diagnostic.

A 23-triangle budget rejects the primary witness before allocation, while 24
accepts it. A quality threshold of `0.98` rejects because the generated value
`40/41` is strictly lower. Foreign handles, unknown provenance, non-finite or
non-positive sizing, invalid budgets, annular caps, and the cut wall also reject
before topology is exposed.

## Not claimed

No volume mesh, per-face override hierarchy, persisted policy wire, annular or
cylindrical surface mesh, curved element, exact arc realization, adaptive,
anisotropic, boundary-layer, or production unstructured mesher, Python or
Studio surface, solver integration, or demo is claimed by this case.

This slice admits only the current provider's rectangle face cycles, which are
exactly closed, exactly orthogonal, and exactly normalized in binary64.
Approximate or arbitrarily rotated CAD frames — cycles whose corners, axis
orthogonality, or unit axes hold only to a tolerance — are not claimed; the
exact-frame admission defects that reject them are exercised as
`eqiora-geometry` `cfg(test)` unit falsifiers, outside this case's registered
facade evidence target.

The remaining bit-exact `hypot` witnesses are x86-64 Linux/glibc evidence: Rust
`f64::hypot` resolves to glibc's libm there. Every frozen diagonal equals the
correctly rounded binary64 result, so any correctly rounded libm reproduces
them. Rust does not guarantee correctly rounded `hypot` across libm versions;
the mixed-axis diagnostic therefore uses the geometric check above.
Cross-platform mesh-byte identity is not claimed.

## Run

```bash
cargo test -p eqiora --test cad_authored_face_reference_mesh
cargo run -p eqiora-verify -- run --case geometry.cad-authored-face-reference-mesh
```

The evidence composes the source-bound owner through the existing Geometry,
Mesh, and complete region-correspondence envelopes. It exercises both cap
orientations, all supported and unsupported v2 face classes, exact topology and
quality values, sizing-boundary mutants, stale/foreign handles, work budgets,
quality rejection, and independent Geometry-versus-Mesh policy identity.
