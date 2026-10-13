# RFC 0081: Exact circular-hole planar geometry

- Status: Superseded for wire and naming contracts
- Authors: Eqiora contributors
- Created: 2026-07-28
- Related RFCs and evidence:
  [RFC 0079](0079-authored-planar-geometry-artifact.md),
  [RFC 0080](0080-geometry-backed-semantic-admission.md), and
  [`geometry.exact-circular-hole-geometry`](../verify/geometry/exact-circular-hole-geometry/README.md)

The current wire and membership contract is
[`geometry.planar-circular-hole-geometry-v2`](../verify/geometry/planar-circular-hole-geometry-v2/README.md),
with public replay and semantic admission covered by
[`geometry.exact-circular-hole-geometry`](../verify/geometry/exact-circular-hole-geometry/README.md).
It has no classification tolerance and requires complete, exactly-once edge/face
membership. The v1 decoder and tolerance-bearing constructors described below
are retired; this RFC retains their historical motivation and derivation.

## Summary

One private kind of the common canonical geometry owner retains an
axis-aligned rectangle with one strictly interior exact circular hole.
Rectangle bounds, circle centre and radius, classification tolerance, and
named exact entity sets determine its canonical bytes and content identity.
Numerical refinement does not.

The existing kind-erased geometry reference admits this family into the
Semantic Model without adding a semantic geometry-kind switch. Chordal mesh
Realization is a dependent slice with a separately frozen error oracle.

## Motivation

The authored straight-edged family in RFC 0079 can represent a polygonal hole,
but that makes the polygon the Model's shape. Refining the circle would then
change geometry identity and force a new Model, mixing shape meaning with a
numerical approximation choice.

Conversely, a general arc B-rep would require canonical start/sweep/orientation
rules and binary64 arc/segment and arc/arc topology predicates before any
consumer needs them. A standalone circle plus CSG would give one region many
boolean-tree spellings without a canonical equivalence policy. This RFC takes
the narrow exact family needed by the first cylinder path and leaves those
distinct capabilities closed.

## Exact geometry

`CanonicalGeometryV1` privately owns:

- two `[lower, upper]` axis bounds in metres;
- one circle centre `[x, y]` and positive radius in metres;
- one positive producer classification tolerance in metres;
- canonical named entity sets;
- exact canonical JSON bytes; and
- `sha256(schema || NUL || bytes)`.

Its fixed entity enumeration is:

| Dimension | Entities |
|---|---|
| 0 | four rectangle corners in lexicographic `(x, y)` order |
| 1 | x-lower, x-upper, y-lower, y-upper, circular hole |
| 2 | one rectangle-minus-circle face |

Named sets may group entities, such as both y boundaries as `walls`. Names,
dimensions, membership validation, member normalization, uniqueness, and
`(dimension, name)` canonical order are the RFC 0079 rules.

Every coordinate is finite and signed zero normalizes to positive zero. Bounds
increase strictly and their spans remain finite. Radius and tolerance are
finite and positive. Radius plus tolerance and each centre-to-side distance
remain finite; every distance is strictly greater than radius plus tolerance.
Thus tangency and a residual clearance no greater than the classification
tolerance fail closed.

This is binary64 analytic identity, not an exact-real predicate claim. Centre
and radius are stored directly rather than reconstructed from sampled points.

## Wire and identity

The separate closed schema and domain separator are:

```text
eqiora.planar-circular-hole-envelope/v1
```

The encoding, kind, and unit are:

```text
eqiora.canonical-json/v1
axis-aligned-rectangle-with-circular-hole-v1
metre
```

Wire field order is:

```text
schema, encoding, kind, length_unit, tolerance_m, bounds,
circle { center, radius_m }, entity_sets
```

Binary64 values use RFC 0079's `serde_json` shortest round-trip spelling,
including `.0` for integral finite values and the canonical lowercase exponent
form. Struct fields retain the underscores shown above; enum values alone use
kebab case.

The digest is:

```text
sha256(
  b"eqiora.planar-circular-hole-envelope/v1"
  || 0x00
  || canonical_json
)
```

The existing straight-edged envelope remains closed and byte-identical. The
new family does not add a variant to that wire or reuse its digest domain.

## External admission and budgets

`CanonicalGeometryLimits` supplies the encoded-byte, vertex, face, loop-index,
entity-set-count, and entity-set-member ceilings. The family checks its fixed
four-corner, one-face, four-outer-loop-index cardinality against those ceilings
even though it does not allocate declaration-sized geometric collections.

Admission checks the byte ceiling, decodes private unknown-field-denying wire
types, verifies closed schema/encoding/kind/unit vocabulary, checks entity-set
budgets, reconstructs through the sole validating constructor, and requires
byte-for-byte equality with the input. Equivalent noncanonical JSON is
rejected rather than normalized at the trust boundary.

## Semantic admission

`CanonicalGeometryV1` gains one private enum variant. Semantic admission
borrows the canonical owner directly and projects only:

- derived digest;
- ambient and topological dimensions, both two; and
- the dimension of an exact named entity set.

No compatibility view or constructor from independent facts is added. RFC 0080's exact
closed-bundle, parent-relative boundary selection, derived support retention,
and artifact-free behavior therefore apply without a change in `eqiora-sem`.
A geometry boundary physical Port remains rejected because entity-set
dimension does not supply a non-Cartesian embedding.

## Separation from Realization

Circle sampling count, chord phase, mesh spacing, mesh-quality thresholds, and
geometric approximation budgets are absent from this artifact. The dependent
chordal slice binds its output to this source digest and independently verifies
the boundary, area, and perimeter deficits. A later curved or NURBS
Realization may replace it without changing this exact geometry or Model.

## Verification

The independently frozen witness uses the DFG-shaped values:

```text
bounds = [[0.0, 2.2], [0.0, 0.41]]
circle center = [0.2, 0.2]
circle radius = 0.05
tolerance = 1e-12
cylinder = [4], inlet = [0], outlet = [1], walls = [2, 3], fluid = [0]
```

A non-implementing Opus 5 lane constructed the bytes three ways without
reading Rust. The frozen result is:

```text
bytes: 511
sha256: b00123472a596e8289820cabaee20d52cdf81b5572fa9ce58ff17cdaa00046d9
```

The registered Rust evidence executes that oracle and must agree exactly.
Falsifiers cover wire vocabulary and spelling, every applicable resource
budget, signed zero, every geometric predicate, named-set validity and
identity sensitivity, unchanged straight-edged identity, and unchanged
kind-erased semantic bundle behavior.

## Compatibility and architecture

No existing wire, digest, Model v7 byte, structural fingerprint, or semantic
diagnostic changes. The original exact-circle owner was subsequently folded
into the equally non-forgeable common opaque canonical owner, removing that
public type while preserving this wire and its kind-specific replay entry.

## Nonclaims

This RFC adds no polygonal or curved Realization, triangulation, mesher,
geometry-to-mesh correspondence, source syntax, artifact discovery or storage,
package manifest, physical boundary embedding, flow lowering, multiple
circles, arbitrary outer loop, general arc, ellipse, spline, NURBS, B-rep,
CSG, boolean, CAD kernel, exact-real topology kernel, 3D geometry, geometric
equivalence, drag/lift/Strouhal reference, or cylinder-flow demonstration.
