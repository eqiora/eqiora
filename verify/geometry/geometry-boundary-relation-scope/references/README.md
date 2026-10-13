# References

The claim, non-claims and precommitted falsifier list come from the accepted
geometry-boundary Relation-scope contract, *Admit one named geometry-boundary
Relation scope with fixed-side normals* (titled *…with inlet/outlet normals* at
the freeze). The base revision is
`a5c122f550fbd3e2b83c6b7745a1deb7fbb0b200`.

The contracts this package consumes without extending:

- [RFC 0080](../../../../rfcs/0080-geometry-backed-semantic-admission.md) —
  closed-bundle admission, parent-relative entity-set lookup, detector
  precedence, and the boundary-physical Port rejection kept here.
- [`geometry.planar-circular-hole-geometry-v2`](../../planar-circular-hole-geometry-v2/README.md) — the current exact
  circular-hole family, its fixed boundary enumeration (x-lower, x-upper,
  y-lower, y-upper, then the circle) and the named sets `inlet`, `outlet`,
  `walls`, `cylinder`. The two-edge membership of `walls` is why no per-member
  wall normal is frozen.
- [RFC 0079](../../../../rfcs/0079-authored-planar-geometry-artifact.md) — the
  straight-edged sibling family, present in the fixture only as the family that
  projects no normal.

Sibling registered cases whose claims this package must not contradict:
[`geometry.geometry-backed-semantic-admission`](../../geometry-backed-semantic-admission/README.md)
and
[`geometry.exact-circular-hole-geometry`](../../exact-circular-hole-geometry/README.md).
The first registered the opposite Relation outcome before this slice; see
*Sequencing* in [the case README](../README.md).

No published DFG benchmark, flow, drag, lift or Strouhal value is evidence
here. Nothing in this package is numerical: the only quantities frozen are two
exact unit normals.
