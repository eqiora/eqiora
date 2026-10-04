# Geometry-backed semantic admission verification

This case proves one exact closed bundle admission path from canonical geometry
content into Semantic Kernel spatial support. The positive artifact is RFC
0079's independently derived dyadic square with a centred square hole. Its
frozen digest is
`e6f8e17ac215ef37ca3c9de07b9979e34f13412a5de11dc9240ea1def8130030`;
`fluid` is dimension 2, while `exterior` and `hole` are dimension 1.

The ordinary graph-authored Model selects `fluid`, selects `hole` relative to
that region, defines a two-component spatial Cartesian continuum Field, and
scopes a divergence Relation to the region. Admission derives volume and
boundary support from the artifact and accepts ordinary Field and Relation
typing. A later typed-residual query succeeds without receiving the artifact
again, proving that `KernelProgram` retained derived support rather than a
borrow.

The sharp parent-relative falsifier supplies two valid artifacts. A boundary
whose own parent artifact lacks `foreign-only` rejects even though the other
artifact defines that name; the mirrored boundary on the defining artifact is
valid. Other falsifiers cover missing, unreferenced, duplicate, and permuted
bundles; declaration-only references; reversed region/boundary dimensions;
multiple regions sharing one digest; spatial extent mismatch; and the absence
of an admitted affine embedding for boundary Fields or a Cartesian embedding for physical Ports.
A continuous Relation may now consume the admitted boundary support; the sibling
[`geometry-boundary-relation-scope`](../geometry-boundary-relation-scope/README.md)
case owns that acceptance and its exact boundary. Boundary-physical Ports remain
rejected in this witness, as do Fields defined on its grouped hole or a curved boundary.
Separate focused product tests admit Fields on exact Cartesian boundaries and individual
straight planar edges, including oblique edges; they reject a segment belonging to a foreign
face in the same artifact. This case does not claim their numerical realization.

Run:

```bash
cargo test -p eqiora --test geometry_backed_semantic_admission
cargo run -p eqiora-verify -- run --case geometry.geometry-backed-semantic-admission
```

This case does not claim source syntax, `box` synthesis, artifact discovery or
storage, mesh admission, correspondence, Realization, numerical lowering,
boundary-to-parent geometric incidence, curved or exact-circle geometry,
embedded manifolds, 3D, CAD, or the cylinder-flow demo.
