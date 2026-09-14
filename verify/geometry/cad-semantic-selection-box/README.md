# CAD semantic selection over one bounded box workflow

This case proves the first intentionally narrow CAD vertical slice without
making CAD-kernel topology part of Eqiora's meaning. One exact millimetre STEP
cuboid is imported through the isolated Truck adapter. A fully constrained XY
rectangle is extruded and intersected with that stock. The result must equal
one exact three-dimensional Cartesian Semantic Domain.

The accepted chain is content-bound from Model and complete source bytes
through CAD design/build evidence, Geometry Identity, a deterministic
six-tetrahedron mesh, and ordinary geometry-to-mesh correspondence. Build
evidence records the exact adapter/kernel versions, three normalized closed
six-plane observations, separate source/modeling/classification tolerances,
and an explicit no-repair disposition. Raw Truck faces, ordering, and objects
never cross the adapter seam.

Application viewport triangles and semantic-table rows both create the same request:
`(Geometry Identity digest, Domain ID)`. The resolved selection exposes exact
body/boundary geometry, mesh membership, applied Relations, and
boundary-physical Ports. Renderer-local triangle indices cannot be used as
selection identity. A changed geometry revision rejects the old request.

Cross-revision retention uses the existing exact geometry-revision association
artifact over two independently accepted plans. Selection retention succeeds
only through its total one-to-one successor relation; a selection from any
other revision fails closed. Dimension-edit transaction preview and commit are
not claimed. The separately
registered
[`geometry.fixed-reference-interface-identity-2d`](../fixed-reference-interface-identity-2d/README.md)
case owns the missing, split, merged, and ambiguous association falsifiers that
this CAD path reuses.

The registered Rust target also rejects changed/truncated STEP bytes, unit and
CAD-policy drift, adapter identity substitution, open or multiple solids,
non-planar faces, non-axis-aligned box topology, stale selection, and foreign
regeneration revisions. Repair and raw kernel-face selection are stronger
type-level exclusions: V1 has no repaired disposition or public face-rank field
that a caller could submit.

Browser Studio contains a fixed, explicitly noncanonical example projection
with local TypeScript and Playwright checks for stale selection, unknown fields,
keyboard table/viewport parity, responsive layout, and accessibility. Those UI
checks do not exercise this plan and are not an additional Cargo evidence target.
This case's registered claim stops at the canonical application projection.

Run:

```bash
cargo test --locked -p eqiora --features cad-truck --test cad_semantic_selection
cargo run --locked -p eqiora-verify -- run --case geometry.cad-semantic-selection-box
```

This case does not claim general STEP/assembly support, curved topology,
healing, a general Boolean kernel, persistent B-rep schemas, universal topology
naming, a general CAD editor, additional feature types, ALE, or remeshing.
