# Planar circular-hole Geometry v2 foundation

This case verifies one provenance-neutral, scale-independent exact Geometry v2
content family: finite axis-aligned rectangle bounds, one finite positive
circle with strict positive side clearance, and complete named membership for
five edges and one face. Its closed wire contains no classification tolerance,
provider identity, authored graph digest, build receipt, or lineage.

The accepted authored-CAD graph is one separate producer route. Its analytic
build projects retained and created construction lineage into crate-private,
dimension-carrying result handles bound to the exact graph identity. Atomic
naming accepts only complete, exactly-once, homogeneous membership. This proves
how that producer obtains Geometry content; it does not turn provenance into
persisted Geometry meaning.

The scale family applies factors `2^-40`, `1`, and `2^40` uniformly. Every
member retains identical name and dimension membership, while each metrically
different Geometry has its own canonical bytes and digest and replays through
the closed v2 decoder. Unknown and duplicate fields, tolerance/classification
fields, reordered keys or members, and equivalent noncanonical numeric spelling
reject.

The tolerance-bearing `eqiora.planar-circular-hole-envelope/v1` wire is retired
and rejected by both the circular decoder and generic replay. Public construction
and authored geometry now share this single content identity. Incomplete naming,
vertex selections, and overlapping aliases are intentionally unsupported.
Two independent derivations fix the ordinary provenance-neutral
v2 wire at 491 bytes, plain SHA-256
`f9a278430c0033f2b0ec148b66d4608cf1f2b559cb46aa31ac9e7259861a26f3`, and
schema-framed identity
`c1226bdfc83a5539f21ecced9afe180c60c5f4ca07a952711e3f3529213dee14`.

Run:

```console
cargo test -p eqiora-geometry --lib \
  cad_authored_result_topology::tests::registered_planar_circular_hole_geometry_v2_evidence
cargo run -p eqiora-verify -- run \
  --case geometry.planar-circular-hole-geometry-v2
```

This foundation exposes no public result-topology naming API. It is not a
source-owned mesh correspondence, imported-mesh classifier, generic B-rep,
general Boolean/lineage system, Python workflow, mesh, solver, or scientific
benchmark claim.
