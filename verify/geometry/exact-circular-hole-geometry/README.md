# Exact circular-hole geometry verification

This case proves one canonical analytic geometry family: an axis-aligned
rectangle with exactly one circular hole. The DFG-shaped witness stores the
rectangle bounds, circle centre and radius, and complete, exactly-once named
edge and face membership. It stores no polygon, chord count, mesh size, or
approximation tolerance, so changing a later numerical realization cannot
change this geometry identity.

The independent Python oracle constructs the compact JSON independently
and freezes exactly 491 bytes plus the framed digest
`c1226bdfc83a5539f21ecced9afe180c60c5f4ca07a952711e3f3529213dee14`.
The Rust evidence executes that oracle, constructs the same geometry from
deliberately noncanonical entity-set order, compares complete bytes and digest,
then externally decodes those bytes through the bounded closed wire.

Falsifiers cover signed zero, alternative number and field spellings, unknown
wire vocabulary, every applicable decoder budget, invalid names/dimensions/
members, reversed or non-finite bounds, non-finite predicate arithmetic,
non-positive or non-finite radius, tangency, and non-positive clearance.
The retired tolerance-bearing v1 wire, partial memberships, vertex selections,
and overlapping aliases are no longer admitted. Existing straight-edged
identity tests remain unchanged.

The same graph-authored Model selects exact `fluid` and parent-relative
`cylinder` sets. The existing `CanonicalGeometryV1` path derives their
dimensions and volume support without exposing a new semantic geometry-kind
switch or retaining the artifact. Artifact-free reconstruction continues to
reject geometry-backed consumers.

Run:

```bash
python3 verify/geometry/exact-circular-hole-geometry/oracle.py
cargo test -p eqiora --test geometry_backed_semantic_admission
cargo run -p eqiora-verify -- run --case geometry.exact-circular-hole-geometry
```

This case does not claim chordal or curved Realization, mesh generation or
correspondence, non-Cartesian boundary embedding, physical Ports, flow
lowering, source syntax, artifact discovery, multiple holes, general arcs,
ellipses, splines, NURBS, B-rep, CSG, booleans, CAD, 3D, drag/lift/Strouhal
values, or the cylinder-flow demo.
