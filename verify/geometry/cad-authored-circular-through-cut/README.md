# Authored circular through-cut

This case proves one topology-changing operation on the common immutable CAD
authoring owner. A fully constrained rectangular face is extruded in positive
z; a strictly interior circle on its end cap is then cut through all in
negative z. The result remains exact analytic meaning: the circle and cylinder
are never replaced by chords, elements, or renderer facets.

The same graph can produce the current provenance-neutral planar Geometry
through complete result-topology naming. That separate claim, including its
491-byte DFG witness, is covered by
[`geometry.planar-circular-hole-geometry-v2`](../planar-circular-hole-geometry-v2/README.md).
The retired tolerance-bearing planar section path is no longer admitted.

Opus 5 and Fable 5 independently derived the geometry and topology before
implementation. Both routes agreed on the 1292-byte v2 wire, digest
`00acb9494fc7dea8f1f2500d1316cb3315130a965a24179b3eb1b10345058b47`,
seven face observations, body volume, surface area, and complete lineage. The
accepted binary64 gate is pure relative `4e-15`, derived from the permitted
seven-term sum error and separated from every precommitted algebra mutant by
more than eleven orders of magnitude.

The build receipt keeps the identity-only base modeling tolerance separate
from the Boolean request and effective policy. A second witness with base
`1e-10 m` and requested/effective Boolean tolerance `1e-11 m` rejects base,
minimum, or maximum clamping. The analytic profile is named explicitly and
reports zero by-construction positional, area, and volume discrepancies plus
`repair = none`; it is not presented as Truck or another numerical kernel.

Relative to the six-face predecessor, four lateral faces are retained
unchanged, both caps retain provenance while gaining an inner boundary cycle,
and one cylindrical cut wall is created. Deleted, split, and merged inventories
are explicitly empty. Edge and vertex counts remain nonclaims because CAD
providers may represent cylindrical seams differently.

Run:

```bash
python3 verify/geometry/cad-authored-circular-through-cut/oracle.py
cargo test -p eqiora-geometry --test cad_authored_circular_through_cut
cargo run -p eqiora-verify -- run --case geometry.cad-authored-circular-through-cut
```

This is not a general feature DAG, B-rep or CSG schema, multiple/blind-hole
system, production CAD-kernel Boolean, healing or per-entity tolerance system,
mesh, solver, Python/Studio projection, or renderer claim.
