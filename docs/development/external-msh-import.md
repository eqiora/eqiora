# Geometry-bound external MSH import

An externally supplied MSH file records a mesh, not proof that Eqiora ran Gmsh.
Import belongs to the existing common Mesh owner. The accepted occurrence binds
original bytes, explicit coordinate units, import limits, exact Geometry, cells
and complete Geometry correspondence. It does not record a target mesh size or
claim a generator release ran.

The first profile admits MSH 4.1 affine triangles on planar regions and affine
tetrahedra on convex polyhedra. Cells remain as supplied. It shares the bounded
MSH parser, quality gate and correspondence derivation with the existing provider
observation path. Both dimensions enforce the requested boundary-facet budget.
Unsupported geometry, cells, units and incomplete boundary coverage reject with
a diagnostic before a Mesh is published.

Rust requests use `MshImportPolicyV1::metres(...)`; Python requires
`coordinate_unit="m"`. The serialized unit discriminator stays private.
Coordinates require an explicit unit declaration. The first profile accepts
metres; it never infers units from Geometry or guesses a scale. Quality, boundary-facet and source-byte limits have defaults with explicit
overrides. Fixed parser count and decoded-memory/work ceilings also apply. Geometry authoring is separate.

## Source identity and persistence decision

External import has its own private provider identity and effective import
policy. The provider version `4.1` identifies the MSH format profile, not an
executable release. It retains the bytes needed for parser and correspondence replay in the
existing common Mesh envelope. It does not add another artifact registry.

The existing Gmsh observation path retains its current identity and behavior.
Its raw-byte native constructor checks the observation's mesh and Geometry
binding; raw bytes alone cannot authenticate subprocess execution. The Python
CLI producer additionally checks and runs its configured executable.
Synthetic byte fixtures must use external import instead of claiming CLI origin.

Adding a distinct origin, policy and common Mesh kind must preserve the meaning
and canonical encoding of existing lineage v1 and common Mesh v4 occurrences.
The additive encoding preserves omission and field order for old kinds; it does
not reinterpret old bytes or need an epoch change. Older readers reject the new
kind; forward compatibility is not claimed. Imported occurrences cannot decode as Gmsh occurrences, and
replay rejects inconsistent source, unit, policy or accepted-resource substitutions.
A coherent valid new source or policy is a new occurrence with its own identity. A change that
cannot preserve those existing semantics requires revisiting this decision
before implementation.

## Independent expectations committed before implementation

Use authored small meshes with analytic geometry rather than Eqiora-generated
expected values:

- A unit square split on its diagonal has two original triangles, area one and
  four boundary edges. Import preserves both triangles and binds every edge to
  the canonical boundary. An affine skew of that square uses its independently
  calculated determinant and boundary coordinates.
- A unit tetrahedron has volume one sixth and four triangular boundary facets.
  Import preserves that one tetrahedron and all four facets.
- Both examples round-trip through the common Mesh bytes with unchanged origin,
  units, cells and Geometry binding. Changing an origin, unit or retained source
  cannot preserve an accepted imported occurrence.
- Foreign or stale Geometry, missing boundary coverage, outside vertices,
  inverted/degenerate cells, malformed input and unsupported cell families fail.
- Boundary-facet limits fail in both dimensions, and parser byte/count/work
  limits remain effective before allocation or publication.
- Python delegates the same request and produces equal Rust canonical identity;
  omitted or unsupported units and invalid policy overrides return typed errors.

These checks establish import and existing P1 geometry behavior. They do not
claim P2, new solvers, new physics or generator execution.
