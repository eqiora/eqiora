# Handle-first polygon and convex-polyhedron authoring

Proposed API, awaiting risk review before implementation:

- Rust `GeometryGraph::polygon(vertices_m: Vec<[f64; 2]>, tolerance_m: f64)`
  constructs one simple filled polygon. Vertices describe a cyclic boundary;
  either winding is accepted by the existing planar-region validator.
- Rust `GeometryGraph::convex_polyhedron(vertices_m: Vec<[f64; 3]>,
  faces: Vec<Vec<usize>>, tolerance_m: f64)` constructs one closed convex shell.
  Faces are outward-oriented vertex loops. The existing exact convex-shell
  constructor rejects wrong orientation, nonplanarity, nonconvexity and open shells.
- Both return the existing `GeometryOperation`. Its `region()` is the interior;
  `boundaries()` follows authored polygon-edge or face order. A handle remains
  owned by its original graph and operation. `build(named_topology)` maps handles
  to the existing canonical vertex/edge/facet enumeration before atomic naming.
  Coordinate equality is used only during this primitive's construction to map
  its own validated authored topology, never to rebind foreign geometry selections.
- Python projects these as keyword-only `polygon(vertices, tolerance)` and
  `convex_polyhedron(vertices, faces, tolerance)`. Numeric coordinates and positive
  tolerance are metres, consistent with existing graph methods. No meshing is
  implied; geometry validation and units remain Rust-owned.

Reuse `PlanarRegion` and `CanonicalGeometryV1::from_convex_polyhedra`; introduce
no geometry family, provider registry or persisted schema. Constructor validation
runs before publishing operation handles. Canonicalization must not change which
physical authored edge or face a named handle means. Face-list permutations,
cyclic loop rotations and vertex-table permutations preserve named Geometry
identity when names follow the same physical entities. Reversing a shell's
outward orientation rejects rather than silently repairing it.

Add a shared Rust geometry bounds query for current finite families. Python
bounds and repr consume it without a Cartesian-only expect. Planar polygon
selection/viewing already consumes the shared PlanarRegion. Three-dimensional
polyhedron viewing retains the existing typed unsupported-projection diagnostic;
it must not panic or pretend a two-dimensional projection is three-dimensional.

Independent test expectations are precommitted before product changes. For the
2D skew box, b=(1/4,-1/2) m and B=((9/8,-21/16),(3/2,9/2)) m give
area det(B)=225/32 m² and bounds x=[-17/16,11/8], y=[-1/2,11/2].
For the 3D skew box, b=(1/4,-1/2,3/4) m and
B=((15/16,-3/32,11/64),(5/4,3,-51/32),(0,5/2,15/4)) m give
volume det(B)=15625/1024 m³ and bounds
x=[5/32,87/64], y=[-67/32,15/4], z=[3/4,7]. These are dyadic
rational inputs; signed area and determinant expectations come from expanding
the two-by-two and three-by-three determinants, independently of Eqiora.

Tests cover named physical entities after canonicalization, foreign graph and
operation handles, non-finite coordinates/tolerance, degenerate polygons/shells,
self-intersection, open/nonplanar/nonconvex shells and incorrect outward winding.
They claim exact geometry authoring only: no P2 solver, mesh import, CAD history,
curved geometry, nonconvex volume or additional viewer capability.
