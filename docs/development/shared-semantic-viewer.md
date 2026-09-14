# Shared semantic viewer V0--V3

The shared viewer starts with one narrow, read-only presentation path for current
accepted Python values. The implemented slice stops at V3: a private typed
scene (V0), shared Geometry/Mesh/Selection interaction (V1), scalar
`FieldOutput` inspection (V2), and installed-wheel rich display (V3).
Studio integration and every V4-or-later concern remain separate work.

## Ownership boundary

Rust is the only semantic projection owner. `_compose_view` admits exact
`Geometry`, `Mesh`, and scalar `FieldOutput` objects, validates their identity
and correspondence, then snapshots immutable little-endian `f64`/`u32`
buffers. The browser cannot infer selections, field association, units, model
ownership, or Mesh ownership. A scalar field is admitted only when its exact
Mesh is present in the same scene.

`eqiora.viewer.scene/v0-private` is an implementation detail, not a public or
persisted artifact. It may be replaced atomically before 1.0. Buffer digests
record the owner-produced snapshot; camera, visibility, colour, and scale range
are explicitly presentation state and never enter accepted scientific
identity. The scale is a linear range of accepted values, not a derived
scientific result.

The current projection is deliberately bounded:

- planar current `Geometry`, with analytic circles tessellated by an explicit
  presentation-only 64-chord policy;
- two-dimensional triangular or quadrilateral `Mesh` values;
- exact edge/face named-selection membership supplied by Geometry topology or
  Mesh correspondence;
- scalar vertex- or cell-associated `FieldOutput` values; and
- copied scenes below the private layer and byte budgets.

Face selection on a line-only Geometry projection is reported unavailable;
composing the exact Mesh supplies its exact mapped face selection. Vertex
selection interaction is also explicitly unavailable in V0--V3. Unsupported
geometry, 3D cells, vector/tensor fields, other associations, trajectories,
foreign owners, and fabricated mappings fail closed.

## Renderer and host choice

The maintained browser spike uses Three.js 0.185.1. It fits this slice because
it provides a small reusable WebGL scene/control layer while Eqiora retains
all topology and field meaning. `vtk.js` would add useful volume, contour,
streamline, and large scientific-data pipelines, but those are outside V0--V3
and would make a renderer library an accidental semantic owner. Reconsider it
only when an accepted later slice requires those operations.

One plain TypeScript mount function owns rendering, picking, controls, and
cleanup. The anywidget adapter is only a host shim over that same function;
there is no second renderer. Picking reports an accepted Mesh
cell or nearest accepted vertex coefficient, never an interpolated value.
Quadrilateral triangulation is renderer-only and exact cell edges remain
separate.

`eqiora.View` retains accepted objects until `close()`, lazily imports the exact
base dependency `anywidget==0.11.0` only for rich display, and ships its
JavaScript/CSS inside the wheel. Importing `eqiora` does not load anywidget,
ipywidgets, or traitlets. Without a rich host, deterministic `text/plain`
remains available. Widget disposal,
listener removal, animation-frame cancellation, GPU resource disposal, and
accepted-object release are explicit lifecycle steps.

`eqiora.colab.prepare()` is the sole Colab host adapter. It validates installed
and loaded distribution identity before importing the Colab API, restarts a
stale runtime, supplies the native Gmsh prerequisite, enables the custom widget
manager, and reports package paths and versions. It does not own scene meaning,
rendering, package installation, or a general notebook environment manager.

## Verification boundary

Focused Rust/Python product tests falsify foreign identity, unsupported shape,
association, buffer mutation, and optional-dependency leakage. TypeScript
contract tests falsify malformed browser transport, and the maintained
Playwright spike exercises shared controls, selection, scalar inspection,
picking, and disposal. These are product checks, not registered scientific
evidence: no rendered pixel, tessellation, colour, camera, or browser output is
an oracle for a scientific claim.
