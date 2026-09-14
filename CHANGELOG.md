# Changelog

Eqiora follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and
Semantic Versioning. During alpha, compatibility changes remain possible and
are recorded here.

## [Unreleased]

### Fixed

- Kept the maintained Colab wake notebook on one coherent Matplotlib installation
  when the runtime preloads the renderer, and moved scalar colorbars onto
  Matplotlib's public `Figure.colorbar` path.

### Removed

- Retired the GTK3/Tauri Studio shell and its native-only authored-CAD,
  Python-export, and packaged DC-drive workflows. The retained browser Studio
  is an explicit interaction preview over fixed projections and has no native
  runtime dependency.

## [0.1.0a14] - 2026-09-14

### Added

- Added event-aware ODE terminal and accepted-step integral functionals with explicit forward Parameter JVPs through localized event times and resets. Exact event sides, derivative coordinates, tolerances, and Result lineage survive canonical replay and ordinary Python execution.
- Added unit-bearing event and forward-sensitivity controls directly to the existing Tsitouras45 request, keeping the internal policy representation out of the public numerical API.
- Added native source, Rust, and Python equality, inequality, and complementarity
  conditions with an explicit bounded active-set realization for finite affine
  real-scalar Models. Plans retain operand-specific coherent-SI tolerances, and
  Results retain independently rechecked active/inactive condition measurements.
- Added fixed-volume scalar Law storage with polynomial time chain rules, exact
  source, Model, and fingerprint identity, independently checked accumulation,
  continuous-State admission, artifact replay, and constant-capacity numerical
  term admission. Transient spatial thermal execution remains unsupported.
- Added named test functions with exact zero-trace restrictions to the single
  authored `weak_form` surface in source and Python.
- Added an installed Python property-composition specimen that executes through
  an exact local package lock, then reproduces Model and Result identities offline.
- Added shared Cartesian Region assembly and exact Field/DOF recovery for complete,
  nonoverlapping chains of scalar conservation regions, including interface flux.

### Changed

- Advanced Model/Transaction artifacts to v27, Source identity to v19, structural
  fingerprints to v22, common Plan and Result artifacts to v5, and common
  Trajectory artifacts to v3. Earlier pre-1.0 encodings reject instead of entering
  compatibility paths.
- Replaced the authored `primal_form` and global `test(field)` APIs with named
  component-owned `test(..., zero_on=...)` and `weak_form(...)`. Rust language
  consumers now use `RelationCondition`; the displaced `Equation` and `LexResult`
  exports are removed.
- Moved typed algebraic Field and gauge structure into solver admission before
  provider selection and removed redundant direct realization constructors.
  Exact block ranges, trace quotients, arbitrary constraints, and typed plural
  scalar quotient admission remain future work.

## [0.1.0a11] - 2026-09-13

### Added

- Added `ModuleRef.record`, `ModuleRef.enum`, and `ModuleRef.space` to Python
  authoring. Independent components can now consume public nominal types from
  an explicit local or locked package graph without copying provider declarations.

### Changed

- Unified local and imported nominal type emission through one qualified syntax
  path. Record constructors, enum members, and finite-space counts preserve the
  exact import alias, while private, unknown, transitive, and equal-shaped foreign
  declarations reject through the existing Module resolver.
- Kept `ImportedRecord` distinct from locally owned `Record` while sharing their
  nominal construction behavior, so the published stubs expose each descriptor's
  exact member value type.

## [0.1.0a9] - 2026-09-11

### Added

- Added opt-in structured execution profiling to Python `run()` and `submit()`.
  Immutable profiles report nested run, setup, solve, time-step, assembly,
  nonlinear-iteration, linear-solve, backend, and post-processing phases plus
  nonlinear convergence observations.
- Added shared static tracing callsites across numerical kernels and the Faer
  backend, with exact profiling checks for ODE and steady/transient cylinder
  execution.

### Changed

- The transient cylinder example can print a compact timing summary. Profiling
  remains disabled by default and disabled runs preserve the same numerical
  result bits without installing a collector.
- Corrected the Rust quick start to use the current `eqiora::compiler` facade
  and the canonical source syntax accepted by the published alpha.

## [0.1.0a8] - 2026-09-10

### Added

- Unified source and Python modeling through `eqiora.Module`, explicit imports,
  structural dimensions, imported pure operators, named connectors, grouped
  relations, boundary families, evolution, events, and sampled components.
- Added exact integers, Boolean predicates, nominal enums, static arrays and
  slices, bounded indexed equations and reductions, flat records and typed
  buses, scalar partial derivatives, analytic scalar properties, and real tables.
- Added deterministic project resolution over local, bundled, and Git sources,
  exact dependency locks, and offline vendoring.
- Added equation-derived region assembly for coupled linear scalar Q1 equations
  and fixed-reference FSI, with exact Field ownership through assembly and
  recovery of physical boundary fluxes, loads, and reactions.
- Added typed Result observables, ordered native parameter batches, first-order
  derivative products, and bounded JAX `vmap` composition.
- Added smooth scalar ODE terminal observables and accepted-step Simpson
  integration independent of requested output cadence. These functionals cover
  the complete fixed Run interval; events, resets, derivatives, and spatial-time
  composition remain unsupported.
- Added exact convex-polyhedral Geometry and authenticated correspondence to
  supplied tetrahedral meshes. This adds neither tetrahedral mesh generation
  nor new PDE/FSI execution.
- Added source and Python steady scalar conservation Laws retaining explicit
  outward flux and source through Model replay. Storage and moving volumes
  remain unsupported.

- Added the `eqiora-language-server` stdio preview with versioned full-document
  diagnostics, formatting, symbols, folding, Markdown hover, definition navigation,
  and reference lookup across open workspace modules through the shared editor
  analysis service. Workspace folders containing `eqiora.toml` also load exact local
  package sources without writing a lock or store. Workspace analysis runs on a
  bounded background worker, discards superseded document versions, and accepts
  request cancellation while an editor operation waits for the current snapshot.

- Added `Eqiora.Fluid@0.4.0` and `Eqiora.Solid@0.4.0` with field-driven vector
  velocity, displacement, and traction boundaries. Installed-distribution
  vendoring provides the specialized `Eqiora.Fluid.Incompressible@0.6.0` and
  `Eqiora.Solid.LinearElasticity@0.6.0` packages and their exact dependencies.

### Changed

- Moved the sole checked-in package source tree under `eqiora-api` so Cargo
  archives contain the same bundled package inputs as source and Python builds.

- Linear solver requests now select an explicit planning objective or a complete
  algorithm/preconditioner/reduction/provider combination. Tolerances alone do
  not define a complete request. Execution-required reductions are admitted
  before candidate ranking and exact selection.
- Converged Model/Component signatures and file-owned module/import syntax.
  Python symbolic equalities use explicit `equation(lhs, rhs)`.
- Advanced current Model/Transaction artifacts to v25, Plans and Results to v3,
  Trajectories to v2, Source identity to v17, and structural fingerprints to v20.
  Recompile models and regenerate saved execution artifacts and package locks;
  obsolete pre-1.0 decoders and aliases are removed.

- Refreshed the exact Diffsol runtime to 0.16.2 and re-bound its backend
  identity and registered scientific evidence to that release.

- Refreshed the shared Base64 dependency to 0.23 and the Python distribution
  build baseline to maturin 1.15.0.

### Removed

- Removed displaced Python `Source`/builder paths in favor of `eqiora.Module`,
  raw public FSI solve entries, and redundant compatibility scaffolding.

- Removed the flat `ConnectionDecl::ports()` compatibility view; consumers now
  use structurally segmented Port paths.

## [0.1.0a7] - 2026-09-02

### Added

- Added natural equation authoring to the Python `Source` projection and
  dimension inference for model-local aliases.

- Added bounded recognition for ideal-gas Euler conservation meaning and
  reusable scalar conservation semantics across one-, two-, and
  three-dimensional Cartesian domains.

- Added exact accepted-State recovery after cancellation and retained admitted
  formulation, material, mesh, and recognition lineage through common Plans.

### Changed

- Converged mesh generation on `MeshPlan`, common scalar execution on
  conservation meaning, and elasticity on shared continuum kinematics,
  isotropic material, element-input, constraint, and observation primitives.

- Prepared immutable execution structure, accepted actions, and reusable Faer
  factorization ownership once per Run rather than rebuilding them per solve or
  accepted step.

- Unified common spatial Plan identity and transient Run schedules while
  preserving exact recognition and realization ownership.

- Reduced documentation and CI duplication by compiling documentation products
  once per source, reusing equivalent lane conclusions, and making Colab the
  sole maintained hosted-notebook path.

### Removed

- Removed public physics-specific dimension aliases and the displaced Marimo
  and Jupyter example paths instead of retaining pre-1.0 compatibility shims.

## [0.1.0a6] - 2026-08-31

### Added

- Added exact scalar property contracts to the common compiler and execution
  path, including Python-authored bindings and affine coefficient evaluation.

- Added authored scalar primal Formulations, directional Stokes correspondence,
  and inspectable Plan identity for the selected formulation.

- Added canonical `math.pi`, typed model-local `let`, and coherent derived
  dimension aliases to Eqiora Language and its bounded Python Source projection.

- Added optional scalar Field initialization and exact local file I/O for
  resolved Plans, complete Results, and complete Trajectories. Rust also gains
  a bounded common transient `RunRequest` artifact.

### Changed

- Generalized admitted equations across additive orientations, including
  equivalent `A = -B` and whole-equation reversal forms, without changing the
  canonical residual meaning.

- Increased the steady and startup Cylinder presentation meshes and retimed the
  startup media so the accepted state change is visible at the documented scale.

### Fixed

- Fixed clean Google Colab Gmsh startup and corrected the Cylinder gallery's
  accessible figure inventory.

### Removed

- Removed unsupported Python Source choice knobs and duplicate transitional ML
  artifact aliases instead of carrying pre-1.0 compatibility shims.

## [0.1.0a5] - 2026-08-29

### Added

- Added bounded canonical artifacts for the resolved common Plan, restartable
  State, complete Trajectory, and complete Result. Decoding is native-owned,
  bounded, and bound to exact upstream identities; installed Python exposes
  byte round trips without becoming a second serialization authority.

- Added inspectable effective Formulations and exact caller overrides across
  the admitted common spatial path. Stokes consumes an explicit mixed
  Formulation, conservative flow consumes an integral Formulation, and
  unsupported or incompatible choices fail before numerical execution.

- Added the shared semantic `eqiora.View` V0-V3 projection for Geometry, Mesh,
  selections, and scalar vertex/cell Fields. The optional installed-wheel
  viewer uses one Rust-owned semantic scene across the standard display path
  and Marimo; Studio integration, 3D/vector/tensor/trajectory viewing, and cloud
  delivery remain later #625 milestones.

### Changed

- Advanced the package-compilation sidecar to the sole current v2 wire and
  semantic-canonicalization epoch 2. Pre-1.0 v1 compilation records are no
  longer decoded; retained historical bytes are archival only.

- Changed installed-Python locked package compilation from a self-contained
  root Model projection to a root public Component bound to caller-owned
  Geometry and Parameters. The resulting ordinary Model now enters the common
  Mesh, Plan, and Run lifecycle, with exact package-compilation lineage retained
  on Model, Plan, and Run and no package-specific resolver or Run semantics.

- Converged common execution on one native Plan sum and one ordinary
  Model/Formulation/Realization boundary. Portable realization graphs are now
  authoritative for common spatial Plans, solver decisions occur before
  numerical work, and compiler lowering is split by responsibility rather than
  physics-name dispatch.

- Converged static scalar, elasticity, and Stokes execution on the common
  Result owner and retired their displaced raw run and root observation/output
  seams. Dynamic Results retain complete common Trajectories and family-specific
  scientific evidence without duplicating the product output authority.

## [0.1.0a4] - 2026-08-28

### Added

- Added bounded installed-Python `eqiora.lang.Source` authoring for the complete
  equations-only steady-cylinder Component. Immutable expression/support handles
  and a one-Component draft emit readable deterministic `.eqi`; direct Source
  compilation and emitted-file compilation both enter the existing Rust parser,
  type checker, lowerer, Geometry binder, and compiler. This is not a Python
  equation runtime, second lowerer, arbitrary language projection, or stable AST
  schema.

### Changed

- Accepted transient MINI States now expose typed pressure point samples and
  signed intrinsic-2D boundary-force action pairs. Samples retain exact
  State/Field/Mesh/point lineage; forces retain exact State/GeometrySelection/
  Mesh lineage and distinguish force on the fluid domain from the equal-and-
  opposite force on the selected boundary. The cylinder gallery presents
  these quantities without assigning benchmark values or scientific acceptance.

- Added one explicitly unverified transient-cylinder product example with
  matching plain-Python, Marimo, and clean Jupyter sources. All three compose
  the same installed-package Geometry, Gmsh Mesh, steady bootstrap, transient
  Plan/State/Run, typed cell-average vorticity, and caller-owned Figure path;
  they now retain ten accepted startup outputs through 0.1 s. The static gallery
  presents that same product Result through a poster, WebM/MP4, first/final
  reduced-motion still, and visible text description. It remains largely
  symmetric startup flow, not a developed wake, and adds no benchmark values,
  tolerances, or scientific acceptance claim. Its GitHub-backed Jupyter source
  also becomes an Open in Colab entry only in the `0.1.0a4` documentation build;
  a clean runtime installs the pinned release without maintainer-owned Drive state.

- Transient `State.curl(velocity_field)` now derives an immutable typed
  cell-average vorticity snapshot from accepted two-dimensional MINI velocity,
  retaining exact State, Field, Mesh, support, unit, and operator lineage.
  The optional Matplotlib adapter renders that cell-associated snapshot through
  the existing `plot_scalar_field` entry point, and the wheel ships the
  equations-only transient-cylinder Component used by the installed profile.

- Geometry-backed transient MINI/P1 flow now advances a nonzero cylinder-wake
  state through correction-form Newton. The resolver uses direct sparse LU for
  the bounded saddle-point Jacobian while preserving its canonical CSR lineage.

- Removed the application-shaped `TransientNavierStokesReference2d` facade and
  its hand-built fixed-mesh evidence. Transient product work now converges on
  the common `Geometry -> Mesh -> Model -> Plan -> State -> Run -> Result`
  lifecycle; the surviving collocated FVM reference owns its model fixture.

- `eqiora.compile` is now keyword-only and accepts exactly one of `path=` or
  `source=`. Definitions-only component source can be closed directly with a
  Python-authored `Geometry` and coherent-SI Parameter values; source owns
  dimensions and abstract support names, while Python is the sole source of
  concrete shape. The exact-cylinder example now follows this route through
  the common root `resolve` / `run` lifecycle. No positional compatibility
  overload or application-specific binding path is retained.

- Root `eqiora.resolve(model, ...)` now owns Model-driven steady and transient
  flow planning with caller-owned Geometry and Mesh and typed spatial, solve,
  scaling, and time policies. Mesh resolution is planning-only; generation
  produces the accepted Mesh.

- Gmsh sizing is one complete mesher-owned policy with an explicit global
  volume target. The common Geometry and Mesh lifecycles replace the retired
  rich-Mesh notebook display and legacy fixed-mesh cylinder stacks.

- The Studio packaged DC-drive protocol no longer presents frozen a3 lineage
  as verification of the current release. Its v2 payload labels the current
  compilation, Run, and binding lineage unverified while retaining the a3 case
  identifier as historical attribution; product execution no longer depends
  on an embedded verification manifest.

## [0.1.0a3] - 2026-08-23

### Changed

- Automatic Python meshing for the bounded exact-cylinder workflow now invokes
  Gmsh 4.15.2 and imports its linear 2D MSH 4.1 output. Missing, mismatched, or
  failed Gmsh executions reject explicitly; the former reference-mesh fallback
  is not retained.

- The exact-cylinder Result and Gallery pressure image now use the accepted
  662-vertex, 1,210-triangle Gmsh realization. This does not
  claim arbitrary geometry, curved elements, adaptive sizing, or cross-platform
  generated-mesh byte identity.

## [0.1.0a2] - 2026-08-21

### Added

- Added `eqiora.compile_package(...)` for one exact installed-Python locked
  Model Package path. Callers provide an explicit content-addressed store,
  canonical resolution bytes, and a bare root-local Model selector; discovery,
  authoring, installation, registry/network access, execution, and Studio
  workflows remain outside the claim.

- Added `eqiora.check_package_conformance(...)` for one explicit locked package
  closure and the exact `eqiora.package.structural-conformance-v1` profile. Its
  immutable in-process report proves structural replay agreement only; it is
  not scientific verification, execution support, trust, certification, or a
  Studio workflow.

- Added one exact serial-host prescribed dynamic-solid publication path with a
  content-addressed standalone Realization, retained prior and accepted-next
  two-Field States, and a Run whose sole output is the accepted-next State.
  This remains the bounded accepted unit-cube reference, not a general
  standalone structural time-integration API.

- Added one failure-atomic external-boundary-provider path for the exact
  prescribed dynamic-solid occurrence. An application-created, already-connected
  subprocess supplies the admitted boundary-displacement candidate; Eqiora
  records a complete provider-occurrence artifact and a separate two-output Run
  linking it to the unchanged accepted-next State. The first verified provider
  is limited to ordinary-GIL CPython 3.12 with NumPy 2.1.0; the direct
  singleton-output Run, process launch authority, and broader coupling remain
  unchanged.

- Added an optional `eqiora.matplotlib` adapter that plots the accepted
  exact-cylinder P1 pressure Result as a caller-owned, headless-saveable
  Matplotlib Figure without making Matplotlib a base dependency.

- Added `eqiora.solid.solve_mixed_boundary_elasticity(...)` and a packaged
  Python example for the accepted exact-v4 structural case. Studio and Python
  now consume one Rust-owned Model-to-Run result, and the optional Matplotlib
  adapter can render its original and explicitly scaled displacement meshes.

- Added an explicit `eqiora.fsi.FixedMeshMonolithic` intent and inspectable
  Model-bound Plan for the accepted two-step FSI case. Studio and Python share
  one Rust-owned resolver; the common Run returns a common Result whose exact
  Trajectory and typed state evidence feed the general scalar and deformed
  Field adapters.

### Changed

- Removed the unreleased Python `FixedReferenceFsiStep`,
  `FixedReferenceFsiResult`, and `solve_fixed_reference_fsi(...)` names. Exact
  spatial Fields now live only on `State`/`FieldSnapshot`, while
  partition, interface, energy, residual, solve, and assembly observations are
  selected through `fixed_mesh_monolithic_evidence(result)`.

- Replaced the demo-specific `plot_fixed_reference_fsi(...)` entry point with
  `plot_scalar_field(...)` and `plot_deformed_field(...)` over common
  `Trajectory` and exact Model-bound `FieldRef` values. No compatibility alias
  is retained.

- The exact-cylinder fluid path now returns the common `Result`,
  `FieldSnapshot`, and `Mesh` owners. Scientific observations move behind
  `fluid.steady_stokes_evidence(result)`, while the demo-specific
  `CircularHoleSteadyStokesResult` and `plot_pressure(...)` surfaces are
  removed in favor of `plot_scalar_field(result, field=...)`.

- Reset the pre-release Model artifact epoch to one current contract. Rust's
  generation-selecting codec API and `eqiora::compatibility`, Python
  `eqiora.compatibility`, Model/Transaction v1--v7 runtime support, and
  control-v1 dispatch were removed. Their ordinary replacements are Rust
  `ModelDocument::{compile, define, replay}`, unversioned current Model and
  Transaction artifact owners, Python `eqiora.compile`, `Model.define`, and
  `eqiora.replay`, plus `eqiora.control/v2`. Old Model/Transaction bytes and
  control-v1 requests reject; no automatic migration is provided.

- The pre-release `eqiora::numerics` facade no longer re-exports
  `SteadyStokesGeometryBinding2d` or
  `solve_resolved_steady_stokes_geometry_mini_2d`. Application consumers use
  `eqiora::api::CircularHoleSteadyStokesResult2d`; lower-level numerical
  composition remains available from its owning `eqiora-numerics` crate.

- The pre-release `eqiora::numerics` facade no longer re-exports the ten
  low-level fixed-reference FSI construction and solution names. Application
  consumers use `eqiora::api::FixedReferenceFsiResult2d`; lower-level
  numerical composition remains available from `eqiora-numerics`.

- Assembly, meshing, geometry, and solver contracts now use their owning crate
  and `eqiora` namespace paths; aliases formerly exposed from numerics,
  artifact, and realization have been removed.

- Package identity and exact in-memory resolution remain available by default;
  directory package authoring, replay, and installation now require the
  `package-filesystem` facade feature.

- Split harmonic ALE naming into the Realization-level
  `P1HarmonicMeshMotionPolicy` and the executable numerical
  `P1HarmonicMeshMotionAction`; the ambiguous pre-release Rust names have no
  compatibility aliases. Artifact wire names and canonical bytes are
  unchanged.

- Renamed the Rust `eqiora-fabric` crate to `eqiora-backend-rayon`, its
  `threaded` facade feature to `rayon`, and the `eqiora-diff` crate to
  `eqiora-differentiation`. The pre-publication Rust names have no
  compatibility aliases.

## [0.1.0a1] - 2026-07-23

The first public alpha establishes one coherent, evidence-gated project
boundary:

- a small typed semantic kernel, Eqiora Language frontend, canonical
  transactions, reference hybrid execution, and scalar Operator IR;
- bounded scalar elliptic FEM/FVM, solver, time, differentiation, artifact,
  package, geometry, I/O, CPU, CUDA, and MPI vertical slices;
- an immutable Python modeling API with synchronous and asynchronous native
  execution, structured diagnostics, explicit NumPy/DLPack ownership, and
  bounded PyTorch/JAX differentiation adapters;
- a thin Studio projection over the same canonical model and typed application
  service;
- versioned artifacts, falsifying verification cases, a public capability
  matrix, and exact release-candidate manifests.

This release supports ordinary-GIL CPython 3.11–3.14 on
manylinux x86-64. It does not claim macOS or Windows wheels, free-threaded
Python, GPU wheels, bundled MPI, a complete physics/component catalogue,
stable-1.0 compatibility, or safety certification.

Detailed claims and nonclaims are the responsibility of the
[capability matrix](docs/capability-matrix.md) and registered
[`verify/`](verify/) cases rather than this summary.

[Unreleased]: https://github.com/nkiyohara/eqiora/compare/v0.1.0a7...HEAD
[0.1.0a7]: https://github.com/nkiyohara/eqiora/releases/tag/v0.1.0a7
[0.1.0a6]: https://github.com/nkiyohara/eqiora/releases/tag/v0.1.0a6
[0.1.0a5]: https://github.com/nkiyohara/eqiora/releases/tag/v0.1.0a5
[0.1.0a4]: https://github.com/nkiyohara/eqiora/releases/tag/v0.1.0a4
[0.1.0a3]: https://github.com/nkiyohara/eqiora/releases/tag/v0.1.0a3
[0.1.0a2]: https://github.com/nkiyohara/eqiora/releases/tag/v0.1.0a2
[0.1.0a1]: https://github.com/nkiyohara/eqiora/releases/tag/v0.1.0a1
