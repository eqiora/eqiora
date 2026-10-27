# Changelog

Eqiora follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and
Semantic Versioning. Until 1.0, releases may include breaking changes; they are
listed here.

## [Unreleased]

- Faer complex CG (HPD) and BiCGSTAB (General, complex-symmetric, Hermitian, HPD)
  now execute Identity/Jacobi through the same typed Krylov owner as real solves.
  Planner v6 admits these ten exact tuples; native work remains bounded by the
  requested budget and acceptance uses the original residual and physical checks.


### Changed

- Linear solution acceptance is constructed with `LinearSolution::accept`,
  `accept_with_execution` or `accept_with_verifier`. The former free functions
  are removed; all methods retain the same independent residual verification.
- Host solver planning v6 uses explicit work forecasts and scoped numeric
  payload bounds where existing assembly owners supply size facts. Exact manual
  controls remain unchanged; forecasts do not grant prepared-state reuse or
  predict elapsed time or process RSS. Current common Plan artifacts use v24;
  previous Plan schemas are rejected rather than migrated.
- Reference complex BiCGSTAB supports Identity/Jacobi on General, complex-symmetric,
  Hermitian and Hermitian-positive-definite operators; HPD CG also supports Jacobi.
  Identity MINRES admits complex Hermitian/HPD through the existing realification,
  with the original iteration budget and final typed residual acceptance.
  Complex Reference CG/MINRES work estimates remain unknown until realification
  and conversion costs are modeled.
  Unknown-shape Fast planning keeps sparse LU first, then CG/MINRES before BiCGSTAB
  among iterative fallbacks after reduction priority.

## [0.1.4] - 2026-10-27

### Added

- Finite real and complex Models use the shared automatic solver planner,
  retaining exact provider choices and admission reasons. Steady and transient
  MINI flow use shared admission for their supported solver policies.
- Anchored one-dimensional Q1 diffusion supports compatible CG policies,
  including Jacobi, for checked primal and derivative solves. The represented
  SPD proof covers its bounded piecewise-diffusion profile; general tangents
  and other unproved positive-definiteness claims remain excluded.
- An installed heterogeneous-conductivity study combines fields and physical
  observations, multi-load fitting with SciPy, held-out predictions, ordered
  parameter datasets, and saved failure/retry history. Physical inputs, mesh
  resolution and direct/iterative solver choices can be changed. The package
  example and guide use one canonical source.
- First-order spatial trajectory products compose implicit-midpoint stages,
  endpoints and observation clocks, with grouped solve evidence and solve-free
  reacceptance of saved evaluations. Adaptive/event/moving-geometry derivatives
  and higher derivatives remain outside this boundary.
- Common workflows add coupled nonlinear Region equations, supported mixed,
  tetrahedral and ALE execution, field point/path sampling and affine
  integration, temporal interpolation/window observations, and replayable
  affine triangle refinement within their documented execution profiles.
- Measured CSV data and typed observations share exact numeric-unit
  normalization and common result export. Finite stochastic laws, typed
  moment/probability estimands and ensemble reductions distinguish sample
  gradients from expectation gradients.
- Accepted-boundary adaptive Run continuation and supported spatial trajectory
  derivatives compose with installed Python JAX/PyTorch interfaces. These are
  products of native Eqiora programs, not arbitrary framework differentiation.

### Changed

- Current Model and Model Transaction envelopes are v49, structural semantic
  fingerprints are v45, and Realization envelopes are v11. Recompile displaced
  artifacts from authored inputs; retired codecs are not silently migrated.
- Differentiable evaluation records use v4. Python evidence exposes grouped
  `primal_solves` instead of the singular `primal_solve`; regenerate displaced
  development evaluation records.

- Removed the Rust compatibility aliases `PoissonSolution1d`,
  `PoissonComparisonRow`, and `SteadyStokesScaleProfile2d`. Use
  `ScalarEllipticFvmSolution1d`, `ScalarEllipticComparisonRow1d`, and
  `IncompressibleFlowScaleProfile2d`, respectively.

## [0.1.3] - 2026-10-11

### Added

- Finite real and complex Models, with typed values, now work with the shared
  linear and Newton solvers, time integration, sensitivities, Hermitian
  eigenproblems, harmonic response, and finite-window Fourier observations.
  Coordinate bases, source identities, and replay history are kept with them.
- Energy variations for scalar and elastic problems, complex scalar Q1 weak
  forms, and finite spectral forms can be written in a model and run through the
  ordinary Model, Formulation, Plan, Run, and Result workflow. Solver selection
  distinguishes Hermitian from non-Hermitian operators.
- New calculus on coordinates: products and partial derivatives with
  dimensions, integrals over bounded intervals including moving endpoints,
  coordinate pullbacks, radial diffusion, nonlocal kernel actions, and Result
  observations on a stated support. Each works within its documented execution
  profile.
- Curl and cross products in oriented Cartesian coordinates, directional
  derivatives, boundary trace targets, and explicit regularity for weak test
  functions. Vector moments on compatible tetrahedral meshes reach shared linear
  Plans.
- Physical interfaces between Cartesian regions, with one-sided scalar, vector,
  and tensor traces. Scalar and vector interface laws that you write run through
  the common Cartesian plans, with continuity and transmission stated as weak
  conditions. An interface implies no continuity or flux law of its own, and
  interfaces on general simplicial meshes cannot be executed yet.
- Common linear plans gain scalar and vector storage per Field, histories mixing
  differential and algebraic unknowns, kinematic displacement boundaries, initial
  assignments, and planar P1 vector fields.
- Planar MINI/P1 constraints derived from the model's equations, with a spatial
  method chosen per Region, static and backward-Euler solves, and replay of Plans,
  States, and Results. MINRES is used when the mixed operator is known to be
  symmetric and BiCGSTAB otherwise. Constant Lagrange-multiplier modes are
  checked; general full rank, inf-sup stability, and convergence are not
  established.
- Prescribed conservative transport of a scalar and conservation on moving
  volumes, within the supported common-plan profiles.
- Better editor support from the compiler: typed hover, completion, go to
  definition, find references, workspace refresh, and atomic application of document changes
  using UTF-16 positions.
  Python wheels include highlighting, completion, and hover for Eqiora source
  cells in Jupyter, compiled natively in IPython.

### Changed

- This release stays in the `0.1.x` series. The `0.2.0` candidate was never
  published and has been superseded; its unused tag has been deleted, leaving
  `0.2.0` free for a future release.
- Model and Model Transaction formats are now v45, with updated structural
  fingerprints and source identity. Files in older formats are rejected rather
  than migrated; regenerate them with this release.
- Shared mathematical forms and linear solver providers are prepared once and
  reused across repeated runs. MINI transport and the natural outlet flux are now
  derived from the model's equations. The Reynolds-100 wake example is still not
  a validated benchmark.
- Migrating fluid–structure interaction to the common equation-driven runtime is
  not finished.

### Fixed

- Newton probes on the reference domain stay inside the valid domain of the
  coordinate map, and weak boundary forms respect the regularity declared for
  their Field.
- Rank-deficient equation blocks, singular initial points, and irregular DAE
  tangents are rejected with diagnostics that point to the source.
- Simultaneous event resets are ordered by explicit priority, reduced dimension
  powers stay exact, and the editor keeps the correct document text when files
  are closed and reopened.
- Boundary operators whose operand is not defined on the whole parent support
  are rejected, as are interface traces whose selected side does not match the
  interface or Field support. The order and orientation of interface normals
  survive Model replay.

## [0.1.2] - 2026-09-15

### Added

- A Reynolds-100 Kármán vortex street replaces the short cylinder start-up
  example. The Python and Colab versions run until the wake has developed and
  report drag and lift coefficients, the front-to-rear pressure difference, and a
  sampled Strouhal number, with accessible media for the site. The result is not
  a validated benchmark.

### Changed

- The Kármán wake is now the main example in the README, on the documentation
  home page, and in the Gallery, with reproducible poster, reduced-motion, WebM,
  and MP4 media.

### Fixed

- The periodic verification gate runs only the `host-cpu` environment by
  default and reports physical MPI-CUDA cases as not selected unless that
  environment is requested.
- Newton rejects linear tolerances loose enough to accept a zero correction
  while the nonlinear residual is above its target, and an exhausted line search
  reports its best residual and step scale.
- Profiling memory no longer grows during long runs: phases reported only in
  aggregate keep one representative event alongside their full counts and
  timings.
- `eqiora.View.add()` accepts scalar cell-based `DerivedFieldSnapshot` values
  such as `State.curl(...)`, which the Colab wake notebook needs.
- The Kármán example, notebook, site sources, and media are tested from the
  source distribution, and the View stub's import of the derived field type is
  fixed.

## [0.1.1] - 2026-09-14

### Added

- `eqiora.colab.prepare()` checks the notebook runtime, restarts a stale Colab
  kernel before importing host packages, enables the custom widget manager, and
  prints the installed package versions.

### Changed

- The viewer is part of the base installation, with `anywidget==0.11.0` loaded
  only when a notebook needs it. The `viewer` extra has been removed.
- The Colab notebook uses `eqiora.colab.prepare()` and the shared viewer instead
  of its own environment setup and Matplotlib plots.

### Fixed

- Colab environment checks map diagnostic modules to their distributions
  explicitly, so Matplotlib's `mpl_toolkits` namespace no longer causes a
  failure.

## [0.1.0] - 2026-09-14

### Added

- The voltage-divider, sampled-control, property-composition, and transient
  heated-body examples run from the installed package. Their package locks,
  Models, Plans, Results, replay, and physical balances are checked against
  independently derived values.
- Numerical resolution is driven by the model's equations and formulations for
  the current scalar, elasticity, Stokes, transient-flow, and FSI paths. Fields,
  Domains, Connections, constraints, gauges, solver and provider choices, and
  placement survive Plan replay and reach execution.
- FSI runs as an ordinary problem with three Domains and two Connections. One
  mapping between Fields and degrees of freedom drives partitioning, assembly,
  State extraction, interface reactions, Result recovery, and the Python topology
  views.

### Changed

- Transient MINI runs prepare assembly once, evaluate local contributions once
  per nonlinear iterate, and compute full residuals and reactions only for
  accepted steps. Faer reuses its symbolic analysis while the sparsity pattern is
  unchanged.
- Profiles report totals, self time, call counts, and repeated solver phases
  separately. Trajectory times come from the grid of accepted steps.
- Versions move from Cargo `0.1.0-alpha.14` and Python `0.1.0a14` to `0.1.0`.
  The API is still pre-1.0: published releases never change, but later pre-1.0
  releases may break compatibility.

### Fixed

- The Colab wake notebook keeps a single consistent Matplotlib installation when
  the runtime preloads it, and scalar colorbars use the public
  `Figure.colorbar` API.

### Removed

- The GTK3/Tauri Studio shell and its native-only CAD authoring, Python export,
  and packaged DC-drive workflows. The browser Studio remains as an interaction
  preview with no native runtime dependency.
- Single-problem FSI execution records, positional Field recovery, duplicate
  assembly paths, and outdated solver lifecycle entry points.

## [0.1.0a14] - 2026-09-14

### Added

- ODE functionals evaluated at the final time or integrated over accepted steps
  can now include events, with forward parameter JVPs that account for event
  times and resets. Event sides, derivative coordinates, tolerances, and Result
  history survive replay and Python execution.
- The Tsitouras45 request takes event and forward-sensitivity settings with
  units directly.
- Equality, inequality, and complementarity conditions in source, Rust, and
  Python, solved with an active-set method for finite affine real scalar Models.
  Plans keep a tolerance per operand in coherent SI units, and Results record
  which conditions were active, rechecked independently.
- Storage terms for scalar Laws on fixed volumes, with polynomial time chain
  rules, accumulation checks, replay, and constant-capacity terms. Transient
  spatial thermal problems still cannot be executed.
- Named test functions with zero-trace restrictions in `weak_form`, in source
  and Python.
- An installed Python property-composition example that runs from a local
  package lock and reproduces its Model and Result identities offline.
- Shared Cartesian Region assembly and Field recovery for chains of
  non-overlapping scalar conservation regions, including interface flux.

### Changed

- Model/Transaction artifacts move to v27, Source identity to v19, structural
  fingerprints to v22, Plan and Result artifacts to v5, and Trajectory artifacts
  to v3. Older encodings are rejected.
- `primal_form` and the global `test(field)` are replaced by named
  `test(..., zero_on=...)` functions owned by a component and `weak_form(...)`.
  Rust code uses `RelationCondition`; the `Equation` and `LexResult` exports are
  removed.
- Algebraic Field and gauge structure is checked during solver selection, before
  a provider is chosen, and redundant realization constructors are removed. Block
  ranges, trace quotients, arbitrary constraints, and multi-region scalar
  quotients are not supported yet.

## [0.1.0a11] - 2026-09-13

### Added

- `ModuleRef.record`, `ModuleRef.enum`, and `ModuleRef.space` in Python
  authoring, so a component can use public types from another local or locked
  package without copying their declarations.

### Changed

- Local and imported types are emitted with the same qualified syntax. Record
  constructors, enum members, and finite-space counts keep the import alias, and
  private, unknown, transitive, or merely same-shaped foreign declarations are
  rejected by the Module resolver.
- `ImportedRecord` stays separate from a local `Record` but is constructed the
  same way, so the stubs show each member's value type.

## [0.1.0a9] - 2026-09-11

### Added

- Optional execution profiling for Python `run()` and `submit()`. Profiles cover
  the run, setup, solve, time steps, assembly, nonlinear iterations, linear
  solves, backend calls, and post-processing, and include nonlinear convergence
  data.
- Shared tracing points in the numerical kernels and the Faer backend, with
  profiling tests for ODE and steady and transient cylinder runs.

### Changed

- The transient cylinder example can print a short timing summary. Profiling is
  off by default, and runs without it produce bit-identical results with no
  collector installed.
- The Rust quick start now uses the `eqiora::compiler` API and the source syntax
  of the published alpha.

## [0.1.0a8] - 2026-09-10

### Added

- `eqiora.Module` unifies source and Python modeling, with explicit imports,
  structural dimensions, imported pure operators, named connectors, grouped
  relations, boundary families, evolution, events, and sampled components.
- Integers, Boolean predicates, enums, static arrays and slices, indexed
  equations and reductions over bounded ranges, flat records, typed buses, scalar
  partial derivatives, analytic scalar properties, and real-valued tables.
- Deterministic project resolution over local, bundled, and Git sources, with
  dependency locks and offline vendoring.
- Region assembly derived from the equations for coupled linear scalar Q1
  problems and fixed-reference FSI, including recovery of boundary fluxes,
  loads, and reactions.
- Typed Result observables, ordered parameter batches, first-order derivative
  products, and limited JAX `vmap` support.
- Smooth scalar ODE observables at the final time, and Simpson integration over
  accepted steps independent of the output schedule. They cover the whole run
  interval and do not support events, resets, derivatives, or space-time terms.
- Convex polyhedral Geometry that can be matched to a supplied tetrahedral mesh.
  There is no tetrahedral mesher and no new PDE or FSI execution.
- Steady scalar conservation Laws in source and Python that keep their outward
  flux and source through Model replay. Storage and moving volumes are not
  supported.
- An `eqiora-language-server` preview over stdio with diagnostics, formatting,
  symbols, folding, Markdown hover, go to definition, and find references across
  open workspace modules. Workspace folders with an `eqiora.toml` load local
  package sources without writing a lock or store. Analysis runs in the
  background, drops outdated document versions, and supports request
  cancellation.
- `Eqiora.Fluid@0.4.0` and `Eqiora.Solid@0.4.0`, with field-driven velocity,
  displacement, and traction boundaries. Vendoring from an installed
  distribution provides `Eqiora.Fluid.Incompressible@0.6.0` and
  `Eqiora.Solid.LinearElasticity@0.6.0` with their dependencies.

### Changed

- The package sources now live under `eqiora-api`, so Cargo archives contain the
  same bundled packages as source and Python builds.
- A linear solver request names either a planning objective or a complete
  combination of algorithm, preconditioner, reduction, and provider; tolerances
  alone are not enough.
- Model and Component signatures and module and import syntax are finalized for
  this series. Python equations use `equation(lhs, rhs)`.
- Model/Transaction artifacts move to v25, Plans and Results to v3, Trajectories
  to v2, Source identity to v17, and structural fingerprints to v20. Recompile
  models and regenerate saved artifacts and package locks.
- Diffsol is updated to 0.16.2, with its backend identity and verification
  evidence re-recorded against that version.
- Base64 is updated to 0.23 and the Python build to maturin 1.15.0.

### Removed

- The old Python `Source` and builder APIs (use `eqiora.Module`), the raw public
  FSI solve functions, and leftover compatibility code.
- The flat `ConnectionDecl::ports()` view; use the segmented Port paths.

## [0.1.0a7] - 2026-09-02

### Added

- Natural equation syntax in the Python `Source` API, and dimension inference
  for model-local aliases.
- Recognition of ideal-gas Euler conservation laws within a limited scope, and
  reusable scalar conservation for 1D, 2D, and 3D Cartesian domains.
- A cancelled run returns its last accepted State, and common Plans record the
  formulation, material, mesh, and recognition they were built from.

### Changed

- Mesh generation goes through `MeshPlan`. Scalar execution is built on
  conservation laws, and elasticity on shared kinematics, isotropic material,
  element-input, constraint, and observation code.
- Execution structure and Faer factorizations are prepared once per run rather
  than for every solve or step.
- Spatial Plans share one identity scheme, and transient runs share one
  scheduling scheme.
- Documentation is built once per source, CI reuses equivalent results, and
  Colab is the only maintained hosted-notebook option.

### Removed

- Physics-specific dimension aliases and the Marimo and Jupyter example paths.

## [0.1.0a6] - 2026-08-31

### Added

- Scalar properties in the common compiler and execution path, including
  bindings written in Python and affine coefficients.
- Scalar primal Formulations written by the user, directional Stokes
  correspondence, and a Plan identity that shows the selected formulation.
- `math.pi`, typed model-local `let`, and derived-dimension aliases in the
  language and its Python `Source` API.
- Optional initial values for scalar Fields, and local file I/O for resolved
  Plans, Results, and Trajectories. Rust gains a common transient `RunRequest`.

### Changed

- Rearranged forms of an equation, such as `A = -B` or the two sides swapped,
  are accepted and give the same residual.
- The steady and start-up cylinder examples use finer meshes, and the start-up
  animation is retimed so the change is visible.

### Fixed

- Gmsh start-up in a fresh Google Colab session, and the accessible figure list
  in the cylinder gallery page.

### Removed

- Unsupported options in the Python `Source` API and duplicate transitional ML
  artifact aliases.

## [0.1.0a5] - 2026-08-29

### Added

- Saving and loading of resolved Plans, restartable States, Trajectories, and
  Results. Decoding is done natively with size limits and checks that the
  artifact matches the objects it depends on; Python reads and writes the same
  bytes.
- Inspection of the effective Formulation, with caller overrides, on the common
  spatial path. Stokes uses a mixed Formulation and conservative flow an
  integral one; unsupported or incompatible choices fail before any numerical
  work.
- `eqiora.View` for Geometry, Meshes, selections, and scalar vertex and cell
  Fields. The optional viewer uses one Rust-built scene for both the standard
  display and Marimo. Studio integration, 3D, vector, tensor, and trajectory
  views, and cloud delivery are planned under #625.

### Changed

- Package compilation records use the v2 format only; v1 records can no longer
  be read.
- Compiling a locked package in Python now binds a public root Component to the
  Geometry and Parameters you supply. The resulting Model goes through the usual
  Mesh, Plan, and Run steps, and records the package it came from.
- Common execution uses one native Plan type and one Model, Formulation, and
  Realization interface. Solver decisions are made before numerical work, and
  compiler lowering is organized by responsibility rather than by physics.
- Static scalar, elasticity, and Stokes runs return the common Result, and
  their older run and output functions are removed. Dynamic Results keep their
  full Trajectories.

## [0.1.0a4] - 2026-08-28

### Added

- `eqiora.lang.Source` in Python can describe the complete steady-cylinder
  Component. It emits readable `.eqi` source, and compiling either the Source
  object or the emitted file goes through the same Rust compiler. It is not a
  Python equation runtime or a stable AST format.

### Changed

- Accepted transient MINI States provide typed pressure samples at points and
  signed boundary forces in 2D, distinguishing the force on the fluid from the
  equal and opposite force on the boundary. The cylinder gallery shows these
  quantities without benchmark comparison.
- A transient cylinder example in plain Python, Marimo, and Jupyter versions,
  all sharing the same geometry, Gmsh mesh, steady start, transient run,
  cell-average vorticity, and plotting code, and saving ten outputs up to 0.1 s.
  The gallery shows it as a poster, WebM and MP4 video, a reduced-motion still,
  and a text description. The flow is still a nearly symmetric start-up, not a
  developed wake, and the example has not been verified. In the `0.1.0a4`
  documentation it can be opened in Colab, which installs the pinned release.
- `State.curl(velocity_field)` computes cell-average vorticity from 2D MINI
  velocity, and the Matplotlib adapter plots it with `plot_scalar_field`. The
  wheel includes the transient-cylinder Component.
- Transient MINI/P1 flow on a geometry now advances a nonzero cylinder wake
  with Newton's method in correction form, using sparse LU for the saddle-point
  Jacobian.
- The `TransientNavierStokesReference2d` API and its hand-built fixed mesh were
  removed in favor of the common
  `Geometry -> Mesh -> Model -> Plan -> State -> Run -> Result` workflow.
- `eqiora.compile` takes keyword arguments only, with exactly one of `path=` or
  `source=`. A component file with only definitions can be compiled directly
  with a Python `Geometry` and SI parameter values: the source defines dimensions
  and named supports, and Python supplies the concrete shapes. The cylinder
  example uses this with the common `resolve` and `run` functions.
- `eqiora.resolve(model, ...)` plans steady and transient flow from the Model
  with your Geometry and Mesh and typed spatial, solver, scaling, and time
  settings. Resolving a mesh only plans it; generating it produces the Mesh.
- Gmsh sizing is a single policy owned by the mesher, with an explicit global
  target size. The old rich-Mesh notebook display and fixed-mesh cylinder code
  are removed.
- The Studio DC-drive example no longer presents the frozen 0.1.0a3 results as
  verification of the current release; its current compilation and run are
  marked unverified.

## [0.1.0a3] - 2026-08-23

### Changed

- The Python cylinder example meshes automatically with Gmsh 4.15.2 and imports
  its linear 2D MSH 4.1 output. If Gmsh is missing, the wrong version, or fails,
  meshing stops with an error; the old reference-mesh fallback is gone.
- The cylinder Result and Gallery pressure image use the resulting Gmsh mesh with
  662 vertices and 1,210 triangles. Arbitrary geometry, curved elements, adaptive
  sizing, and identical meshes across platforms are not supported.

## [0.1.0a2] - 2026-08-21

### Added

- `eqiora.compile_package(...)` compiles a Model from a locked package in an
  installed Python environment, given a content-addressed store, the resolution
  bytes, and the name of a root-local Model. It does not discover, author, or
  install packages, access the network, or run the model.
- `eqiora.check_package_conformance(...)` checks a locked package against the
  `eqiora.package.structural-conformance-v1` profile. The report confirms that
  the package replays consistently; it is not scientific verification.
- A serial prescribed dynamic-solid example that publishes a standalone
  Realization, the previous and next two-Field States, and a Run whose output is
  the next State. It covers only the unit-cube reference case.
- An external boundary-provider path for the same dynamic-solid case: an
  already-connected subprocess supplies the boundary displacement, and Eqiora
  records it and links it to the next State in a separate Run. It is verified
  only with ordinary-GIL CPython 3.12 and NumPy 2.1.0.
- An optional `eqiora.matplotlib` adapter that plots the cylinder's P1 pressure
  as a Matplotlib Figure you own and can save headlessly. Matplotlib is not a
  base dependency.
- `eqiora.solid.solve_mixed_boundary_elasticity(...)` and a packaged Python
  example for the mixed-boundary elasticity case. Studio and Python share one
  Rust implementation, and the Matplotlib adapter can draw the original and
  scaled displaced meshes.
- An `eqiora.fsi.FixedMeshMonolithic` option with an inspectable Plan for the
  two-step FSI case. Studio and Python share one resolver, and the Result's
  Trajectory and states work with the general scalar and deformed-field plots.

### Changed

- The unreleased Python names `FixedReferenceFsiStep`,
  `FixedReferenceFsiResult`, and `solve_fixed_reference_fsi(...)` are removed.
  Fields are read from `State`/`FieldSnapshot`; partition, interface, energy,
  residual, solve, and assembly data come from
  `fixed_mesh_monolithic_evidence(result)`.
- `plot_fixed_reference_fsi(...)` is replaced by `plot_scalar_field(...)` and
  `plot_deformed_field(...)`, which take a `Trajectory` and a Model-bound
  `FieldRef`.
- The cylinder flow example returns the common `Result`, `FieldSnapshot`, and
  `Mesh`. Its scientific observations come from
  `fluid.steady_stokes_evidence(result)`, and `CircularHoleSteadyStokesResult`
  and `plot_pressure(...)` are replaced by
  `plot_scalar_field(result, field=...)`.
- Model artifacts use a single current format. The Rust `eqiora::compatibility`
  module and version-selecting codec API, Python `eqiora.compatibility`, support
  for Model/Transaction v1–v7, and control-v1 are removed. Use Rust
  `ModelDocument::{compile, define, replay}`, Python `eqiora.compile`,
  `Model.define`, and `eqiora.replay`, and `eqiora.control/v2`. Older Model and
  Transaction files and control-v1 requests are rejected without migration.
- The `eqiora::numerics` facade no longer re-exports
  `SteadyStokesGeometryBinding2d` or
  `solve_resolved_steady_stokes_geometry_mini_2d`. Applications use
  `eqiora::api::CircularHoleSteadyStokesResult2d`; lower-level code can still use
  the `eqiora-numerics` crate.
- The `eqiora::numerics` facade no longer re-exports the ten low-level
  fixed-reference FSI construction and solution names. Applications use
  `eqiora::api::FixedReferenceFsiResult2d`; lower-level code can still use
  `eqiora-numerics`.
- Assembly, meshing, geometry, and solver types are reached through their own
  crates and `eqiora` namespaces; the aliases in numerics, artifact, and
  realization are removed.
- Package identity and in-memory resolution are available by default; authoring,
  replaying, and installing packages from directories requires the
  `package-filesystem` feature.
- Harmonic ALE mesh motion is split into the Realization-level
  `P1HarmonicMeshMotionPolicy` and the numerical `P1HarmonicMeshMotionAction`.
  The previous Rust names are gone; artifact formats are unchanged.
- The Rust crate `eqiora-fabric` is renamed `eqiora-backend-rayon` and its
  `threaded` feature `rayon`, and `eqiora-diff` is renamed
  `eqiora-differentiation`. The old names have no aliases.

## [0.1.0a1] - 2026-07-23

The first public alpha. It contains:

- a small typed semantic kernel, the Eqiora Language frontend, model
  transactions, reference hybrid execution, and a scalar operator IR;
- scalar elliptic FEM/FVM and initial solver, time integration,
  differentiation, artifact, package, geometry, I/O, CPU, CUDA, and MPI
  components, each covering a narrow case;
- a Python modeling API with synchronous and asynchronous native execution,
  structured diagnostics, explicit NumPy/DLPack ownership, and PyTorch and JAX
  differentiation adapters for limited cases;
- a thin Studio interface over the same model and application service;
- versioned artifacts, verification cases, a public capability matrix, and
  release manifests.

Wheels are for ordinary-GIL CPython 3.11–3.14 on manylinux x86-64. There are no
macOS, Windows, free-threaded Python, or GPU wheels, MPI is not bundled, the
component catalogue is incomplete, and there is no 1.0 compatibility promise or
safety certification.

The [capability matrix](docs/capability-matrix.md) and the cases in
[`verify/`](verify/) describe exactly what is and is not supported.

[Unreleased]: https://github.com/eqiora/eqiora/compare/v0.1.4...HEAD
[0.1.4]: https://github.com/eqiora/eqiora/releases/tag/v0.1.4
[0.1.3]: https://github.com/eqiora/eqiora/releases/tag/v0.1.3
[0.1.2]: https://github.com/nkiyohara/eqiora/releases/tag/v0.1.2
[0.1.1]: https://github.com/nkiyohara/eqiora/releases/tag/v0.1.1
[0.1.0]: https://github.com/nkiyohara/eqiora/releases/tag/v0.1.0
[0.1.0a14]: https://github.com/nkiyohara/eqiora/releases/tag/v0.1.0a14
[0.1.0a11]: https://github.com/nkiyohara/eqiora/releases/tag/v0.1.0a11
[0.1.0a9]: https://github.com/nkiyohara/eqiora/releases/tag/v0.1.0a9
[0.1.0a8]: https://github.com/nkiyohara/eqiora/releases/tag/v0.1.0a8
[0.1.0a7]: https://github.com/nkiyohara/eqiora/releases/tag/v0.1.0a7
[0.1.0a6]: https://github.com/nkiyohara/eqiora/releases/tag/v0.1.0a6
[0.1.0a5]: https://github.com/nkiyohara/eqiora/releases/tag/v0.1.0a5
[0.1.0a4]: https://github.com/nkiyohara/eqiora/releases/tag/v0.1.0a4
[0.1.0a3]: https://github.com/nkiyohara/eqiora/releases/tag/v0.1.0a3
[0.1.0a2]: https://github.com/nkiyohara/eqiora/releases/tag/v0.1.0a2
[0.1.0a1]: https://github.com/nkiyohara/eqiora/releases/tag/v0.1.0a1
