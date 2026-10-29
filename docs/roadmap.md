# Roadmap

Eqiora is developing reusable physical models, numerical methods, and Python
and Studio workflows. This roadmap shows priorities and the dependencies that
make later features possible. For functionality available today, start with
the [capability matrix](capability-matrix.md).

## Direction

Write the textbook equation, choose the method, solve it fast on CPU threads, MPI
or GPU, and differentiate the result. Well-known equations and boundary
conditions are one `import` away. You can still write any equation yourself and
mix it with standard components. Decided in
[#1742](https://github.com/eqiora/eqiora/issues/1742).

What sets this apart from tools that already solve user-written equations:

- you write the strong form as it appears in a textbook, not a hand-derived weak form;
- the method is chosen separately from the model, so the same model runs with FEM
  or FVM on CPU, MPI or GPU by changing the Plan;
- results can be differentiated through the solve and composed with JAX and PyTorch; and
- units and types are checked, and lumped and field models live in one system.

Rust is why it is fast and installs with `uv add eqiora`. It supports the
direction; it is not the headline.

Work is ranked by the user outcome it unblocks:

| Promise | Work |
| --- | --- |
| Textbook equations by import | standard library ([#1743](https://github.com/eqiora/eqiora/issues/1743)), direct boundary values ([#1736](https://github.com/eqiora/eqiora/issues/1736)) |
| Little code | defaults and name lookup ([#1737](https://github.com/eqiora/eqiora/issues/1737)), solver selection ([#366](https://github.com/eqiora/eqiora/issues/366)) |
| Choose the method | P2/Q2 and Taylor–Hood ([#1738](https://github.com/eqiora/eqiora/issues/1738)), substitutable solvers ([#1707](https://github.com/eqiora/eqiora/issues/1707)) |
| Fast on CPU, MPI and GPU | external comparison ([#1739](https://github.com/eqiora/eqiora/issues/1739)), run performance ([#802](https://github.com/eqiora/eqiora/issues/802)), portable execution ([#1568](https://github.com/eqiora/eqiora/issues/1568)) |
| Differentiable | solve and trajectory derivatives ([#1565](https://github.com/eqiora/eqiora/issues/1565), [#1566](https://github.com/eqiora/eqiora/issues/1566)) |
| Runs anywhere | macOS and Windows wheels ([#1734](https://github.com/eqiora/eqiora/issues/1734), [#1741](https://github.com/eqiora/eqiora/issues/1741)), Colab ([#1735](https://github.com/eqiora/eqiora/issues/1735)) |
| People find and trust it | Discussions, citation, JOSS ([#1740](https://github.com/eqiora/eqiora/issues/1740)) |

Deferred until the core fluid, solid and thermal path meets this direction:
quantum, kinetic and plasma models, electrochemistry and semiconductors, order
parameters, PSpice import, Studio, and a native CAD provider. These issues
carry the `status:deferred` label.

## Next release

The next release checkpoint brings the language and Python APIs together around
ordinary circuit, thermal, sampled-system, and property workflows. Its remaining
work includes:

- records and enums for structured values and buses;
- named connectors, exact boundaries, and component composition;
- conservation Laws with storage, physical flux, and sources;
- observables, spatial and trajectory functionals;
- inequalities and complementarity with explicit numerical enforcement;
- authored weak and integral forms; and
- corresponding Python APIs, examples, package composition, and saved artifacts.

Analytic properties need typed inputs, validity rules, and formal partials.
Data-backed properties then add exact tables and specified interpolation.
State-dependent accumulation uses expression differentiation; trajectory
functionals need integration history that is independent of plotting cadence.
These dependencies are part of the release work.

The broader tensor, complex-number, and eigenproblem portfolio follows this
checkpoint. GPU/XLA exploration also follows the release.

## Implemented foundations

### Meaning, language, packages, and artifacts

Typed quantities, equations, activations, Ports, and connections support
continuous and sampled systems. Components expand into complete models;
packages use explicit versions and can reopen offline. Spatial supports,
field-valued interfaces, and complete exterior Port families connect component
models to geometry. Models, Plans, States, Results, and Trajectories can retain
their relationships when saved and loaded.

### Spatial numerics and physics

Current paths include scalar elliptic FEM/FVM on Cartesian meshes in 1D–3D,
linear elasticity, steady Stokes flow, fixed-domain transient flow, and
fluid–structure interaction. Simplex and hypercube geometry, quadrature,
P0/P1/Q1 and MINI spaces support these methods. Mesh and field interchange
includes selected Gmsh, VTU, XDMF, and HDF5 workflows.

The matrix describes the available combinations of geometry, spaces, boundary
conditions, and solvers. General adaptive meshing, higher-order elements, and
broader constitutive libraries are future work.

### Execution and differentiation

Reference execution supports scalar continuous dynamics, periodic updates,
and localized zero crossings with resets. CPU, CUDA, and MPI paths are
available for the problems and environments listed in the matrix. Scalar
operators provide values, Jacobian-vector products, and vector-Jacobian
products; current differentiation also includes implicit and selected spatial
and event calculations.

## Multiphysics

The fluid–structure work progresses from fixed-reference models to moving
meshes and field transfer:

```text
Geometry and mesh correspondence
  → fixed-reference monolithic 2D fluid–structure interaction
    → fixed-topology moving-mesh interaction in 2D and tetrahedral 3D
      → remeshing and field transfer
        → trajectory export and derived datasets
```

Current implementations cover specific models and meshes in this sequence.
Next extensions include broader interfaces and material models, more capable
solvers, and larger distributed runs. Shape sensitivities depend on the
corresponding geometry, mesh-motion, and transfer calculations.

## Public clients

### Python

Python is the main workflow for authoring, compiling, meshing, running, and
plotting. It also provides package compilation, saved artifacts, NumPy arrays,
CPU DLPack exchange, and optional PyTorch/JAX differentiation adapters. The
[Python guide](python/README.md) lists installation requirements and examples.

Planned work includes more complete component and PDE authoring, richer
results and visualization, framework GPU support, and additional platforms.

### CLI and editors

The CLI compiles and checks `.eqi` models. The language-server
preview adds diagnostics, formatting, symbols, and navigation. Execution and
result workflows remain separate from the current compile/check tools.

### Studio

Studio provides model inspection, diagnostics, reference execution, selected
FEM/FVM workflows, scalar edits with undo/redo, and example applications.
Future work expands CAD editing, mesh and vector-field visualization, imported
and adaptive fields, 3D views, and localization.

## Next capabilities

| Area | Next work | Depends on |
| --- | --- | --- |
| Geometry and CAD | Curved and multi-body geometry, sketches, features, imports, and meshing | Geometry editing and regeneration; Python and Studio authoring |
| External providers | A second-language implementation of prescribed solid-boundary behavior | Existing connected-subprocess interface |
| Physics libraries | Thermal slab, elasticity patch, and Couette–Poiseuille workflows, followed by thermoelasticity and conjugate heat | Reusable material and boundary models; numerical forms for the selected method |
| Finite elements | Broader structural and fluid element libraries | Typed forms, supported trial/test spaces, and assembly |
| Time and hybrid systems | General implicit DAEs, simultaneous event ordering, modes, multi-rate loops, and fault statecharts | Continuous integration and reset semantics |
| Scale and execution | Stronger preconditioners, distributed results, checkpoint/restart, GPU-resident assembly and solve, then multiple GPUs | The corresponding serial solver and distributed data layout |
| Differentiation and optimization | Trajectory adjoints and checkpointing, then moving-mesh, remeshing, and CAD sensitivities | The matching forward calculations and their saved history |

Finite-volume methods use conservative face fluxes; they do not depend on a
finite-element weak-form representation. Coupled optimization builds on the
individual physics and sensitivity paths needed by the chosen objective.

## Gallery sequence

The planned demonstrations are:

1. laminar cylinder wake;
2. thin cylindrical-shell collapse;
3. Turek–Hron FSI3;
4. Stokes dissipation shape optimization;
5. three-dimensional Taylor–Green flow;
6. notched-plate phase-field fracture;
7. dam break around an obstacle; and
8. electric-motor multiphysics.

Cylinder flow develops the fluid and output workflows needed by later coupled
examples. Stokes design optimization follows FSI so it can reuse geometry,
mesh motion, steady-flow results, and differentiation. The motor example
combines electromagnetic, circuit, rotating, thermal, and coolant models and
therefore comes later. The [Gallery](https://eqiora.org/gallery/) contains the
walkthroughs currently available.

## Example systems

Four compact systems guide the system-modeling work: multi-rate feedback with
an algebraic loop, a bouncing ball with resets, a DC motor with a discrete
controller, and a fault statechart with a thermal plant. The packaged
DC-motor/controller example is already available; the other examples require
the corresponding time, event, and statechart features above.

See the [benchmark roadmap](verification/benchmark-roadmap.md) for planned
numerical comparisons and the [issue tracker](https://github.com/nkiyohara/eqiora/issues)
for individual features and their dependencies.
