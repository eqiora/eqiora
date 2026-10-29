# Capability and Maturity Matrix

Find the problem types and workflows available in Eqiora, along with their
current requirements and planned extensions. Start with the executable problems
below, then open the area file for detailed status.

## Executable problem classes

Choose a problem class, then follow its case details for setup and assumptions.
The detailed rows in each area file separate implementation, execution,
verification, and maturity.

Native execution uses the common Plan/Result lifecycle. Raw scalar assembly and
linearization helpers used only inside that lifecycle are private; direct scalar
FVM, simplicial assembly, transport assembly, and MINI step assembly forwarding
exports have been removed. Their existing numerical kernels and common consumers
remain. Artifact JSON budgets are owned by `JsonLimits`; diagnostic records reuse
the kernel severity, source span, and patch types.

| Problem class | What runs today | Case details |
|---|---|---|
| Stationary scalar elliptic PDE | Poisson problems on generated Cartesian meshes in 1D–3D run through finite elements or finite volumes; one fixed 2D triangular finite-element path is also covered. | [`numerics.poisson-fem-fvm`](../verify/numerics/poisson-fem-fvm/README.md), [`numerics.cartesian-poisson-3d-fem-fvm`](../verify/numerics/cartesian-poisson-3d-fem-fvm/README.md) |
| Steady incompressible Stokes flow | A 2D triangular finite-element reference handles all-velocity and mixed static-pressure boundary cases. | [`fluid.mixed-static-pressure-mini-stokes-2d`](../verify/fluid/mixed-static-pressure-mini-stokes-2d/README.md), [`fluid.port-closed-si-mini-stokes-2d`](../verify/fluid/port-closed-si-mini-stokes-2d/README.md) |
| Fixed-domain transient incompressible flow | A 2D cell-centred finite-volume Navier–Stokes reference runs on a generated Cartesian mesh. | [`fluid.cell-centered-navier-stokes-fvm-2d`](../verify/fluid/cell-centered-navier-stokes-fvm-2d/README.md) |
| First-order dynamic linear solid | One 2D prestrained linear-solid step runs as the solid side of fixed-reference FSI. | [`fsi.fixed-reference-monolithic-step-2d`](../verify/fsi/fixed-reference-monolithic-step-2d/README.md) |
| Fixed-reference monolithic 2D FSI | One conforming 2D fluid–structure model runs for one backward-Euler step. | [`fsi.fixed-reference-monolithic-step-2d`](../verify/fsi/fixed-reference-monolithic-step-2d/README.md) |
| Fixed-topology moving-mesh 2D FSI | One serial 2D fluid–structure trajectory moves the mesh while keeping its topology fixed. | [`fsi.fixed-topology-ale-monolithic-2d`](../verify/fsi/fixed-topology-ale-monolithic-2d/README.md) Physical velocity boundaries use `fsi::FixedReferenceFsiBoundary<D>` directly in both fixed-reference and ALE callers; the ALE-only alias is removed. |
| Fixed-topology moving-mesh 3D FSI | A serial tetrahedral 3D fluid–structure trajectory runs for multiple first-order time steps. | [`fsi.fixed-topology-ale-monolithic-3d`](../verify/fsi/fixed-topology-ale-monolithic-3d/README.md) |
| Conservative remeshing transfer for 2D FSI | One serial 2D transition transfers displacement, velocity, and pressure between two accepted meshes before continuing. | [`fsi.remeshing-transfer-2d`](../verify/fsi/remeshing-transfer-2d/README.md) |
| Explicit first-order ODE | Analytic reference ODEs run through the reference path and adaptive integrators. | [`time.canonical-first-order`](../verify/time/canonical-first-order/README.md), [`time.diffsol-adaptive`](../verify/time/diffsol-adaptive/README.md) |
| Constant mass-matrix DAE | A constant-mass system, including a rank-deficient accepted initial pair, runs through the adaptive path. | [`time.diffsol-adaptive`](../verify/time/diffsol-adaptive/README.md) |
| Residual-native implicit DAE | One state-dependent, index-one implicit residual problem executes with consistent initialization. | [`time.general-implicit-dae`](../verify/time/general-implicit-dae/README.md) |
| Localized event and reset | One falling zero crossing is localized, its grouped implicit reset is applied, and integration restarts from the accepted post-event state. | [`hybrid.registered-event`](../verify/hybrid/registered-event/README.md) |
| Fixed-step restart | Resume an implicit-Euler simulation from a saved checkpoint. | [`artifacts.implicit-time-restart-lineage`](../verify/artifacts/implicit-time-restart-lineage/README.md) |
| Transient PDE through FVM | 2D examples cover conservative scalar transport and incompressible flow on generated Cartesian meshes. | [`fluid.cartesian-advection-diffusion-fvm-2d`](../verify/fluid/cartesian-advection-diffusion-fvm-2d/README.md), [`fluid.cell-centered-navier-stokes-fvm-2d`](../verify/fluid/cell-centered-navier-stokes-fvm-2d/README.md) |
| Spatial-periodic scalar transport | One 2D cell-centred FVM case connects opposite Cartesian boundaries and transports a scalar conservatively across the seam. | [`fluid.cartesian-periodic-transport-fvm-2d`](../verify/fluid/cartesian-periodic-transport-fvm-2d/README.md) |
| Sampled electromechanical trajectory | One serial DC-motor model combines continuous physical relations with an exact periodic sample-and-hold controller. | [`hybrid.packaged-dc-motor-controller`](../verify/hybrid/packaged-dc-motor-controller/README.md) |

## Coverage index

Detailed rows live in one file per area. Update the area file in the same pull
request as the capability change.

- [Problem classes: detailed status](capabilities/00-problem-classes.md)
- [Canonical semantics and model identity](capabilities/01-canonical-semantics-and-model-identity.md)
- [Language, packages, and authoring](capabilities/02-language-packages-and-authoring.md)
- [Geometry, CAD, and spatial selection](capabilities/03-geometry-cad-and-spatial-selection.md)
- [Mesh, quadrature, spaces, and assembly](capabilities/04-mesh-quadrature-spaces-and-assembly.md)
- [Spatial discretization and numerical methods](capabilities/05-spatial-discretization-and-numerical-methods.md)
- [Physics and reusable component libraries](capabilities/06-physics-and-reusable-component-libraries.md)
- [Time, DAE, events, and hybrid semantics](capabilities/07-time-dae-events-and-hybrid-semantics.md)
- [Signals, controls, and system simulation](capabilities/08-signals-controls-and-system-simulation.md)
- [Linear, nonlinear, eigen, and time solvers](capabilities/09-linear-nonlinear-eigen-and-time-solvers.md)
- [Differentiation, optimization, uncertainty, and data](capabilities/10-differentiation-optimization-uncertainty-and-data.md)
- [CPU, distributed, accelerator, and cloud execution](capabilities/11-cpu-distributed-accelerator-and-cloud-execution.md)
- [Artifacts, provenance, I/O, and interoperability](capabilities/12-artifacts-provenance-i-o-and-interoperability.md)
- [Rust API, Python, and ecosystem integration](capabilities/13-rust-api-python-and-ecosystem-integration.md)
- [Studio: complete modeling and analysis UX](capabilities/14-studio-complete-modeling-and-analysis-ux.md)
- [Code generation, embedded, and real-time execution](capabilities/15-code-generation-embedded-and-real-time-execution.md)
- [Verification, validation, quality, and OSS maturity](capabilities/16-verification-validation-quality-and-oss-maturity.md)

## How to read the matrix

| Mark | Meaning |
|---|---|
| ✅ | Present for the exact scope stated in the row |
| 🟨 | Narrow, provisional, design-only, or incomplete |
| ⬜ | Absent |
| — | Not applicable |

The columns describe four independent aspects:

| Column | Question |
|---|---|
| **C — Contract** | Is there a typed, versioned semantic or lowered contract? |
| **X — Execution** | Does at least one real end-to-end path execute? |
| **V — Verification** | Is there reproducible evidence for the stated claim? |
| **M — Maturity** | Is the capability broad, robust, documented, performant, and suitable for general users? |

Read each status together with the row's scope. Execution and verification are
separate: a runnable path may have no registered verification. Eqiora is pre-1.0;
no technical product area is marked fully mature.
