# Physics and reusable component libraries

[Capability matrix](../capability-matrix.md) · [legend](../capability-matrix.md#how-to-read-the-matrix)

| Capability | C | X | V | M | Current implementation and planned extensions |
|---|:---:|:---:|:---:|:---:|---|
| Scalar Poisson/diffusion | ✅ | ✅ | ✅ | ⬜ | Manufactured 1D–3D Cartesian FEM/FVM and one fixed-simplex path |
| Transient heat conduction | 🟨 | 🟨 | 🟨 | ⬜ | 1D diffusion and time foundations exist; canonical multidimensional path remains |
| Convection and radiation heat transfer | ⬜ | ⬜ | ⬜ | ⬜ | Boundary laws, view factors, participating media, and coupling remain |
| Thermal contact | ⬜ | ⬜ | ⬜ | ⬜ | Contact resistance, gap conductance, and nonlinear coupling remain |
| Linear elasticity | ✅ | ✅ | ⬜ | ⬜ | One dimension-typed `IsotropicElasticityContinuum<D>` owns displacement/stress Relation identity, Lamé lineage, conservative load, complete boundary meaning, constitutive reduction, and integration measure across static Q1/P1, dynamic P1, and FSI consumers. The admitted profiles are explicitly intrinsic 2D with resultants per unit out-of-plane thickness and full 3D with volume integration. The 2D Model executes standalone through Cartesian Q1/CSR/CG with patch, balance, and convergence evidence; exact package releases prove reusable laws and boundaries; and the bounded tetrahedral ALE FSI slice executes 3D solid mechanics over multiple backward-Euler steps. Plane stress/strain, axial/beam/shell reductions, general standalone 3D analysis, wider boundary data, elements, materials, performance, scale, and a broad solid library remain outside the claim; [`solid.mixed-boundary-elasticity-2d`](../../verify/solid/mixed-boundary-elasticity-2d/README.md), [`fsi.fixed-topology-ale-monolithic-3d`](../../verify/fsi/fixed-topology-ale-monolithic-3d/README.md) |
| Nonlinear solid mechanics | ⬜ | ⬜ | ⬜ | ⬜ | Finite strain, objective rates, tangents, and robust increments remain |
| Structural dynamics | ✅ | ✅ | ⬜ | ⬜ | Bounded first-order 2D and 3D semantic/package meaning exists for displacement, velocity, positive density, momentum, and velocity/traction boundary roles. A 2D prestrained P1 state executes in fixed-reference FSI, and the method-neutral 3D release executes as the solid side of a multi-step tetrahedral fixed-topology ALE trajectory. Standalone stepping, restart/continuation, damping, modal, harmonic, response-spectrum, finite-strain, and contact paths remain; [`solid.dynamic-linear-solid-semantics-2d`](../../verify/solid/dynamic-linear-solid-semantics-2d/README.md), [`fsi.fixed-topology-ale-monolithic-3d`](../../verify/fsi/fixed-topology-ale-monolithic-3d/README.md) |
| Beams, shells, membranes, and trusses | ⬜ | ⬜ | ⬜ | ⬜ | Element families, locking control, sections, and orientations remain |
| Plasticity and viscoplasticity | ⬜ | ⬜ | ⬜ | ⬜ | Constitutive library, return mapping, and verification remain |
| Hyperelasticity and viscoelasticity | ⬜ | ⬜ | ⬜ | ⬜ | Material models and incompressibility treatments remain |
| Damage, fracture, and fatigue | ⬜ | ⬜ | ⬜ | ⬜ | Phase field/cohesive/XFEM, crack growth, and life models remain |
| Composite materials | ⬜ | ⬜ | ⬜ | ⬜ | Laminates, anisotropy, failure criteria, and progressive damage remain |
| Rigid-body and multibody dynamics | ⬜ | ⬜ | ⬜ | ⬜ | Joints, constraints, contacts, flexible bodies, and initialization remain |
| Incompressible laminar flow | ✅ | ✅ | ⬜ | ⬜ | Bounded coherent-SI 2D Newtonian paths cover steady all-velocity/mixed-pressure MINI execution, collocated cell-centered FVM transient Navier--Stokes, fixed-topology ALE FSI, and one conservative remesh handoff. The FVM path verifies transient-consistent pressure--velocity coupling, checkerboard action, block-independent mass/momentum acceptance, oriented replay, and affine boundary-pressure balance. Immutable 3D releases lower the same conservative meaning, and one serial-host tetrahedral ALE FSI slice executes it. General standalone 3D CFD, wider boundaries, dynamic cross-method convergence, turbulence, compressibility, production preconditioning, distributed/device transient execution, and general time methods remain; [`fluid.cell-centered-navier-stokes-fvm-2d`](../../verify/fluid/cell-centered-navier-stokes-fvm-2d/README.md), [`fsi.fixed-topology-ale-monolithic-3d`](../../verify/fsi/fixed-topology-ale-monolithic-3d/README.md), [RFC 0072](../../rfcs/0072-collocated-incompressible-finite-volume.md) |
| Compressible flow | ⬜ | ⬜ | ⬜ | ⬜ | Conservative laws, Riemann fluxes, shocks, and thermodynamics remain |
| Turbulence modeling | ⬜ | ⬜ | ⬜ | ⬜ | RANS, LES, wall treatment, transition, and verification hierarchy remain |
| Multiphase flow | ⬜ | ⬜ | ⬜ | ⬜ | Free surface, phase field, VOF, Euler-Euler, cavitation, and surface tension remain |
| Porous-media flow | ⬜ | ⬜ | ⬜ | ⬜ | Darcy/non-Darcy, multiphase, transport, and constitutive laws remain |
| Passive scalar transport | ✅ | ✅ | ⬜ | ⬜ | One generic scalar Field with coherent-SI temperature dimension executes a zero-source fixed-Cartesian 2D advection--diffusion Relation through upwind/orthogonal-TPFA/backward Euler. This is not yet a species, thermal-component, reaction, or source-library claim; [`fluid.cartesian-advection-diffusion-fvm-2d`](../../verify/fluid/cartesian-advection-diffusion-fvm-2d/README.md), [RFC 0069](../../rfcs/0069-conservative-cell-centered-transport.md) |
| Species transport and reactions | ⬜ | ⬜ | ⬜ | ⬜ | Advection-diffusion-reaction, chemistry, and stiffness handling remain |
| Combustion | ⬜ | ⬜ | ⬜ | ⬜ | Kinetics, reduced mechanisms, turbulence interaction, and radiation remain |
| Fluid-structure interaction | ✅ | ✅ | ⬜ | ⬜ | Bounded conforming 2D fixed-reference FSI reuses one finalized operator across CPU, one-host MPI, one-device CUDA, and host-staged MPI+CUDA. Serial-host 2D and tetrahedral 3D moving slices reuse the ordinary fluid/solid/conserving-interface meaning with harmonic P1 mesh motion, affine GCL correction, and first-order fixed-topology ALE stepping; the 3D slice adds direct/exact-package authoring, claim-local complete Jacobian-column verification, cubic path gates, three-state lineage, and temporal refinement. Ordinary accepted steps retain occurrence-local residual, constraint, interface, GCL, geometry, and lineage checks but no implementation-equivalence audit fields. A separate 2D transition proves conservative remeshing handoff. Partitioned coupling, production preconditioning/remeshing, 3D remeshing, distributed/device ALE, nonlinear structure, and sensitivities remain; [`fsi.fixed-reference-monolithic-step-2d`](../../verify/fsi/fixed-reference-monolithic-step-2d/README.md), [`fsi.fixed-topology-ale-monolithic-2d`](../../verify/fsi/fixed-topology-ale-monolithic-2d/README.md), [`fsi.fixed-topology-ale-monolithic-3d`](../../verify/fsi/fixed-topology-ale-monolithic-3d/README.md), [`fsi.remeshing-transfer-2d`](../../verify/fsi/remeshing-transfer-2d/README.md), [RFC 0070](../../rfcs/0070-dimension-parametric-tetrahedral-ale-fsi.md) |
| Conjugate heat transfer | ⬜ | ⬜ | ⬜ | ⬜ | Solid-fluid interface semantics and coupled execution remain |
| Electrostatics | ⬜ | ⬜ | ⬜ | ⬜ | Scalar foundation is related but no physics-qualified library exists |
| Magnetostatics | ⬜ | ⬜ | ⬜ | ⬜ | H(curl), gauges, nonlinear materials, and force extraction remain |
| Low-frequency electromagnetics | ⬜ | ⬜ | ⬜ | ⬜ | Eddy currents, coils, circuits, and motion remain |
| Full-wave electromagnetics | ⬜ | ⬜ | ⬜ | ⬜ | Frequency/time domain, ports, radiation, and absorbing boundaries remain |
| Static scalar circuit components | ✅ | ✅ | ⬜ | ⬜ | `Eqiora.Electrical.Basic@0.1.0` packages Pin, IdealVoltageSource, Resistor, and Ground through the ordinary exact resolver; nonlinear/transient devices and a broad catalog remain; [`packages.offline-model-package`](../../verify/packages/offline-model-package/README.md) |
| Electrical machines and drives | 🟨 | 🟨 | ⬜ | ⬜ | One exact-packaged scalar ideal linear DC motor, viscous lumped load, and proportional controller execute in the host-serial reference path; switching, saturation, field models, broad drive libraries, and production execution remain; [`hybrid.packaged-dc-motor-controller`](../../verify/hybrid/packaged-dc-motor-controller/README.md) |
| Acoustics | ⬜ | ⬜ | ⬜ | ⬜ | Pressure, thermoviscous, structural-acoustic, and radiation models remain |
| Piezoelectricity | ⬜ | ⬜ | ⬜ | ⬜ | Electromechanical constitutive coupling remains |
| Geomechanics | ⬜ | ⬜ | ⬜ | ⬜ | Soil/rock models, consolidation, seepage, and failure remain |
| Lubrication and tribology | ⬜ | ⬜ | ⬜ | ⬜ | Reynolds equation, contact, thermal, and wear coupling remain |
| Particle tracing | ⬜ | ⬜ | ⬜ | ⬜ | Lagrangian particles, forces, collisions, and two-way coupling remain |
| Discrete element method | ⬜ | ⬜ | ⬜ | ⬜ | Contact detection, particle laws, and parallel execution remain |
| Quantum/wave equations | 🟨 | 🟨 | ⬜ | ⬜ | Finite-basis stationary Hermitian operators use the shared typed matrix-pencil path, with independent small-matrix quantum and structural consumers. Spatial wave equations, time-dependent complex states and complete physics packages remain unsupported. |
| Reduced-order physical components | ⬜ | ⬜ | ⬜ | ⬜ | Data-driven and projection-based components with validity domains remain |
| Material database | ⬜ | ⬜ | ⬜ | ⬜ | Units, temperature/rate dependence, uncertainty, citations, and versioning remain |
| Validated component library | ⬜ | ⬜ | ⬜ | ⬜ | Thermal, mechanical, fluid, electrical, and control libraries require independent V&V |
| Explicit structural dynamics | ⬜ | ⬜ | ⬜ | ⬜ | Central difference, contact, material failure, mass scaling, and energy balance remain |
| Rotordynamics | ⬜ | ⬜ | ⬜ | ⬜ | Gyroscopic systems, bearings, critical speeds, Campbell diagrams, and instability remain |
| Creep and stress relaxation | ⬜ | ⬜ | ⬜ | ⬜ | Time-temperature material laws, consistent updates, and life assessment remain |
| Electrochemistry | ⬜ | ⬜ | ⬜ | ⬜ | Species, charge, reactions, interfaces, thermodynamics, and stiff coupling remain |
| Battery cell and pack models | ⬜ | ⬜ | ⬜ | ⬜ | Electrochemical/thermal/aging cell models, circuits, pack cooling, and controls remain |
| Semiconductor devices | ⬜ | ⬜ | ⬜ | ⬜ | Drift-diffusion, Poisson coupling, quantum corrections, contacts, and circuits remain |
| Plasma physics | ⬜ | ⬜ | ⬜ | ⬜ | Charged species, fields, collisions, kinetics, and multiscale stiffness remain |
| Geometrical optics and ray tracing | ⬜ | ⬜ | ⬜ | ⬜ | Sources, materials, surfaces, polarization, scattering, and detector models remain |
| Wave optics | ⬜ | ⬜ | ⬜ | ⬜ | Complex fields, propagation, diffraction, nonlinear media, and coupling remain |
| Radiative transport | ⬜ | ⬜ | ⬜ | ⬜ | Angular discretization, scattering, participating media, and acceleration remain |
| Microfluidics | ⬜ | ⬜ | ⬜ | ⬜ | Rarefaction, electrokinetics, surface effects, and particle/cell coupling remain |
| Additive-manufacturing process | ⬜ | ⬜ | ⬜ | ⬜ | Moving heat source, material activation, phase change, stress, and path provenance remain |
| Welding and heat-treatment process | ⬜ | ⬜ | ⬜ | ⬜ | Moving sources, metallurgy, distortion, residual stress, and calibration remain |
| Composite curing | ⬜ | ⬜ | ⬜ | ⬜ | Cure kinetics, heat, shrinkage, tooling contact, and process optimization remain |
| 0D–1D–3D model coupling | ⬜ | ⬜ | ⬜ | ⬜ | Typed dimensional interfaces, conservative transfer, time negotiation, and evidence remain |
| Submodeling and global-local analysis | ⬜ | ⬜ | ⬜ | ⬜ | Boundary transfer, error control, iteration, and provenance remain |
| Homogenization and representative volume elements | ⬜ | ⬜ | ⬜ | ⬜ | Boundary ensembles, effective tensors, uncertainty, and scale linkage remain |
| Concurrent multiscale coupling | ⬜ | ⬜ | ⬜ | ⬜ | Scale-separated domains, handshake regions, stability, and parallel execution remain |
| Atomistic–continuum coupling | ⬜ | ⬜ | ⬜ | ⬜ | Particle/field identity, interface forces, time scales, and conservation remain |
| Pore–continuum coupling | ⬜ | ⬜ | ⬜ | ⬜ | Upscaling/downscaling, interface fluxes, moving topology, and uncertainty remain |
| Monolithic multiphysics coupling | 🟨 | 🟨 | ⬜ | ⬜ | One bounded 23-by-23 reference residual jointly solves scalar electrical, rotational, signal, differential, and algebraic coordinates; general block assembly/solvers and libraries remain; [`hybrid.packaged-dc-motor-controller`](../../verify/hybrid/packaged-dc-motor-controller/README.md) |
| Partitioned multiphysics coupling | 🟨 | ⬜ | ⬜ | ⬜ | Coupling identity exists; transfer, fixed-point/Quasi-Newton, rollback, and evidence remain |
| Coupling stability and conservation diagnostics | 🟨 | 🟨 | ⬜ | ⬜ | One dimensioned in-memory `PhysicalSample` path reaccepts component/junction residuals and checks electromechanical power, stored-energy change, physical losses, and separately named backward-Euler numerical dissipation; general interface stability, iteration, and time-lag diagnostics remain; [`hybrid.packaged-dc-motor-controller`](../../verify/hybrid/packaged-dc-motor-controller/README.md) |

