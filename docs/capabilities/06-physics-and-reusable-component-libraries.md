# Physics and reusable component libraries

[Capability matrix](../capability-matrix.md) · [legend](../capability-matrix.md#how-to-read-the-matrix)

| Capability | C | X | V | M | Current implementation and planned extensions |
|---|:---:|:---:|:---:|:---:|---|
| Scalar Poisson/diffusion | ✅ | ✅ | ✅ | ⬜ | Manufactured 1D–3D Cartesian FEM/FVM and one fixed-simplex path |
| Transient heat conduction | 🟨 | 🟨 | 🟨 | ⬜ | 1D diffusion and time foundations exist; canonical multidimensional path remains |
| Convection and radiation heat transfer | ⬜ | ⬜ | ⬜ | ⬜ | Boundary laws, view factors, participating media, and coupling remain |
| Thermal contact | ⬜ | ⬜ | ⬜ | ⬜ | Contact resistance, gap conductance, and nonlinear coupling remain |
| Linear elasticity | ✅ | ✅ | ⬜ | ⬜ | One isotropic elasticity model serves static Q1/P1, dynamic P1 and FSI in 2D (per unit thickness) and 3D. The 2D model runs standalone on Cartesian Q1 with patch, balance and convergence checks; 3D solids run inside the tetrahedral moving-mesh FSI case. Not yet: plane stress/strain choice, beams and shells, standalone 3D analysis, more elements and materials; [`solid.mixed-boundary-elasticity-2d`](../../verify/solid/mixed-boundary-elasticity-2d/README.md), [`fsi.fixed-topology-ale-monolithic-3d`](../../verify/fsi/fixed-topology-ale-monolithic-3d/README.md) |
| Nonlinear solid mechanics | ⬜ | ⬜ | ⬜ | ⬜ | Finite strain, objective rates, tangents, and robust increments remain |
| Structural dynamics | ✅ | ✅ | ⬜ | ⬜ | First-order 2D and 3D solid dynamics (displacement, velocity, density, momentum) run as the solid side of fixed-reference 2D FSI and multi-step 3D moving-mesh FSI. Not yet: standalone time stepping, restart, damping, modal and harmonic analysis, finite strain, contact; [`solid.dynamic-linear-solid-semantics-2d`](../../verify/solid/dynamic-linear-solid-semantics-2d/README.md), [`fsi.fixed-topology-ale-monolithic-3d`](../../verify/fsi/fixed-topology-ale-monolithic-3d/README.md) |
| Beams, shells, membranes, and trusses | ⬜ | ⬜ | ⬜ | ⬜ | Element families, locking control, sections, and orientations remain |
| Plasticity and viscoplasticity | ⬜ | ⬜ | ⬜ | ⬜ | Constitutive library, return mapping, and verification remain |
| Hyperelasticity and viscoelasticity | ⬜ | ⬜ | ⬜ | ⬜ | Material models and incompressibility treatments remain |
| Damage, fracture, and fatigue | ⬜ | ⬜ | ⬜ | ⬜ | Phase field/cohesive/XFEM, crack growth, and life models remain |
| Composite materials | ⬜ | ⬜ | ⬜ | ⬜ | Laminates, anisotropy, failure criteria, and progressive damage remain |
| Rigid-body and multibody dynamics | ⬜ | ⬜ | ⬜ | ⬜ | Joints, constraints, contacts, flexible bodies, and initialization remain |
| Incompressible laminar flow | ✅ | ✅ | ⬜ | ⬜ | 2D Newtonian flow runs as steady MINI Stokes, transient cell-centred FVM Navier–Stokes, moving-mesh FSI and one remesh; 3D flow runs inside the tetrahedral FSI case. Not yet: standalone 3D CFD, turbulence, compressible flow, production preconditioners, distributed or GPU transient runs; [`fluid.cell-centered-navier-stokes-fvm-2d`](../../verify/fluid/cell-centered-navier-stokes-fvm-2d/README.md), [`fsi.fixed-topology-ale-monolithic-3d`](../../verify/fsi/fixed-topology-ale-monolithic-3d/README.md), [RFC 0072](../../rfcs/0072-collocated-incompressible-finite-volume.md) |
| Compressible flow | ⬜ | ⬜ | ⬜ | ⬜ | Conservative laws, Riemann fluxes, shocks, and thermodynamics remain |
| Turbulence modeling | ⬜ | ⬜ | ⬜ | ⬜ | RANS, LES, wall treatment, transition, and verification hierarchy remain |
| Multiphase flow | ⬜ | ⬜ | ⬜ | ⬜ | Free surface, phase field, VOF, Euler-Euler, cavitation, and surface tension remain |
| Porous-media flow | ⬜ | ⬜ | ⬜ | ⬜ | Darcy/non-Darcy, multiphase, transport, and constitutive laws remain |
| Passive scalar transport | ✅ | ✅ | ⬜ | ⬜ | One generic scalar Field with coherent-SI temperature dimension executes a zero-source fixed-Cartesian 2D advection--diffusion Relation through upwind/orthogonal-TPFA/backward Euler. This is not yet a species, thermal-component, reaction, or source-library claim; [`fluid.cartesian-advection-diffusion-fvm-2d`](../../verify/fluid/cartesian-advection-diffusion-fvm-2d/README.md), [RFC 0069](../../rfcs/0069-conservative-cell-centered-transport.md) |
| Species transport and reactions | ⬜ | ⬜ | ⬜ | ⬜ | Advection-diffusion-reaction, chemistry, and stiffness handling remain |
| Combustion | ⬜ | ⬜ | ⬜ | ⬜ | Kinetics, reduced mechanisms, turbulence interaction, and radiation remain |
| Fluid-structure interaction | ✅ | ✅ | ⬜ | ⬜ | 2D fixed-reference FSI runs on CPU, MPI, CUDA and MPI+CUDA; 2D and 3D moving-mesh FSI run serially, and one 2D case transfers fields across a remesh. Not yet: partitioned coupling, 3D remeshing, distributed or GPU moving-mesh FSI, nonlinear solids, sensitivities; [`fsi.fixed-reference-monolithic-step-2d`](../../verify/fsi/fixed-reference-monolithic-step-2d/README.md), [`fsi.fixed-topology-ale-monolithic-2d`](../../verify/fsi/fixed-topology-ale-monolithic-2d/README.md), [`fsi.fixed-topology-ale-monolithic-3d`](../../verify/fsi/fixed-topology-ale-monolithic-3d/README.md), [`fsi.remeshing-transfer-2d`](../../verify/fsi/remeshing-transfer-2d/README.md), [RFC 0070](../../rfcs/0070-dimension-parametric-tetrahedral-ale-fsi.md) |
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

