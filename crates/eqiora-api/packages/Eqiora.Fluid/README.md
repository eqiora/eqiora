# Eqiora.Fluid

The development release `Eqiora.Fluid@0.5.0` owns fixed-unit incompressible
fluid equations. `SteadyStokesBalance2d` is the volume law; `SteadyStokes2d`
composes it with `VelocityTractionInterface2d`. Each has a separate `Forced`
variant requiring a physical force-potential Field. Unforced components require
no zero-potential Field.

`ForcedInertialStokes2d` adds inertia without velocity advection.
`ForcedConservativeNavierStokes3d` includes conservative advection and inertia.
These existing dynamic contracts require a physical force-potential Field. The 2d/3d suffixes
state the supported spatial dimensions; they are not inferred.

`VelocityTractionInterface2d` and `VelocityTractionInterface3d` use the shared
`Eqiora.Mechanics.Interfaces@0.3.0` boundary connector. No-slip, traction-free,
normal-pressure, inward-normal velocity and prescribed velocity/traction
boundaries remain separate contracts. Normal velocity and prescribed vector
data are typed expressions on the boundary, including traces of body Fields.
The normal-pressure outlet retains its physical pressure Field contract. Normal velocity is positive
inward; traction and pressure use the outward normal. Numerical owners may
reject boundary data outside their supported profiles.

The imported reference example uses concrete Cartesian domains. A complete
exterior cannot yet be constructed from a Component's formal boundary arguments;
Curved Geometry-bound complete-exterior composition also remains unsupported. The
ordinary cylinder example keeps its handwritten form until that general support
composition is available. This release does not infer ambient dimension.
