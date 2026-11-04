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

The imported reference example uses concrete Cartesian domains. Components can
also form a complete exterior from their formal boundary arguments. The shared
Geometry owner checks the bound selections for complete, disjoint coverage,
including grouped walls and circular boundaries in the cylinder example.
Same-support boundary connections preserve the exact Geometry and parent;
normal-dependent data still require a normal supported by the numerical method.
This release does not infer ambient dimension.
