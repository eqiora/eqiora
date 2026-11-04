# Eqiora.Fluid

The development release `Eqiora.Fluid@0.5.0` owns fixed-unit incompressible
fluid equations. `SteadyStokesBalance2d` is the volume law; `SteadyStokes2d`
composes it with `VelocityTractionInterface2d`. Each has a separate `Forced`
variant requiring a physical force-potential Field. Unforced components require
no zero-potential Field.

`InertialStokes2d` adds inertia without velocity advection.
`ConservativeNavierStokes3d` includes conservative advection and inertia. Their
`Forced` variants preserve the force-potential contract. The 2d/3d suffixes
state the supported spatial dimensions; they are not inferred.

`VelocityTractionInterface2d` and `VelocityTractionInterface3d` use the shared
`Eqiora.Mechanics.Interfaces@0.3.0` boundary connector. No-slip, traction-free,
normal-pressure, inward-normal velocity and prescribed velocity/traction
boundaries remain separate contracts. Prescribed data are typed expressions
on the boundary, including traces of body Fields. Normal velocity is positive
inward; traction and pressure use the outward normal. Numerical owners may
reject boundary data outside their supported profiles.
