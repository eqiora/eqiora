# Eqiora.Fluid.Incompressible

`SteadyStokesWithPotential2d` and `ConservativeNavierStokesWithPotential3d`
are incompressible Newtonian flow laws: steady in 2D, and transient in
conservative form in 3D. The enclosing Model supplies the velocity, pressure,
and force-potential Fields.

`NewtonianMechanicalInterface2d` and `NewtonianMechanicalInterface3d` expose
the velocity trace and the Cauchy traction (with respect to the outward
normal) on the whole exterior boundary as
`Eqiora.Mechanics.Interfaces::VelocityTractionBoundary` Ports.

The components contain only equations. The enclosing Model supplies domains,
Fields, and boundary data, and the Realization chooses the mesh,
discretization, and solver.
