# Eqiora.Fluid

`SteadyStokes2d` is a ready-made two-dimensional steady incompressible
Newtonian model. It is built from the public `SteadyNewtonianBalance2d` and
`VelocityTractionInterface2d` components, so a project can use the whole model
or replace either part.

The same package provides zero, normal, and field-driven boundary conditions.
`PrescribedVelocity2d` and `PrescribedTraction2d` take vector Fields defined on
the fluid body and prescribe their traces as the boundary velocity or as the
traction with respect to the outward normal.
