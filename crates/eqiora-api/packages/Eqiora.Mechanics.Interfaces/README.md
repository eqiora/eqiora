# Eqiora.Mechanics.Interfaces

`VelocityTractionBoundary` is a boundary connector that pairs velocity with
traction (taken with respect to the outward normal), so their product is
power. Quasistatic solids use a different connector that pairs displacement
with traction.

`ZeroVelocity2d`, `ZeroTraction2d`, `ZeroVelocity3d`, and `ZeroTraction3d`
set the velocity or traction to zero on the boundary they are bound to. The
enclosing Model supplies the body and boundary, and the Realization chooses
the numerical treatment.
