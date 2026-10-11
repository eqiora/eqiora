# Eqiora.Solid.LinearElasticity

`IsotropicBalanceWithPotential2d` is the 2D small-strain isotropic balance law,
with Lamé parameters and a conservative load potential.
`IsotropicMechanicalInterface2d` exposes the displacement trace and the
traction (with respect to the outward normal) as `QuasistaticMechanicalBoundary`
Ports. `FixedDisplacement2d` and `ZeroTraction2d` are homogeneous boundary
conditions.

`IsotropicElastodynamicsWithPotential2d` and
`IsotropicElastodynamicsWithPotential3d` provide first-order dynamics:

```text
derivative(displacement) - velocity = 0
density * derivative(velocity) - div(stress(displacement))
  - grad(load_potential) = 0
```

The matching `ElastodynamicMechanicalInterface2d` and
`ElastodynamicMechanicalInterface3d` components expose the velocity trace and
elastic traction through `Eqiora.Mechanics.Interfaces::VelocityTractionBoundary`,
which pairs velocity with traction rather than displacement with traction as
in the quasistatic case.

Each Field is a spatial vector on its body. The enclosing Model and
Realization supply the initial fields, discretization, solver, and execution
target.
