# Eqiora.Fluid.InertialStokes

`Eqiora.Fluid.InertialStokes@0.1.0` provides one two-dimensional
incompressible Newtonian volume law with fluid inertia and a conservative
load potential. `InertialStokesWithPotential2d` declares

```text
density * derivative(velocity)
  - div(2 * dynamic_viscosity * symmetric_part(grad(velocity))
        - isotropic_lift(pressure))
  - grad(force_potential) = 0

div(velocity) = 0.
```

The package contains only this volume law. Boundary conditions come from
`Eqiora.Fluid.Incompressible` and `Eqiora.Mechanics.Interfaces`; the mesh,
elements, time method, initial state, pressure reference, solver, and any FSI
coupling are chosen elsewhere. The law has no Navier–Stokes advection term.
