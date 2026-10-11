# Eqiora.Mechanics.BoundaryLoads

Provides `NormalPressureTraction2d`, a normal-pressure load for the
velocity/traction connector in `Eqiora.Mechanics.Interfaces`. It takes a
pressure Field defined by the enclosing Model and contributes only

```text
terminal flux - parent-outward normal(isotropic pressure) = 0.
```

The connection reverses the sign of the terminal flux at the connected port,
so a positive pressure pushes inward on the body. Because the pressure is a
separate Field, the enclosing Model controls its definition and parameters.

The package contains no numerical treatment: quadrature, trace space,
pressure gauge, mesh, solver, and coupling are chosen elsewhere. Execution has
so far been verified only for a spatially constant pressure, in the
mixed-boundary MINI Stokes case.
