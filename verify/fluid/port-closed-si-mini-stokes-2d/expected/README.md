# Expected evidence

The direct and exact-package Models have distinct semantic identities but the
same equation roles and physical data. For each Cartesian `(axis, side)` role,
the lowerer must retain the exact Boundary identity and report `TraceZero`.

The admitted package chain is

```text
verification root
  -> Eqiora.Fluid@0.5.0
       -> Eqiora.Mechanics.Interfaces@0.3.0
  -> Eqiora.Mechanics.Interfaces@0.3.0
```

The root's direct mechanics dependency supplies the explicit zero terminals;
the fluid dependency supplies the same exact nominal Connector transitively.
Resolution must converge on one exact neutral release identity.

Using the RFC 0045 physical mesh and scale profile, both authoring forms must
produce the same dimensionless sparse system. The accepted physical solution
is

```text
u = (0, 0) m/s,
p = 0.75 Pa/m * (x - 2 m),
integral p = 0,
gauge multiplier = 0,
integrated grad(q) = (6, 0) N/m,
essential reaction = (-6, 0) N/m.
```

The test does not re-claim MINI convergence or general congruence scaling;
those remain owned by the registered RFC 0043 and RFC 0045 cases. It proves
that a public physical boundary package reaches that unchanged path without
package dispatch.
