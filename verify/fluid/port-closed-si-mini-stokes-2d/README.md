# Port-closed coherent-SI MINI Stokes

This case verifies RFC 0046's first executable fluid boundary. The exact
`Eqiora.Mechanics.Interfaces@0.3.0` package owns one nominal velocity/traction
Connector and zero terminals. The exact
`Eqiora.Fluid@0.5.0` package depends on it and binds the complete
exterior velocity trace and parent-outward Newtonian traction to those Ports.

The direct fixture writes four `trace(velocity) = 0` Relations. The packaged
fixture connects four generated fluid Ports to `ZeroVelocity2d` terminals.
After ordinary package elaboration, the same name-independent Stokes lowerer
must report `TraceZero` on the corresponding axis/side roles. Both Models then
use the existing exact Field-wise v2 plan, coherent-SI normalization, MINI
assembly, reference MINRES, and physical reconstruction. The test requires
equal canonical CSR/RHS data and equal physical velocity, pressure, gauge,
force, and reaction evidence.

The root dynamic-viscosity Parameter is forwarded unchanged into both the
volume and boundary Component slots; elaboration does not fabricate
occurrence-local Parameters. Direct and exact-package coefficient expressions
agree in primal evaluation, forward JVP, and evaluated reverse-mode VJP. A
boundary law bound to a separately declared Parameter is rejected even when
its current value is equal to the volume coefficient. [RFC
0055](../../../rfcs/0055-component-parameter-terms.md) records this identity
rule.

Near-miss variants also replace the shared viscosity direction with an
equal-valued independent Parameter, one terminal with `ZeroTraction2d`, or a
compatible unresolved Port. Canonical lowering rejects the independent
coefficient and must retain `FluxZero` or `PortBinding` for the boundary
variants;
RFC 0047 makes `FluxZero` the exact zero normal-pressure case and therefore
selects an empty pressure-constraint list. The live Port must still fail before
inspecting a supplied invalid mesh. This ordering prevents unresolved coupling
from being silently interpreted as either zero velocity or zero traction.

Run:

```bash
cargo test --locked -p eqiora --test port_closed_si_mini_stokes_2d
cargo run --locked -p eqiora-verify -- run --case fluid.port-closed-si-mini-stokes-2d
```

This evidence does not claim a general natural/open-boundary solve, live Port
or trace transfer, nonzero boundary data, fluid-fluid coupling, a solid velocity law,
displacement-to-velocity conversion, structural dynamics, ALE, or FSI.

The current consolidated Fluid release also has an unforced imported witness:
`u=(U,0)` and `p=0` on the same rectangular mesh, with inward normal speed `U`
on the left, inward speed `-U` on the right, and zero traction above and below.
Zero strain and divergence satisfy the volume law, and the two normal signs
prescribe the same rightward translation. Two independently specified speeds
check that the original caller Parameter controls every velocity vertex while
no force-potential Field is introduced. Pressure and MINI bubble coefficients
are zero. This uses the existing `2e-10` physical reconstruction tolerance;
it does not claim generic externally bound Geometry exterior composition.
