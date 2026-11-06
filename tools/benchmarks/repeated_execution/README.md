# Repeated-Run comparability

The [frozen a7 cylinder target](https://github.com/eqiora/eqiora/issues/802)
remains unresolved. Timing the same inputs is insufficient when the weak
boundary condition has changed.

## Mathematical boundary

The a7 skew volume convection omitted the natural outlet term included by the
[authored transport correction](https://github.com/eqiora/eqiora/issues/1251):

```text
B(u, v) = (rho / 2) integral_outlet (u . n) (u . v) ds
```

Here `n` is the outward normal and `v` is the velocity test function. The authored
outlet prescribes zero **physical stress**. Omitting `B` therefore changes its
natural weak condition; this is not a pressure-gauge difference or a term that
weak incompressibility cancels.

The [physical-stress half-flux test](../../../crates/eqiora-numerics/src/form_compiler/region/tests/nonlinear.rs#L419)
and [constant-throughflow test](../../../crates/eqiora-numerics/src/simplicial_navier_stokes/verification.rs#L170)
verify that correction in their stated settings. They do not establish accuracy
of the fixed ten-step cylinder trajectory. Steady Stokes and zero-velocity
oracles do not bound that nonlinear trajectory either.

## What the measurements establish

The [driver](run.py) freezes the mesh and physical initial data, retains controls
and results, and times the ordinary Run. It has **no cross-version cylinder
field acceptance gate**. Its decay workload separately checks the independent
`exp(-t)` bound. Neither successful per-version acceptance nor the
[reported field differences](../../../docs/performance/2026-10-24-repeated-runs.md)
defines an allowable cross-version tolerance.

With matched builds, environments and measurement controls, comparisons using
the same current semantics can support attribution of performance changes.
They cannot fulfill the frozen a7 target of a twofold improvement with unchanged
mathematics. Do not remove the authored outlet term or relax controls to obtain that
claim.

## Minimum evidence still needed

Compare all ten outputs offline: vertex velocities, MINI bubble coefficients,
absolute pressures and the complete Run controls. Independently scale and
evaluate the omitted outlet defect against the unchanged method-owned residual
allowances. Keep this separate from timing and from each version's own accepted
residual. These comparison and defect checks are not implemented by this harness.
They are necessary falsification checks, not a sufficient equivalence contract:
passing them would not resolve the a7 target. No perturbation bound or new
tolerance is asserted here.
