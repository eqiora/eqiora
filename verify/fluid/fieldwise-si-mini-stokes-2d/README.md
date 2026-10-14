# Field-wise coherent-SI MINI Stokes bridge

This case joins two previously separate claims without adding a fluid-specific
shortcut. Both a direct-flat Model and an ordinary exact-package compilation
through the current Model owner lower to
the same package-neutral two-dimensional Stokes roles. Those exact Domain and
Field identities then select a field-wise Realization: `(P1+bubble)^2` for
velocity, P1 for pressure, retained canonical data for the force potential,
and one Realization-owned zero-pressure-integral constraint.

The physical fixture is

```text
Omega = (0 m, 4 m) x (0 m, 2 m),
mu    = 6 Pa s,
q     = 3 Pa (x / (4 m) - 1/2),
u     = 0,
p     = q.
```

Thus `grad(q) = (0.75 Pa/m, 0)`, the pressure integral is zero, the
body-force resultant is `(6 N/m, 0)`, and the complete-boundary reaction is
`(-6 N/m, 0)`, all per unit out-of-plane thickness. The affine pressure is
represented exactly by P1, so this bridge verifies semantic identity, scaling,
artifact lineage, reconstruction, gauge, and equilibrium rather than claiming
a second MINI convergence study.

## Two numerical coordinate profiles

The same physical Model is realized twice:

| Profile | `L` | `U` | `P` | derived `G = U/L` | derived `Theta = P U L` |
| --- | ---: | ---: | ---: | ---: | ---: |
| A | 4 m | 0.5 m/s | 0.75 Pa | 0.125 1/s | 1.5 W/m |
| B | 4 m | 1 m/s | 1.5 Pa | 0.25 1/s | 6 W/m |

Both have dimensionless viscosity `mu U/(P L) = 1`. Their finalized matrices
are therefore identical, while their dimensionless body forces are `(4, 0)`
and `(2, 0)` and their right-hand sides differ. Their Realization digests must
also differ. Reconstruction must nevertheless recover equivalent physical
velocity, pressure, gauge multiplier, body-force resultant, and reaction.
Neither scale profile changes Model bytes or physical geometry.

## Ordinary execution and durable identity

The physical rectangle is divided into `4 x 2` Cartesian cells and then into
16 connected affine triangles. The imported mesh envelope, its digest,
intrinsic dimension, boundary closure, and its exact binding in current Realization
are validated before assembly. Both fixtures follow the same ordinary path:

```text
source / exact package
  -> current canonical Model
  -> package-neutral Stokes lowering
  -> exact Field-wise current Realization
  -> direct dimensionless MINI assembly
  -> reproducible reference MINRES
  -> coherent-SI reconstruction
  -> linked Run-manifest v2 identity
```

current Realization must decode and re-encode byte-for-byte and reproduce its
domain-separated digest. Run-manifest v2 retains its existing identity and
provenance meaning; its presence is not execution attestation.

After the current Realization resolver succeeds, the same exact Domain, velocity
and pressure Fields, spaces, gauge constraint, symmetric-congruence scales,
solver, and host requirement project into one connected typed portable
Realization DAG. The finalizer consumes that graph; projection leaves the
typed authoring value untouched. Equation-aware negative probes author typed
plans directly; current mixed wire tests cover decode closure without reconstructing
a request from obsolete JSON. The separate
`artifacts.realization-run-wire` case owns the committed current canonical-byte
golden fixture.

Production never materializes a mixed-unit matrix. A bounded verification-only
oracle, independent of the SI scaling adapter but intentionally reusing the
already verified MINI local assembly, checks every coefficient and RHS row
across the complete reduced velocity, pressure, and gauge blocks:

```text
A_hat z = D A(D z) / Theta,
b_hat   = D b / Theta.
```

This is stronger than one sampled probe and detects one-sided scaling,
incorrect block order, a forged gauge scale, or inverse reconstruction.
Acceptance also requires exact finalized CSR
symmetry, an independently recomputed dimensionless true residual, zero
physical velocity, the affine physical pressure, zero pressure integral and
gauge multiplier, and componentwise reaction-plus-body-force balance.
Direct and package paths independently pass the ordinary canonical role
lowerer and must agree in dimensionless algebra and all reported physical
evidence. The separate registered RFC 0044 case owns byte-level
verification-private identity normalization; this case does not duplicate it.

Run the registered evidence with:

```sh
cargo test --locked -p eqiora --test fieldwise_si_mini_stokes_2d
cargo run --locked -p eqiora-verify -- run --case fluid.fieldwise-si-mini-stokes-2d
```

## Bounded claim

This verifies one connected 2D Cartesian rectangle, affine triangles, one
exact conservative affine force potential, complete homogeneous velocity
trace, MINI/P1 with one global pressure constraint, and the reproducible
identity-preconditioned reference MINRES backend. It does not claim natural,
open, slip, or live-Port boundaries; nonconservative vector forcing; other
stable pairs or stabilization; disconnected domains; automatic scale choice;
transient or Navier--Stokes flow; production preconditioning; parallel solve,
MPI, CUDA, fluid-structure interaction, a general result artifact, or
execution attestation.
