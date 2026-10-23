# Canonical Relation to first-order time execution

This case starts from a validated Semantic Kernel model, lowers its residual
Relation to scalar SSA Operator IR, structurally proves the derivative
Jacobian, and executes the resulting backend-neutral `TimeProblem` through the
optional Diffsol adapter.

The explicit fixture proves a full monomial derivative Jacobian and is
normalized to `y_dot = f`. The index-one fixture proves a constant monomial
Jacobian with an exact algebraic zero row, remains a rank-deficient mass
matrix, and enters BDF with the complete pair accepted by the common
initializer. The adapter reports `Provided` and preserves both State and rate.

Two further fixtures prove complete dense constant matrices. One is full rank;
the other is singular despite having no literal zero row. Eqiora interprets
the final `f64` coefficients as exact binary rationals and recomputes rank with
arbitrary-precision elimination, so neither fixture depends on a numerical
rank tolerance. Both execute through BDF and are checked against analytic
trajectories. Their canonical Parameter symbols are also ordered by the same
SSA lowering and differentiated through `f_p dp`; the constant derivative
proof supplies `M_p = 0`, so full and singular mass sensitivities are checked
analytically without a second model representation.

The residual roots intentionally differ from state order and use signs and
scales other than the identity. The lowering must recover the same explicit
ODE without evaluating the model at sample points. A second canonical model
uses a state-dependent derivative coefficient and must fail closed before it
can enter the admitted first-order seam. The independent
[`time.general-implicit-dae`](../general-implicit-dae/README.md) case proves
that this valid Relation enters the distinct residual-native seam instead.

Every admitted fixture also round-trips a content-addressed lowering witness.
The witness links model digest/revision, Relation, state order, equation class,
the complete derivative matrix, and exact rank, then independently rechecks
the coefficients and replays rank against Operator IR. Forged coefficients,
forged ranks, and over-limit exact-rank replay fail closed. A separate time-run
manifest links that witness to the exact `TimePlan`, adapter/version, accepted
report, and output digests. The first-order v1 manifest rejects the
reference-only `ImplicitEuler` method until a residual-native provenance
artifact exists.

Run only this case with:

```console
cargo run -p eqiora-verify -- run --case time.canonical-first-order
```
