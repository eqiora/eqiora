# Adaptive ODE and mass-matrix DAE

This case exercises one backend-neutral `TimeProblem` and `TimePlan` through
the optional Diffsol adapter. It compares requested dense samples against:

- a smooth scalar exponential-decay solution under Tsitouras 5(4);
- a stiff analytic tracking solution under variable-order BDF;
- an index-one mass-matrix DAE with `M = diag(1, 0)` under BDF;
- analytic parameter sensitivity of `y' = -k y` under both methods;
- analytic BDF sensitivities of Parameter-independent full and rank-deficient
  mass-matrix systems;
- a localized zero crossing, externally committed reset, and explicit restart.

The DAE receives the common owner's accepted initial pair `(x,z)=(0,1)`
and `(x_dot,z_dot)=(1,0)`; the unused algebraic adapter rate is zero.
Diffsol must preserve that pair under the `Provided` policy. Fresh consistency
solving belongs to the common initializer and is exercised separately by
[`time.canonical-first-order`](../canonical-first-order/README.md).
Every sampled state is checked against both the analytic trajectory and the
algebraic invariant. Provider-local reinitialization is not part of this claim.

The adapter separately rejects general `F(t,y,ydot)=0` residuals and explicit
Runge--Kutta for mass-matrix systems. Mass-matrix sensitivity additionally
fails unless the system explicitly proves `M_p = 0`, preventing an omitted
`M_p y_dot` term. This evidence does not claim Eqiora event ordering,
automatic backend reset semantics, adjoints, Parameter-dependent mass
sensitivity, or a general implicit DAE backend. Root results are proposals
only; the test commits a post-event state outside Diffsol and restarts the same
lowered problem. The admitted forward sensitivity consumes the same primal,
state-JVP, and parameter-JVP actions as the Eqiora differentiation boundary;
it is not a second model semantics.

Run only this case with:

```console
cargo run -p eqiora-verify -- run --case time.diffsol-adaptive
```
