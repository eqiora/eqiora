# General implicit DAE reference verification

`time.general-implicit-dae` verifies the smallest problem that Eqiora's
constant-coefficient first-order form cannot represent. Its residual is

```text
(1 + z) (x_dot + x) = 0
z - x^2 = 0
```

with exact solution `x(t) = exp(-t)` and `z(t) = exp(-2t)`. Because the
derivative coefficient `1 + z` depends on state, `FirstOrderProgram` rejects
the Relation and `GeneralImplicitProgram` retains the full residual plus its
analytic state/derivative JVP.

The explicit variable partition is `(x: differential, z: algebraic)`.
Consistent initialization holds `x = 1` and `z_dot = 0` fixed while solving
for `x_dot = -1` and `z = 1`. This is the semi-explicit index-one
initialization mode documented for IDA; it does not assume that every accepted
residual has index one.

## Recorded results

| Steps | Step | Terminal L2 error | Observed order | Constraint error |
|---:|---:|---:|---:|---:|
| 10 | 1.0000e-1 | 2.2116138367007383e-2 | — | 2.7755575615628914e-17 |
| 20 | 5.0000e-2 | 1.1234336075507990e-2 | 0.977184635941 | 0.0 |
| 40 | 2.5000e-2 | 5.6626276671572370e-3 | 0.988371289924 | 0.0 |
| 80 | 1.2500e-2 | 2.8428606953024140e-3 | 0.994128265510 | 0.0 |

The deterministic implicit-Euler oracle must show monotonically decreasing
terminal error, observed order above `0.9`, and terminal algebraic-constraint
error below `1e-12`. The machine-readable source is
[`../../verify/time/general-implicit-dae/expected/convergence.csv`](../../verify/time/general-implicit-dae/expected/convergence.csv).

The same evidence target also checks `x_dot^2 - 1 = 0`. The Relation is
classified as nonlinear in its derivative, and a provided consistent pair
with `x_dot = +1` advances along `x(t) = t`. The opposite branch is equally
valid; the test checks that the branch is chosen by the problem data, not by a
lowering heuristic.

The case round-trips three provenance records: the replayable general lowering
witness, the supplied and accepted initial-pair envelopes, and the run manifest
that links both to `ImplicitEuler`, the backend identity, the plan, and the
output digests. It also checks that decoder dimension limits, partition drift,
accepted-pair digest drift, and explicit Runge--Kutta methods are rejected.

## Scope

The case covers one semi-explicit index-one DAE with a state-dependent
coefficient, solved with a dense reference Newton solve, and one explicitly
chosen branch of a scalar residual that is nonlinear in its derivative. It does
not cover automatic branch selection, a production IDA adapter, adaptive BDF,
arbitrary-index or structural-index analysis, sparse nonlinear algebra,
checkpoint/restart lineage, DAE sensitivities, or hybrid DAE events.
