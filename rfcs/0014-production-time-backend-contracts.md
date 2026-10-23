# RFC 0014: Production time backend contracts

- Status: Draft implementation
- Authors: Eqiora contributors
- Created: 2026-07-18

## Summary

Eqiora classifies continuous canonical Relations into an explicit ODE,
first-order mass-matrix system, or general implicit DAE before selecting an
execution path. The first two use `TimeProblem`; the third uses the distinct
residual-native `ImplicitDaeProblem`. Both share `TimePlan` without pretending
their action vocabularies are interchangeable. The first optional production
adapter uses Diffsol 0.16 for Tsitouras 5(4), BDF, consistent initialization,
root proposals, and continuous forward sensitivities. A deterministic
implicit-Euler oracle verifies one semi-explicit index-one general residual
without transferring model time, event ordering, or reset meaning to a
library. The residual-native path also versions one accepted semantic
checkpoint and an acyclic parent-to-child restart edge without serializing
backend history.

## Motivation

The reference interpreter deliberately couples a transparent backward-Euler
step and dense Newton solve to the normative activation calendar. Replacing it
inside that interpreter would make an adaptive library part of model meaning.
Conversely, giving every backend its own ODE configuration, trajectory type,
and callback model would recreate the duplicate-plan problem already removed
from linear algebra.

The canonical Relation form is more general than the equation class accepted
by an ODE library:

```text
canonical                         lowered first-order projection
F(t, y, y_dot, p) = 0      →      M(t, p) y_dot = f(t, y, p)
```

That arrow is a compiler proof obligation. A residual that cannot be factored
into the right-hand form must remain a `GeneralImplicitDae`; an adapter must
reject it instead of inventing a mass matrix.

## Proposed design

### Ownership path

```text
canonical Relation network
        ↓ equation-class lowering
TimeProblem + TimeSystem       ImplicitDaeProblem + ImplicitTimeSystem
        ↓                               ↓
production first-order adapter   residual-native adapter / reference oracle
        └──────────── TimePlan / capability admission ────────────┘
        ↓ owned TimeSolution + TimeExecutionReport
first-order: TimeLoweringEnvelopeV1 + RootRegistrationEnvelopeV1
             + TimeRunManifestV1
residual-native: GeneralImplicitTimeLoweringEnvelopeV1
                 + supplied/accepted ImplicitTimeInitialDataEnvelopeV1
                 + ImplicitTimeRunManifestV1
                 + accepted ImplicitTimeCheckpointEnvelopeV1
                 + ImplicitTimeRestartManifestV1
        ↓ verification / output artifacts
```

`eqiora-time` is L2. It owns the first-order and residual-native problems, the
shared plan, derivative actions, root-proposal DTO, result/evidence vocabulary,
and a small residual-native reference oracle, but executes no third-party
solver. `eqiora-backend-diffsol` is an optional L3 adapter. Diffsol, nalgebra,
and faer types remain private to that crate. Canonical model and wire schemas
contain none of them.

`eqiora-runtime` performs the first canonical projection. It consumes an
immutable `KernelProgram` plus scalar Operator IR, checks continuous
activation, captures revision-local initial and Parameter values, and proves
the complete constant derivative Jacobian directly from SSA structure. Each
finite coefficient is then interpreted as the exact binary rational represented
by its `f64` bits, and arbitrary-precision rational elimination recomputes the
rank. A full monomial Jacobian may be permuted, signed, and scaled; the runtime
normalizes it to `y_dot = f(t,y)`. Every other non-zero-rank constant matrix
remains a full or rank-deficient mass matrix, and the latter requires
consistent initialization even when its residual basis contains no literal
zero row. All projections derive the state JVP from the same Operator IR. They
never classify by evaluating sample states or by using a floating-point rank
threshold.

State-dependent coefficients and nonlinear derivative dependence do not enter
the first-order seam. `GeneralImplicitProgram` admits only those two structural
obstructions, retains `y` and `y_dot` as independent inputs, derives an
explicit differential/algebraic partition from effective structural
derivative dependence (including exact cancellation),
and exposes residual plus paired JVP actions. A Relation that has a valid
constant first-order projection is rejected from this wider path so that one
equation never acquires two competing lowerings.

The proof is retained, not discarded after callback construction.
`TimeLoweringProof` records Relation ID, state order, the complete row-major
constant derivative matrix, and its recomputed exact rank. A monomial-row view
is derived only when normalizing an explicit ODE. `TimeLoweringEnvelopeV1`
binds the proof to model digest and semantic revision; loading it with the
model independently reconstructs scalar Operator IR, compares every
coefficient, and replays exact rank. `TimeRunManifestV1` then links the
lowering digest to the exact `TimePlan`, adapter/version, accepted
equation/initial-condition report, and output digests. Residual-native
checkpoint and restart artifacts compose with those outputs below; the
first-order run wire remains unchanged.

### Equation classes

`TimeEquationClass` has three explicit cases:

- `ExplicitOde`: `y_dot = f(t,y)`;
- `MassMatrix { Full | RankDeficient }`: `M(t)y_dot = f(t,y)`;
- `GeneralImplicitDae`: arbitrary `F(t,y,y_dot)=0`.

`TimeProblem` represents only the first two. Its constructor rejects the third
because the available action vocabulary could not express it faithfully.
The common initializer owns consistency and supplies an accepted State/rate pair.
`TimeProblem` carries that pair; a native adapter must preserve it, including
rank-deficient and row-mixed mass matrices, rather than re-solving supplied data
or inferring algebraic coordinates from mass diagonal zeros.

The Diffsol adapter admits Tsitouras 5(4) only for `ExplicitOde`. BDF admits
the explicit and mass-matrix classes. `ImplicitDaeProblem` represents the
third class through `F(t,y,y_dot)` and
`F_y dy + F_y_dot dy_dot`; Diffsol rejects it. The deterministic reference
backend admits only `ImplicitEuler`. The optional SUNDIALS IDA adapter consumes this same residual-native seam for
serial dense BDF execution with native-version checks and accepted history.
Common Plan/Run composition and broader provider capabilities remain separate
delivery work; the native product tests do not establish those claims.

### Residual-native initialization and reference execution

`ImplicitDaeProblem` binds one residual/JVP provider, a state-coordinate
partition, an initial state, an initial derivative, and an initial-condition
policy. `Provided` means the complete pair is already accepted as consistent.
For `SolveConsistent`, the first reference slice follows the semi-explicit
index-one IDA convention: differential states and algebraic derivatives remain
fixed while Newton solves algebraic states and differential derivatives.

The reference backend applies fixed implicit Euler. At each accepted step it
solves

```text
F(t_n, y_n, (y_n - y_(n-1)) / h) = 0
```

with a dense Jacobian assembled from the analytic paired JVP, scale-aware
pivoting, bounded Newton iterations, and a damped line search. It is a
deterministic falsification oracle, not an adaptive, sparse, or production DAE
solver. It does not infer structural index, repair arbitrary inconsistent
systems, or promise convergence for every residual admitted by the action
contract.

### Numerical plan and result

`TimePlan` is the only numerical time policy at this boundary. It records:

- method;
- initial model time and an initial adaptive step guess or fixed-step bound;
- relative and per-state absolute tolerances; and
- strictly increasing requested output times.

Requested output times are not internal adaptive steps and do not become
ClockDomains. `TimeSolution` owns finite time-major state samples and a report
containing one atomic adapter name/release identity, method, exact equation
class, and initial-condition policy. Name and release cannot be assembled
independently after execution. No library matrix or borrowed callback buffer
escapes.

`TimeRunManifestV1` serializes the same validated plan rather than adding a
second time configuration. Its constructor rejects state-tolerance shape,
method, equation-class, and initial-condition drift against the linked
lowering witness and backend report.

The original v1 lowering/run artifacts describe only the first-order
projection and reject `ImplicitEuler`. Residual-native runs use dedicated v1
envelopes: the lowering records state order, effective partition, and structural
reason; separate initial-data envelopes retain the supplied guess and
backend-accepted pair; the run manifest links both to the plan, adapter report,
and outputs.

An `ImplicitTimeCheckpointEnvelopeV1` records an accepted `(t, y, y_dot)` pair
in the lowering's canonical state order. Construction and external validation
rebuild scalar Operator IR from the immutable model and replay the residual
norm under the stored acceptance tolerance. Decoder dimension and finite-value
checks apply before replay. The checkpoint references no run; its digest can
therefore appear in the parent run's sorted outputs without a cycle.

Checkpoint-derived restart input is an ordinary
`ImplicitTimeInitialDataEnvelopeV1` with `Provided` policy. A separate
`ImplicitTimeRestartManifestV1` links parent run, checkpoint, that exact child
initial artifact, and child run. It requires the child plan to start at
checkpoint time and both child input/accepted identities to equal the derived
artifact. A future IDA adapter must reuse these semantic identities rather than
put library-native vectors, BDF history, factorization state, or controller
memory into this wire. Those data require a distinct durable-payload contract.

### Callback failure boundary

Eqiora actions return structured `Diagnostic` values, while Diffsol callbacks
are infallible Rust closures. The adapter retains the first action diagnostic,
poisons the private library output with non-finite values to stop integration,
and returns the original diagnostic at the public boundary. Coloring is
disabled because Diffsol's automatic sparsity probe intentionally injects NaNs,
which would be indistinguishable from an Eqiora action failure at this seam.
Explicit sparse patterns can replace this dense first slice later.

### Differentiation

`ParametricTimeSystem` extends the same primal system with matrix-free actions:

```text
f_y dy
f_p dp
y0_p dp
```

`ForwardSensitivityProblem` binds a finite parameter point, and the adapter
integrates coordinate-basis sensitivities under a separate validated error
plan. For a mass-matrix problem it additionally requires
`MassParameterDependence::Independent`, an explicit proof that `M_p dp = 0`.
The canonical constant-derivative lowerer provides that proof and derives
Parameter order plus `f_p dp` from the same scalar SSA program. Diffsol BDF
then integrates

```text
M s_dot = f_y s + f_p
```

for full and rank-deficient constant mass matrices. The fail-closed default is
`Unspecified`; Parameter-dependent mass cannot silently omit its
`-M_p y_dot` contribution. This is not a second model semantics, a dual-number
simulator, a discrete time-step adjoint, or hybrid event sensitivity.
Parameter-dependent mass sensitivities, adjoints, adjoint checkpoint
scheduling, and general hybrid derivatives remain gates under RFC 0011. The
first narrow transversal explicit-ODE event is now connected through the
content-linked root registration below; that does not widen the supported
event classes.

### Root and reset ownership

`RootFunctions` supplies zero-crossing values only. A backend receives it only
inside `RegisteredRootProblem`, alongside a `RootRegistrationId` and
`RootRegistrationProof`. Diffsol localizes the first sign change before the
declared horizon and returns a `RootProposal` containing registration identity,
time, root index, and pre-event state. A proposal does not commit:

- crossing direction;
- simultaneous-event grouping;
- event priority;
- periodic-tick coincidence; or
- reset state.

The root index is local to one ordered callback registration and is never a
canonical Activation identity. Routing `root_index == 0` directly to the first
Event Activation would permit a proposal from another model revision, guard,
or lowering to be accepted accidentally.

`RootRegistrationEnvelopeV1` closes that ambiguity. Its canonical wire links
the immutable model digest/identity/revision and `TimeLoweringEnvelopeV1`
digest to a complete ordered partition of Event Activation IDs. Structurally
identical guards with the same direction form one atomic group. The wire does
not copy guard expressions; external validation reloads the model, lowers each
guard to scalar Operator IR, checks its state/Parameter/time symbols, and
reconstructs the partition independently. Decoding bounds both callback count
and total Activation references and rejects non-canonical order, duplicates,
overlap, or malformed identities.

`CanonicalRootSet` consumes the opaque artifact digest and proof, rebuilds each
callback from canonical semantics in proof order, and independently rejects an
incomplete, split, or combined partition. A proposal retains that digest, and
the reset/saltation boundary rejects a mismatched registration before using its
index. Pointer identity, vector position alone, and unlinked structural
inference are not admissible provenance.

The Eqiora hybrid layer decides those semantics. The first admitted vertical
slice selects one structurally identical explicit-ODE event group, enforces its
crossing direction, solves its grouped monomial implicit reset, produces its
saltation linearization, and explicitly restarts the same lowered system from
the post-event state. The adapter deliberately does not configure Diffsol's
automatic reset facility. Distinct simultaneous guards, tick coincidence,
priority, rejected-proposal resume, mode-dependent flow, and general DAE
events still require the full scheduler contract.

## Dependency and compatibility policy

Diffsol is exact-pinned at 0.16.2 behind `diffsol-runtime`. That release has
unconditional public re-exports for both nalgebra and faer host families, so
both upstream features are enabled even though the current adapter executes
with `NalgebraLU`. Its nalgebra 0.35 graph requires Rust 1.89, which is now the
single production-workspace MSRV. Cargo cannot encode a feature-specific MSRV,
so the MSRV gate executes all production features instead of publishing a
lower default-only support claim. [RFC 0059](0059-production-msrv-contract.md)
records the MSRV correction and the preceding 0.16.1 BDF safety update. The
0.16.2 upgrade retains that fix while changing BDF solver selection and
parameter-scaled forward-sensitivity tolerances; the registered BDF, mass,
sensitivity, and restart evidence is rerun at the new exact pin.

This is an explicit compatibility boundary, not an unrecorded toolchain bump.
Upstream feature or MSRV changes are reviewed when the exact pin changes.

Exact rank replay uses `num-rational` 0.4.2 and its pure-Rust arbitrary-
precision integer dependency. Both support Rust 1.60 or newer and are
MIT/Apache-2.0 dual licensed. Arbitrary-precision types remain private to
`eqiora-time`; untrusted lowering artifacts are checked against an explicit
decoder dimension limit before cubic elimination begins.

## Alternatives considered

### Put adaptive stepping into the semantic interpreter

Rejected. It would turn library error control and internal steps into
canonical activation behavior and obscure the small reference oracle.

### Genericize the simulator over dual numbers

Rejected as the canonical derivative design. It entangles solver internals,
state storage, and backend scalar support. Explicit primal/JVP/VJP actions
compose with forward, reverse, symbolic, or hand-written differentiation.

### Use Diffsol automatic reset

Rejected at the Eqiora boundary. A library-local reset cannot implement
simultaneous Relation/tick grouping and atomic `Pre`/`Next` semantics.

### Call every residual a DAE

Rejected as a support claim. A mass-matrix DAE and a general fully implicit DAE
have different admissibility and initialization requirements.

## Verification

`time.diffsol-adaptive` checks:

- Tsitouras 5(4) against smooth exponential decay;
- BDF against an analytic stiff tracking mode;
- BDF on an index-one `diag(1,0)` mass-matrix DAE from an inconsistent guess,
  including the algebraic invariant;
- Tsitouras and BDF forward parameter sensitivity against
  `dy/dk = -t exp(-kt)`;
- BDF forward sensitivity for Parameter-independent full and rank-deficient
  mass matrices, plus rejection when `M_p = 0` is unproven; and
- Tsitouras and BDF root proposals before and after an externally committed
  reset/restart.

Admission tests reject explicit Runge--Kutta for mass matrices and reject
general implicit DAEs for every Diffsol method.

`time.canonical-first-order` starts from validated canonical Relations. Its
explicit fixture permutes and scales residuals before structural
normalization. Its index-one fixture proves `diag(1,0)`, performs consistent
initialization, and runs BDF. Two additional fixtures prove a dense full matrix
and a dense singular matrix with no literal algebraic row, then run both
through BDF against analytic trajectories and analytic Parameter sensitivities.
A fifth valid Relation with a state-dependent derivative coefficient must
return `EQ0705` from `FirstOrderProgram`; the independent general-residual case
then proves that the rejection preserves a valid wider lowering rather than
discarding the model.

The same case round-trips lowering and run artifacts, rejects a forged
derivative coefficient, rejects a forged exact rank and an over-limit replay,
rejects a run report whose method differs from the plan, and rejects the
reference-only `ImplicitEuler` method from the first-order v1 run artifact.

`time.general-implicit-dae` starts from the canonical residual
`(1+z)(x_dot+x)=0`, `z-x^2=0`. It checks first-order rejection,
residual-native admission and analytic JVP, differential/algebraic partition,
consistent initialization from an inconsistent guess, first-order terminal
convergence under step refinement, and algebraic constraint preservation at
roundoff. This evidence is deliberately limited to one semi-explicit
index-one fixture and the dense deterministic reference oracle.

The same target separately admits `x_dot^2-1=0`, retains a provided
`x_dot=+1` branch, checks its paired JVP, and advances the analytic `x(t)=t`
trajectory. It verifies explicit branch ownership, not automatic branch
selection or general nonlinear-DAE robustness. The case also round-trips the
dedicated general-lowering, supplied/accepted initial-data, and run artifacts;
bounded decode, forged partition, foreign accepted-pair digest, and explicit-RK
admission fail closed.

`artifacts.implicit-time-restart-lineage` records the accepted point after one
reference step, replays its residual from canonical Operator IR, derives exact
`Provided` child initial data, and links parent and child runs without a digest
cycle. The restarted terminal state agrees with an uninterrupted two-step run.
Checkpoint value/dimension drift, a missing parent output edge, child start-time
drift, and a direct parent/child cycle fail closed. This does not verify
adaptive/BDF history continuation or adjoint checkpoint schedules.

`hybrid.registered-event` starts from a canonical bouncing-ball model and
round-trips the root-registration artifact. It rejects callback/Activation
resource excess, non-canonical order, incomplete linkage, and a proposal with
another registration digest. The accepted registered callback drives Diffsol
localization, canonical grouped reset and saltation, then an explicit restart;
impact and subsequent flight are checked against analytic values.

## Unresolved questions

- Structural-index analysis and residual-native admission beyond the current
  variable-coefficient/nonlinear-derivative proof boundary.
- Production IDA ownership/FFI containment and library-version compatibility.
- Sparse Jacobian/mass pattern transport without NaN probing.
- Distinct simultaneous root/tick grouping, priority, and
  resume-after-rejected-proposal in the hybrid scheduler.
- Parameter-dependent mass sensitivities, time adjoints, general hybrid
  differentiation, derivative provenance, adjoint checkpoint schedules, and
  backend-native durable checkpoint payloads.
