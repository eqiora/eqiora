# RFC 0011: Implicit differentiation contracts

- Status: Draft implementation
- Authors: Eqiora contributors
- Created: 2026-07-17

## Summary

Eqiora differentiates a converged implicit Relation through one lowered
`LinearizedRelation` contract with primal, Jacobian-vector, and
vector-Jacobian actions; normal and transposed linear systems are two
orientations of the same solver contract, while hybrid jumps supply their own
event-time and saltation linearization.

## Motivation

The canonical model already states equations as implicit Relations. For one
activated and discretized system, write

```text
R(w, p) = 0,
```

where `w` contains the unknown values solved by the realization and `p`
contains selected design parameters. At a converged point `(w*, p)`, the
first-order relation is

```text
R_w dw + R_p dp = 0.
```

It gives both principal sensitivity modes:

```text
forward:  R_w dw = -R_p dp
adjoint:  R_w^T lambda = J_w^T
gradient: dJ/dp = J_p - lambda^T R_p.
```

Differentiating Newton or Krylov iterations would make a mathematical
derivative depend on convergence history, damping, stopping conditions, and
backend implementation. It also retains every reverse-mode iteration unless a
special transformation recognizes the solve. The derivative of a converged
implicit relation needs none of those details: it needs actions of `R_w`,
`R_p`, and their transposes, followed by ordinary linear solves.

The scalar Operator IR is a topologically ordered SSA expression program. It
is therefore a natural first implementation of primal, JVP, and VJP actions,
but it is not the differentiation contract itself. A symbolic transform,
forward or reverse AD backend, generated device program, or handwritten
physics action must be able to implement the same interface without changing
model meaning.

## Proposed design

### Layering and ownership

```text
Semantic Relation R = 0                       canonical meaning
             |
             v
Operator IR + explicit variable binding       lowered realization
             |
             v
LinearizedRelation at (w*, p)                 primal + JVP + VJP
      |                              |
      v                              v
R_w / R_p actions             transposed actions
      |                              |
      +--------------+---------------+
                     v
              oriented linear solve
                     |
          forward / adjoint analysis
```

The Semantic Kernel remains unchanged. `eqiora-ir` owns the backend-neutral
linearization contract and the scalar SSA implementation. `eqiora-solver`
owns linear actions, transpose capability, solver policy, and convergence
evidence. `eqiora-numerics` may implement the same lower contract for one
assembled discretization, while a separate analysis layer composes either
implementation into forward and adjoint algorithms. Analysis algorithms do
not enter the Semantic Model or method-specific assembly.

Hybrid event-time sensitivity is owned by the hybrid execution layer because
it requires guard, flow, reset, and transition-side semantics that are not
properties of a smooth residual expression.

### Linearized Relation

The contract is scalar-representation-parametric even though the first scalar
SSA executor implements only `f64`:

```text
LinearizedRelation<S>
    unknown_dimension()
    parameter_dimension()
    residual_dimension()
    primal(residual)
    jvp(Unknown | Parameter | Both, residual_tangent)
    vjp(residual_cotangent, Unknown | Parameter | Both)
```

The object represents one immutable linearization point. Its `primal` action
evaluates `R(w*, p)`. Its JVP evaluates

```text
R_w dw + R_p dp,
```

and its VJP evaluates both components

```text
(R_w^T c, R_p^T c).
```

The tagged tangent/cotangent selection represents the direct sum explicitly.
It lets state- or parameter-only Jacobian views omit a mathematically zero
seed or discarded output without allocating a zero scratch vector on every
matrix-free action.

Callers own output buffers. Shape mismatch, non-finite point data,
non-finite derivatives, or an unsupported derivative action returns a stable
diagnostic. An implementation cannot silently substitute finite differences
for a declared exact derivative source.

For every finite tangent `(dw, dp)` and cotangent `c`, an admitted
implementation satisfies the duality invariant

```text
<c, jvp(dw, dp)> = <vjp(c), (dw, dp)>
```

under its declared floating-point tolerance. This identity is the primary
backend-independent conformance check for JVP/VJP pairing.

### Explicit variable binding

Operator inputs are assigned one of three roles after lowering:

```text
Unknown     one coordinate of w
Parameter   one coordinate of p
Frozen      fixed at this linearization point
```

Roles are explicit and retain dense slot order. They are not inferred from
`SymbolRef`: a `Field`, `Port`, initial condition, or control may be an
unknown, a selected design variable, or frozen under different analyses.
Semantic `Parameter` is a useful default-selection source at a higher layer,
not proof of the derivative role inside Operator IR.

Frozen inputs have no tangent coordinate and receive no returned cotangent.
This keeps time, coordinates, fixed boundary data, and inactive controls out
of an analysis without pretending that their mathematical derivative is
zero in every possible analysis.

### Scalar SSA implementation

The scalar executor evaluates primal values once in instruction order. JVP
propagates one tangent alongside each primal value. VJP seeds residual roots
and accumulates instruction cotangents in reverse order. Repeated reads and
shared subexpressions accumulate into the same dense input slot.

Integer power handles exponent zero explicitly so that the derivative of
`x^0` is zero without evaluating `x^-1` at `x = 0`. Division and negative
powers retain the same finite-evaluation policy as the primal IR.

The first implementation remains `f64` because canonical values and the
host-local solver are currently `f64`. This is an exact capability statement,
not a type-system claim that all future linearizations are `f64`. Other scalar
executors implement `LinearizedRelation<S>` independently; the whole
simulator is not made generic over a dual number.

### Spatial expression and assembly implementation

Spatial lowering retains canonical Parameter IDs and the canonical revision's
default values in one immutable scalar-expression template. An application
may derive another immutable complete numerical point by replacing only its
explicitly selected Parameters; every unselected model or geometry coordinate
remains frozen at the template value. Coordinate and Parameter tangents compose
in the same analytic JVP pass. Design role remains an analysis choice: callers
explicitly select and order typed `SpatialDesignCoordinate` values. The first
coordinate kinds are a canonical model Parameter and one bound of a Cartesian
Domain; realization-local mesh-vertex indices are not promoted into model
meaning.

For the first Cartesian scalar elliptic slice, Q1 FEM and orthogonal TPFA FVM
retain their method-native unknowns and assemble

```text
R(w, p) = A(p) w - b(p) = 0
```

into `AssembledLinearizedRelation`. Sparse CSR supplies `R_w`; a dense
row-major action stores the few selected columns of `R_p`. FEM includes
constitutive, source-load, and eliminated essential-boundary terms. TPFA
includes transmissibility, cell-source, and boundary-flux terms. This storage
choice is not a semantic distinction and can be replaced behind
`LinearizedRelation` when Parameter dimension warrants sparse or matrix-free
actions.

The Parameter-action evidence is a 2D canonical Poisson model under both
discretizations. It does not establish nonorthogonal FVM, mixed/high-order
finite elements, natural-boundary differentiation in the Cartesian execution
path, or a continuous-objective contract.

### Static application program and accepted evaluations

The bounded scalar-elliptic application separates identity that is fixed for a
compiled analysis from numerical state accepted at one Parameter point:

```text
DifferentiableProgram
  = (Model artifact, Realization, ordered inputs, output, shapes,
     scalar/device/derivative/solver contract)

DifferentiableEvaluation(p)
  = (Program identity, exact p, accepted primal, R_w/R_p,
     output linearization, solve evidence)
```

The program retains one lowered template and its default evaluation. Evaluating
another complete finite input vector immutably binds the selected Parameters,
keeps all unselected Parameters at their canonical Model values, finalizes and
solves through the same exact Realization, and publishes an evaluation only
after the primal and paired linearizations are accepted. It neither mutates the
canonical Model nor changes the program's Model/Realization identity.

Every evaluation owns its point, accepted primal, relation, output projection,
and solve evidence. A caller may therefore retain the reverse action for one
point while the same program evaluates another point concurrently. Repeating
`p0 -> p1 -> p0` must reproduce the default accepted result without state
leaking between evaluations. Bound model, exact Realization, finalized system,
and accepted solution form one owned handoff; the only linearization operation
borrows that handoff and cannot accept a caller-supplied model, Realization, or
solution. Identity is not inferred from assembled bytes because distinct
Parameter points may have equal primal systems and different derivatives.
Mismatched shapes, non-finite values, inadmissible coefficients, and foreign
Realizations fail before publication.

This is an application-level numerical binding, not a Semantic Model mutation,
artifact migration, or general runtime override mechanism. The first
implementation remains one 2D generated-Cartesian scalar Field, Q1 FEM or
orthogonal TPFA FVM, host-serial native `f64`. Python exposure, framework
operators, batching, persistence, GPU, and distributed evaluation remain
separate capabilities.

### Fixed-topology geometry action

Shape and mesh differentiation enter through the local
reference-to-physical map, not through a second residual API. For one accepted
entity map and selected design direction,

```text
x = chi(xi, p)       J = d chi / d xi
dx = chi_p dp        dJ = d(chi_p dp) / d xi
```

`AffineGeometryLinearization` carries the primal map together with origin,
Jacobian, and measure JVPs. Spatial-expression coordinate JVPs consume `dx`.
Q1 FEM additionally differentiates inverse-Jacobian basis gradients and cell
measure. Orthogonal TPFA differentiates cell/source measure, facet measure,
cell-center distance, and transmissibility. Both assemble the resulting
column into the same `AssembledLinearizedRelation` as model-Parameter actions.

The first realization maps a selected Cartesian Domain bound to every axis
vertex while preserving its normalized coordinate. This is a global affine
pullback: topology, axis alignment, and TPFA orthogonality remain invariant.
The verified 2D constant-load Poisson case compares every method-native state
sensitivity and an adjoint objective gradient against independently recompiled
centered differences for Q1 FEM and TPFA FVM. An area-preserving log-aspect
parameterization then consumes the adjoint gradient through Armijo
backtracking and approaches the square stationary shape from a nonsquare
start.

The next realization preserves that same action on explicit simplex
connectivity. `SimplicialMeshVelocity` maps one canonical design coordinate to
finite vertex velocities in a mesh revision; local operators still consume
only `AffineGeometryLinearization`. A mesh is accepted only when every affine
cell has positive signed Jacobian and passes a recorded scale-invariant
mean-ratio threshold. Duplicate cells, isolated vertices, and non-manifold
facets also fail before assembly. This establishes cell-local affine
injectivity, not global non-overlap.

The objective side uses one row of the shared paired output linearization:

```text
DenseOutputLinearization = ([J], J_w, J_p)
J_h = integral_Omega source(x, p) u_h(x, p) dx.
```

`J_p` includes coordinate/source/measure actions and the direct derivative of
eliminated essential values. `AcceptedOutputLinearization` validates its layout against the accepted relation;
`adjoint_output_gradient` with output cotangent `[1]` uses the existing transpose solve. Exact
accepted-point identity across serialized artifacts remains run-provenance
work; the numerical contract checks finiteness and dimensions.

The verified 2D P1 case uses distorted triangular connectivity and compares
all state actions plus both Domain-bound compliance gradients with separately
compiled, rebuilt, quality-checked, and solved centered differences. The
objective value also converges under refinement to the independent
Fourier-sine compliance series for a rectangle. The simplex topology and
geometry contracts are runtime-dimensional, with separate tetrahedral unit
evidence; the PDE derivative claim remains 2D.

These are discretize-then-differentiate results on fixed topology. They do not
claim mesh-independent Hadamard shape calculus, global mesh injectivity,
remeshing, adaptive refinement, topology change, nonorthogonal FVM,
mixed/high-order elements, or a production optimizer.

### Derivative source and provenance

The numerical action, not its construction technique, is the stable contract.
The realization/run record identifies the derivative source:

```text
OperatorIrForward
OperatorIrReverse
Symbolic
Automatic
Handwritten
FiniteDifferenceReference
```

This vocabulary is provenance, never canonical semantics. Finite difference
is allowed as a small conformance oracle and explicit fallback capability, not
as an unreported implementation behind JVP/VJP.

### Normal and transposed linear actions

`LinearOperator` continues to mean one callable linear action. A separate
`TransposeLinearOperator` capability provides `apply_transpose`; operators
that cannot supply it do not implement the trait. This avoids a default
transpose method that fails only after an adjoint solve has started.

An Eqiora-owned `Transposed` view swaps dimensions and implements ordinary
`LinearOperator::apply` by calling the source transpose action. Consequently
the solver has one execution path:

```text
solve(normal action, rhs, plan)
solve(Transposed(action), rhs, plan)
```

There is no duplicated `solve_transpose` policy API and no second tolerance or
preconditioner configuration. The solve report records the action orientation
so evidence and provenance cannot confuse the two systems. For real `f64`,
transpose is the required adjoint action. A later complex-scalar RFC must
distinguish non-Hermitian transpose from conjugate transpose explicitly.

Matrix-free iterative adapters can consume the oriented action directly.
Direct solvers may prepare one immutable operator and reuse a factorization
for multiple right-hand sides. Safe cross-call reuse additionally requires a
versioned operator/linearization identity; no cache may guess identity from a
pointer or matrix shape. The prepared-solver lifecycle is introduced with
that artifact identity rather than added now as an uncheckable cache hint.

### Materialized canonical direct-output composition

An accepted relation/output pair may additionally borrow its exact
`CanonicalCsrSystemView` through
`AcceptedOutputLinearization::new_with_canonical_state_jacobian`. This is an
application-established association, not identity inferred from CSR bytes,
shape, allocation, or solution agreement. Construction validates the accepted
primal and complete relation/output/source layout.

`LinearSolveRequest::solve_canonical_oriented` combines that coefficient
source with one separately derived right-hand side and an existing
`LinearOperatorOrientation`. The source retains its captured primal right-hand
side; it is never substituted for `-R_p dp` or `J_w^T`. The bounded faer
`SparseLu` path factors the original canonical coefficients and applies the
normal or transpose solve of that factor. It reports and independently checks
the submitted orientation and derivative problem.

Backend acceptance is necessary but not sufficient because a same-shape
foreign canonical source can satisfy its own residual. Before publishing an
output tangent or gradient, differentiation therefore replays the returned
vector through the accepted relation's JVP or VJP and compares that residual
with the solve report's target. Existing pairs constructed with `new` retain
the matrix-free route. This composition adds no explicit transpose CSR,
prepared-factor lifecycle, cross-call factor reuse, Stokes symmetry shortcut,
or general materialization requirement.

### Jacobian operator views

At one `LinearizedRelation`, four views follow mechanically:

```text
StateJacobian              dw -> R_w dw
Transposed(StateJacobian)   c -> R_w^T c
ParameterJacobian          dp -> R_p dp
Transposed(ParameterJacobian)
                            c -> R_p^T c
```

These views implement solver linear-action traits through JVP/VJP. They do not
assemble a dense Jacobian unless a selected realization requests an assembly
artifact. A handwritten matrix-free relation and scalar SSA relation are
therefore interchangeable at the sensitivity algorithm boundary.

### Forward and adjoint algorithms

Forward sensitivity evaluates `R_p dp`, negates it, and solves the state
Jacobian action. Adjoint sensitivity solves the transposed state Jacobian for
the objective-state cotangent, then evaluates the parameter VJP and combines
it with the direct objective-parameter cotangent.

Both algorithms independently verify the linear residual using the oriented
action accepted by `eqiora-solver`. Neither observes nonlinear iteration
history. A failed or insufficiently converged primal solve is not
linearizable evidence: the residual at the claimed point must meet the
analysis tolerance before sensitivity results are accepted.

Time integration lowers each accepted step to its discrete implicit residual.
The discrete adjoint differentiates that step relation and records the exact
time-discretization realization. It does not differentiate an adaptive
controller or reuse continuous-adjoint formulas while claiming a discrete
gradient.

The first verified time slice uses residual-native implicit Euler. For one
accepted step,

```text
G(y_next; y_previous, p)
  = F(t_next, y_next, (y_next - y_previous) / h, p) = 0.
```

The step `LinearizedRelation` declares `y_next` as its unknown and the direct
sum `[y_previous, canonical model Parameters]` as its parameter vector. Model
time and `h` are frozen realization data. Its actions are the exact chain-rule
composition

```text
G_y_next     = F_y + F_y_dot / h
G_y_previous =      - F_y_dot / h
G_p          = F_p,
```

with the corresponding transposed projection for VJP. Scalar Operator IR
still sees Field, Derivative, and Parameter inputs as distinct coordinates;
only this method-specific projection couples them. The implementation neither
records nor differentiates the dense reference Newton iterations.

`DiscreteStepLinearization` adds accepted previous/next state, exact start/end
time, canonical state order, and common model-Parameter point to that relation.
Before reverse accumulation, the trajectory algorithm validates every layout,
accepted primal residual, adjacent state/time boundary, and explicitly
declared checkpoint boundary. It then solves each `G_y_next^T` in reverse,
passes the leading previous-state cotangent to the prior step, and accumulates
the trailing common model-Parameter cotangent.

The first trajectory evidence covers four fixed implicit-Euler steps and one
separately validated content-addressed semantic restart after step two. The
artifact layer proves parent/checkpoint/Provided-child lineage; the numerical
layer independently proves that the decoded checkpoint state and time match
both adjacent step relations before any transpose solve. This does not
differentiate serialization or claim BDF history, adaptive step selection,
optimal checkpoint scheduling, backend-native checkpoint payloads, or
derivative-run provenance. The first objective has one terminal-state
cotangent and one direct common-Parameter term; running/intermediate objective
terms remain a later extension.

### Hybrid differentiation

A smooth `LinearizedRelation` is valid only within one mode and activation
regime. At a guard crossing, the hybrid layer must additionally provide:

- guard gradients and event-time sensitivity;
- the reset Jacobian;
- pre- and post-transition flow values;
- a saltation action updating state tangents/cotangents;
- explicit behavior for grazing, simultaneous transitions, and Zeno
  termination.

The saltation action composes between smooth step linearizations. Omitting it
is not an approximation that may be hidden behind ordinary JVP/VJP. A hybrid
gradient request is admitted only when its guard/reset/event class has explicit
saltation evidence; otherwise it fails closed.

The first verified seam implements this composition for one localized,
transversal explicit-ODE event. For guard `g(t,y,p)=0`, reset
`y^+ = rho(t,y^-,p)`, fixed-time pre-event sensitivity `S^-`, and

```text
d = g_t + g_y f^- != 0,
```

it evaluates

```text
tau_p = -(g_y S^- + g_p) / d,
Xi = rho_y + (f^+ - rho_y f^- - rho_t) g_y / d,
S^+ = rho_y S^- + rho_p + (rho_y f^- + rho_t - f^+) tau_p.
```

`eqiora-time` owns only the evaluated flow/guard/reset linearization and this
composition. `eqiora-runtime` lowers the canonical guard and every reset
Relation in its structural event group through scalar Operator IR, obtains
guard/reset derivatives with JVP actions, and solves the grouped implicit
reset. The initial reset solver admits a constant full-monomial `Next`
Jacobian; this is a checked capability boundary, not a canonical restriction.

The linearization operation accepts a localized point and explicit guard
tolerance. Canonical crossing direction is enforced and an exactly zero
transversality denominator fails as grazing. The first production path now
obtains that point from a content-registered explicit-ODE root proposal,
rejects registration drift, applies the grouped reset/saltation, and explicitly
restarts from the post-event state. Distinct guards that become simultaneous
numerically, periodic ticks, priority, coupled/nonlinear `Next` solves, DAE
events, post-event mode changes, trajectory adjoints, and checkpoint lineage
through event transitions remain unsupported. Consequently
`differentiation.hybrid-event` and
`hybrid.registered-event` are narrow verified capabilities, not a claim of
general hybrid AD.

## Alternatives considered

### Genericize the simulator over dual numbers

This makes simple forward mode easy, but propagates a differentiation
implementation choice through state storage, solvers, event logic, FFI, GPU
kernels, and public APIs. Reverse mode and handwritten or symbolic actions
still need separate treatment. Rejected as the canonical design. A dual-number
executor may implement the same JVP contract as an optional backend.

### Differentiate nonlinear and linear solver iterations

This requires retaining or replaying convergence history and makes gradients
depend on damping, tolerances, and backend-specific control flow. It is useful
only when the algorithm itself is the object of differentiation. Rejected for
model sensitivity; implicit differentiation is selected.

### Materialize every Jacobian

Dense or sparse matrices enable familiar transpose and factorization paths but
discard matrix-free structure and can dominate memory for PDE systems.
Rejected as a universal contract. Assembly is a realization choice behind the
same JVP/VJP actions.

### Put an optional transpose method on every operator

A default method returning "unsupported" makes adjoint capability a runtime
surprise and encourages accidental fallbacks. Rejected. Transpose is a
separate trait bound and capability.

### Add separate normal and transpose solver configuration types

This duplicates tolerances, algorithms, preconditioners, and residual policy,
and permits the two paths to drift. Rejected. Orientation wraps the action;
the solver plan remains unique.

### Infer design variables from semantic node kinds

The same field or parameter can be active in one optimization and frozen in
another. Automatic inference would mix analysis selection with model meaning.
Rejected; binding roles are explicit after lowering.

## Compatibility and migration

No Semantic Kernel node, source construct, canonical model wire, or activation
meaning changes. `ScalarOperatorIr::evaluate` remains the primal convenience
path and is implemented consistently with the new linearization evaluator.

`eqiora-solver` gains a transpose-capability trait and oriented view. Existing
operators and normal solves remain source-compatible except where solve-report
construction changes internally to record orientation. External backends
already return Eqiora-owned reports through the acceptance function and do not
construct reports directly.

The additive canonical-oriented request permits a canonical coefficient
source, derivative-specific right-hand side, and transposed orientation to
coexist. External backends may reject that combination with their existing
structured capability diagnostic; they must not substitute the source's
primal right-hand side, relabel the orientation, or materialize a second source.
The old accepted-output constructor and all matrix-free callers remain
source-compatible.

The scalar implicit case, one assembled spatial residual under two
discretizations, and one time-step residual now have independent evidence.
The spatial application API additionally separates one static program identity
from immutable accepted evaluations at explicit numerical Parameter points;
the default-point convenience methods preserve their previous behavior. The
first narrow hybrid saltation case also has analytic evidence. Rust API
stabilization and wire representation of program, evaluation, derivative
provenance, and linearization identity remain separate decisions in the
Realization/run-provenance RFC.

## Verification

- Compare scalar SSA primal values with the canonical expression evaluator.
- Compare JVP against analytic derivatives and centered finite differences.
- Compare VJP against analytic derivatives and verify JVP/VJP duality on a
  nonsymmetric multi-equation relation.
- Reject wrong point, role, tangent, cotangent, and output dimensions before
  evaluation; reject every non-finite result.
- Solve one converged nonlinear `R(w, p) = 0` for forward sensitivity and
  compare against an analytic or independently differenced solution map.
- Solve `R_w^T lambda = J_w^T` using an actual VJP-backed transposed action,
  then compare the total parameter gradient with finite differences.
- Require a nonsymmetric `R_w` so a mistaken normal solve cannot pass.
- Verify normal and transposed true residuals independently through the same
  solver acceptance path.
- Bind a nonsymmetric canonical 5x5 relation whose primal right-hand side is
  zero and whose distinct derivative right-hand side is nonzero; run ordinary
  direct normal and transposed output differentiation first, then reject a
  provider reading the primal RHS, a transpose route returning the normal
  solution, and a same-shape foreign canonical source at their named residual
  boundaries.
- Differentiate one canonical 2D Poisson problem with explicitly selected
  coefficient, source, and essential-boundary Parameters. Under both Q1 FEM
  and TPFA FVM, compare every forward state component and an adjoint objective
  gradient with independently recompiled centered differences.
- Through the same static program identity, accept one non-default Parameter
  point and compare its primal/JVP/VJP with independent rebuilds; verify
  `p0 -> p1 -> p0` repeatability and concurrent evaluation isolation, and reject
  invalid points or a solution/linearization pair from different points.
- Verify one residual-native implicit-Euler step by JVP/VJP duality,
  nonsymmetric normal/transposed solves, analytic state constraints, and
  centered finite differences of the independent discrete solution map.
- Verify four accepted residual-native implicit-Euler steps in reverse across
  one independently validated semantic checkpoint/restart edge; reject state,
  time, layout, or Parameter-point discontinuity before transposed solves.
- Verify the canonical bouncing-ball first impact against analytic event-time,
  reset, and saltation derivatives, including crossing-direction rejection.
- Verify a content-linked production root proposal selects that exact canonical
  group, rejects a foreign registration, and restarts to an analytic
  post-impact flight sample.

The smooth verification capability is named
`differentiation.implicit-relation`; it does not imply spatial, time-dependent,
hybrid, GPU, or distributed differentiation. The separate
`differentiation.materialized-direct-output` capability covers only one
host-local real-`f64` faer `SparseLu` reference with independently replayed
normal/transposed relation actions; it does not claim prepared factors,
reuse, other backends, or Stokes E2. The separate
`differentiation.spatial-poisson-fem-fvm` capability covers only the Cartesian
Poisson boundary and discretizations stated above. The separate
`differentiation.discrete-implicit-step` capability covers only the accepted
residual-native implicit-Euler step and explicit parameter layout above. The
`artifacts.implicit-time-restart-lineage` capability covers semantic restart
identity but not cotangent accumulation or checkpoint scheduling. The
`differentiation.checkpointed-trajectory-adjoint` capability composes four
fixed implicit-Euler steps across one such validated boundary; it does not add
a checkpoint scheduler or backend-history derivative. The separate
`differentiation.hybrid-event` capability has the explicit localized-event
boundary above. `hybrid.registered-event` adds only the registered
proposal-to-restart path for the same event class.

## Research basis

- [JAX custom derivative rules](https://docs.jax.dev/en/latest/notebooks/Custom_derivative_rules_for_Python_code.html#example-implicit-function-differentiation-of-iterative-implementations)
  derive a VJP from the converged fixed-point equation instead of reversing
  through an unbounded iterative implementation.
- [PETSc `KSPSolveTranspose`](https://petsc.org/release/manualpages/KSP/KSPSolveTranspose/)
  treats `A^T x = b` as a first-class linear solve over the same solver object;
  Eqiora retains the symmetry while making transpose availability explicit on
  matrix-free actions.
- [Diffsol 0.16 sensitivity documentation](https://docs.rs/diffsol/0.16.2/diffsol/#forward-sensitivity-analysis)
  separates forward sensitivity, adjoint checkpointing/backward passes, root
  events, and reset handling. It informs the future adapter boundary but does
  not define Eqiora semantics.
- Kong, Payne, Zhu, and Johnson,
  [“Saltation Matrices: The Essential Tool for Linearizing Hybrid Dynamical
  Systems”](https://arxiv.org/abs/2306.06862) (v3, 2024), identifies the
  saltation matrix as the sensitivity update across a hybrid jump.

## Security, safety, and governance

Derivative callbacks obey the same panic, finite-value, allocation, and FFI
boundaries as primal operators. Untrusted artifacts cannot select arbitrary
native derivative code. Backend-native derivative source, generated module
identity, scalar representation, and numerical policy are recorded in run
provenance.

Incorrect gradients can silently corrupt optimization and design decisions.
Therefore capability names require executable evidence, finite differences
remain an independent small oracle, and hybrid requests outside the verified
localized transversal seam fail closed. An RFC review is required before
stabilizing the differentiation API or claiming a new equation class.

## Unresolved questions

- The stable wire identity for a linearization and prepared factorization.
- The scalar trait and complex adjoint convention beyond the first `f64`
  executor.
- Checkpoint scheduling and memory budgets for long discrete adjoints.
- Sparse coloring and batched JVP policy for assembled Jacobians.
- Nonsmooth constitutive laws and generalized derivative vocabulary.
- Generalized derivatives for grazing events and composition of distinct
  simultaneous saltations.
