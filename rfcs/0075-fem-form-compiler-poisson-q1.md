# RFC 0075: FEM form compiler, Cartesian Q1 Poisson slice

- Status: Accepted. The bounded private 2D Cartesian Q1 Poisson slice is
  implemented and verified; the second consumer and the shadow-path deletion
  remain deferred.
- Authors: Claude (contract and oracle), Codex (implementation)
- Created: 2026-07-25
- Related RFCs and evidence: RFC 0007 (canonical spatial operators),
  RFC 0020 (local action kernel boundary), RFC 0053 (discrete block system),
  RFC 0056 (pure calculus and support maps),
  [AI-authored platform strategy](../docs/development/ai-authored-platform-strategy.md),
  `verify/numerics/cartesian-poisson-fem-fvm`,
  `verify/differentiation/spatial-poisson-fem-fvm`,
  `verify/numerics/global-matrix-free-action`

## Summary

Derive the Cartesian Q1 Galerkin local matrix, load vector, and paired
residual/JVP/VJP actions for the `org.example.poisson` strong relation from the
typed semantic residual through one private, proof-carrying derivation, and
accept it only against a pre-committed independent oracle.

## Motivation

Physics capabilities currently re-earn their proof machinery. Across
independent physics owners in `eqiora-numerics`, the same structural roles are
re-created: `api.rs` ×7, `assembly.rs` ×4, `acceptance.rs` ×4, `element.rs` ×4,
`newton.rs` ×3 — 22 files and 11,178 lines of candidate repeated surface.
`acceptance.rs` ×4 is the conformance oracle itself, rebuilt per physics.

Under agent authorship the binding constraint is mechanical verifiability. An
agent can safely emit a form; it cannot safely hand-write a physics module with
a bespoke oracle. This RFC tests one hypothesis, on the smallest problem that
can falsify it:

> Can the *compiler* own the proof of its translation class, so that a physics
> instance supplies only witness data?

Poisson is chosen because it has no inf-sup requirement, no gauge, no
stabilization, and an existing registered case with analytic reference — so a
failure here is a failure of the hypothesis, not of the problem.

## Proposed design

### Boundary of this slice

This is an **FEM** lane. A universal weak-form IR is rejected on its merits:
conservative finite-volume face fluxes are method-foreign to a variational form
(RFC 0007). The FVM path is untouched and keeps its existing evidence.

No new crate, no new public type, no new language. The work is a private module
`crates/eqiora-numerics/src/form_compiler.rs`, registered privately from
`src/lib.rs`. Promotion of any contract into `eqiora-ir` is out of scope and
gated on amendment A1's audit-compression test.

### Input is not sufficient as it stands

`RelationDecl::residuals` carries componentwise **strong** residuals. The source
program does not carry test or trial functions, measures, integrals, element
families, quadrature, DOF layout, or weak boundary-term disposition, and field
declarations do not distinguish unknown from coefficient, frozen, or design
role. Supplying that structure fail-closed is the substance of this slice, not
an assumption.

### Execution path

```text
packages/org.example.poisson/src/main.eqi
  -> eqiora-lang AST
  -> eqiora-compiler
  -> eqiora_sem::KernelProgram::typed_relation_residual
  -> scalar-elliptic fail-closed recognizer
  -> private DerivedScalarGalerkinForm + derivation certificate
  -> reference Q1 evaluator
  -> eqiora_assembly::LocalContribution
       |-> assembly packet / CSR / existing solve
       `-> matrix-only LocalLinearActionIr projection
```

`LocalContribution` is the output target because it owns a local matrix **and**
RHS. `LocalLinearActionIr` is insufficient alone: Poisson needs a source term,
which RFC 0020 deliberately excludes from local action.

### Admitted subset, closed

The derived plan represents exactly:

- `coefficient * dot(grad(test), grad(trial))`;
- `source * test`;
- exact relation and field identities;
- complete essential boundary discharge through the zero trace of the test field;
  the prescribed trial trace may be nonzero.

Any other structure **fails**. No generic callback, no escape hatch, no
backend identifier becomes mathematical meaning.

Admitted symbol kinds are `Field` (exactly one, the unknown), `Parameter`, and
`SpatialCoordinate`. Admitted expression kinds are `Constant`, `Symbol`, `Neg`,
`Add`, `Sub`, `Mul`, `PowI`, `SpatialCoordinate`, `UnaryMath(Sin)`, `Gradient`,
`Divergence`, and `Trace`. Every other `KernelNode` variant — including `Pre`,
`Next`, `Derivative`, every `Port` kind, `Time`, `Div`, `SymmetricPart`,
`IsotropicLift`, `NormalComponent`, and `PureOperatorApplication` — is rejected
at admission rather than ignored.

### Derivation rules

The certificate is a list of entries, each binding one versioned rule ID to the
exact source relation identity, the source node identity it consumed, and the
weak-term slot it produced. The rule set for this slice is closed:

| Rule ID | Effect |
| --- | --- |
| `fem.derive.v1.test-pairing` | Multiply the strong residual by a test function drawn from the declared test space. |
| `fem.derive.v1.divergence-by-parts` | Rewrite `-div(k grad u) * v` as `k grad u . grad v` minus a boundary flux term. |
| `fem.derive.v2.boundary-discharge.zero-test-trace` | Discharge that boundary flux term because the test space vanishes on the essential boundary. Requires the essential boundary to cover the complete boundary of the relation scope. |
| `fem.derive.v1.source-pairing` | Rewrite `f` as `f * v`. |

A derivation that terminates with any strong-form node unconsumed, or that
applies a rule outside this table, fails admission.

**Slot convention.** The local matrix is indexed `[test][trial]`: row index is
the test slot, column index is the trial slot. The certificate records the slot
of every produced term. Because this operator is self-adjoint the convention is
numerically invisible, so it is checked structurally against the certificate,
never inferred from matrix symmetry.

### Role inventory

Admission enumerates and consumes exactly once: the single unknown field, every
parameter, every volume relation, and every boundary relation reachable from
the relation scope. `Domain`, `Representation`, and `Activation` nodes are
consumed as scope and space selection rather than as residual terms, and are
enumerated separately. A node consumed zero times or more than once fails.

### Quadrature policy

The Realization selects the rule; admission checks it and the certificate
records it. Selection is never inferred from the integrand.

| Term | Integrand degree per axis | Required declared exactness |
| --- | --- | --- |
| `k grad u . grad v` on an affine map | 2 | `>= 2` |
| `f * v` with a non-polynomial source | not polynomial | `>= 3`, recorded as a bounded-error term |

`QuadratureRule::polynomial_exactness` is declarative evidence and is read as a
declaration, never as proof. A rule whose declared exactness is `None`, or below
the required value, fails admission. This makes under-integration a fail-closed
admission error rather than a tolerance question.

### Bounded compilation

| Bound | Limit |
| --- | --- |
| Expression DAG nodes per relation | 4096 |
| Derivative order | 1 |
| Integral terms per relation | 8 |
| Quadrature points per cell | 64 |
| Local DOFs per cell | 32 |
| Temporaries per local evaluation | 256 |

### Eligibility predicate

The compiled path is taken if and only if all hold: the relation scope is a
2D Cartesian box; the discretization is `ContinuousGalerkin` over
`HypercubeQ1Space`; there is exactly one scalar unknown; every boundary of the
scope carries an essential relation with closed unknown-independent scalar
spatial data; the Galerkin test is zero on that complete boundary; and every
admission gate above passes. Any other configuration takes the existing path
unchanged. The predicate is total and fail-closed: an unrecognized
configuration never falls through to the compiled path.

### Admission gates

All must pass before assembly; each fails closed.

| Gate | Required output |
| --- | --- |
| Static semantics | Replayed typed residual: every node carries dimension, shape, frame, and nominal support; every root matches relation scope. Consumes `KernelProgram::typed_relation_residual`. |
| Role assignment | Explicit inventory of unknowns, coefficient/frozen fields, parameters, volume relations, and boundary relations. Every semantic node consumed exactly once; ambiguity and leftover relations fail. |
| Derivation certificate | Closed versioned rule IDs binding exact source relation and node identities to each weak term: test pairing, negative-divergence integration by parts, volume terms, complete boundary disposition. |
| Realization compatibility | Exact test and trial space, reference cell, local DOF ordering, geometry map, quadrature policy. Declared quadrature exactness is recorded, never treated as inferred proof. |
| Bounded compilation | Limits on DAG nodes, derivative order, integral terms, quadrature points, local DOFs, temporaries, and generated work. |

Passing every gate makes the form **compiler-admissible**. That is not a
capability claim. Only the registered case below makes it verified.

### Reused contracts

`DiscreteSpace` / `HypercubeQ1Space` for basis tabulation; `AffineGeometryMap`
and `QuadratureRule` for geometry and integration; `ScalarSpatialExpression`
with its `evaluate_jvp` / `evaluate_vjp` for derivative actions;
`LinearizedRelation` as the existing linearization boundary;
`LocalContribution` as output. The public `ScalarEllipticCartesianModel` API is
preserved unchanged.

### Bounded claim

The exact `org.example.poisson` strong relation lowers through a private
proof-carrying 2D Cartesian Q1 Galerkin form, produces local matrix and RHS
contributions and paired residual/JVP/VJP actions on the CPU reference path,
and agrees with independent analytic, assembled, convergence, and conservation
oracles.

### Nonclaims

Source-level weak forms; arbitrary expressions; natural, Robin, or mixed
boundary conditions; vector, mixed, or nonlinear forms; simplex, high-order, or
adaptive spaces; native or JIT code generation; a public form IR; CUDA or MPI
execution; and any performance property.

### Stop condition

If the admitted subset requires a special case at essentially every term, stop
and report rather than widening toward a general weak-form IR.

**Correction, draft 3.** The original second sentence — proceed only if this
slice deletes or simplifies a hand-written path — is unmeetable by
construction, and was accepted before anyone checked that. The retained
evaluator serves every configuration the eligibility predicate excludes: other
dimensions, other spaces, finite volume. Nothing about a 2D Q1 slice can retire
it. Measured outcome: 1,347 insertions against 50 incidental deletions, with the
hand-written evaluator still live.

So amendment A1's audit-compression test is **not yet met**. This slice
increases the number of invariant-bearing hand-written formula sites rather than
reducing it, and deletes no implementation. What it establishes is that the
derivation is correct and independently falsifiable — not that it pays for
itself.

The replacement condition, which can actually be evaluated:

> Audit compression is demonstrated when the compiled path subsumes the whole
> domain of one hand-written evaluator so that evaluator is **deleted**, or when
> a second instance is added and shown to need only witness data. Until one of
> those holds, the abstraction is on probation and no further physics is built
> on it.

**Partly met, draft 4 — and an earlier draft of this paragraph overstated it.**

What holds: the derivation is runtime-dimensional over 1D–3D Cartesian Q1, and
the hand-written production evaluator is gone. `CartesianEllipticCell` delegates
and its `compiled` field is no longer an `Option`, so a fallback is structurally
impossible rather than merely unused. Both construction sites must produce either
an admitted form carrying a semantic certificate or a compiled witness-data form.

What does not hold: amendment A1 requires **all** of its five conditions, and one
of them is that *at least two hand-written implementations, or one primal/JVP/VJP
triple, are deleted*. One evaluator was retired. The design-sensitivity site
below survives. Reading "the formula-site count fell" as satisfying A1 is
selecting the clause that suits the result, and this RFC previously did exactly
that.

The accounting: production is **+104 / −103**, a net gain of one line. Total
volume did not fall. The gain is structural — one site instead of two — and that
is real but partial. Nobody should read this as evidence that the compiler
shrinks the codebase, because it has not.

The generic witness-data path does **not** carry a semantic certificate. Only the
admitted path does. That distinction is part of the claim boundary.

**What was not consolidated.** The claim is about the *primal* evaluator.
`linearize_scalar_elliptic_cartesian_fem` still tabulates basis gradients and
forms its load term directly, so the design-sensitivity path remains a
hand-written invariant-bearing site with its own arithmetic. It was outside this
slice and is not covered by the certificate. Anyone counting formula sites should
count it: one was retired, one remains, and it is the obvious next target.

## Alternatives considered

**A public `eqiora-form-compiler` crate with a general weak-form IR.** Rejected.
It spends the abstraction budget before the hypothesis is tested, requires a
public compatibility promise for a design with zero consumers, and reopens the
universal-variational-form question that RFC 0007 settled on FVM grounds. A
private module can be deleted; a published IR cannot.

**Continue hand-written vertical slices per physics.** Rejected as the default,
retained as the fallback. It is mathematically safe and has produced correct
results, but it re-earns the oracle per physics — 22 files of repeated role
surface at four physics — and under agent authorship the review cost of that
growth is the binding constraint. If this slice fails its stop condition, this
alternative is what Eqiora continues with, and that outcome is informative.

**Generate from the existing `calculus/` proof seam.** Rejected for this slice.
That module is a bounded tensor-calculus and support-map checker whose standard
operators are only `SymmetricPart` and `IsotropicLift`; it has no quadrature,
basis, or DOF concept and explicitly excludes general weak forms. Extending it
would enlarge a central proof surface before the hypothesis is tested.

## Compatibility and migration

No public API changes. `ScalarEllipticCartesianModel` keeps its signature.

Migration is shadow-first: the derived evaluator is routed through
`CartesianEllipticCell::evaluate` for exactly the Q1 slice, and the existing
closed-form path is retained as test-only shadow code. Every local contribution
and the final CSR and RHS are compared against the current path before the old
path is considered for removal. Numerical agreement alone does not authorize
removal — the derivation certificate and mutant gates must also hold.

`lower_cartesian_q1_diffusion_local_action` continues to project the matrix
portion to `LocalLinearActionIr`, unchanged in meaning.

## Verification

The oracle below is **pre-committed**: it was derived analytically from the
bilinear basis before any implementation was read, and per amendment A4 the
implementing agent may not author, tune, or relax it.

### Reference values

Node ordering is **production bit ordering**, matching
`HypercubeQ1Space::tabulate` and `CartesianMesh::entity_vertices`, which both
index a vertex by its axis bits `(v >> axis) & 1`:

```
n0 = (x0,      y0     )   bits 00   lower-left
n1 = (x0 + hx, y0     )   bits 01   lower-right
n2 = (x0,      y0 + hy)   bits 10   upper-left
n3 = (x0 + hx, y0 + hy)   bits 11   upper-right
```

Production ordering is retained and the oracle is expressed in it. Introducing a
permutation at the evidence boundary would add a place for an off-by-one to
hide. With `alpha = hy/(6 hx)`, `beta = hx/(6 hy)`:

```
K = k * ( alpha * [[ 2,-2, 1,-1],     +  beta * [[ 2, 1,-2,-1],
                   [-2, 2,-1, 1],                [ 1, 2,-1,-2],
                   [ 1,-1, 2,-2],                [-2,-1, 2, 1],
                   [-1, 1,-2, 2]]                [-1,-2, 1, 2]] )
```

Unit square, exact rationals, `6*K`. The diagonal partners are `(0,3)` and
`(1,2)`, which is the ordering's signature:

```
[ 4 -1 -1 -2 ]
[-1  4 -2 -1 ]
[-1 -2  4 -1 ]
[-2 -1 -1  4 ]
```

Element fixture cell — **non-square, off-origin**:
`x0=0.25, y0=0.5, hx=0.25, hy=0.5, k=1`. `hx != hy` separates the two derivative
contributions so a swapped `alpha`/`beta` cannot survive; `x0,y0 != 0` prevents
a source bug from hiding behind symmetry.

```
K = [ 0.83333333333333326 -0.58333333333333326  0.16666666666666666 -0.41666666666666663]
    [-0.58333333333333326  0.83333333333333326 -0.41666666666666663  0.16666666666666666]
    [ 0.16666666666666666 -0.41666666666666663  0.83333333333333326 -0.58333333333333326]
    [-0.41666666666666663  0.16666666666666666 -0.58333333333333326  0.83333333333333326]
```

Exact load for `f = 2 pi^2 sin(pi x) sin(pi y)`, by separable analytic
integration rather than quadrature:

```
F = [0.42549571438121403, 0.47482060177589164, 0.24287139083576761, 0.2710258553802215]
```

Exact load for constant `f = 1`: `hx*hy/4 = 0.03125` at every node.

### The n-dimensional generalization, draft 4

The 2D form above is the `d = 2` case of

```
K_ij = sum_a (1/h_a) * s_a * prod_{b != a} (h_b * m_b)
  s_a = +1 if i_a == j_a else -1
  m_b = 1/3 if i_b == j_b else 1/6
```

over the tensor-product basis, with the vertex index as its axis bit pattern.
This reproduces the frozen 2D matrix in exact rationals.

3D fixture, `hx=1/4, hy=1/2, hz=1/8, k=1`. All three edge lengths differ so no
axis permutation survives. **`192 * K` as exact integers** — 192 is the least
common multiple of the entry denominators, computed rather than guessed:

```
[  28,   6,  12,   2, -18, -13, -10,  -7]
[   6,  28,   2,  12, -13, -18,  -7, -10]
[  12,   2,  28,   6, -10,  -7, -18, -13]
[   2,  12,   6,  28,  -7, -10, -13, -18]
[ -18, -13, -10,  -7,  28,   6,  12,   2]
[ -13, -18,  -7, -10,   6,  28,   2,  12]
[ -10,  -7, -18, -13,  12,   2,  28,   6]
[  -7, -10, -13, -18,   2,  12,   6,  28]
```

Constant-source load: `prod(h)/2^d = 1/512` at every node.

Draft 3 published this table scaled by 144, where the entries are not integers.
Integer truncation turned `9/2` into `4`, and the published row sum was `1/144`
rather than zero — a matrix that is not a Q1 stiffness matrix at all. The
implementing agent refused to build against it and derived the correct values
independently, which is the only reason it was caught. The formula and the
underlying rational arithmetic were always right; the presentation was not. This
is why amendment A4 forbids an implementer from fitting an implementation to an
oracle, and why the dual-oracle gate now exists.

### Structural invariants

Every row of `K` sums to zero; `K` is symmetric with positive diagonal; the load
vector sums to the exact cell integral of the source; in 2D, `alpha` and `beta`
depend only on the aspect ratio. These hold in 1D, 2D, and 3D and are asserted
on exact rationals before any scaling is published.

### Tolerances, step, and direction

Fixed here so that no tolerance is chosen by the implementer.

| Comparison | Tolerance |
| --- | --- |
| `K` against the closed form | `1e-14` relative. The integrand is exactly degree 2 per axis on an affine map, so an admitted rule integrates it exactly and only floating-point error remains. |
| `F` against the analytic load | `2.5e-2` relative. The source is non-polynomial, so this is a bounded-error term. Measured error is `1.3e-2` for exactness 3 and `6.6e-1` for exactness 1, so the band admits the declared rule and rejects under-integration by a factor of 26. |
| Constant-source `F` | `1e-14` relative; that integrand is exactly bilinear. |
| Shadow CSR and RHS against the existing path | `1e-15` relative, entry by entry, with identical sparsity pattern. |
| JVP against centered differences | `1e-7` relative. |
| VJP against the scalar directional derivative | `1e-7` relative. |

Centered-difference step: `h = 1e-6`. Truncation is `O(h^2) ~ 1e-12` and
round-off is `O(eps/h) ~ 2e-10`, so the `1e-7` band clears the noise floor by
more than two orders of magnitude while still rejecting a corrupted column.

Direction vector: `v_i = sin(i + 1)` over the local or global index, normalized
to unit L2. Deterministic, dependent on no RNG, and non-degenerate in every
component so a single zeroed column cannot hide.

### Mutant set

Each mutant must be rejected by a **named** gate. A mutant that survives means
the gate set is incomplete and is itself a reportable defect.

| # | Mutation | Rejecting gate |
| --- | --- | --- |
| M1 | Flip the integration-by-parts sign | derivation certificate; sign disagreement against `K` |
| M2 | Drop one essential boundary relation | role assignment — every node consumed exactly once |
| M3 | Duplicate a boundary relation on one domain | role assignment — ambiguity fails rather than silently resolving |
| M4 | Swap `alpha` and `beta` | element fixture on the non-square cell; invisible on a square cell |
| M5 | Swap test/trial gradient indices | certificate slot convention, checked structurally; numerically invisible for this symmetric operator |
| M6 | One-point quadrature for the source term | quadrature admission — declared exactness below the required value fails before assembly; the load comparison confirms it |
| M7 | Scale the flux coefficient | element comparison; also breaks the balance residual |
| M8 | Reclassify `source_scale` as an unknown | role assignment |
| M9 | Corrupt one Jacobian column | JVP against independently rebuilt centered differences |
| M10 | Return the primal residual as the VJP | VJP against a scalar directional derivative — **not** JVP/VJP duality, which is self-referential |
| M11 | Omit the source term in assembly | conservation: boundary reactions plus integrated source |
| M12 | Off-by-one in local-to-global DOF mapping | shadow comparison of final CSR and RHS |

M5 and M10 are precisely the mutants a generated-versus-generated check would
pass. They are why a hand-derived oracle must exist independently.

### Registered evidence

The bounded claim splits into two evidence items because they cannot share a
mesh. The ordinary generated Realization builds `vec![cells; dimension]` over
the package's unit square, so every cell it produces is square — a non-square
cell is unreachable through the package path, and a square cell cannot
distinguish `alpha` from `beta`.

**E1 — element fixture** (focused tests in `eqiora-numerics`). Constructs the
non-square off-origin cell directly, without a package or a Realization, and
compares `K`, both load vectors, the certificate rule sequence and slots, and
the JVP/VJP actions against the values above. This item owns M1, M4, M5, M6,
M7, M9, and M10.

**E2 — package path**
(`verify/numerics/compiled-cartesian-poisson-q1-2d/` and
`crates/eqiora/tests/compiled_package_poisson_form.rs`). Compiles the actual
`org.example.poisson` package through the ordinary explicit Q1 Realization on
its unit square. Shadow-compares every local contribution and the final CSR and
RHS against the existing path, replays the four-level continuous-L2 convergence
floor of 1.9 and the relative balance limit of 2e-11, and runs the eligibility
predicate. This item owns M2, M3, M8, M11, and M12.

Neither item alone establishes the claim. E1 without E2 proves an element
nobody reaches; E2 without E1 cannot see a swapped aspect ratio.

Both must explicitly run `numerics.cartesian-poisson-fem-fvm`,
`differentiation.spatial-poisson-fem-fvm`, and
`numerics.global-matrix-free-action`.

## Security, safety, and governance

No unsafe code, no new trust boundary, no irreversible action. Compilation is
bounded in DAG size, derivative order, quadrature points, local DOFs, and
temporaries, so a malformed or adversarial package cannot induce unbounded work.

Governance: the implementing agent must not author, tune, or relax the oracle,
expected values, tolerances, or falsifiers for its own implementation
(amendment A4). Wiring a pre-committed fixture is permitted; owning the evidence
content is not. The integrator role is per slice and runs the affected closure.

## Unresolved questions

These are deferred deliberately. None of them blocks implementation: each has a
defined behaviour for this slice, and the open part concerns a later one.

- Which second physics consumer best falsifies the audit-compression claim —
  the elasticity patch, or the thermal slab? The elasticity patch exercises
  vector fields and is the stronger test, but is a larger step.
- Should the derivation certificate be persisted in the artifact wire, or
  remain an in-process proof? It stays in-process for this slice; persisting it
  would create a compatibility promise the slice does not need.
- Whether the shadow path is deleted at the end of this slice or after the
  second consumer. It is retained for this slice; deleting it early removes the
  strongest available oracle.

## Revision history

- **Draft 2** (`2026-07-26`). Re-frozen after the contract review returned eight
  blocking defects, all genuine. Draft 1 specified no bounded-compilation
  limits, no certificate rule IDs or slot convention, no role-inventory scope,
  no tolerances or finite-difference step, and left quadrature policy in
  Unresolved questions while requiring it as an admission gate — so an
  implementer would have had to author the M1, M5, and M6 gates itself, which
  amendment A4 forbids. Draft 1 also stated the oracle in counter-clockwise
  ordering while production uses bit ordering, and demanded a non-square cell
  from a package path that can only produce square ones. The oracle values are
  re-derived in production ordering; the numerical content is unchanged.
