# Tensor contractions and local maps

These [target-language](core.md) rules specify bounded tensor and local-map operations.
The current spatial profile executes full-coordinate ranks one through four via
`component`, `permute_axes`, `transpose`, `outer`, `contract`, `matrix_trace` and
`componentwise_product`, plus the three-vector `cross`. Uniform real/complex source evaluation is independently
checked by [the tensor contraction case](../../verify/language/tensor-contractions/README.md).
Finite-map trace and identity, and real determinant/inverse with composition execute through the
ordinary source/native/Python Model/Plan/Result path. Hermitian `inner` remains a specified target.

## Axes and construction

Tensor axes are ordered and zero-based. Every axis retains its exact spatial frame or admitted
component-space role. Equal extent is not a frame conversion. `tensor<Pa, 2, 2, 2, 2>` uses
full tensor coordinates, not a compressed symmetric matrix. A symmetric numerical sample
does not change its declared type or eliminate coordinates.

`tensor_value(frame = frame_ref, components = nested_array)` explicitly constructs a spatial
tensor. Array nesting fixes axis order and extent; the final axis varies fastest. Components
must have one compatible scalar type and a frame-compatible shape. The frame reference is
exact caller-bound mathematical context, not a string. No array becomes a spatial tensor
without this constructor or an equally explicit typed external binding.

The current coefficient-authoring profile uses `frame = body`, where `body` is an
exact admitted Cartesian support. It projects the existing model-global frame and
uses the support's ambient dimension; the uniform Parameter retains no spatial
support. Native `DraftParameter.with_frame` and Python `Parameter(frame=...)`
retain the same explicit support reference during source construction. Components
must be closed scalar expressions; named model values, including Parameter aliases, reject.
Arithmetic using already framed Parameters retains its ordinary expression graph. Arbitrary
local frames remain a separate capability. Nested component arrays admit ranks one
through four and must have the ambient extent on every axis.

`component(T, indices = (i, j, ...))` returns the selected scalar. It requires one bounded
integer per axis. `permute_axes(T, order = (...))` explicitly reorders axes and retains their
roles. A permutation contains every axis exactly once. Neither operation changes dimensions.

`outer(A, B)` concatenates the axes of `A` followed by those of `B` and multiplies their
component values, without conjugation. `contract(A, B, axes = ((a, b), ...))` sums over the
listed pairs of axes. A pair identifies an axis of `A` and an axis of `B`; contracted axes
must have equal extents and compatible frame/dual roles. An axis may occur only once.
The result orders uncontracted axes of `A` first, then those of `B`, preserving each order.
The contraction multiplies physical dimensions and never conjugates implicitly.

`cross(a, b)` is the bilinear cross product of two spatial Cartesian three-vectors
in the model-global right-handed frame, with `(e0, e1, e2)` positively oriented.
Its components are `(a1*b2-a2*b1, a2*b0-a0*b2, a0*b1-a1*b0)`; exchanging operands
reverses the sign. Physical dimensions multiply. Real and complex operands share
this rule without implicit conjugation. Two-dimensional vectors, channel arrays,
nominal coordinate spaces and foreign supports reject; no implicit embedding or
local-frame conversion occurs. Python authoring uses `eqiora.lang.cross(a, b)`.
The retained pure-operator graph supplies evaluation and differentiation. This
algebraic operation alone supplies no curl, oriented trace or compatible finite element.

`inner(A, B)` instead contracts all corresponding components after conjugating the first
operand. The two tensor shapes and axis roles must agree. Rank-two `transpose(T)` swaps its
two axes without conjugation. Use `matrix_trace(T)` for the algebraic diagonal contraction;
`trace(T, on = boundary)` remains a boundary restriction and is never selected by rank.

Scalar multiplication is distinct from componentwise tensor multiplication. Explicit
`componentwise_product(A, B)` requires matching shapes and roles and returns their component
products; it is neither a matrix product nor a contraction. There is no broadcasting or
Einstein summation triggered by repeated names.

## Complete rank-four constitutive specimen

```eqiora
model ElasticResponse(
  support body: volume(ambient_dimension = 2),
  input stiffness: tensor<Pa, 2, 2, 2, 2> on body,
  input strain: tensor<1, 2, 2> on body,
  output stress: tensor<Pa, 2, 2> on body
) {
  relation constitutive on body {
    stress = contract(stiffness, strain, axes = ((2, 0), (3, 1)));
  }
  observable energy_density: Pa on body = 0.5 * inner(strain, stress);
}
```

Bind a fixed 2D physical support and its exact Cartesian orthonormal frame. Bind uniform real
fields with the following stiffness components in that frame; all components not listed are
exactly zero. No spatial solve, boundary condition, or initialization is required for this
constitutive evaluation. A later elasticity model supplies displacement kinematics, force
balance, and its own boundary/initial conditions.

| Components, using zero-based indices | Value |
|---|---|
| `C0000` | 10 Pa |
| `C1111` | 20 Pa |
| `C0011`, `C1100` | 3 Pa each |
| `C0101`, `C0110`, `C1001`, `C1010` | 4 Pa each |

The explicitly expanded equations, independent of the contraction implementation, are:

```text
stress00 = 10 Pa * strain00 + 3 Pa * strain11
stress11 = 3 Pa * strain00 + 20 Pa * strain11
stress01 = 4 Pa * strain01 + 4 Pa * strain10
stress10 = 4 Pa * strain01 + 4 Pa * strain10
```

For pure shear `strain01 = strain10 = 0.01` and zero normal components, both shear stresses
are 0.08 Pa and energy density is 0.0008 Pa. The engineering shear is 0.02, not 0.01.
Dropping one off-diagonal contribution produces the wrong stress and energy. For normal
strain `strain00 = 0.01`, `strain11 = 0.02`, with zero shear, stresses are 0.16 Pa and
0.43 Pa and energy density is 0.0051 Pa.

This binding has the stated minor and major symmetries, but the source operation accepts a
full tensor and proves none of them from its name. The energy expression is the intended
elastic potential for this symmetric binding; an arbitrary nonsymmetric replacement cannot
inherit that physical interpretation merely because the contraction type-checks.

The initial source profile exposes full coordinates only. A future compressed symmetric
representation must name its mapping explicitly. Engineering shear, Mandel, and Voigt
coordinates cannot share an untagged array or an implicit factor-of-two conversion.

## Local endomorphisms and inverse action

For a finite map on one exact space, `matrix_trace(A)` sums diagonal coefficients and retains
their dimension. `determinant(A)` has coefficient dimension raised to the space extent.
`inverse(A)` swaps source and target and reciprocates coefficient dimensions. `apply(inverse(A), x)`
is an inverse action, not a source-level choice of numerical solver. Trace and determinant
reject a map between distinct spaces unless a separately admitted identification is explicit.

```eqiora
space Channels = orthonormal(first, second);
let A: map<1, Channels, Channels> = linear_map(Channels, Channels, [[2, 3], [5, 7]]);
let x: coordinates<V, Channels> = coordinates(Channels, [11 [V], 13 [V]]);
let y: coordinates<V, Channels> = apply(A, x);
let recovered: coordinates<V, Channels> = apply(inverse(A), y);
```

Direct row multiplication gives `y = [61 V, 146 V]`. The determinant is -1 and the inverse
matrix is `[[-7, 3], [5, -2]]`, recovering `[11 V, 13 V]`. The unequal off-diagonal entries
detect accidental transposition. The algebraic trace is 9, not a boundary field.

For a direction `H = [[1, 0], [0, 0]]`, direct differentiation of the 2x2 inverse formula gives
the inverse derivative `[[-49, 21], [35, -15]]`. This agrees with `-inverse(A)*H*inverse(A)`
using map composition, and provides a separate directional check rather than relying only
on `A*inverse(A) = identity`.

The numerical reference owner uses scaled row-pivoted LU, rather than expanding the determinant
or inverse into a factorial-size scalar expression. There is no special 4×4 cutoff. Focused
independent checks include 6×6 constitutive maps and coupled 9×9 and 16×16 maps, with inverse
JVP/VJP products; these are finite matrix sizes, independent of physical-space dimension.
`identity(S)` is dimensionless and retains the exact declared basis, including explicit duality.
It defaults to real coefficients and also admits an explicitly declared complex map context.
Algebraic trace preserves either real or complex coefficients without conjugation.

A singular matrix such as `[[1, 2], [2, 4]]` has no admitted inverse. Execution does not substitute
a pseudoinverse, diagonal shift, or truncated spectrum. The binary64 reference profile admits an
inverse only when its computed reciprocal infinity-norm condition estimate exceeds 64 machine
epsilons. A zero numerical pivot reports singularity or unresolved arithmetic; factorization
success is not an exact regularity proof. The determinant remains differentiable at singular
matrices through its cofactors. An unrepresentable underflowed determinant reports a numerical
failure instead of claiming mathematical singularity. Normalization also rejects if it would
round a nonzero coefficient to zero. Cofactor evaluation avoids accepting a zero derivative
caused only by underflow in an intermediate determinant/inverse product.

Resource admission remains separate from the mathematical operators: scalarization charges the
retained coefficient inventory against its existing component-work budget, and numerical
factorization bounds its work before allocation. Full inverses currently reuse a factorization
within one scalar row, not across all output rows. Complex inverse/determinant execution,
general provider selection for these local operations, and spatial finite-coordinate solves
remain outside this profile.

Reject repeated/out-of-range contraction axes, mismatched frames, foreign same-sized spaces,
implicit symmetric compression, and unsupported rank or extent before allocation/evaluation.
The initial bounded tensor profile supports ranks through four; its concrete element-count
limits belong to the common resource profile, not a per-material exception.

Pure tensor compositions preserve one exact Cartesian volume or boundary support.
On a boundary, tensor extents use the parent ambient dimension, not the boundary's
measure dimension. Operands on different boundaries, different parents, or a volume
and its boundary do not become interchangeable; apply explicit trace or normal
operations first. Conserving-interface supports remain outside this pure profile.

## Oriented Cartesian calculus

`curl(u)` uses a right-handed Cartesian frame and the derivative axis last:
`grad(u)[i,j] = ∂u_i/∂x_j`. In three dimensions it returns
`(∂y u_z − ∂z u_y, ∂z u_x − ∂x u_z, ∂x u_y − ∂y u_x)`.
In two dimensions, vector curl returns the scalar `∂x u_y − ∂y u_x`,
and scalar curl returns `(∂y f, −∂x f)`. Shape and ambient dimension select
these explicit conventions; a 3D scalar curl, channel array, or nominal basis
vector does not silently embed into physical space. Curl divides the operand's
unit by length and retains its exact volume support.

On an exact boundary of that volume, `tangential_trace(u)` means `n × u`
in 3D and `n_x u_y − n_y u_x` in 2D. This oriented trace differs from the
unoriented tangential projection `u − n(n · u)`: reversing the normal reverses
the trace. It retains the operand's unit and acquires the exact boundary support.
Equal coordinates or equal box sizes cannot substitute another parent or face.
Python authors the same expressions with `eqiora.lang.curl` and
`eqiora.lang.tangential_trace`.

Source Model expressions may select a boundary explicitly with
`trace(u, on=wall, from=body)`, `normal(u, on=wall, from=body)`, or
`tangential_trace(u, on=wall, from=body)`. `on` names an exact boundary Domain;
optional `from` must name its exact parent volume. Omitting `on` uses the owning
Relation's boundary scope. An explicit `on` also permits a trace in a `let`
outside a Relation, but consuming it in an equation still requires the same
boundary support. The retained trace/normal node carries the target identity.
These selectors do not introduce physical interfaces, continuity or flux balance.

Authored weak forms retain `curl`, `cross`, `normal`, and `tangential_trace` as typed
operators. Trace, normal and tangential trace accept the same exact `on`/`from`
selectors; their retained target must equal the integral support. H(div) tests
admit a normal trace, while a full trace still requires H1. H(curl) admits a
tangential trace; L2 admits neither boundary trace. A Cartesian Model Domain supplies its own ambient dimension; one
physical vector trial is admitted without a separate Geometry binding.
`tangential_trace` requires integration on an exact boundary of the operand's
support. Complex forms use explicit `inner` pairings to retain conjugate-linear
test dependence; `cross` itself remains bilinear. Current authored-form artifacts
use `eqiora.authored-form/v15`; previous epochs are rejected. This authoring and
replay support does not by itself establish strong/weak correspondence, discharge
boundary conditions, or select a numerical vector-space realization. A one-trial vector
form can describe only one equation of a coupled Model; it is not a checked
mixed system. Mixed correspondence still requires the complete velocity/pressure
equation and test inventory.

The bounded exact weak-residual comparator expands first-gradient curl and 3D
cross products through the same pure component definitions as the Model calculus.
It requires physical operands on the trial's exact Cartesian volume; foreign
supports and nominal vector frames reject before algebraic cancellation. Curl
operands in this profile are direct Fields, tests, or variation directions.
Focused tests compare 2D/3D curl-energy first and second variations against
independently expanded antisymmetric gradient pairs, and complex cross pairings
against signed component rows with explicit conjugation.

Boundary-term component comparison also expands direct-field `tangential_trace`
through the shared tangential lift used by Model `normal`. Each formal outward-normal
component carries its exact boundary identity. The active integral must select a
boundary of the trial's volume; a foreign parent, volume measure, missing boundary
scope, or a typed Model normal from another boundary rejects before cancellation.
Independent 2D/3D real and complex component tests check the signed rows and explicit
conjugation. This is a local boundary-term comparison, not a curl integration-by-parts
certificate or an executable vector boundary condition.

For a planar scalar field, `dot(curl(eta),curl(u))` equals
`dot(grad(eta),grad(u))`. This pairing reaches the existing scalar Q1 Poisson
correspondence, Plan, solve, and Result replay. The unit-square test with four
cells, unit source, and zero essential conditions has the independently derived
central value `3/32`. This scalar realization does not provide vector curl-curl
admission, curl integration-by-parts boundary
discharge, or compatible edge elements.

For a 2D scalar trial, a retained strong `curl(curl(u))` also reaches this
Q1 path. Its two exact shared curl definitions establish
`curl(curl(u)) = -div(grad(u))`; the correspondence records the distinct
`fem.derive.v1.planar-scalar-curl-curl-by-parts` rule and retains the original
source occurrence. Focused real/complex tests replay the Plan and Result and
check the unit-square coefficient `3/32`, or `(3/32)(1+2i)` for load `1+2i`.
Reversing only the strong operator or weak stiffness fails correspondence.
This reduction requires the direct scalar composition; a sign inserted between
curls, vector reductions and 3D curl-curl are not silently identified with it.
It does not provide the general vector curl integration-by-parts certificate.

Spatial weak tests default to H1. A declaration such as `test v:1 for u in hcurl;`
selects a continuum regularity hypothesis independently of any element or mesh.
The closed labels are `h1`, `hcurl`, `hdiv`, and `l2`; global finite tests have no
spatial regularity label. H(curl) and H(div) tests require physical vectors.
The weaker profiles admit value pairings and direct first curl or divergence,
respectively; L2 admits value pairings. Full `zero_on` traces require H1.
Products inside derivatives and higher derivative compositions remain outside
these weaker profiles. Canonical projection decoding rechecks these restrictions;
the declaration does not prove regularity of a solution or admit a numerical space.
Python `Component.test(..., regularity="hcurl")` emits the same declaration.

For a real or complex physical 3-vector, the explicit Rust inspection
`eqiora_numerics::check_authored_spatial_formulation(program, projection)` checks a
bounded strong-implies-weak correspondence without selecting a numerical method.
It recognizes one direct shared `curl(curl(u))` occurrence on a Cartesian box,
with signed linear algebraic value terms. Each of the six exact box faces must
carry either a homogeneous full-trace law for `u` or the homogeneous natural
law `tangential_trace(curl(u)) = 0`. The test's `zero_on` inventory must
match exactly the full-trace faces; natural faces leave the test unrestricted. The curl Green identity is
`∫Ω v·curl(curl(u)) = ∫Ω curl(v)·curl(u) − ∫∂Ω (n×v)·curl(u)`.
The full zero test trace or `n × curl(u) = 0` discharges its surface term.
When every face is natural, the conditional Green correspondence also admits
H(curl) tests; full-trace faces retain the stronger H1 requirement.
The latter uses `(n×v)·curl(u) = −v·(n×curl(u))`; a zero normal component
`n·curl(u) = 0` is insufficient and rejects. The shared tangential lift and
curl definitions bind that distinction to the retained source nodes. Complex fields use
`inner(curl(v),curl(u))` and conjugated test value pairings. The checker
reuses live source identities, correspondence replay, variation authentication,
unit checking, and exact polynomial comparison; it does not sample field values.
This explicit inspection leaves general vector authoring available. Tangential-only
essential laws, nonzero natural curl data, reverse implication, uniqueness, vector numerical
admission, compatible edge elements, and Maxwell execution remain outside it.

For twice continuously differentiable fields in a fixed Cartesian frame,
`div(curl(u)) = 0`, `curl(grad(f)) = 0`, and
`curl(curl(u)) = grad(div(u)) − div(grad(u))` in 3D. Mixed partials commute
under this regularity assumption. The Laplacian remains `div(grad(.))`;
these equalities do not assert that a finite-element or finite-volume
representation preserves the differential identities.

A spatial directional derivative contracts the coordinate axis of the gradient
with an explicitly declared physical direction whose Cartesian frame and spatial
support are compatible. For a scalar `f`, use `contract(grad(f),a,axes=((0,0),))`;
for a vector `u`,
use `contract(grad(u),a,axes=((1,0),))`, since the derivative axis is last.
The direction is not normalized implicitly: dimensions are `[f][a]/length`
or `[u][a]/length`. A velocity direction therefore gives a rate, while a
dimensionless direction gives a derivative per length. This physical-coordinate
contraction is distinct from a variation with respect to a function-space trial.

For smooth 3D fields and an outward-oriented, piecewise smooth boundary,
the bilinear integration-by-parts pairing is
`∫Ω curl(u) · v = ∫Ω u · curl(v) + ∫∂Ω (n × u) · v`.
Each side has units `[u][v] length²`. A Hermitian pairing must explicitly
conjugate the corresponding factor. Weak extensions require admitted trace
and regularity spaces; this syntax alone does not provide those spaces.

The reference evaluator admits real explicit coordinate polynomials on Cartesian
volumes, including nested gradient/divergence/curl compositions. The admitted
arithmetic is constants, Parameters, coordinates, addition/subtraction, multiplication,
nonnegative integer powers and polynomial pure component maps; division and
nonpolynomial functions are outside this profile. Parameters are read from the
actual evaluation bindings. Native `EvaluationPoint` also admits
Cartesian faces using the complete parent coordinate inventory and the exact
face position; normal orientation is distinct from a one-sided limit request.
This profile does not reconstruct unknown spatial Fields, evaluate complex
spatial polynomials, supply general boundary charts, solve Maxwell equations,
or provide compatible edge elements. Unsupported expressions reject explicitly.

## Finite Hamiltonians and declared evolution conditions

A finite Hamiltonian is an energy-valued linear map on a declared orthonormal basis.
For example, `H:map<complex<J>,S,S>` acts on dimensionless
`psi:coordinates<complex<1>,S>`. The evolution equation keeps the physical conversion
explicit: `derivative(psi)=math.complex(0,-1)/hbar*apply(H,psi)`, with
`hbar:J*s`. Finite map sums, compositions, adjoints and tensor products can be bound
as Parameters through the same typed component IR used during execution.

For a closed system, declare both conditions on its ordinary time policy:

```python
psi = model.field("psi")
H = model.parameter("H")
policy = policy.with_hermitian_parameter(H).with_conserved_norm(
    [psi], target=1.0, tolerance=1e-10, dimension=eqiora.Dimension()
)
plan = eqiora.resolve(model, temporal=policy)
```

Both `Tsitouras45` and `ImplicitMidpoint` expose these immutable policy methods.
The first checks the complete bound matrix before division by hbar or other
numerical scaling; tiny unequal conjugate entries cannot disappear before this
check. It validates the bound binary64 Parameter, not symbolic arithmetic before
its construction. The Parameter must belong to the exact Model. The same declaration can check a
Hermitian observable operator before taking the real part of its expectation.

The second declares the sum of squared magnitudes of complete selected Fields.
The target and tolerance have the squared physical dimension of those Fields.
Admission structurally proves a homogeneous autonomous linear generator and its
selected norm identity; sampling the vector field does not establish linearity.
Every accepted `State` must satisfy the tolerance. No value is normalized or
projected to make it pass. Output interpolation can fail this condition even when
accepted integration steps preserve the norm; choose suitable output boundaries
and tolerances explicitly. These conditions survive Plan/State replay and cannot
be mixed with references from a different Model.

Current conservation admission does not prove nonlinear or time-dependent
invariants, event resets or forward-sensitivity directions. Such requests fail
explicitly; ordinary time evolution without these declarations retains its wider
admitted scope. Finite basis extent is independent of physical-space dimension;
there is no two-level or 4×4 bound in these conditions. Focused product tests cover
a two-level example and equivalent explicit/composed 2×3-factor Hamiltonians.
The [registered finite-Hamiltonian case](../../verify/time/finite-hamiltonians/README.md)
proves the stated two-level and 2×3 spectral/Cayley claims. Python global-phase and
explicit basis-change invariance have separate focused product tests.
