# Tensor contractions and local maps

These [target-language](core.md) rules specify bounded tensor and local-map operations.
The current spatial profile executes full-coordinate ranks one through four via
`component`, `permute_axes`, `transpose`, `outer`, `contract`, `matrix_trace` and
`componentwise_product`. Uniform real/complex source evaluation is independently
checked by [the tensor contraction case](../../verify/language/tensor-contractions/README.md).
The local inverse-map and Hermitian `inner` profiles below remain specified targets.

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

A singular matrix such as `[[1, 2], [2, 4]]` has no admitted inverse. Execution may not replace
it with a pseudoinverse, diagonal shift, or truncated spectrum. An ill-conditioned invertible
map has a separate numerical admission/failure policy; a successful factorization is not an
exact proof of mathematical regularity. Complex inverse derivatives use the shared real-linear
derivative owner when that profile is admitted.

Reject repeated/out-of-range contraction axes, mismatched frames, foreign same-sized spaces,
implicit symmetric compression, and unsupported rank or extent before allocation/evaluation.
The initial bounded tensor profile supports ranks through four; its concrete element-count
limits belong to the common resource profile, not a per-material exception.
