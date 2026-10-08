# Finite mathematical constraints

A `relation` owns an ordered list of mathematical conditions. Equality remains ordinary
`left = right` syntax. An inequality is explicit and non-strict:

```eqi
relation contact {
  2[N/m] * gap - force = load;
  complementarity(0[m] <= gap, force >= 0[N]);
  inequality(gap <= 4[m]);
}
```

`inequality(a <= b)` and `inequality(b >= a)` have the same retained orientation: the first
operand is greater than or equal to the second. Both operands must be invariant real scalars
with the same physical type and support. Strict inequalities are not mathematical constraint
syntax.

`complementarity(first, second)` takes two explicit nonnegativity predicates. The compiler
normalizes `0 <= value` and `value >= 0` to the nonnegative physical operand, then retains the
requirement that both operands are nonnegative and at least one is zero. The operands may have
different dimensions, such as a gap in metres and a force in newtons, but must have identical
support. A bare expression does not imply nonnegativity, and a reversed sign is rejected.

These conditions are Model meaning. An active-set numerical realization separately names every
inequality and complementarity condition, supplies coherent-SI operand tolerances, and bounds its
complete enumeration. Solver success is insufficient: execution re-evaluates the
original equality, ordering, signs and complementary zero operand before publishing a Result.
The selected active/inactive state and measured original operands remain inspectable after exact
Plan and Result replay.

The active-set executable profile is finite, nonspatial, affine and real-scalar. It does not infer
a penalty, clipping, smoothing or differential inclusion. Geometric contact search, spatial
contact assembly, friction, nonlinear constraints and augmented or penalty formulations require
separate explicit formulations.


## Finite nonlinear execution

A finite real or complex nonlinear equality system can use `solve.Newton`. Equality-only
systems need no enforcement policy. If inequalities are authored, supply
`solve.StrictInterior(margins=...)`. Every authored inequality requires a positive margin in
its own physical dimension. Execution accepts only points whose original inequality slack is
strictly greater than that margin; complementarity is excluded. These numerical margins
restrict the study domain without changing the Model's non-strict inequalities.

For `w*w=p`, author `inequality(p>=0)` and `inequality(w>=0)`. Margins of `1e-8` restrict
execution to `p>1e-8` and `w>1e-8`; they do not admit every arbitrarily small positive p.
`p=0` rejects before Newton, even if a small approximate positive w would satisfy a residual
tolerance. The numerical seed is supplied with
`State.initial(plan, fields=(InitialField(model.field("w"), value=1.0),))`.
Every exact Field needs one complete finite coherent-SI value. `value=` accepts real or
complex scalars and rectangular nested component arrays; shapes must match exactly, with no
broadcasting or reshaping. Units and nominal bases come from the exact Field. Imaginary
components in real Fields, spatial associations, foreign Fields, duplicates and omissions
reject. A source `initial` equation is not a numerical seed. This replaces the pre-1.0
`scalar_value=` argument. Native callers use `CommonInitialField::finite` with the exact
`ValueShape` and row-major real/imaginary component pairs.

Residual acceptance uses the real Euclidean norm of the normalized equality components.
By default, each equality uses one coherent-SI unit of its operand dimension. Override it
with `scaling={model.constraint("root", 0): (scale, Dimension(...))}` in `resolve`.
Scales must be finite and positive with the exact equality dimension; both parts of a
complex component use the same scale. The same scales divide the residual, unknown
Jacobian and Parameter partial rows, so Newton updates and acceptance agree. The Plan
retains every effective scale, including defaults, in canonical condition order. Wrong
units, foreign conditions, and nonrepresentable normalization reject. This normalization
does not imply an unknown-space metric or a physical energy norm.

Newton uses the shared component Operator IR's real-linear AD and the Plan's exact
SparseLU/Identity/Fast provider policy. It requires strictly improving or converged interior
trial points within the requested iteration and backtracking budgets. Before publishing a
Result, the original equations and inequalities are reevaluated and the accepted-point AD
Jacobian must have full rank, even if the seed already has zero residual. This is local
regularity of the represented Jacobian, not a global uniqueness or branch-tracking theorem.

Plan v10 retains Newton controls and margins; finite State v2 binds the complete numerical seed;
Result v12 retains that State, original residual acceptance and distinct nonlinear/linear
records. Replay rechecks original conditions and local regularity. A zero-update record must
retain an already accepted seed exactly. Python `result.solve` returns `NonlinearSolveSummary`
for this profile, including nonlinear iteration count and residual bounds. Accepted-point reduced sensitivities use the common differentiable Program. Global branch
selection, complementarity derivatives and spatial nonlinear constraints remain separate.
