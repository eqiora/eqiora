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


## Strict-interior nonlinear execution

A finite real-scalar nonlinear equality system can instead use `solve.Newton` with
`solve.StrictInterior(margins=...)`. Every authored inequality requires a positive margin in
its own physical dimension. Execution accepts only points whose original inequality slack is
strictly greater than that margin; complementarity is excluded. These numerical margins
restrict the study domain without changing the Model's non-strict inequalities.

For `w*w=p`, author `inequality(p>=0)` and `inequality(w>=0)`. Margins of `1e-8` restrict
execution to `p>1e-8` and `w>1e-8`; they do not admit every arbitrarily small positive p.
`p=0` rejects before Newton, even if a small approximate positive w would satisfy a residual
tolerance. The numerical seed is supplied with
`State.initial(plan, fields=(InitialField(model.field("w"), scalar_value=1.0),))`.
Every exact Field needs one finite coherent-SI scalar value. Spatial associations, foreign
Fields, duplicates and omissions reject. A source `initial` equation is not a numerical seed.

Newton uses the existing scalar Operator IR's exact AD and the Plan's exact
SparseLU/Identity/Fast provider policy. It requires strictly improving or converged interior
trial points within the requested iteration and backtracking budgets. Before publishing a
Result, the original equations and inequalities are reevaluated and the accepted-point AD
Jacobian must have full rank, even if the seed already has zero residual. This is local
regularity of the represented Jacobian, not a global uniqueness or branch-tracking theorem.

Plan v6 retains Newton controls and margins; finite State v2 binds the complete numerical seed;
Result v7 retains that State, original residual acceptance and distinct nonlinear/linear
records. Replay rechecks original conditions and local regularity. A zero-update record must
retain an already accepted seed exactly. Python `result.solve` returns `NonlinearSolveSummary`
for this profile, including nonlinear iteration count and residual bounds. This primal
lifecycle does not yet expose reduced-solution derivatives, global branch selection,
complementarity derivatives or spatial nonlinear constraints.
