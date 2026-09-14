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

These conditions are Model meaning. A numerical realization must separately name every
inequality and complementarity condition, provide coherent-SI operand tolerances, and bound its
complete active-set enumeration. Solver success is insufficient: execution re-evaluates the
original equality, ordering, signs and complementary zero operand before publishing a Result.
The selected active/inactive state and measured original operands remain inspectable after exact
Plan and Result replay.

The current executable profile is finite, nonspatial, affine and real-scalar. It does not infer
a penalty, clipping, smoothing or differential inclusion. Geometric contact search, spatial
contact assembly, friction, nonlinear constraints and augmented or penalty formulations require
separate explicit formulations.
