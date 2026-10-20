# Conditional scalar values

`if condition then value else value` is a memoryless value expression. The condition must be
Boolean. Both branches are checked statically for value type, dimensions, shape, nominal
support, and activation, but evaluation computes only the selected branch. Both branches are
required. Comparisons use the exact mathematical endpoints, with no solver tolerance and no
automatically generated event.

Real scalar operators can use this for guarded square roots and piecewise laws:

```eqiora
operator guarded_root(input area: V^2): V =
  if area >= 0[V^2] then math.sqrt(area) else 0[V];

operator conductance(input voltage: V, input forward: S, input reverse: S): A =
  if voltage >= 0[V] then forward * voltage else reverse * voltage;
```

The guarded root returns 2 V at 4 V² and 0 V at -1 V², without evaluating the square root of a
negative number. With forward conductance 2 S and reverse conductance 0.5 S, voltages
`[-4, 0, 3] V` give `[-2, 0, 6] A`. Batches apply the same scalar evaluator to each row, with
fresh memoization and limits on total work and output; they never evaluate both branches and
choose afterwards.

In Python, write this expression as `q.if_else(condition, then_value, else_value)`, using
`q.quantity(value, unit)` for typed thresholds and branch constants. Python builds both branch
expressions, and numerical evaluation selects one. All three operands keep their Source and
formal ownership, so an inactive branch cannot capture a declaration from another source.

`math.abs`, binary `math.min`/`math.max`, `math.clamp`, `math.sign`, and `math.step` are built
on the same conditional, with the endpoints given in the [numeric catalog](numeric-catalog.md).
Binary extrema return their first argument on a tie. Finite IndexSet reductions use the same
comparison and selection nodes, preserving ordinal order and exact integer values, although
their source syntax is different. Clamp accepts equal bounds but rejects an inverted interval
before evaluating its value; this domain requirement remains part of the expression.

The native `ScalarOperatorIr::linearize_typed` computes a derivative at one finite real scalar
input point, with inputs in `symbols()` order. It keeps live input dependencies, evaluates only
the active branch, and returns the usual point-bound linearization. The conductance slopes are
0.5 S below zero and 2 S above zero; at the switching point the ordinary derivative is rejected.
The guarded-root slopes are 0.25/V at 4 V² and 0/V at -1 V²; its switching point is rejected
too, because a square root that is active at zero has no ordinary derivative there.

The derivative is rejected whenever evaluation lands on a comparison boundary, including either
endpoint of a clamp and its equal-bound case. Boolean predicates are not differentiated, and
you cannot assert smoothness yourself; equal branch values do not prove that the join is
differentiable. Generalized derivatives and proofs of smooth joins are not available. Symbolic
first and second `partial` graphs admit the [smooth scalar calculus](operators.md),
including quotients and elementary functions, while value branches and absolute-value
expressions remain outside that formal profile. A derivative computed at one point does not
make the same branch valid at another point.

A piecewise value creates no history, hysteresis, reset, or crossing detection; use an explicit
event for those. General nonsmooth solves, complex ordering, analytic functions beyond this
catalog, and automatic event insertion are not provided.
