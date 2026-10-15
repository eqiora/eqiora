# Accepted mapped first-order products

This case owns coordinate composition, not a new PDE solver or pointwise derivative.
`EvaluationMapProducts` borrows a complete accepted map and uses the Plan's exact
shared/occurrence-local input declaration. Every point/seed action uses
the ordinary JVP/VJP. `Retain` never resolves another primal; `Recompute` explicitly reaccepts
one frozen point, verifies its entire original acceptance evidence and reuses it across seeds. Shared
reverse contributions are summed in request order and retained as typed base-space covectors;
JVP/shared/combined grid shapes exclude intrinsic coordinates, and empty grids retain their signatures;
they are never averaged or obtained by filtering failed members.

## Independent derivation

For `f(s,x) = (sx+x², 2s−3x)`, elementary differentiation gives `Ds=(x,2)` and
`Dx=(s+2x,−3)`. Set `s=2`, `x=[1,3,1]`, `ds=1/2`, `dx=[1,−2,1/4]` and output
cotangents `[(1,2),(−1,1),(2,−1)]`.

- JVPs are `[(9/2,−2),(−29/2,7),(3/2,1/4)]`.
- Shared reverse contributions are `[5,−1,0]`, hence their sum is `4`.
- Occurrence-local reverse contributions are `[−2,−11,11]`.
- Both dual pairings are `99/4`.

All values are exactly representable binary64 arithmetic in this bounded specimen. No
finite-difference step, observed solve output or tuned tolerance determines an expectation.
The arithmetic specimen exercises the production coordinate assembly/accumulation helpers;
it does not fabricate accepted evaluations or replace the pointwise evaluator.

Separately accepted Q1 and TPFA products are compared in every occurrence of nested
`[2,3]` point axes and `[2,2]` seed axes, including point-major, seed-major and interleaved
layouts. Original request order controls shared accumulation in every layout. Exact
primal/action evidence remains point-local after other Program evaluations.
The released-state probe independently evaluates each original point and compares both seed
products plus the shared request-order sum, without changing the analytic specimen below.

Plan-admission probes reject foreign/duplicate shared Parameter selections and inconsistent
shared values before execution. Product-admission probes reject
missing/foreign primal members, foreign/stale derivative evidence, wrong typed product count,
same-width reordered tangent bases, same-width foreign Model output signatures, wrong shape, nonfinite
input/sum, checked byte/stride overflow and insufficient budgets. Empty maps, singleton
scalar axes, zero seeds, all-shared and all-mapped coordinates have explicit checks.
The numerical-budget falsifier separately counts each JVP's primal/tangent/point buffers,
each VJP's primal/gradient/point buffers, and the mapped/shared reverse aggregates. Empty
axes cannot hide overflow in the final component-bearing shape's logical strides.

The registered exact library entry is named in `case.toml`; the public facade path also has
an ordinary integration test in `crates/eqiora/tests/mapped_products.rs`. The two existing
ordered-map cases remain the primal/terminal composition owners. Mathematical PDE evidence
stays with `differentiation.spatial-poisson-fem-fvm`.

```bash
mise run affected -- --case differentiation.mapped-products \
  --case differentiation.bounded-parameter-study \
  --case differentiation.bounded-parameter-study-private
```

This registered case covers real first-order implicit products. Native Program admission
owns other executable profiles. Nested seed axes do not imply higher derivatives. Complex pairing, Python/JAX/PyTorch batching, sampling, new solvers,
threading, persistence and peak-process-memory guarantees are outside this case.

The storage boundary includes one typed aggregate record per seed and per combined-grid
occurrence, even when that product has no leaves. The focused budget probe independently
counts those records with `size_of::<DifferentiableCotangent>()`; numeric payload counts,
scientific expected values and tolerances are unchanged. Signature metadata, allocator
overhead and transient scratch remain outside this estimate.
