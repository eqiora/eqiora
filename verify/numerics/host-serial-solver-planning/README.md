# Deterministic host-serial solver planning

This case verifies exactly one versioned planning policy over its frozen General
five-member catalog projection. The accepted problem is the normal-orientation,
complete-diagonal, canonical CSR `f64` system

```text
A = [[4, 1], [2, 3]]
b = [6, 8]
x = [1, 2]
```

The ordinary path resolves `Robust`, `Fast`, or `LowMemory`, exposes the exact
candidate, provider descriptors, plan, evidence identity, execution provider,
and ordered stable reasons, then executes the selected request once. The
registered test compares that execution by exact `PartialEq` with the existing
manual `LinearSolveRequest::new(backend, plan).solve(problem)` path, including
the complete `SolveReport`. It independently reapplies `A x`, requires a
componentwise solution error no larger than `2^-40`, and retains the existing
true-residual acceptance.

This public case owns the five real-backend tuple executions, independent true
residual, exact manual/decision `LinearSolution` and complete `SolveReport`
equality, frozen positive decisions, and public observations. The required
companion
[`numerics.host-serial-solver-planning-private`](../host-serial-solver-planning-private/README.md)
owns the malformed catalog, stale identity, profile, capability, precedence,
tie-break, mutation, exact-problem-and-owned-operator identity, zero-work,
one-attempt, and total actual-operator-call falsifiers. The capability is verified
only when both cases pass.

Run:

```bash
python3 verify/numerics/host-serial-solver-planning/references/derive_policy_v3.py
cargo test -p eqiora-solver planning::tests
cargo test -p eqiora --test host_serial_solver_planning
cargo run -p eqiora-verify -- run \
  --case numerics.host-serial-solver-planning \
  --case numerics.host-serial-solver-planning-private
```

## Selection reason vocabulary

Selected reason codes name the caller's ranking preference: `robust-preference`,
`fast-preference`, or `low-memory-preference`. They do not assert that the selected
candidate uses reproducible reduction, direct factorization, or an iterative
method: admission can exclude the preferred candidates.
The v3 inventory also includes the existing reference and faer Identity BiCGSTAB tuples.
The ranking axes remain unchanged; the final ascending-ID tie-break selects Identity
before Jacobi when both are admitted. Unknown or absent diagonal facts reject only
Jacobi candidates, while unavailable provider/capability tuples retain their own reasons.
The exact rational solution and tolerances are unchanged.

## Boundary

`Robust`, `Fast`, and `LowMemory` are names for this literal v3 rule table,
not empirical or universal optimization claims. This case proves no solver or
provider superiority, timing, byte memory, fill, scale, fallback, retry,
portfolio, transport, advisor, learning, mutable discovery, durable decision
artifact, matrix-free, transpose, SPD, symmetric-indefinite, distributed,
threaded, CUDA, MPI, nonlinear, or mixed-precision planning.
