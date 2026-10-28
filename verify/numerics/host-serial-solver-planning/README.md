# Deterministic host-serial solver planning

This case verifies planning policy v9 over 30 real and 28 complex tuples (58 in
all). The real inventory contains the frozen five-member General catalog,
eleven SPD tuples, seven symmetric-indefinite tuples and seven symmetric tuples.
The complex inventory spans four mathematical operator classes.
The real problem is the normal-orientation,
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

This public case owns 30 real and 28 complex tuple executions, independent true
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
python3 verify/numerics/host-serial-solver-planning/references/derive_policy_v9.py
cargo test -p eqiora-solver planning::
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
The v9 inventory also includes the existing reference and faer Identity BiCGSTAB tuples.
Without size facts, Robust/LowMemory retain the specialized CG/MINRES preference before general-purpose BiCGSTAB. Version6 also preserves that specialization within Fast's iterative fallback, after its reduction and sparse-LU priorities; absence of an optional faer provider must not demote CG solely by candidate-ID spelling. The General catalog preference ordering remains unchanged. The final ascending-ID tie-break selects Identity
before Jacobi when both are admitted. Unknown or absent diagonal facts reject only
Jacobi candidates, while unavailable provider/capability tuples retain their own reasons.
The exact rational solution and tolerances are unchanged.

## Typed complex catalog

The independent oracle freezes four genuine complex 2x2 systems. Reference
BiCGSTAB admits Identity and Jacobi with Reproducible reduction on General,
complex-symmetric, Hermitian and HPD profiles. HPD also admits Reference CG with
Identity or Jacobi. Hermitian and HPD profiles admit Reference MINRES with
Identity or Jacobi through the existing realification. Faer CG on HPD and
BiCGSTAB on all four classes support Identity/Jacobi with Fast reduction; each
class also retains Faer Identity/Fast SparseLU. The complex inventory contains
14 Reference and 14 Faer tuples. The Hermitian fixtures have nonreal
off-diagonal entries. HPD is established independently by leading principal
minors 4 and 11; it is not inferred from a Field declaration. The separate
Hermitian matrix `[[1,2i],[-2i,1]]` has positive diagonal entries, determinant -3
and eigenvalues -1 and 3, so it remains indefinite.

Jacobi requires known complete structural diagonal presence. CG and MINRES
additionally require actual finite strictly positive real diagonal values;
BiCGSTAB requires finite nonzero typed values. MINRES uses `W=1/sqrt(D)` and the
symmetric congruence `W A W` in correction coordinates. Every recovered Jacobi
iterate must satisfy the original residual target and physical predicate within
the original budget; a small scaled residual cannot admit the original problem.
The oracle checks complete diagonal presence and an unproved diagonal, with explicit rejection reasons for each
unproved Jacobi candidate. A separately stored numerical zero diagonal rejects
at preconditioner construction, while Identity still solves the nonsingular
system. Conjugate-transposed Jacobi actions conjugate the delegated diagonal.
Without shape facts, HPD retains the declared specialized CG preference before
BiCGSTAB for Robust/LowMemory; the ordinary Fast preference remains faer SparseLU.

Exact rational complex arithmetic derives forward values `[1+2i, -2+i]` and
conjugate-transposed values `[2-i, 1+3i]` for each fixture, with separately frozen
right-hand sides. Every tuple executes both actions under the existing residual
acceptance and unchanged componentwise `2^-40` bound. Normal-orientation planning
retains exact manual/decision equality and full objective traces. An absent
optional provider records `catalog.provider-unavailable`; reference remains
eligible, while a required Fast reduction can leave no admitted candidate.

## Boundary

`Robust`, `Fast`, and `LowMemory` are names for this versioned v9 rule table,
not empirical or universal optimization claims. This case proves no solver or
provider superiority, timing, process memory, fill, scale, fallback, retry,
portfolio, transport, advisor, learning, mutable discovery, durable decision
artifact, matrix-free, unconjugated transpose, distributed,
threaded, CUDA, MPI, nonlinear, or mixed-precision planning. Generic HPD evidence
does not claim ordinary finite or spatial HPD mathematical recognition.

## Bounded work and numeric payload

The existing public test also supplies independent two-coordinate systems for
all eleven declared real SPD tuples, seven symmetric-indefinite tuples and seven
symmetric tuples. The SPD matrix `[[4,1],[1,3]]` has leading minors 4 and
11 and maps `[1,2]` to `[6,7]`. The indefinite matrix `[[1,2],[2,1]]` has eigenvalues
3 and -1 and maps `[1,-1]` to `[-1,1]`; the same coefficients also exercise
an asserted symmetric profile without a stronger property claim. The zero-diagonal
symmetric matrix `[[0,1],[1,0]]` maps `[1,2]` to `[2,1]`: Identity paths remain
admitted while Jacobi fails its independent nonzero-diagonal prerequisite.
Reference MINRES and BiCGSTAB use Identity/Jacobi/Reproducible, and Faer
BiCGSTAB uses Identity/Jacobi/Fast. MINRES Jacobi requires finite strictly positive
real diagonal entries; this does not establish symmetry or definiteness.
Each compatible tuple executes normal and transpose actions with distinct RHSs,
its exact provider and unchanged original residual acceptance. Symmetry alone
adds no nonsingularity, unique-solution or minimum-norm promise. This does not prove ordinary recognition of every
matrix class or delivery of undeclared provider tuples.

Versioned work units use typed rows `n`, stored slots `z`, solve count `H`,
structural preparations `S` and numeric preparations `F`, with `1 <= S <= F <= H`.
Structural setup is `S(n+z)`. LU numeric setup is `Fn^3` and each solve is
`2n^2+3n`, an explicitly conservative dense envelope rather than sparse-fill
prediction. CG each iteration is `2z+8n+jn`; BiCGSTAB is `4z+20n+2jn`, where
`j=1` for Jacobi and 0 for Identity. Their numeric setup is `Fjn`. MINRES uses
`k=min(forecast,n)`. Real Identity MINRES uses the base proxy
`M=k(2z+8n)+4nk(k+1)` for retained-basis actions and reorthogonalization.
Real Jacobi MINRES adds numeric setup `3Fn` and per-solve work
`5n+4nk+nk(k+1)+k(k+1)(2k+1)/6` to `M`: declared weights for initial scaling/norm,
`W A W`, correction recovery and repeated projected triangular solves. It counts
original-residual acceptance `Hk(2z+5n)` across the horizon. The other listed
methods add `H(2z+5n)` as a final independent acceptance allowance. These
scoring weights do not count every intermediate provider callback or executed action. Complex Reference CG/MINRES estimates stay
unknown; these real MINRES formulas do not model realification.
All arithmetic is checked; missing dimensions or overflow stay unknown. These
are declared ranking units, not a FLOP, timing, convergence or optimality claim.
Identity/Jacobi forecasts are separate; omitted forecasts use the iteration
ceiling. Binding and requested output costs remain outside this typed solve.

The numeric payload adds CSR/RHS `sizeof(usize)(n+1+z)+8(n+z)` to supported
Reference real workspace bounds. Normal CG uses `8((5+j)n+ceil(n/1024))` bytes;
normal BiCGSTAB uses `8((12+j)n+ceil(n/1024))`. The bound accounts requested
heap numeric buffers, including the owned solution and reduction partials.
Transposed real actions account an additional work vector instead of partials.
It excludes caller guesses, assembly copies, retained outputs, stack scalars,
container metadata, allocator rounding and process RSS. MINRES has no proved
numeric-payload bound, including with Jacobi. Unknown bounds cannot
satisfy an explicit payload cap. Public tests prove preflight denial before
backend invocation, preserve manual controls, and check changed actual CSR
size against the retained cap without changing the original decision audit.

A declared repeated horizon can change the proxy preference in the generic
planner. It never grants reuse authority. The existing bounded `Plan.for_run`
path independently rederives workload and retention facts: compatible Faer sparse
LU retention can forecast `S=F=1`, while iterative providers prepare per solve.
One-shot Plans use `H=S=F=1`. This case does not add a whole-Run measurement or
extend that bounded retention proof. Existing no-size catalog oracles retain their literal
fallback ranking and now name that limitation in each admitted trace.

For `LowMemory`, known numeric payload bounds take precedence over unknown ones.
Among candidates whose payload is unproved, the declared iterative-before-LU
preference remains; a smaller work proxy is not evidence of smaller memory use.

Reference complex CG/MINRES work is unknown until the expanded real system and
conversion/typed-acceptance work are independently modeled. The native complex
BiCGSTAB proxy is unchanged. Unknown dimensions retain the shape-unavailable
reason; known dimensions report the unmodeled realification.
