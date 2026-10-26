# Linear backend verification

This case checks that one Eqiora-owned `SolverPlan` and `LinearOperator`
contract can drive the independent reference and faer adapters without moving
backend types into canonical model or spatial APIs.

The executable evidence includes:

- reference CG and faer CG agreement on a symmetric positive-definite system;
- faer BiCGSTAB on a nonsymmetric manufactured system;
- exact faer BiCGSTAB SPD Identity/Jacobi Fast tuples on the independent
  `[[4,1],[1,3]]` matrix, with normal and transposed actions, distinct exact
  right-hand sides/solutions, retained provider/policy, and true-residual acceptance;
- Faer partial-pivot sparse LU on a precommitted exact-rational,
  non-structurally-symmetric canonical CSR system;
- zero-work acceptance of the exact initial guess, one factor-and-solve attempt
  for the ordinary path, and fail-closed handling of one inconsistent
  rank-deficient system;
- exact capability admission for General, symmetric-positive-definite, and
  symmetric-indefinite operators with only Identity, Fast, and `f64`;
- rejection of matrix-free or hand-built `LinearProblem` values without the
  same captured canonical CSR owner;
- identity and Jacobi adapter actions;
- one exact faer 0.24.4 provider release bound before execution and retained
  consistently by the producer report and accepted execution receipt;
- independent `||b - A x||_2` acceptance through the Eqiora operator;
- explicit rejection of a reproducible-reduction request that faer does not
  currently guarantee; and
- the same canonical Poisson model revision solved by reference and faer.

The sparse-LU path converts validated row-major storage to the same mathematical
column-major matrix, then calls Faer low-level symbolic, numeric, and solve APIs.
Numeric factorization and solve receive `Par::Seq` explicitly; process-global
parallelism wrappers are structurally excluded. Eqiora replays the true residual
through its own operator before accepting the returned values.

The frozen sparse-LU claim is host-local, `f64`, replicated, normal-orientation, captured-CSR,
Identity, Fast, and single-worker. It is not a matrix-free, transpose,
threaded, distributed, accelerator, Realization-admission, performance, scale,
fill, memory, pseudo-inverse, least-squares, or singularity-diagnosis claim.

The provider's complete capability-inventory assertion additionally names real
symmetric and complex General, complex-symmetric, Hermitian and
Hermitian-positive-definite sparse LU. Their focused numerical tests live in
`eqiora-backend-faer`; this case does not extend the frozen real oracle or claim
registered complex numerical evidence. Mixed real/complex provider inventories
are projected to the real scalar boundary by the existing spatial admission owner.

Run:

```bash
cargo test -p eqiora-backend-faer
cargo test -p eqiora --test faer_spatial
cargo run -p eqiora-verify -- run --case numerics.linear-backends
```

See [RFC 0010](../../../rfcs/0010-execution-backend-contracts.md) and the
[backend strategy](../../../docs/development/library-and-accelerator-strategy.md)
for ownership and graduation rules.

## Pre-committed sparse-LU evidence

`expected/sparse-lu-contract.json` and `oracle/sparse_lu_oracle.py` freeze the
exact-rational oracle for the proposed `SparseLu` direct algorithm under the
`SolverPlan` contract of
[RFC 0010](../../../rfcs/0010-execution-backend-contracts.md). They were written
by an agent that does not implement the slice, before any implementation
existed, and later amended by that same agent to bind the acceptance threshold
the implementer is not permitted to author. Neither the original nor the
amendment read, ran, or saw any implementation.

The registered Rust target invokes this exact Python oracle with the frozen
digest, then consumes the same JSON fixture for its CSR, RHS, expected solution,
capability tuples, plan tolerances, error ceiling, and early-exit boundary.
Neither the implementation nor its Rust test owns those values.

```bash
python3 verify/numerics/linear-backends/oracle/sparse_lu_oracle.py --summary
```

See [the oracle reference](references/sparse-lu-oracle.md) for the witnesses,
the falsifiers, the acceptance threshold and the plan it binds, the fixture
digest, and the nonclaims.

## Ordinary provider continuity

The `provider_continuity` module binds one compiled Model and authenticated
2-by-2-cell unit-square mesh to the ordinary TPFA Plan/Result path. Both exact
Reference CG/Identity/Reproducible and pinned faer CG/Identity/Fast receive the
same captured CSR and RHS. Independent integral cell balance fixes diagonal 6,
neighbor entries -1, RHS 1/4, and solution 1/16. Each output and original-operator
residual is checked independently, with the existing component bound 5e-13 and
requested residual target 5e-13; agreement alone is insufficient.

Successful solves of 2A fail original-A acceptance. Stale same-ID adapter or
library releases, unknown providers, unsupported reduction and changed supplied
backend capabilities reject before solver work. An explicit Reference fallback
under an admitted faer identity fails provenance acceptance. Exact Plan/Result
identities differ while Model, mesh, CSR/RHS and Field meaning remain the same.
The fixture and falsifiers were frozen in
[#454](https://github.com/eqiora/eqiora/issues/454#issuecomment-6428433551) before
implementation. See the [adapter migration record](../../../crates/eqiora-backend-faer/README.md)
for dependency ownership, update checks and the removal surface. This is one
steady real SPD host TPFA boundary, without a performance, general interchangeability
or published-artifact claim. The sparse-LU oracle above is unchanged.
