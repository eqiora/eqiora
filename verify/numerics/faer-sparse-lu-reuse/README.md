# Accepted faer sparse-LU symbolic reuse

This case keeps the independently derived two-element Q1 systems and solutions in
[`analytic.json`](expected/analytic.json) and [`symbolic.json`](expected/symbolic.json).
The public product test compares cold solves with one backend-neutral run-local prepared
provider session for `p0 -> p1 -> p2`, including exact solutions and accepted reports.

The assembly owner supplies a complete canonical structural encoding. Exact equality uses
that entire encoding; its digest is observational only. Faer additionally compares the actual
canonical CSR row offsets and column indices before retaining symbolic state. Changed values
recompute numeric LU, while a changed structural identity or actual topology rebuilds symbolic
LU before numeric factorization. A failed numeric candidate cannot replace accepted factors.

Run both authorities with:

```console
cargo run --locked -p eqiora-verify -- run \
  --case numerics.faer-sparse-lu-reuse \
  --case numerics.faer-sparse-lu-reuse-private
```

The state is ephemeral and provider-private. This case makes no persistent, cross-process,
CUDA, MPI, multi-RHS, timing, or scale claim.
