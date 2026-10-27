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


The ordinary linear scalar heat-storage Run also connects to this optional prepared
provider seam. It checks the selected Plan/provider/resource profile, prepares at most
one session per Run, and supplies the existing assembly owner's exact structure and
canonical problem after one numeric assembly per step. A backend that declares no
preparation uses its ordinary solve; a preparation or solve failure does not trigger a
fallback. Faer retains responsibility for exact coefficient equality and numeric
refactorization when coefficients change.

The separate focused `scalar_heat_prepared` product tests compare cold and warm Runs
against the existing independently derived heat recurrences. They cover real and complex
Fields through the admitted real-coordinate solve, changing capacity and time step,
provider and original-equation rejection, cancellation, and cold resume. Original spatial
acceptance and execution-provenance validation precede State publication and the next
prepared call. Any later error drops the sole Run-local session, including factors that
passed linear acceptance before outer spatial rejection; no failed-step retry continues
with that owner. These focused tests add no registered verification or maturity promotion
to this case. They establish no retryable-trial rollback, nonlinear or cross-Run reuse,
performance benefit, or completion of the broader prepared-execution lifecycle issue.
