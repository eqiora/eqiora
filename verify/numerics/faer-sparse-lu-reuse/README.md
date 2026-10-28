# Accepted faer sparse-LU symbolic reuse

This case keeps the independently derived two-element Q1 systems and solutions in
[`analytic.json`](expected/analytic.json) and [`symbolic.json`](expected/symbolic.json).
The public product test compares cold solves with one backend-neutral run-local prepared
provider session for `p0 -> p1 -> p2`, including exact solutions and accepted reports.

The assembly owner supplies a complete canonical structural encoding. Exact equality uses
that entire encoding; its digest is observational only. Faer additionally compares the actual
canonical CSR row offsets and column indices before retaining symbolic state. Changed values
recompute numeric LU, while a changed structural identity or actual topology rebuilds symbolic
LU before numeric factorization. Every prepared solve requires an explicit action.
Successful corrections update tentative factors; the caller commits only after its full
physical and lineage acceptance. A failed solve poisons the action until discard, and
neither failure nor caller rejection can replace the last committed factors. Invalid
operation sequences reject before mutation; an action with no solves can commit without
inventing factorization work.

Run both authorities with:

```console
cargo run --locked -p eqiora-verify -- run \
  --case numerics.faer-sparse-lu-reuse \
  --case numerics.faer-sparse-lu-reuse-private
```

The state is ephemeral and provider-private. This case makes no persistent, cross-process,
CUDA, MPI, multi-RHS, timing, or scale claim.


The ordinary scalar heat-storage and real nonlinear Q1 backward-Euler Runs also
connect to this optional prepared provider seam. They check the selected
Plan/provider/resource profile, prepare at most one session per Run, and pass the
existing assembly owner's exact structure and canonical problem at every solve.
Nonlinear corrections may reuse symbolic preparation while changed Jacobian
coefficients require fresh numeric factors. A backend declaring no preparation uses
its ordinary solve; preparation or solve failure never triggers fallback.

The separate focused `scalar_heat_prepared` product tests retain the independently
derived linear heat recurrences, real/complex Fields through real coordinates,
capacity/time-step changes, original-equation/provider rejection, cancellation and
cold resume. The nonlinear witness adds unchanged two-field transport/diffusion
inputs, cold/warm accepted equivalence, an exact-zero physical control, actual
correction residual/provenance and saved reopening. These product tests add no
registered nonlinear scientific verification or maturity promotion to this case.

The outer scalar, standalone/common MINI and ALE FSI owners commit only after their
last fallible physical/trajectory/lineage/Time acceptance. A checked-profile rejection
also poisons the action before provider work. On rejection they discard tentative
resources and preserve accepted physical State. The public Run still ends and drops
its session on an error; restarting is cold. Private provider rollback/retry evidence
does not imply public same-session Run retry. Multiple factor sets may coexist during
staging, so no peak-memory or performance claim follows from preparation counts.
This does not complete the broader resident-device/prepared-execution issue.
