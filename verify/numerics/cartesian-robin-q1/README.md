# Cartesian Q1 Robin heat boundary

The ordinary scalar heat law supports a Robin boundary on the checked serial CPU
path. The [independent rational reference](reference/README.md) owns the four
fixed cases, exact affine solutions, complete operators, physical units, reaction
signs and numerical bounds. Its expected values do not come from Eqiora.

The [sealed evidence inventory](evidence/2026-11-06/inventory.json) links each
actual source, Model, Mesh, Plan, Result, captured operator and comparison report
to its producer and build receipts. Four build/link records are explicitly labelled
[public projections](evidence/2026-11-06/public-projections.json): only machine-local
filesystem locators are omitted, with the original and projected hashes retained.
All scientific payloads and comparison reports keep their original bytes.
All four comparisons passed all nine checks:
the full uneliminated 9 × 9 matrix and nine loads, the physical free residual,
every coefficient, the reference residual, all three left weak reactions,
and their total. Reaction recovery also passes the separate, stricter bound
against the exported operator; agreement with the reference alone is insufficient.

The producer calls public `CommonSpatialPlan::run_result`. An opt-in test observer
captures the full operator, exact DOF map and typed scales from the finalized
system retained by that same Run. It preserves the Model, Mesh and Plan before
execution and the Result immediately afterward. Plan and Result reopening retain
their canonical bytes. No public executor or thermal-specific numerical path is
introduced.

## Retained attempts and falsifiers

The first attempt completed one Run, then failed a harness shape assertion: a
Cartesian Q1 field has logical shape `[3, 3]`, not `[9]`. Its saved Result and
[failure receipt](evidence/2026-11-06/receipts/attempt1/producer-finish.json) remain
in the archive. The operator was not saved before that process stopped, so this
attempt supplies no complete scientific comparison. A read-only metadata audit
preceded the corrected producer; it did not reconstruct a substitute operator.

The corrected attempt completed four additional Runs, for five campaign Runs in
total. Its source and executable identities are recorded separately from later
registration changes. The [four reports](evidence/2026-11-06/receipts/attempt2/comparison-summary.json)
and [34 rejected payload defects](evidence/2026-11-06/payload-falsifiers/summary.json)
are preserved unchanged. The defects cover omitted or reversed Robin terms,
changes to only the matrix or load, wrong faces, eliminated test traces, missing
endpoint weights and incorrect reactions or heat signs. They require no further
PDE solves.

The original reports retain `preparation_only: true`, a marker emitted by the
standalone mathematical reference. It describes that reference's preparation;
it does not mean that the recorded native-input comparisons were absent. Their
process receipts and `accepted` checks record the comparisons that actually ran.
The compiled executable is identified by its build receipt; executable binaries
are not checked into this archive.

## Scope and regression check

The scientific witnesses cover steady, real scalar heat on a two-by-two
Cartesian Q1 mesh in 2D, with constant positive conductivity, nonnegative constant
Robin coefficient, constant boundary data and a fixed left temperature. They
include zero transfer, heating and cooling through the boundary, and independent
changes to geometry, conductivity and transfer coefficient. The compiler also
checks units, supports, coefficient restrictions and complete boundary coverage.

Quadratic trace matrices have separate pure kernel tests, not a complete Q2
Robin witness here. Transient or nonconstant Robin coefficients, curved meshes,
CPU-parallel, MPI, GPU, TPFA and performance are outside this evidence.

The [registered case](case.toml) uses the existing Cargo library-test runner:

```sh
mise run pr -- --case numerics.cartesian-robin-q1
```

It runs the same four-case public lifecycle and the canonical Python reference,
using fresh home-backed scratch without overwriting the archived campaign.
Python 3 and its standard library are required; a missing interpreter or failed,
missing or invalid comparison report fails the check with retained diagnostics.
