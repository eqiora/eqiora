# MPI distributed verification

This registered case runs one captured complete-CSR system
through an arbitrary-owner distributed layout with one, two, and four MPI
ranks on one host. Collective admission seals the complete verifier, system,
layout, and sole solver plan only after all dynamic execution workspace exists.
The transport-neutral source first derives exact checked allocation extents,
owns each canonical local layout once, and deterministically forms halo groups
from sorted `(owner, receiver, index)` triples. Shards borrow that sole layout
beside their owned-row CSR storage.

All later status records carry an exact phase, iteration, and monotonic
ordinal. The public adapter exposes no unadmitted raw apply, dot, or solve
operation. Rank-order explicit global-index blocks drive the uninterrupted
index/value gather pair, and every rank independently reaccepts and compares
the same complete vector and report.

The admitted solver executes both numerical policies at every rank count.
Reproducible dot products all-gather one partial per rank and fold them in rank
order. Fast dot products use the MPI implementation's native all-reduce. Both
plans solve the same manufactured SPD system with Jacobi preconditioning and
must pass complete-host residual acceptance.

Test-only hooks inject plan, Jacobi, local-action, producer, gather, and host
verifier failures after successful admission. A parent process enforces a
timeout and checks a common diagnostic at one, two, and four ranks.

This case supplies bounded C/X/V for the generic algebra bridge. It does not
execute canonical FEM/FVM lowering or finish, typed durable distributed
artifacts, physical multi-node bridge placement, scalability,
checkpoint/restart, or process-failure recovery. Earlier physical two-node
evidence covers the lower-level halo/reduction/CG case only.

Run on a machine with Open MPI and the rsmpi build dependencies:

```bash
cargo test -p eqiora-backend-mpi --features mpi-runtime,mpi-test-hooks --test mpi_distributed -- --test-threads=1
cargo run -p eqiora-verify -- run --case numerics.mpi-distributed
```

The same bounded process harness checks collective pre-execution selection of
all three native MPI tuples. A rank-local invalid control rejects collectively
before fingerprint exchange; different objectives or controls reject even when
the selected tuple agrees. The group remains usable after each rejection. These
checks supplement the exact planning-projection falsifiers in the existing
private planning case; they do not claim recovery from a failed MPI process.

Group construction stages local validation and all fallible buffer preparation,
then agrees readiness with a stack scalar before communicator duplication. The
registered harness injects a one-rank preparation failure and immediately retries
successfully on one, two and four ranks. It rejects intercommunicators before
readiness (two groups of1+1 or1+3 ranks) while accepting Cartesian intracommunicators.
This is uniform validation-failure handling, not MPI process-failure recovery.
