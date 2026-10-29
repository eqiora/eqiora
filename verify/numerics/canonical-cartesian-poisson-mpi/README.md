# Canonical Cartesian Poisson over MPI

This registered case composes the canonical two-dimensional scalar-elliptic
path with the admitted MPI complete-CSR bridge. The parent compiles
[`models/poisson.eqi`](models/poisson.eqi) exactly once, writes one bounded
canonical current `ModelEnvelope`, and launches the current test binary under one,
two, and four ranks. Children reconstruct `KernelProgram` from those shared
bytes; no rank recompiles or alpha-normalizes source.

For continuous Q1 FEM and orthogonal cell-centered TPFA, every child resolves
`2D + f64 + Distributed + HostCpu { threads: 1 } + CG/Jacobi` with each explicit
`Reproducible` and `Fast` reduction policy, then finalizes replicated assembly
into the sole captured complete-CSR view. Solver controls, equations and
acceptance bounds are identical. Complete system, partition and layout content
must agree across policies; Realization and Run identities must change with
requested/effective reduction provenance.
The test uses rotated-cyclic ownership, so `(global + 1) % ranks` exercises
noncontiguous owner order. Model, Realization, Run, system, partition, and
derived-layout artifacts must round-trip to exact canonical bytes and agree by
digest across every rank. A fresh request reconstructed only from decoded
Realization accessors must reproduce the exact system bytes and digest before
content-DAG validation and collective admission.

After application-owned MPI initialization and process-group observation, but
before numerical run workspace or communication exists, the portable
distributed graph is bound to one transport-neutral logical process-group
slot, the exact rank count, one host worker per partition, and the admitted
solver capabilities. MPI implementation, communicator, provided thread
support, and library versions remain adapter and Run observations; they do
not enter portable Realization identity. This is deliberately not a
pre-communicator rejection claim.

The admitted MPI solve records every actual synchronized admission, halo,
owned action, vector update, reduction, producer-report, owner-gather,
native-acceptance, and result-agreement boundary in dense global order. Trace
storage is bounded from the admitted maximum iteration count and reserved
before the first numerical collective. The receipt keeps this actual trace
behind a fixed nine-step macro DAG rather than expanding the iterative solve
to its maximum possible length.

The solve gathers explicit global indices and values, performs native
serial-host true-residual acceptance on every rank, and agrees the accepted
result. The common execution receipt then independently replays the true
residual against the same complete canonical CSR. Before exposing a complete
host output, the MPI adapter all-gathers one domain-separated summary over
the operator, output, dimension, report, partition, layout, admission,
process group, and complete actual trace. Every rank must agree that summary.
Only after this final receipt agreement does the case invoke the
method-native FEM or FVM finish. Reproducible reductions retain rank-ordered
folds; Fast reductions use the MPI implementation's native all-reduce. Neither
solution-bit identity nor equal iteration counts are required across policies
or rank counts; exact rank agreement within each Run remains required. The finished field must satisfy the analytic
continuous-L2 bound, reaction-or-flux plus source balance, and an explicit
tolerance against a separately resolved one-worker serial reference. A
changed-RHS negative case forms an internally exact content-linked DAG but is
rejected by fresh semantic derivation replay, demonstrating that content
linkage is not a lowering claim.

Run on a host with Open MPI or MPICH and the mpi-rs build dependencies. The
launcher is probed before adding an implementation-specific oversubscription
option:

```bash
cargo test --locked -p eqiora \
  --features mpi \
  --test canonical_cartesian_poisson_mpi \
  -- --test-threads=1
cargo run -p eqiora-verify -- run \
  --case numerics.canonical-cartesian-poisson-mpi
```

This is physical one-host MPI evidence at one, two, and four ranks, with
replicated mesh, assembly, and method finish. It does not claim RFC 0060's
distributed mesh, distributed assembly, or distributed Field; physical
multi-node canonical execution; scaling; hybrid rank/thread execution; MPI plus
CUDA; FSI or MINRES;
process-failure recovery; another scalar, dimension, or mesh family; a
durable execution receipt; or a curated public execution-graph API.

The native witness selects the exact MPI CG/Jacobi plan collectively before
assembly through the shared deterministic planning owner. Its fixed unit
coefficient and full Dirichlet boundary supply the global SPD/diagonal claims;
this is not a generic variable-coefficient classification. Explicit Reproducible
and Fast requirements remain admission constraints. The selected plan enters the
Realization request and binds the replayed complete system and actual process
group before the existing admission, solve, reacceptance and method finish.
The original controls, analytic bounds and ownership comparisons are unchanged.
