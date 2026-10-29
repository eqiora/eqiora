# Fixed-reference FSI distributed solve over MPI in 2D

This case closes the bounded composition between the physical owner-routed
assembly proven by
[`fsi.fixed-reference-distributed-assembly-mpi-2d`](../fixed-reference-distributed-assembly-mpi-2d/README.md)
and the distributed execution lifecycle of [RFC
0058](../../../rfcs/0058-portable-realization-and-execution-graphs.md). It uses
the exact fixed-reference monolithic FSI meaning and finish from
[`fsi.fixed-reference-monolithic-step-2d`](../fixed-reference-monolithic-step-2d/README.md).

The accepted reduced owner-row payloads are the sole source of rank-local CSR
and right-hand-side storage. The case never reconstructs a complete CSR and
repartitions it for the solver. Ghosts and halo exchanges derive only from
off-owner payload columns. A distinct, content-identical complete CSR is used
only for identity and every-rank host verification, never to build local
shards. The full target is not submitted to MINRES; it remains a lineage
witness for reconstructed full pressure-row continuity. Interface action is
reevaluated from local residuals and energy from accepted Fields/quadrature.

At one, two, and four physical MPI ranks on one host, the admitted
symmetric-indefinite reduced system executes through `f64`, reproducible,
identity-preconditioned MPI MINRES. Every rank gathers paired explicit global
owner indices and values, reconstructs the complete candidate, reapplies the
exact finalized CSR, and independently reaccepts its true residual. All ranks
must then agree on the exact output bits and a domain-separated summary of the
operator, output, report, owner/layout/admission identities, process group,
and normalized trace. The ordinary fixed-reference FSI finish then
reconstructs physical velocity, pressure, and displacement; the composite
in-memory receipt itself is not serialized for cross-rank comparison.

The MPI-transport-independent oracle separately runs complete CPU reference
assembly, reference MINRES, and the unchanged FSI finish. It shares the
canonical physics and lowering under test, so it is an independent transport
path rather than an independent formulation. Reduced and full assembly remain
bit-identical before the solve. Each normalized pair satisfies
`|a-b| <= 2e-10 + 2e-10 max(|a|, |b|)`. This applies to dimensionless
algebraic coefficients and, after exact Field identity/support/order/length
checks, to velocity divided by `U`, pressure by `P`, and displacement by `L`.
No dimensionless absolute tolerance is applied directly to heterogeneous SI
arrays. CPU and MPI paths pass their native physics gates independently.

Changing the rank count changes ownership and the floating-point reduction
grouping. The case therefore requires the same Model, finalized reduced/full
operator meaning, solver plan, pressure policy, and tolerance-bounded physical
result, but does not require bit-identical solution values or iteration counts
between one, two, and four ranks. Within a run, all ranks agree exactly on the
accepted complete result and domain-separated execution summary.

The direct composition falsifier gives one rank a locally coherent but forged
row-owner authority. MPI admission must return the same diagnostic everywhere
before numerical work, after which the authoritative solve must still pass.
Reduced/full drift is covered by the assembly-to-FSI binder. Explicit-index
gather/host-reacceptance and synchronized post-admission MINRES local-action
faults remain generic RFC 0058 MPI prerequisites; RFC 0060 retains the
collective mesh/route falsifiers. This case does not relabel them as direct
FSI-composition injections.

Every rank records typed observed MPI implementation/library version, MPI
standard, `mpi-rs`, provided thread support, rank count, and reduction, then
agrees a domain-separated summary of that runtime observation. Assembly
lineage, runtime observation, and the execution receipt remain typed in-memory
evidence. The case does not retag the symmetric-indefinite operator as
`General` or widen the frozen distributed artifact v1 profile merely to create
a Run manifest; a durable symmetric-indefinite distributed Run remains a
separate versioned-artifact capability.

Run on a host with Open MPI or MPICH and the mpi-rs build dependencies:

```bash
cargo test --locked -p eqiora \
  --features mpi \
  --test fixed_reference_fsi_distributed_solve_mpi_2d \
  -- --test-threads=1
cargo run --locked -p eqiora-verify -- run \
  --case fsi.fixed-reference-distributed-solve-mpi-2d
```

This is one fixed-reference implicit monolithic 2D FSI step with a replicated
mesh and replicated physical finish. It does not claim solution-bit identity
between rank counts, a distributed result Field, multiple physical nodes,
performance or scale, a transient FSI trajectory, hybrid rank/thread
execution, GPU or MPI plus CUDA, process-failure recovery, ALE, remeshing, or
a durable distributed execution receipt or symmetric-indefinite distributed
Run artifact.

The native witness now selects its existing reproducible MINRES/Identity tuple
collectively using the fixed-reference method's symmetric-indefinite claim.
The selected plan enters the coupled Realization before MPI assembly; binding
checks the complete verifier and process group while preserving the accepted
owner-row assembly evidence and native FSI finish. This does not derive global
operator properties from a rank-local block, change the independent CPU oracle,
or add a symmetric-indefinite durable Run format.
