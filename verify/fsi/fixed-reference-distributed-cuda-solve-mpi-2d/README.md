# Fixed-reference FSI over host-staged MPI and CUDA in 2D

This case composes the accepted owner-routed MPI assembly and distributed
MINRES path with one resident CUDA sparse action per rank. It uses the exact
fixed-reference monolithic operator and unchanged physical finish proven by
the CPU, MPI, and single-device CUDA parent cases. It does not add another FSI
lowering, assembly rule, solver plan, or result representation.

At one, two, and four ranks on one host, MPI owns halo exchange, Krylov state,
reproducible reductions, explicit-index gather, and complete-host residual
acceptance. CUDA owns only the deterministic rectangular action for the
accepted local row shard. The matrix is uploaded once; every action uploads
`[owned | ghost]`, waits for the sparse operation, downloads owned rows, and
waits before MPI crosses its local-action agreement boundary.

The launcher requires an explicit ordered list of four distinct physical
selectors. Its verification-owned wrapper maps local rank `i` to selector
`i`, exports only that selector through `CUDA_VISIBLE_DEVICES`, and therefore
gives every process the same Realization-local ordinal zero. The runtime then
all-gathers live CUDA UUIDs and rejects missing, nil, or duplicated physical
identity. Selectors may be CUDA indices or UUID selectors accepted by the
installed driver.

```bash
EQIORA_MPI_CUDA_DEVICE_SELECTORS=0,1,2,3 \
cargo test --locked -p eqiora \
  --features mpi-cuda \
  --test fixed_reference_fsi_distributed_cuda_solve_mpi_2d \
  -- --ignored --test-threads=1

EQIORA_MPI_CUDA_DEVICE_SELECTORS=0,1,2,3 \
cargo run --locked -p eqiora-verify -- run \
  --environment physical-mpi-cuda \
  --case fsi.fixed-reference-distributed-cuda-solve-mpi-2d
```

Absence of MPI, CUDA, four explicitly selected devices, or any required
runtime/library is a verification failure, never a skip or host fallback. The
bounded claim excludes GPU-aware MPI, device-resident Krylov or reductions,
GPU assembly, multiple devices per rank, multiple physical nodes, performance,
transient FSI, ALE, remeshing, and a durable composite Run artifact.

The public hosted gate selects the separate `host-cpu` evidence environment.
It validates this complete manifest and reports the physical target as
`not-selected`; it does not claim to repeat the run. Unfiltered verification
and an exact `physical-mpi-cuda` run retain the failure behavior above. Generic
Cargo suites leave the parent orchestration test ignored; the typed evidence
runner is the repository-owned path that activates it.


The ordinary witness now admits this bounded composition before shared solver
selection: real-f64 symmetric-indefinite MINRES/Identity/Reproducible is the sole
compatible delivered tuple. It is the MPI Krylov subplan, not a standalone CUDA
solver. The selected plan enters the original coupled Realization before assembly.

The existing MPI-CUDA adapter authenticates its exact local-action provider,
ordinal-zero device, queue-zero ownership, deterministic action capabilities and
HostStaged transport before the common MPI selected-subplan binding. Rank-local
construction/provider/queue errors and changed exact controls reject collectively;
fixed-size diagnostic hashes check agreement without introducing mismatched
collective extents. Native assembly lineage, live device observation and distinct
physical UUID agreement, complete-host acceptance and FSI finish remain mandatory.
Physical devices are not required to have identical descriptors across ranks.
Composition cost, transfer, synchronization and allocation bounds remain unknown;
this is not automatic placement or a performance-optimal choice claim.
