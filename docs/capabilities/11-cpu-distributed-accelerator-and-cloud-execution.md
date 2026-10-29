# CPU, distributed, accelerator, and cloud execution

[Capability matrix](../capability-matrix.md) · [legend](../capability-matrix.md#how-to-read-the-matrix)

| Capability | C | X | V | M | Current implementation and planned extensions |
|---|:---:|:---:|:---:|:---:|---|
| Serial host execution | ✅ | ✅ | ✅ | ⬜ | Reference and bounded optimized adapters |
| Run-owned CPU threading | ✅ | ✅ | ✅ | ⬜ | Rayon threads run row actions, assembly and fixed-order reductions; MINRES/Jacobi and the complex reproducible solvers give the same results on one and four workers, and the thread placement is saved with the Plan and Result. Not yet: threaded complex spatial assembly, performance claims; [`numerics.threaded-cpu`](../../verify/numerics/threaded-cpu/README.md) |
| Explicit thread-count capability | ✅ | ✅ | ✅ | ⬜ | Bounded host placement and evidence |
| Deterministic threaded policy | ✅ | ✅ | ✅ | ⬜ | Fixed-order reference policies; broad fast/reproducible modes remain |
| NUMA-aware execution | ⬜ | ⬜ | ⬜ | ⬜ | Placement, first touch, affinity, and topology evidence remain |
| SIMD/vectorized kernels | 🟨 | 🟨 | 🟨 | ⬜ | Library/compiler benefits may occur; no explicit portable kernel contract |
| Distributed ownership/layout | ✅ | ✅ | ✅ | ⬜ | Ordered owned/ghost algebraic layouts |
| Halo exchange | ✅ | ✅ | ✅ | ⬜ | Loopback and MPI bounded paths |
| Reproducible collectives | ✅ | ✅ | ✅ | ⬜ | One/two/four partitions and ranks |
| MPI execution | ✅ | ✅ | ✅ | ⬜ | Registered one-host 1/2/4-rank canonical Poisson and fixed-reference FSI bridges plus recorded lower-level physical two-node algebra evidence; neither graph-bound bridge has physical multi-node evidence |
| General multi-node scalability | 🟨 | 🟨 | 🟨 | ⬜ | Recorded lower-level two-node algebra proves functional transport, not generic-bridge placement or scaling |
| Distributed mesh and assembly — fixed FSI slice | ✅ | ✅ | ⬜ | ⬜ | One 2D fixed-reference FSI mesh is partitioned and assembled on 1, 2 and 4 MPI ranks on one host, and the full system is rebuilt on the host for checking; the shards feed the MPI solve below. Not yet: other meshes and PDEs, distributed Fields, multi-node runs, performance; [`fsi.fixed-reference-distributed-assembly-mpi-2d`](../../verify/fsi/fixed-reference-distributed-assembly-mpi-2d/README.md), [`fsi.fixed-reference-distributed-solve-mpi-2d`](../../verify/fsi/fixed-reference-distributed-solve-mpi-2d/README.md) |
| Distributed fixed-reference FSI solve | ✅ | ✅ | ⬜ | ⬜ | One 2D monolithic FSI step solves with reproducible MINRES on 1, 2 and 4 MPI ranks on one host and matches the CPU result within `2e-10` relative and absolute tolerance for velocity, pressure and displacement. Not yet: bit-identical results across rank counts, distributed Fields, transient or moving-mesh FSI, multi-node, GPU; [`fsi.fixed-reference-distributed-solve-mpi-2d`](../../verify/fsi/fixed-reference-distributed-solve-mpi-2d/README.md) |
| Single-device fixed-reference FSI solve | ✅ | ✅ | ⬜ | ⬜ | The bounded 2D path finalizes one shared CPU/CUDA operator and supports MINRES with identity preconditioning and host acceptance. [Product tests](../../crates/eqiora/tests/fixed_reference_fsi_cuda_2d.rs) cover host-side finalization and an explicitly selected physical-device run. No current recorded-device replay verifies this composition. |
| Distributed checkpoint/restart | ⬜ | ⬜ | ⬜ | ⬜ | Durable rank-independent state and repartition restart remain |
| Distributed failure semantics | ⬜ | ⬜ | ⬜ | ⬜ | Detection, containment, retry, shrink/recover, and evidence remain |
| Explicit device identity | ✅ | ✅ | ✅ | ⬜ | Device/transfer/completion contracts |
| CUDA sparse action | ✅ | ✅ | ✅ | ⬜ | Physical `f64` cuSPARSE CSR action |
| CUDA resident Krylov solve | ✅ | ✅ | 🟨 | ⬜ | Public-source selected-device evidence verifies Jacobi-CG/SPD and identity-MINRES/symmetric-indefinite `f64`/`Fast` in bounded complete workflows. The exact identity-BiCGSTAB/general tuple remains implemented and hardware-gated but is not registered as case-level physical evidence. Broader preconditioners, reproducible reductions, operators, scale, and scalar types remain |
| GPU matrix-free execution | 🟨 | ⬜ | ⬜ | ⬜ | Local-action IR exists; accelerator gather/scatter and kernels remain |
| GPU assembly | 🟨 | ⬜ | ⬜ | ⬜ | Device contract exists; production assembly remains |
| GPU preconditioners | 🟨 | 🟨 | 🟨 | ⬜ | Jacobi only; stronger methods remain |
| GPU reproducibility modes | 🟨 | ⬜ | ⬜ | ⬜ | Host true residual checks exist; deterministic device reductions remain |
| Multi-GPU execution | 🟨 | ⬜ | ⬜ | ⬜ | Device identity can grow; residency, partition, transport, and evidence remain |
| ROCm execution | 🟨 | ⬜ | ⬜ | ⬜ | Experimental investigation only; no support claim |
| Vendor-neutral GPU kernels | 🟨 | ⬜ | ⬜ | ⬜ | [Isolated CubeCL experiment](../../experiments/cubecl-local-action/README.md) only: device-independent checks and CUDA/HIP compilation pass on Rust 1.95. Current physical `f64` execution and product integration remain unverified. |
| Accelerator capability negotiation | ✅ | ✅ | ✅ | ⬜ | Typed bounded CUDA admission; broad scalar/layout/operator matrix remains |
| Out-of-core execution | ⬜ | ⬜ | ⬜ | ⬜ | Streaming operators, storage hierarchy, and recovery remain |
| Batch/HPC scheduler integration | ⬜ | ⬜ | ⬜ | ⬜ | Slurm/PBS launch, manifests, staging, monitoring, and collection remain |
| Containerized reproducible execution | 🟨 | 🟨 | 🟨 | ⬜ | CI environments exist; supported user images and hardware matrices remain |
| Cloud execution service | ⬜ | ⬜ | ⬜ | ⬜ | Multi-tenant scheduling, quotas, secrets, storage, and cost evidence remain |
| Remote execution protocol | ⬜ | ⬜ | ⬜ | ⬜ | Content-addressed submission, progress, cancellation, and result retrieval remain |
| Performance benchmark governance | 🟨 | 🟨 | 🟨 | ⬜ | Benchmark roadmap exists; stable baselines and regression gates remain |

## Parallel execution depth

| Capability | C | X | V | M | Current implementation and planned extensions |
|---|:---:|:---:|:---:|:---:|---|
| Per-run isolated thread pool | ✅ | ✅ | ✅ | ⬜ | Run-owned Rayon execution exists; broader nested-parallel policy remains |
| CPU affinity and pinning | ⬜ | ⬜ | ⬜ | ⬜ | Portable topology discovery, policy, reporting, and evidence remain |
| NUMA memory placement | ⬜ | ⬜ | ⬜ | ⬜ | Locality-aware allocation, migration, and first-touch evidence remain |
| Task-graph parallelism | 🟨 | 🟨 | 🟨 | ⬜ | Ordered packet work exists; general dependency-aware heterogeneous scheduler remains |
| Nested parallelism control | ⬜ | ⬜ | ⬜ | ⬜ | Solver/library/thread-pool oversubscription and admission remain |
| MPI communicator ownership | ✅ | ✅ | ✅ | ⬜ | Application-owned initialization plus one mutably borrowed duplicated execution group; subcommunicators and concurrent studies remain |
| Deterministic partition-independent result | ✅ | ✅ | ✅ | ⬜ | Bounded algebraic fixtures; broad nonlinear/time/mesh cases remain |
| Graph-based mesh partitioner | ⬜ | ⬜ | ⬜ | ⬜ | Weighted dual graph, constraints, reproducibility, and adapter isolation remain |
| Dynamic load balancing | ⬜ | ⬜ | ⬜ | ⬜ | Cost measurement, repartition, state migration, and provenance remain |
| Hybrid MPI plus CPU threads | 🟨 | 🟨 | 🟨 | ⬜ | Ingredients exist separately; topology-aware joint evidence remains |
| MPI neighborhood collectives | 🟨 | 🟨 | 🟨 | ⬜ | Explicit halo contract exists; tuned topology-specific adapters remain |
| Nonblocking communication overlap | 🟨 | 🟨 | 🟨 | ⬜ | Bounded halo action; measured assembly/solve overlap remains |
| Reproducible distributed reductions | ✅ | ✅ | ✅ | ⬜ | Reference policy verified; scale/performance modes remain |
| Native fast distributed reductions | ✅ | ✅ | ✅ | ⬜ | MPI native policy passes the admitted one-host 1/2/4-rank bridge; numerical envelope at scale remains |
| GPU stream/queue identity | ✅ | ✅ | ✅ | ⬜ | Device contract names completion and queue; multi-stream scheduling remains |
| Explicit GPU memory residency | ✅ | ✅ | ✅ | ⬜ | Resident Krylov vectors; whole spatial/time workflow residency remains |
| Cold one-shot CUDA transfer threshold | ✅ | ✅ | ⬜ | ⬜ | A privacy-bounded public-source observation records all raw repetitions and verifies by independent replay that no durable crossing occurred over refinements 64–4096 in that environment. This is an environment-specific cold one-shot result, not a portable default or regression limit; [`performance.cuda-csr-transfer-threshold`](../../verify/performance/cuda-csr-transfer-threshold/README.md) |
| GPU memory pool | ⬜ | ⬜ | ⬜ | ⬜ | Run-owned allocation, fragmentation bounds, reuse, and telemetry remain |
| Asynchronous host-device transfer | ✅ | ✅ | ✅ | ⬜ | Typed transfer/completion slices; overlap and broad operators remain |
| Pinned host memory | 🟨 | 🟨 | 🟨 | ⬜ | Adapter implementation detail may exist; public ownership/performance contract remains |
| Unified-memory policy | ⬜ | ⬜ | ⬜ | ⬜ | If admitted, migration and performance must be explicit rather than an implicit copy |
| GPU kernel fusion | ⬜ | ⬜ | ⬜ | ⬜ | IR legality, register/memory tradeoffs, compilation cache, and evidence remain |
| GPU sparse matrix assembly | 🟨 | ⬜ | ⬜ | ⬜ | Device and local-action contracts exist; deterministic insertion/reduction remains |
| GPU direct solver | ⬜ | ⬜ | ⬜ | ⬜ | Factorization lifecycle, memory failure, reuse, and transpose solve remain |
| GPU algebraic multigrid | ⬜ | ⬜ | ⬜ | ⬜ | Backend-neutral policy, hierarchy provenance, and convergence evidence remain |
| GPU-to-GPU peer transfer | ⬜ | ⬜ | ⬜ | ⬜ | Topology detection, access policy, synchronization, and fallback remain |
| NCCL/RCCL collective adapter | ⬜ | ⬜ | ⬜ | ⬜ | Device collectives, stream ordering, reproducibility, and failures remain |
| GPU-aware MPI | ⬜ | ⬜ | ⬜ | ⬜ | Capability probe, device buffer ownership, synchronization, and fallback remain |
| GPUDirect RDMA | ⬜ | ⬜ | ⬜ | ⬜ | Transport admission, topology, registration, security, and evidence remain |
| Single-node multi-GPU | 🟨 | ⬜ | ⬜ | ⬜ | Device identity exists; partition, halo, collectives, and memory balance remain |
| Multi-node multi-GPU | ⬜ | ⬜ | ⬜ | ⬜ | MPI/device topology, transport, solver, restart, and scaling evidence remain |
| CPU-GPU cooperative execution | ⬜ | ⬜ | ⬜ | ⬜ | Work partition, load balance, coherence, and deterministic policy remain |
| Heterogeneous operator placement | 🟨 | 🟨 | 🟨 | ⬜ | Realization placement vocabulary exists; general cost/capability resolution remains |
| Device-side time integration | ⬜ | ⬜ | ⬜ | ⬜ | Resident residual/Jacobian/event execution and accepted-step evidence remain |
| Device-side differentiation | 🟨 | ⬜ | ⬜ | ⬜ | JVP/VJP contracts exist; GPU kernels and transpose solves remain |
| Accelerator profiling trace | ⬜ | ⬜ | ⬜ | ⬜ | Kernel, transfer, memory, communication, and semantic phase correlation remain |
| Roofline/performance diagnostics | ⬜ | ⬜ | ⬜ | ⬜ | Hardware counters, arithmetic intensity, bottleneck explanation, and reproducibility remain |
| Strong-scaling evidence | ⬜ | ⬜ | ⬜ | ⬜ | Controlled problem/hardware suites and efficiency thresholds remain |
| Weak-scaling evidence | ⬜ | ⬜ | ⬜ | ⬜ | Controlled growth, memory, communication, and efficiency thresholds remain |
| Energy and cost accounting | ⬜ | ⬜ | ⬜ | ⬜ | Hardware energy, cloud cost, uncertainty, and run provenance remain |

