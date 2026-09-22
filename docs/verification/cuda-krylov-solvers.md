# CUDA Krylov solver verification

The first CUDA solver conformance gate proves one assembled, single-device,
`f64` path:

```text
validated LinearSystem + sole SolverPlan
    -> fail-closed algorithm/property/preconditioner/reduction admission
    -> one explicit CSR/RHS/initial/Jacobi H2D phase
    -> resident cuSPARSE SpMV + cuBLAS vector iteration
    -> one candidate-solution D2H phase
    -> independent serial-host CSR action + fixed-order norm
    -> accepted LinearSolution + distinct producer/verifier evidence
```

No CUDA type enters Semantic Model, `SolverPlan`, `LinearSolution`, or the L2
device contracts. The adapter exposes CUDA-specific transfer/version/timing
evidence beside the backend-neutral accepted solution.

## Falsifying fixtures

CG uses an asserted SPD matrix, Jacobi preconditioning, and exact solution
`[1, 2]`:

```text
[ 4 -1 ] [1] = [2]
[-1  3 ] [2]   [5]
```

BiCGSTAB uses a general nonsymmetric matrix, identity preconditioning, and the
same exact solution:

```text
[4 1] [1] = [6]
[2 3] [2]   [8]
```

The gate checks:

- `Fast` is the only admitted CUDA reduction policy;
- CG rejects a missing SPD assertion before CUDA discovery;
- Jacobi rejects an invalid diagonal before CUDA discovery;
- CSR and all Krylov vectors stay device-resident during iteration;
- cuBLAS version evidence is present and cuSPARSE workspace is retained;
- initial-guess upload precedes solve completion, which precedes solution
  download;
- CUDA is recorded as the producer and serial host as the verifier;
- independently recomputed host true residual satisfies the sole
  `SolverPlan`; and
- setup, H2D, solve, D2H, verification, and total wall times are distinct.

## Running the physical gate

The test is ignored in ordinary CI because compiling an optional dynamically
loaded adapter does not prove hardware execution. On a machine with a
compatible NVIDIA driver, cuSPARSE 12, and cuBLAS 12:

```bash
CUDA_VISIBLE_DEVICES=<physical-index> \
EQIORA_CUDA_DEVICE=0 \
cargo test -p eqiora-backend-cuda --features cuda-runtime \
  --test contract_boundary \
  cuda_runtime::physical_cuda_krylov_paths_are_independently_accepted \
  -- --ignored --exact
```

After visibility is narrowed, Eqiora ordinal zero names that selected device.
The adapter never falls back to CPU and rewrites provenance. Runtime absence,
missing library symbols, unsupported policy, and numerical breakdown are
structured failures.

## Numerical policy and nonclaims

cuBLAS reductions are backend-native even with atomics disabled. This gate
therefore claims `Fast`, not Eqiora `Reproducible`, and acceptance uses the
independent fixed-order host oracle. The small exact fixtures falsify wiring,
residency, algorithm, and evidence errors; they do not establish performance,
conditioning robustness, or scale.

This slice does not claim GPU assembly, ILU/IC/AMG, pinned host memory, a
memory pool, matrix-free kernels, mixed precision, multiple streams, multiple
GPUs, checkpoint/restart, cross-toolkit bit identity, or real-time scheduling.
Those capabilities require separate gates rather than wider wording around
this result.
