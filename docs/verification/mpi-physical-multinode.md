# Physical multi-node MPI verification

This case runs only on suitable hardware and checks that transport and solves
work correctly across two physical nodes. The ordinary CI case runs several
processes on one host. Both use the same distributed layout, halo, collective,
solver-plan, and true-residual contracts; there is no second solver or
MPI-specific model API.

## Contract under test

```text
one validated global CSR problem
        ↓ balanced owned/ghost partition
one MPI rank per physical node
        ↓ exact MPI_Get_processor_name all-gather
minimum distinct-processor precondition
        ↓
halo action + reproducible/native dot products
        ↓
Jacobi-CG under the sole SolverPlan
        ↓
fresh distributed true-residual acceptance
```

By default the test child makes no claim about physical topology. Setting
`EQIORA_MPI_MIN_PHYSICAL_NODES` enables a runtime check: each rank gets its
processor name from MPI, all ranks gather the full fixed-size names, and the
test rejects a placement with fewer distinct processors than requested. The
value must be positive and no larger than the number of ranks.

Processor names are used only for this check. They are not part of the Semantic
Model, the Realization, the distributed layout identity, solver reports, or the
run wire format. Run provenance still records the MPI implementation and version,
thread support, rank topology, and reduction policy, but hostnames never become
part of portable artifact identity.

## Hardware command

Build the integration test with the system MPI used on every allocated node:

```bash
cargo test --locked -p eqiora-backend-mpi \
  --features mpi-runtime --test mpi_distributed --no-run
```

Then launch the printed `mpi_distributed-*` executable with one rank on each of
at least two physical nodes. Exact launcher flags are scheduler-specific. For
Open MPI, explicitly export the verification inputs to the remote ranks:

```bash
export EQIORA_MPI_TEST_CHILD=1
export EQIORA_MPI_MIN_PHYSICAL_NODES=2
mpirun --map-by ppr:1:node -n 2 \
  -x EQIORA_MPI_TEST_CHILD -x EQIORA_MPI_MIN_PHYSICAL_NODES \
  /path/to/mpi_distributed-HASH \
  --exact mpi_child_executes_halo_and_collectives --nocapture
```

For MPICH/Hydra, the equivalent explicit placement and propagation are:

```bash
mpirun -hosts node-a,node-b -ppn 1 -n 2 \
  -genv EQIORA_MPI_TEST_CHILD 1 \
  -genv EQIORA_MPI_MIN_PHYSICAL_NODES 2 \
  /path/to/mpi_distributed-HASH \
  --exact mpi_child_executes_halo_and_collectives --nocapture
```

Both environment values must be exported to remote ranks, and the executable and
its dynamic MPI and runtime libraries must resolve to the same builds on every
node. The hardware launch is not part of ordinary CI; the default test runs 1, 2,
and 4 ranks on one host, which checks the protocol but not physical distribution.

The check has passed with two ranks on two physical nodes, using one
byte-identical test executable and one MPI runtime build, and its single-node
falsifier fails before any numerical work. That run predates the generic
complete-CSR bridge, so it is evidence for the lower-level halo, reduction, and
CG path only. The test name is kept as an alias for the current child body so the
commands above keep working; the generic bridge needs a new recorded physical run
before it can claim multi-node evidence.

## Falsifying cases

- zero, malformed, or rank-excessive minimum-node input fails before numerical
  work;
- two or more ranks placed on one processor fail the physical-node
  precondition;
- processor names that exceed the bounded exchange buffer fail closed;
- partition ownership, ghost order, halo values, or rank count drift fails;
- reproducible and native collective reductions must both match the exact
  global fixture;
- both Jacobi-CG runs must match the manufactured solution and satisfy a newly
  recomputed global true residual; and
- MPI initialization/finalization and communicator ownership remain with the
  application/test launcher.

## Scope

This case shows functional execution across at least two physical nodes. It
does not cover strong or weak scaling, NUMA placement, network
topology optimization, distributed assembly, checkpoint/restart, elastic
ranks, process-failure recovery, ULFM, or production scheduler integration.
