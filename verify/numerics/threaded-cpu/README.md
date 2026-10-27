# Threaded CPU verification

This case sends canonical 1D, 2D, and 3D Poisson revisions through one-worker
direct execution and a four-worker, run-owned Rayon pool. The resolved
Realization selects the worker count. Indexed P1/Q1 FEM and orthogonal TPFA
packets are evaluated concurrently in bounded batches, then scattered through
the same packet/target/local-entry order as the reference assembler. One FEM
cell packet feeds both the reduced solve system and full reaction system. TPFA
uses one stable sequence containing all source cells followed by every
canonical interior or boundary facet. The assembled CSR operator then exposes
disjoint row actions; reference CG routes every norm and inner product through
the same replicated execution. Fixed 1,024-element logical partials are
evaluated concurrently and combined in index order.

The executable evidence requires:

- exact equation-aware Domain and Field claims projected into the common typed
  portable Realization DAG, then compared with the accepted scalar lowering by
  the finalizer for both one and four host workers;
- pure deployment binding before worker-pool or numerical-system
  materialization, including rejection when the selected host executor cannot
  supply the requested workers;
- one opaque execution admission that seals the exact portable graph,
  canonical CSR fingerprint, solver plan, and exact solver/execution provider
  releases, including their implementation and dependency-release versions;
- the same fixed `SolveWithNativeAcceptance -> ReplayTrueResidualOnHost ->
  AcceptHostComplete` logical DAG for serial and Rayon, with each solver-native
  verification retained rather than hidden and with distinct deployment
  bindings and producer execution reports;
- an additional independently recomputed serial-host true residual and an
  immutable receipt bound to the exact normalized complete-vector fingerprint;
- registered insufficient-capacity, same-ID solver-version, and same-ID
  execution-library-version substitution falsifiers;
- two independently admitted 2D paths—imported-simplex P1 FEM with
  CG/SPD/Identity and generated-Cartesian P0 FVM with
  BiCGSTAB/General/Jacobi—plus rejection of the otherwise structurally valid
  imported-FEM/BiCGSTAB recombination before execution materialization, and
  rejection when legacy compatibility resolution is followed by an
  equation-aware operator-property claim outside its retained exact tuple;
- bit-identical P1/Q1 fields, TPFA cell fields and reconstructions,
  gradients, reactions/fluxes, balances, and every numerical residual,
  iteration, and termination field for one and four workers; execution and
  verification placement are intentionally distinct evidence;
- matching accepted packet/target counts with separate exact placement in
  `AssemblyReport` and `SolveReport`;
- FEM packet counts equal the canonical cell count and target counts equal two;
- TPFA packet counts equal cells plus all facets and target counts equal one;
- unchanged solver-provider identity and implementation version;
- distinct execution-provider identity, exact worker count, and exact Rayon
  1.12.0 library observation in `SolveReport`, receipt, and Run v2 provenance;
- no Rayon library claim on the serial Run;
- rejection of fast reduction and a backend without reproducible support;
- rejection when a Realization requests more workers than the pool admits; and
- rejection of multi-worker execution for an operator without a row-action
  capability.

The same case also executes real Reference MINRES/Jacobi on two independently
specified canonical CSR systems:

- SPD `A=[[4,1],[1,1]]`, `b=[5,2]`, `x=[1,1]`, with leading minors 4 and 3;
- symmetric-indefinite `A=[[4,3],[3,1]]`, `b=[-2,1]`, `x=[1,-2]`, with
  determinant -5 and a finite positive diagonal.

One- and four-worker `pool.solver(...)` execution preserves the exact Reference
solver provider, Rayon execution and verification providers, Reproducible
reduction, Jacobi policy and original two-iteration limit. Literal-matrix
residual replay and independent solutions use the predeclared absolute target
`2^-40` and componentwise error bound `2^-39`; serial/threaded values and all
numerical report fields must also match exactly. The congruence `W A W`, with
`W=1/sqrt(diag(A))`, preserves inertia; a positive diagonal does not establish
SPD. Stored zero and negative diagonal entries reject even with zero RHS.
Normal and transposed oriented actions retain the requested orientation on both
one- and four-worker pools. Transposed CSR row actions use allocation-free column
lookups in the original source-row order, preserving compensated sums. A full
square transposed action uses `O(n² log(degree))` lookups; this is no transpose
speedup claim.

The same registered target also covers every advertised Reference complex
Reproducible tuple (14 at this revision), normal and conjugate-transposed,
with one and four workers. General and complex-symmetric BiCGSTAB, HPD CG,
and Hermitian/HPD MINRES use their admitted Identity/Jacobi combinations.
Independent literal products and inverse-norm bounds in `references/README.md`
check original coordinates; serial parity is supplemental. Complex coordinate
lowering retains the same execution provider, worker count and reduction.

An ordinary complex algebraic Model executes with an explicit Rust
`CommonLinearRequest::with_execution` request and reopens its exact Result.
Plan v26 retains the execution provider release, dependency inventory and
worker count. Both solve and verifier reports must match that Plan; serial
substitution, changed workers and forged verifier metadata fail closed.
Automatic ranking remains host-serial. Nonserial numeric-payload caps reject
as unproved: the serial reduction-buffer bound is not a Rayon memory bound.
These are Rust host-execution claims, not a new Python placement interface.

The execution receipt is an in-memory host contract, not a graph artifact or
general scheduler. This is a replicated, host-local `f64`
generated-Cartesian P1/Q1 FEM,
orthogonal TPFA, and reference-CG claim for dimensions one through three, plus
the bounded real and complex solver systems above. It
claims exact pairing only for the generic Realization admission axes. Exact
space, order, quadrature, and method-specific structure remain the typed plan
validators' responsibility; the capability set is not a second execution
graph. It does not claim imported-simplex or adaptive assembly,
nonorthogonal FVM, fast
accumulation/reductions, faer threading, distributed/device assembly, NUMA
placement, complex spatial assembly, CUDA/MPI execution DAGs, graph persistence, or performance
superiority.

Run:

```bash
cargo test -p eqiora-backend-rayon
cargo test -p eqiora --test threaded_cpu --features rayon
cargo run -p eqiora-verify -- run --case numerics.threaded-cpu
```
