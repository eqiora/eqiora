# Library and accelerator strategy

- Status: Active
- Recorded: 2026-07-17
- Scope: numerical libraries, execution providers, distributed and device
  placement, external data/CAD adapters, and supporting implementation tools

## Purpose and authority

This document owns the durable strategy for deciding what Eqiora implements,
what it delegates to external libraries, and how an adapter earns a support
claim. It is deliberately not an implementation ledger.

The following sources have separate authority:

| Question | Authoritative source |
| --- | --- |
| What can the product do now? | [`docs/capability-matrix.md`](../capability-matrix.md) |
| Which executable evidence supports a claim? | `verify/**/case.toml`, projected by `cargo run -p eqiora-verify -- index` |
| What is the current dependency order? | [`docs/roadmap.md`](../roadmap.md) |
| What are the exact accepted contracts? | [`rfcs/`](../../rfcs/README.md) |
| Which Rust dependency versions ship? | root and separately scoped `Cargo.lock` files |
| Which native runtime/library/device versions executed? | accepted Run evidence and registered environment observations |
| What work remains open? | GitHub Issues and their native parent, sub-issue, and blocking relations |

Do not copy those inventories into this file. A library name below is an
adoption decision or investigation boundary, not by itself a capability claim.

Evidence development is active under [RFC 0089](../../rfcs/0089-resume-claim-local-evidence-development.md).
New adapters use focused product and compatibility tests; they gain verified status only when a
registered, independently derived case proves their exact tuple. Evidence binds the smallest
provider, numerical, and environment projection needed for that claim rather than an unrelated
whole inventory.

## Decision in one sentence

Eqiora owns mathematical meaning, typed lowering and realization contracts,
capability admission, provenance, and its existing small deterministic conformance oracles;
optimized algebra, integration, transport, device execution, parsing, CAD
kernels, persistent storage, and presentation mechanisms enter only through
isolated adapters that consume those contracts.

The execution-provider invariant is:

```text
meaning
  -> lowered contract
  -> realization
  -> adapter
  -> focused positive and failure tests
```

Verification is a separate optional status supplied only by applicable independently derived
evidence.

No execution backend may skip a step by interpreting the Semantic Model
directly. Other extension families have different, equally closed paths:

```text
ModelPackage     -> validated declarations -> elaborated canonical meaning
untrusted bytes  -> bounded data adapter   -> Eqiora-owned artifact
Studio workflow  -> public facade intent   -> transaction or projection
```

A data artifact enters a Model or Realization only through an explicit typed
binding. A Studio workflow neither owns hidden model state nor calls an L3
provider around capability admission.

## Non-negotiable invariants

1. **Meaning does not depend on a provider.** Solver, thread count, process
   topology, device ordinal, library version, and file parser are realization
   or execution facts, never Semantic Model meaning.
2. **Eqiora owns accepted representations.** No third-party matrix, vector,
   parser node, mesh object, CAD face handle, communicator, device pointer, or
   error type appears in the stable Semantic Model or wire schemas.
3. **Reference oracles remain independent.** Adding a production adapter does not make that
   adapter the authority for its own expected result. Replace an obsolete oracle only with a
   genuinely distinct derivation of the same or deliberately revised claim.
4. **Capability admission is exact.** A provider advertises supported tuples, not the Cartesian
   product of independently supported-looking axes. New tuples are explicitly unverified.
5. **Unsupported combinations fail before execution.** There is no silent
   fallback to another solver, scalar, layout, target, reduction, or device.
6. **Reported convergence is not verification.** Focused tests may check a residual or declared
   product outcome, but only registered independently derived evidence upgrades verification.
7. **Environment observations remain bounded.** One device, host, driver,
   process count, or network proves only the recorded configuration. Portable
   support requires a replayable registered case with an honest boundary.
8. **Control and data planes remain separate.** Small versioned plans,
   diagnostics, identities, and provenance do not become bulk array or device
   transport protocols.
9. **Adapters are typed families, not one universal plugin.** Model packages,
   execution providers, data adapters, and Studio workflows share identity and
   provenance discipline while retaining distinct payload schemas and loading
   contracts.
10. **Dependencies are compatibility decisions.** A version bump that affects numerical
    behavior, ABI, hardware, MSRV, safety, or durable provenance is a bounded implementation slice
    with focused tests, not routine text maintenance or a new evidence lane.

## Ownership and layers

### Eqiora owns

- implicit Relation and activation meaning;
- typed identity, dimensions, shape, support, frame, orientation, and revision
  provenance;
- canonical expression and operator IR;
- mesh, geometry-map, quadrature, discrete-space, local-contribution,
  assembly, and sparse-system contracts;
- solver, time, differentiation, device, distributed-layout, and completion
  contracts;
- Realization selection and capability resolution;
- canonical artifacts, run lineage, diagnostics, and evidence schemas;
- deterministic reference evaluators and falsifiers.

### External libraries own

- optimized dense/sparse factorization and Krylov kernels;
- adaptive integration, nonlinear iteration internals, and backend histories;
- thread-pool scheduling, MPI transport, and collective implementations;
- device discovery, contexts, streams, allocations, vendor handles, and tuned
  kernels;
- external file grammar parsing and CAD-kernel topology construction;
- persistent collection, syntax-tree, CLI, and diagnostic-rendering internals.

### Crate boundary

```text
L0/L1  meaning, schema, graph, validation
  |
L2     IR and Eqiora-owned numerical/execution contracts
  |
L3     faer / Rayon / MPI / Diffsol / CUDA / Gmsh / XDMF / HDF5 / CAD adapters
  |
L4     public facade, Python, Studio, CLI and application workflows
```

L3 adapters may accept a library-owned handle at their application-facing
constructor when lifetime ownership requires it—for example, an
application-owned MPI communicator. They must not retain that type in an
Eqiora artifact or return it through an Eqiora contract. “No backend type
crosses the adapter” means no such type crosses the retained or serialized
boundary, not that private adapter code is forbidden from naming its library.

The current ownership split is intentionally specific:

- `eqiora-solver` owns the sole `SolverPlan`, operator, preconditioner,
  capability, and report vocabulary;
- `eqiora-meshing` owns topology, geometry maps, quadrature, and mesh quality;
- `eqiora-assembly` owns anonymous local contributions, constraint maps, and
  backend-neutral sparse algebra;
- `eqiora-numerics` composes discrete spaces and numerical realizations but
  must not duplicate those contracts;
- `eqiora-device`, `eqiora-distributed`, and `eqiora-time` own mechanism-neutral
  execution contracts;
- each external mechanism lives in a dedicated L3 adapter crate.

`eqiora-numerics` remains one publication boundary. Its shared discrete-block,
boundary-normalization, and finalized-linear contracts are deliberately
private, as are method-specific contracts such as the MINI transient form. Its
internal dependency direction is common numerical machinery → solid/fluid →
FSI/ALE. Splitting those families into crates merely to improve compilation
parallelism would force private lowered contracts into the public compatibility
surface or duplicate them. Reconsider a family split only when an independent
scientific requirement makes the shared lowered contracts stable public API;
build speed alone is not such a requirement.

The crate root exports no numerical item directly. Every public item has one
scientific owner path:

| Public owner | Responsibility |
| --- | --- |
| `eqiora_numerics::common` | genuinely cross-family mesh, discrete-space and Field representation, boundary, local-operator, assembled-linearization, design-coordinate, spatial-expression, and step-count contracts |
| `eqiora_numerics::scalar` | scalar elliptic, diffusion, transport, and affine-network realizations |
| `eqiora_numerics::solid` | elasticity and elastodynamics realizations |
| `eqiora_numerics::fluid` | incompressible Stokes and Navier–Stokes realizations |
| `eqiora_numerics::fsi` | fixed-reference fluid–structure interaction |
| `eqiora_numerics::ale` | moving-domain ALE, mesh motion, and remeshing |

The `pub use` declarations in those six modules are the exhaustive ownership
inventory; do not copy the item list into another registry or prelude. A new
item enters the narrowest owning family. If two families need it, move the
underlying contract to `common` only when both are real consumers, not in
anticipation of reuse. Implementation modules and lowered/finalized staging
bridges remain private.

`eqiora::numerics` is a separately checked, deliberately small application
facade. It selects canonical model, finalization, solution, and top-level
lower/solve entry points from the owners above; it does not mirror the complete
numerics crate. Low-level composition and evidence code imports the canonical
owner path directly.

Native and hardware adapters remain opt-in. The default build neither compiles
nor loads a system MPI or CUDA runtime. Optional production features still
share the one workspace MSRV and are compiled by the all-feature MSRV gate;
only a separately locked application or experiment may declare a different
toolchain contract.

`cargo xtask check-layers` must compare the declared layer map with the exact
workspace crate set. Neither an undeclared crate nor a declaration for a
nonexistent future crate is valid; a reserved name is not architecture review.

## Extension families

Eqiora does not expose a dynamic `Plugin` anything-box. The extension family
determines both its meaning and its admissible payload:

| Family | Supplies | Must not own |
| --- | --- | --- |
| `ModelPackage` | Components, Parameters, Connectors, Relations, package dependencies | solvers, mesh methods, placement |
| execution provider | a typed solver/time/distributed/device contract implementation | Semantic Model interpretation |
| data adapter | bounded external syntax to or from Eqiora-owned geometry, mesh, field, or trajectory artifacts | canonical identity reconstructed from file ordering |
| Studio workflow | applicability, intent, commands, and projections over public facade contracts | a second model or execution semantics |

The families may share name, version, digest, dependency, signature, and
provenance rules. They do not share an arbitrary JSON payload or runtime ABI.
Dynamic discovery is introduced only when two independently shipped providers
of one family prove that compile-time registration is the bottleneck.

## Exact capability admission

### Capability tuple

A solver/execution capability is an admitted tuple over all axes that affect
correctness, not a bag of independent sets. Depending on the provider, the
tuple includes:

```text
operator class and orientation
scalar type and precision policy
matrix/operator representation
vector and distributed layout
method and preconditioner
reduction/reproducibility policy
target kind and runtime requirements
thread/rank/device topology
MPI thread-support level when applicable
required library and ABI versions
```

For example, evidence for the exact CUDA tuples `CG + SPD + Jacobi + f64 +
replicated + Fast`, `BiCGSTAB + General + Identity + f64 + replicated + Fast`,
and `MINRES + SymmetricIndefinite + Identity + f64 + replicated + Fast` does
not admit BiCGSTAB on an SPD declaration, Jacobi with BiCGSTAB or MINRES,
reproducible device MINRES, or any other cross-product tuple. A new tuple arrives with focused
positive/failure tests and a capability-matrix update whose verification status remains absent.

Realization-level capabilities follow the same rule. Independent sets of
method, mesh, solver capability, layout, and target are safe only when every
product is intentionally admitted; scalar type belongs to the solver
capability. Heterogeneous and graph-shaped placement uses the portable
execution graph below; new MPI/GPU composition must extend that graph rather
than harden an earlier global target enum into a second scheduler.

The generic Realization boundary represents this as an exact set of
`RealizationCapabilityContext + SolverCapability` pairs. The context nests a
spatial capability `(method, mesh family, dimension envelope)` with layout,
target capacity, and either offline execution or one exact real-time
priority/deadline request. The solver tuple is the sole scalar-type authority.
Space family, order, quadrature, and method-specific structure remain mandatory
checks of their typed plan validators; the generic set neither duplicates
those contracts nor claims to describe the entire execution graph. A basic
compatibility-plan resolution without an operator assertion retains its
nonempty operator-property candidate set. The equation-aware portable
projection seals its claim against that set; a legacy finalizer that does not
use that projection must explicitly seal its known property before operator
materialization. Field-wise resolution checks the property directly, and every
numerical finalizer still proves exact equation identity and coefficients
before execution.

### Graduation checklist

An external adapter graduates only when all applicable items are complete:

1. an Eqiora-owned typed input/output contract identifies one invariant owner;
2. dependency and external types are confined to one adapter boundary;
3. capability discovery rejects unknown versions, unsupported tuples, and
   stale mappings before numerical mutation;
4. one ordinary end-to-end path executes without a test-only semantic bypass;
5. focused ordinary positive and failure tests cover the admitted tuple without a test-only
   semantic bypass;
6. the capability matrix states the same narrow boundary and leaves verification absent unless
   registered independently derived evidence applies exactly;
7. ordinary Run provenance records applicable model/plan/operator identities, provider and
   library versions, scalar/layout, topology, and environment;
8. dependency license, advisory, MSRV, unsafe/FFI, and platform effects pass
   policy review; and
9. the narrowest relevant local gates pass, including every semantically affected registered
   case.

An ignored hardware test or a prose observation is useful investigation, but it does not upgrade
verification. Failed experiments remain valuable when their gate is explicit; they still do not
become support claims.

## Stable execution seams

### Portable realization and execution graph

An equation-aware finalizer owns both the portable typed execution graph and
the finalized operator it executes. The graph contains logical executor,
queue, and process-group slots; native pool handles, communicator identities,
device ordinals, driver/library versions, and completion objects belong to a
deployment or Run, not portable meaning.

Execution follows one fail-closed lifecycle:

```text
finalize operator and graph
  -> bind exact provider capacity and capability tuple
  -> materialize run-owned resources
  -> execute typed steps and record actual boundaries
  -> accept with the native verifier
  -> replay the Eqiora-owned independent host oracle
  -> emit an immutable complete-host-output receipt
```

Binding is pure and seals the stable provider ID, declared Eqiora release,
sorted dependency-release inventory, capacity, and capability tuple
before numerical resource materialization wherever live capacity is already
observable. The producer report carries typed solver, execution, and verifier
provider releases; receipt acceptance requires the selected solver/execution
pair and the contract-required verifier, rejecting even a same-ID version or
library substitution before exposing the result. MPI is the deliberate exception at the
transport edge: communicator duplication and bounded preparation buffers may
precede numerical binding so the adapter can learn the exact process group,
but an unsupported numerical placement still fails before MPI run workspace
and the first numerical collective.

Runtime materialization replaces logical slots with process-unique resource
identities. Device data motion records typed transfers, value generations, and
waited `Completion`/`Fence` values. Distributed execution records the actual
collective trace and requires rank agreement over the accepted result and the
receipt summary. The receipt seals the finalized operator, output, dimensions,
solver report, normalized execution trace, native acceptance, and independent
replay without cloning backend storage or claiming that producer convergence
is acceptance.

Provider and runtime versions remain paired adapter/Run provenance rather than
portable graph fields. For the bounded host path, Run provenance is projected
from the provider releases in the accepted receipt, never reconstructed from
an unrelated application-crate version. Versions of live drivers and native
libraries remain runtime observations rather than static-provider claims. The
accepted receipt is currently an in-memory
execution result; a durable receipt/trace wire and a curated raw graph API are
separate compatibility gates. The capability matrix and registered evidence,
not this strategy, state which exact host, CUDA, and MPI graph tuples are
currently verified.

Run manifest v2 names the primary solver and execution provider and carries one
flat, unique component-version inventory. It is not a role-preserving encoding
of every verifier or nested provider in the execution DAG. The receipt retains
those typed roles in memory; two roles that require different versions under
the same component name fail projection rather than being collapsed. A
role-structured durable schema is introduced only with its first executable
consumer, never by reinterpreting v2.

### Linear algebra

The common seam is mathematical rather than method-specific:

```text
LinearOperator
  dimensions
  operator properties and orientation
  apply
  apply_transpose when admitted

Preconditioner
  apply(residual, correction)

SolverPlan
  method
  relative/absolute tolerance
  maximum iterations
  preconditioner
  reduction policy

LinearSolverBackend
  exact capabilities
  solve(problem, plan, execution) -> SolveReport + vector
```

The CSR oracle and production providers implement this seam directly. Physics
code does not translate a Realization plan into a method-specific config.
`SolveReport` retains producer termination evidence, while acceptance
independently reapplies the operator. Normal and transposed solves are
symmetric capabilities; implicit differentiation never differentiates a
backend's Newton/Krylov iteration history.

Identity and Jacobi are the current stable preconditioner vocabulary. ILU,
incomplete Cholesky, AMG, domain decomposition, restarted GMRES, and sparse
direct methods are admitted only with a problem that can falsify the added
method and with exact construction/provenance policy.

### In-process CPU execution

Rayon is placement behind `eqiora-backend-rayon`, not an ambient global semantic
choice. A Run owns its pool and worker count. Reproducible mode uses indexed
packets and a fixed reduction tree; fast mode may use provider-native ordering
but is a distinct policy and evidence claim.

Host admission binds the exact finalized operator, solver tuple, and worker
capacity before pool and system materialization. Serial and Rayon paths share
the same graph shape and receipt acceptance; provider-native verification and
the independent serial replay remain distinct steps.

Thread count is checked against the actual pool. Nested MPI/Rayon execution
must resolve rank/thread topology once rather than create a pool per inner
operation. NUMA placement, affinity, first-touch policy, and fast reduction
performance remain separate capabilities.

### Distributed CPU execution

`eqiora-distributed` owns:

- the nonempty global algebraic space and unique ownership;
- deterministic owned/ghost ordering and global-to-local maps;
- owned-row CSR shards and sparsity-derived halo plans;
- partition, layout, and distributed-admission identities;
- transport-neutral rank-local problem and solution contracts; and
- the one-process loopback protocol oracle.

`eqiora-solver` owns the shared scalar, solver-plan, reduction-policy, and
solve-report vocabulary consumed here. `eqiora-execution` owns
transport-normalized collective phases, bounded execution traces,
process-group binding, and accepted execution receipts.
`eqiora-backend-mpi` owns MPI wire steps, status and failure records, live
collective fault agreement, and communicator execution. These boundaries do
not claim a durable rank-independent checkpoint or result identity.

MPI implements transport only. Initialization is application-owned; adapters
duplicate or borrow communicators under explicit lifetime rules and never
finalize global MPI state. The requested and actually provided thread-support
levels (`Single`, `Funneled`, `Serialized`, or `Multiple`) are admission and Run
provenance; a weaker provided level fails before distributed execution. A
one-host multi-rank case does not prove physical multi-node placement.
Replicated assembly/finish does not prove distributed mesh, assembly, field
ownership, scalability, restart, or failure recovery.

The transport-neutral graph names a logical process-group slot and exact
process count. After numerical admission, the adapter seals system, partition,
layout/halo, solver-plan, and process-group identities in one token, records a
bounded actual collective trace, independently reaccepts the complete result,
and requires every rank to agree on a domain-separated receipt summary.

Distributed mesh and assembly must precede MPI FSI. MPI+GPU is composition of
an accepted distributed plan and accepted per-rank device execution, not a new
physics lowerer.

[RFC 0061](../../rfcs/0061-mpi-fixed-mesh-fsi.md) fixes the intervening MPI FSI
boundary. RFC 0060's accepted reduced owner-row payloads are the sole source
of rank-local CSR/RHS storage; a distinct content-identical complete CSR is
retained only for identity and every-rank host acceptance, never repartitioned
into a second solver layout. The provider adds exactly reproducible,
identity-preconditioned `f64` MINRES for the finalized symmetric-indefinite FSI
operator. Its sparsity derives the solver halo, explicit global indices derive
the gathered candidate, and the unchanged FSI finish owns physical
acceptance. The full assembly target remains in lineage. Rank-count changes
may change floating-point reduction grouping, so the portable meaning and
accepted tolerances are invariant while bit-identical solutions across rank
counts are not claimed.

### Device execution

The device-neutral seam owns:

```text
DeviceIdentity
MemorySpace / BufferIdentity
TransferDirection / TransferEvidence
QueueIdentity
Completion or Fence
DeviceExecutionReport
```

“Event” is reserved for model/hybrid semantics; asynchronous backend
completion uses `Completion` or `Fence`.

CUDA adapters receive an already finalized Eqiora operator. They may bind it
to cuSPARSE/cuBLAS and resident buffers, but may not re-lower physics or infer
operator properties. Admission records the live driver, individual loaded
library versions, selected device identity and architecture, scalar/index
types, algorithm, atomics mode, stream/handle topology, and workspace policy
where applicable.

The portable graph names a logical queue slot. Deployment binds one selected
device before allocation; runtime materializes a process-unique queue identity
and records real waited completions around input transfer, solve, and output
transfer. Host replay may reconstruct successful fences only as structural
witnesses; it never re-attests the original physical waits.

The fixed-reference FSI CUDA slice follows the same rule at the physics
boundary: CPU and CUDA Realizations may select different placement and
reduction policies, but both must finalize to one exact CSR/RHS fingerprint.
The device adapter consumes only that finalized algebra, and the returned
generic accepted output must pass the unchanged FSI finish. The adapter does
not receive a CUDA-specific fluid, solid, interface, scaling, or pressure
schema.

Cold one-shot, resident/amortized, pinned-memory, pooled-allocation,
matrix-free, GPU assembly, multi-GPU, and MPI+GPU are different protocols.
Evidence for one must not set defaults for another.

The host-reference matrix-free boundary is deliberately lower and narrower.
`PacketLinearSystem` consumes one target of ordered `AssemblyWork`, reuses the
sole constraint projection, and retains mapped packet rows plus RHS without a
global CSR. It proves normal, row, transpose, diagonal, and reference-solver
composition independently of accelerator placement. It is not a device
kernel plan, a memory/performance claim, or evidence that a canonical
Realization can yet omit its independent CSR acceptance oracle.

### Time integration and hybrid execution

Eqiora owns equation-class admission and model-time semantics:

- explicit first-order ODE;
- constant mass-matrix DAE with exact structural rank evidence;
- residual-native `F(t, y, y_dot) = 0` with paired actions;
- accepted initial pairs and semantic checkpoint lineage;
- clocks, root registration, event grouping, reset atomicity, and restart.

Diffsol owns adaptive stepping and its internal solver history for admitted
first-order classes. A root finder returns an uncommitted proposal; Eqiora
owns direction, simultaneous grouping, reset, and explicit restart.
General implicit DAE primal execution remains a separate SUNDIALS IDA
candidate, not a relabelled mass-matrix path. Its first production slice is
limited to primal residual execution, consistent initialization, callback and
ownership safety, exact native-version provenance, and comparison with the
reference oracle. Forward/adjoint sensitivity, quadrature, and
checkpoint/recompute belong to a later IDAS slice and do not block primal IDA.

Backend-native checkpoint payloads are subordinate to semantic checkpoints.
They may accelerate restart but cannot become the only durable identity or
make a Run unreplayable without one library version.

### Differentiation

The canonical relation remains `R(w, p) = 0`. Lowered operators provide
primal, JVP, and VJP actions through a `LinearizedRelation`; oriented solves
provide forward and adjoint sensitivities. Solver iteration internals are not
canonical differentiation semantics.

Time-step adjoints differentiate accepted implicit residuals. Hybrid event
time, reset Jacobians, and saltation remain in the hybrid layer. Adaptive BDF
history, checkpoint schedules, backend-native derivative payloads, FSI/CAD
shape adjoints, ALE sensitivity, and remeshing sensitivity require distinct
registered slices.

### Mesh, geometry, and CAD adapters

External syntax is untrusted input. An adapter must enforce resource bounds,
normalize through Eqiora-owned constructors, and derive accepted identity from
semantic content rather than source ordering or parser handles.

Gmsh input therefore reconstructs a `SimplicialMesh`; paths, entity numbers,
parser state, and ignored sections do not become mesh identity. CAD adapters
return normalized Eqiora observations and semantic selections; kernel face
ordering and B-rep objects do not become Model or boundary identity.

XDMF metadata is a pure two-phase boundary: bounded XML produces typed array
requests, then an explicitly caller-owned resolver supplies complete source
bytes and typed values. The metadata adapter never opens the displayed HDF
locator. Fresh artifact production remains distinct from replay: only exact
agreement with independently loaded expected manifest, mesh, and ordered field
artifacts issues an opaque verified handle.

Native HDF5 resolution is a separate optional L3 adapter composed only at L4.
It accepts one complete caller-owned file image, opens it through Core VFD, and
has no path, directory, URL, or network authority. One serialized operation
fixes the native VOL, disables and restores plugin loading, audits the complete
bounded hard-link tree, and admits only exact standard `u64`/IEEE binary64
`f64`, internally stored, unfiltered, non-VDS datasets with no committed
datatype identity. Exact binding and observed native runtime versions enter
import provenance; handles and binding types do not cross the adapter. This
boundary deliberately does not claim to
contain hostile pre-initialization process state, native-library defects, or
native internal work outside Eqiora's explicit accounting. Those require an
isolated worker. Temporal collections and export remain separate slices, so
accepting XML syntax or one file image cannot silently grant time-series
authority.

Geometry, mesh, FieldSnapshot/Trajectory, XDMF/HDF5, VTU, Gmsh, and ML Dataset
remain separate artifact families. A format adapter consumes accepted durable
state; it does not define the physical field or simulation semantics.

## Dependency decisions

Versions below describe the reviewed line; current toolchain research is owned
by the [language baselines](language-baselines.md). A `Cargo.lock` is
authoritative only for Rust crates in its workspace. System MPI, CUDA driver,
loaded vendor libraries, devices, and other live native facts are discovered
at execution and recorded in accepted Run evidence. “Current” is rechecked
before a material upgrade.

| Area | Decision | Boundary and next gate |
| --- | --- | --- |
| Reference algebra/time | Keep small Eqiora oracles | Deterministic correctness seed only; never grow into a second production stack |
| Host algebra | Adopt exact-pinned `faer` 0.24.4 behind `eqiora-backend-faer` | Exact capability tuples, deployment-bound provider-release provenance, and true-residual acceptance; stronger methods/preconditioners need registered falsifiers. The package metadata names the [faer Codeberg repository](https://codeberg.org/sarah-quinones/faer) as upstream |
| CPU threading | Adopt exact-pinned [Rayon 1.12.0](https://github.com/rayon-rs/rayon) behind a Run-owned pool | Preserve indexed work and fixed reductions for reproducible mode; the accepted Run names the Eqiora Rayon-adapter build and Rayon version only when that path executed; benchmark fast/NUMA separately |
| MPI transport | Adopt [rsmpi 0.8.2](https://github.com/rsmpi/rsmpi/releases/tag/mpi-0.8.2) over a system MPI | The lock pins the Rust binding only. The application owns initialization/finalization; the adapter duplicates a communicator and records live implementation/version plus provided thread support. One fixed-reference 2D FSI slice verifies owner-routed assembly and accepted-shard MPI MINRES at 1/2/4 ranks on one host through [RFC 0061](../../rfcs/0061-mpi-fixed-mesh-fsi.md). Distributed Field output, symmetric-indefinite distributed Run artifacts, multi-node composition, scale, restart, and failure semantics remain |
| ODE/mass DAE | Adopt [Diffsol 0.16.2](https://github.com/martinjrobins/diffsol/releases/tag/diffsol-v0.16.2) behind the time adapter | Exact pin retains the BDF scratch-memory fix and includes parameter-scaled sensitivity tolerances; rerun BDF, mass, sensitivity and restart evidence on every numerical upgrade |
| General implicit DAE primal | Investigate [SUNDIALS IDA 7.8](https://github.com/llnl/sundials/releases/tag/v7.8.0) only when the reference oracle is insufficient | First FFI falsifier covers `IDACalcIC`, primal residual equivalence, callback panic containment, ABI/version, allocation ownership and clean teardown; it does not claim sensitivity or adjoint support |
| General implicit DAE differentiation | Investigate [SUNDIALS IDAS 7.8](https://sundials.readthedocs.io/en/v7.8.0/idas/) only after the primal IDA boundary is accepted and a derivative consumer exists | Forward sensitivity, quadrature, adjoint checkpoint/recompute consistency and derivative-run provenance form a distinct slice; IDAS is not bundled into primal admission |
| NVIDIA execution | Keep cudarc 0.19.9 plus narrow dynamic cuSPARSE/cuBLAS bindings as the implementation baseline | `cuda-12000` remains the binding ABI baseline, not “current CUDA”. Existing single-device conformance gates and the accepted host-staged MPI-CUDA case bound the physical checks for an upgrade. Historical public-source performance observations retain their original runtime tuple; they do not establish performance for the upgraded binding. Neither implies a general CUDA, reproducible-device, hardware-support-matrix, or scale claim. Follow [cuSPARSE](https://docs.nvidia.com/cuda/cusparse/index.html) and [cuBLAS](https://docs.nvidia.com/cuda/cublas/) reproducibility rules |
| Cross-vendor kernels | Keep [CubeCL 0.10](https://github.com/tracel-ai/cubecl/releases/tag/v0.10.0) in an unpublished experiment | Do not graduate while ordinary required `f64`, production MSRV, physical ROCm, cross-device values, cache identity and scale gates fail |
| Visualization compute | Prefer `wgpu` at the L4 Studio boundary when a real rendering slice begins | Do not use it as the first numerical solver backend. Select an exact line only after rechecking the Studio toolchain and [wgpu package MSRV](https://crates.io/crates/wgpu) |
| Gmsh parsing | Keep the narrow ASCII/binary MSH 4.1 decoder owned by `eqiora-io-gmsh` | Decode only the admitted linear-simplex grammar under Eqiora count, byte, work, and fallible-allocation budgets, then reconstruct through owned mesh constructors. Do not widen this adapter into a general MSH parser without a typed consumer and evidence |
| XDMF metadata | Retain exact `quick-xml` 0.42.0 behind an Eqiora-owned streaming grammar and resource budget | Parse only the admitted XDMF 3 Uniform simplex subset into typed HDF array requests; the XML adapter performs no I/O and never treats parser acceptance as an Eqiora mesh/Field claim. Caller-resolved and native-HDF5 execution remain distinct L4 compositions. See [quick-xml](https://github.com/tafia/quick-xml) |
| VTK XML UnstructuredGrid | Reuse exact `quick-xml` 0.42.0 behind an independent Eqiora-owned VTU grammar and resource budget | Admit only the one-Piece ASCII homogeneous affine-simplex profile, then reconstruct through shared mesh/Field invariants and L4 provenance replay. Inline/appended binary, compression, multiple pieces and export graduate independently. See the [VTK XML file-format specification](https://docs.vtk.org/en/latest/vtk_file_formats/vtkxml_file_format.html) |
| HDF5 storage | Retain exact `hdf5-metno` 0.13.0 with its static bundled HDF5 behind `eqiora-io-hdf5` | Current runtime evidence observes HDF5 2.1.0. The adapter accepts only a complete borrowed file image, fixes native VOL, suppresses plugins during the serialized operation, audits the whole hard-link tree, and resolves one fully preflighted `u64`/`f64` batch. CMake is a fresh-build prerequisite. Dependency/runtime upgrades or broader grammar rerun or replace the claim-local compatibility evidence for their exact profile. See [hdf5-metno](https://github.com/metno/hdf5-rust) and [HDF5 file images](https://support.hdfgroup.org/documentation/hdf5/latest/group___f_a_p_l.html) |
| CAD kernel | Retain exact `truck-modeling` 0.6.0, `truck-stepio` 0.3.0 and `truck-topology` 0.6.0 for the bounded adapter | Continue excluding `truck-shapeops`; `deny.toml` confines exact unmaintained exceptions `RUSTSEC-2026-0196` and `RUSTSEC-2024-0370` to the reviewed graph. Widen only with topology, healing, naming and advisory evidence. See [Truck](https://github.com/ricosjp/truck) |
| Persistent graph | Benchmark before adopting `imbl`; no production version is selected | If an experiment begins, recheck an exact maintained release at or beyond the reviewed 7.0.1 safety fix, then prove snapshot isolation, atomic validation behavior, deterministic ordering and representative graph-scale benefit. See [imbl](https://github.com/jneem/imbl) |
| Incremental syntax | Consider Rowan only with the first incremental parsing/LSP slice; no production version is selected | Rowan may store a lossless tree, while Eqiora retains grammar, recovery, typed lowering, formatting rules and stable projections/canonical bytes. Compare incremental identity, parse/format/reparse idempotence and diagnostic-span stability before adoption. See [Rowan](https://github.com/rust-analyzer/rowan) |
| Diagnostics | Use [miette](https://github.com/zkat/miette) only as a CLI presentation adapter | Stable `EQxxxx`, spans, graph paths and machine-readable diagnostics remain Eqiora types; `miette::Report` is never a library return type |
| CLI | Use [clap](https://github.com/clap-rs/clap/releases) for user-facing tools | Library APIs remain independent; patch updates follow ordinary dependency review |
| External FEM framework | Do not make [Fenris](https://github.com/InteractiveComputerGraphics/fenris) canonical | Its own status disclaims general production use/API stability, and its object model would bias the method-neutral FEM/FVM boundary |

Dependency exceptions are exact and removable. An unmaintained transitive
notice without a patched version may be accepted only when confined to one
optional adapter, documented in `deny.toml`, and re-audited before scope grows.
An actual vulnerability is not waived merely because a dependency is optional.

## Evidence admission

Admission is open to claim-local evidence under RFC 0089. Add a registered case only for a durable
claim that focused adapter tests cannot adequately falsify.

The machine-readable evidence inventory is generated, never transcribed:

```bash
cargo run --locked -p eqiora-verify -- check
cargo run --locked -p eqiora-verify -- index
cargo run --locked -p eqiora-verify -- run --case <exact.case-id>
```

`status = "verified"` requires a structured evidence target and the required
case directory. The closed runners admit workspace Cargo integration tests and
repository-owned installed-wheel Python gates; neither accepts a shell command
or free-form arguments. Local unit, TypeScript, Playwright, ignored hardware,
and experiment tests remain valuable validation; until a registered case owns
their exact claim they do not justify a ✅ verification gate in the capability
matrix.

A portable device or distributed observation records enough immutable input
and output to replay all non-hardware acceptance checks without the original
machine. A prose report without the raw observation remains historical context,
not registered verification. Performance observations additionally record,
where applicable, hardware, topology, driver and individual libraries,
precision, compiler flags, operator/matrix ordering, algorithm,
preconditioner, tolerances, setup/transfer inclusion, warmup, repetitions, and
raw samples.

## Deferred gates

These are capability gates, not promises that an enum or crate should be added
now:

- restarted GMRES, sparse direct solve, ILU/IC/AMG and domain decomposition;
- adaptive/unstructured distributed assembly, fast reductions and NUMA policy;
- production primal IDA and larger nonlinear/sparse general-DAE execution;
- IDAS forward/adjoint sensitivity and quadrature, adaptive/BDF trajectory
  adjoints, checkpoint scheduling, backend-native durable payloads and
  derivative-run provenance;
- physical multi-node canonical bridges, scale, repartition restart and
  process-failure semantics;
- resident/amortized CUDA protocols, pools/pinned memory, GPU assembly,
  matrix-free device gather/action/scatter, deterministic device policy and
  multi-GPU;
- durable execution-receipt/trace wires and a curated raw execution-graph API;
- ROCm and cross-vendor value/cache/scale evidence;
- graph-scale persistent collections and incremental syntax/LSP;
- XDMF/HDF5 temporal import and broader temporal profiles, VTU binary/appended
  payloads, compression, multiple pieces and export, broader Gmsh, and broader
  ML Dataset profiles or storage adapters;
- complete application-facing capability preview, placement, data-plane and
  progress/cancellation contracts.

Implementation starts only when its consumer and focused positive/failure tests are concrete.
Verified promotion remains unavailable without applicable independently derived evidence. Empty
provider names, target enums, and method configs do not count as progress.

Investigation may also start when a current workflow demonstrates a concrete resource,
convergence, or robustness need. Add a threshold artifact, hosted observation protocol, falsifier,
or admission envelope only when it can change a concrete support decision. A candidate enters the
supported vocabulary after its typed contract and focused positive/failure tests pass; verified
promotion additionally requires applicable registered evidence.
AMG, restarted GMRES, and field split remain three distinct product contracts rather than one
shared gate.

## Rejected or superseded approaches

- **Backend types in model or artifact schemas.** Rejected because execution
  representation would become canonical meaning and compatibility debt.
- **Delete the reference path after adopting a library.** Rejected because it
  removes a useful independent product fallback and may invalidate accepted evidence.
- **Build an entire production sparse/time/device stack in Eqiora.** Rejected;
  it duplicates mature mechanisms without strengthening semantic ownership.
- **Advertise capabilities as independent axis sets.** Superseded by exact
  tuples because cross-products admit unverified combinations.
- **Treat `HostCpu`, `CudaGpu`, a thread count, or a provider name as support.**
  Rejected; support is an admitted executable tuple with focused product tests. Verification is
  reported separately.
- **Use a universal dynamic plugin registry now.** Rejected until one typed
  provider family has multiple independently shipped implementations and a
  demonstrated discovery need.
- **Use wgpu as the first scientific solver backend.** Rejected; visualization
  portability does not supply sparse numerical maturity.
- **Make CubeCL mandatory.** Rejected while required scalar, toolchain,
  hardware and scale gates remain unsatisfied.
- **Adopt Fenris as canonical FEM.** Rejected while its status and object model
  conflict with the production, method-neutral contract.

## Red-team checklist

- Defaults can leak backend semantics even when types do not. Tolerances,
  ordering, pivoting, algorithms and preconditioners must remain explicit.
- A library residual may be recursive or preconditioned. Recompute acceptance
  against the original accepted operator.
- A GPU launch proves neither value correctness nor useful acceleration.
- A cold no-crossing benchmark says nothing about resident repeated action.
- Fast reductions can change nonlinear convergence or event decisions;
  reproducible and fast policies are distinct contracts.
- A one-host MPI run is not multi-node evidence; a two-node run is not scale.
- Replicated assembly is not distributed assembly, and gathered output is not a
  distributed result artifact.
- Rank-local Rayon pools can oversubscribe nodes unless topology is resolved
  once for the Run.
- Sparse direct and Krylov disagreement may expose an invalid
  model/realization rather than a provider bug.
- Device reproducibility depends on toolkit/library version, architecture, SM
  count, algorithm, atomics, streams, handles and workspace policy.
- Persistent collections may reduce clones while worsening locality and
  allocation fragmentation; benchmark representative transactions.
- Incremental syntax storage can change formatting and diagnostic spans;
  golden and idempotence tests remain authoritative.
- External parsers may silently ignore unsupported sections; every adapter
  must reject semantics it cannot reconstruct. The owned Gmsh decoder admits
  only its explicit MSH 4.1 subset.
- CAD face indices are not semantic boundary identity.
- Adding several libraries in one slice obscures causality. Graduate one
  provider behavior at a time.

## Open design questions

- Which production problem first justifies scalar-type or mixed-precision
  generalization beyond the current `f64` contracts?
- Which stronger preconditioner has two real consumers and a falsifying scale
  problem?
- Which physical topology and recovery contract is required beyond fixed
  distributed mesh/assembly?
- Which residual size, sparsity, or robustness requirement first exceeds the
  general-DAE reference oracle and justifies IDA?
- How should the verified host packet projection lower into a resident device
  gather/action/scatter plan without making packet storage or mesh numbering
  part of `LocalLinearActionIr`?
- Which dependency/MSRV longevity policy should begin with the first stable
  public release?

Questions remain here only while they affect the strategy. Once a bounded
consumer exists, the decision moves to an RFC and implementation Issue.
