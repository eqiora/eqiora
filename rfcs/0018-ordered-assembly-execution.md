# RFC 0018: Ordered assembly execution

- Status: Implemented reference/threaded assembly and host packet action
- Authors: Eqiora contributors
- Created: 2026-07-18

## Summary

Eqiora lowers entity-local numerical work to an indexed stream of finite
assembly packets; serial and parallel adapters may evaluate packets under
different placement, but they scatter packets and local entries in one fixed
logical order into typed assembly targets.

## Motivation

`eqiora-numerics` currently owns local contributions, constraint maps, COO
accumulation, CSR storage, spatial realization, linearization, and solver
orchestration. This remains understandable at the present scale, but adding
Rayon, MPI owner routing, device COO, and external sparse assembly directly to
the crate would turn it into the backend container the architecture is meant
to avoid.

The existing `CooAssembler` is also inherently serial. Parallelizing each
spatial loop in place would duplicate error handling and create race-dependent
floating-point accumulation. Letting `eqiora-backend-rayon` depend on
`eqiora-numerics` would add an L3-to-L3 dependency explicitly rejected by RFC
0010. The reusable boundary is the ordered contribution stream between local
operator evaluation and a finalized algebraic system.

## Proposed design

### Crate ownership

A new L2 crate, `eqiora-assembly`, owns one executable contract:

```text
LocalContribution     anonymous finite dense matrix + rhs
AssemblyMap           local rows/columns -> free/fixed global algebra
AssemblyTarget        one typed square output system and its dimension
AssemblyPacket        one local contribution + maps to one or more targets
AssemblyWork          indexed pure packet evaluation
AssemblyBackend       work + plan -> systems + report
PacketLinearSystem    work + one target -> mapped packet action + rhs
CooAssembler          small deterministic reference accumulator
CsrMatrix             Eqiora-owned finalized host sparse representation
```

This is not a vocabulary-only crate. In the first implementation it has two
backends (direct reference and run-owned Rayon), two consumers
(`eqiora-numerics` and `eqiora-backend-rayon`), and one canonical P1 FEM path.

`eqiora-numerics` continues to own discrete spaces, geometry/coefficient
contexts, spatial local operators, canonical realization, reconstruction, and
physics evidence. Its `LocalOperator` returns the L2 `LocalContribution`.
Assembly types are imported from their owning `eqiora-assembly` crate; the
public facade exposes that contract through `eqiora::assembly`, not through
`eqiora::numerics`.

`eqiora-backend-rayon` depends only on the new L2 contract and the existing solver/
realization vocabulary. Neither L3 crate depends on the other. MPI and CUDA
adapters may later consume the same packet contract without entering spatial
semantics.

### Plans, targets, and packets

An `AssemblyPlan` contains a nonempty ordered list of nonempty square targets.
`AssemblyTargetId` is an opaque typed ordinal scoped to one plan. A packet
contains exactly one `LocalContribution` and a nonempty, target-ordered set of
`TargetAssemblyMap` values. Target IDs in a packet must be unique and valid for
the plan.

One local evaluation may feed multiple algebraic views. The initial P1 FEM
work uses two targets:

```text
target 0: reduced free-DOF system used by the solve
target 1: uneliminated full system used for reaction evidence
```

Each cell evaluates diffusion, source, basis gradients, and quadrature once.
Its packet carries a constrained map for the reduced target and an
unconstrained map for the full target. Natural-boundary packets likewise map
to both. Target ordinals are realization-local and do not enter Semantic Model
or artifact identity.

`AssemblyPacket::new` validates local/map shape and target uniqueness before a
backend observes it. Global target bounds are checked during ordered scatter.
All fixed values and contribution entries must be finite.

### Work and deterministic failure

`AssemblyWork` is an object-safe, `Sync` indexed action:

```text
packet_count()
evaluate(packet_index) -> AssemblyPacket
```

Every index has stable meaning for the lifetime of one call. Evaluation must
be pure with respect to assembly output: it may read immutable mesh, geometry,
quadrature, fields, and coefficients, but it may not mutate a global matrix or
depend on execution order. Backend scheduling therefore cannot acquire
mathematical meaning.

The backend checks plan shape and packet count before allocating output. The
first failing logical packet is the reported failure. No `AssemblyResult`
escapes unless every packet and every target finalizes successfully. Internal
partial accumulators are dropped on failure.

### Reference and Rayon execution

The reference backend evaluates packets from index zero upward and scatters
each packet immediately. Within a packet, targets are processed by ascending
target ID; within each map, local rows and columns retain row-major order. COO
entries finalize in ascending global `(row, column)` order into CSR.

The Rayon backend uses the exact pool owned by `CpuThreadPool`. It evaluates a
bounded fixed batch of consecutive packet indices concurrently, collects the
indexed results in logical order, selects the lowest failing index, and only
then scatters the successful batch through the same reference accumulator.
Batch size affects memory and scheduling only. It cannot change the scatter or
floating-point expression order.

Assembly and solve placement are recorded separately even when they share one
pool. `AssemblyReport` retains exact execution identity, worker topology, and
accepted packet/target counts. `SolveReport` retains solver execution. A
caller cannot infer that one report applies to the other.

The first Rayon assembly backend has only the reproducible policy. Fast
thread-local maps followed by an unordered merge, graph coloring with dynamic
ownership, atomics, and backend-native device COO are later policies with
separate evidence.

### Host-local packet action

`PacketLinearSystem::from_work` consumes one target of the same pure
`AssemblyWork`, in increasing packet order, and invokes
`AssemblyPacket::project` as the sole constraint and local-to-global mapping
rule. It retains canonical mapped rows per packet and the accumulated RHS; it
does not construct or retain global CSR storage. A temporary canonical
coordinate accumulation applies the same exact-zero structural-row gate as
reference assembly, then is discarded. Fixed columns disappear from the
homogeneous operator and contribute only through the projected RHS.

The resulting `PacketLinearOperator` implements complete-vector normal,
row-range, transpose, and diagonal actions through caller-owned buffers. A
successful finite action does not allocate merely to return its output. Normal
action visits packet, projected row, then ascending global column and adds into
the selected global row. Transpose action visits the same immutable sequence
and adds each reversed contribution into its global column. Row-range action
scans the same packets and therefore preserves the per-output expression order
even when disjoint ranges are placed independently.

This is a deterministic host reference representation, not a claim that
packet-local storage is smaller or faster than CSR. The first registered case
uses identity-preconditioned reference CG and an independently assembled CSR
oracle. General source assembly without local packets, threaded solve evidence,
distributed ownership, device gather/scatter, heterogeneous local-action IR,
and a canonical Realization that never materializes its independent CSR oracle
remain separate gates.

### Initial canonical path

`solve_scalar_elliptic_linear_fem_with_assembly` receives an
`AssemblyBackend`. The existing entry point supplies the reference backend and
preserves source ergonomics. Canonical resolved execution gains a corresponding
explicit-assembly path used by the threaded verification case.

The returned P1 solution retains both its `AssemblyReport` and `SolveReport`.
The first claim is one-dimensional P1 FEM with essential or mixed endpoint
conditions. FVM, Cartesian 2D/3D, simplex, tangent assembly, distributed owner
routing, and device residency remain unclaimed until migrated and verified.

## Alternatives considered

### Add Rayon directly to `eqiora-numerics`

This is the shortest implementation and avoids moving types, but makes a
specific scheduler part of the spatial realization crate and invites MPI/CUDA
branches beside every cell loop. It has low initial cost and poor architectural
faithfulness. Rejected.

### Let `eqiora-backend-rayon` depend on `eqiora-numerics`

This isolates Rayon source but creates an L3-to-L3 edge from adapter to a
concrete spatial realization. It also prevents non-spatial assemblers from
using the contract and contradicts RFC 0010. Rejected.

### Parallel thread-local COO maps followed by a tree merge

This is likely faster than ordered scatter, but the merge tree changes with
worker/task partition and therefore changes cancellation behavior. It is a
promising `Fast` policy, not reproducible evidence. Deferred.

### Color the mesh and scatter same-color cells concurrently

Coloring can provide unique writers and avoid atomics, but color construction,
ordering, and balance become new realization artifacts. It is valuable for a
later production assembler and matrix-free execution, but unnecessary for the
first falsifiable boundary. Deferred.

### Send one precomputed vector of contributions to Rayon

This parallelizes only bookkeeping after local physics has already run
serially. It proves less than the required assembly boundary and increases
memory without exposing useful concurrency. Rejected.

The selected design has the highest migration cost of the immediate options,
but it is mathematically faithful to entity-local weak-form evaluation,
preserves the reference accumulation tree, and provides an executable seam for
host, distributed, and device adapters.

## Compatibility and migration

No Semantic Kernel or wire schema changes. Rust assembly types are owned and
exported only by `eqiora-assembly`; facade users import them through
`eqiora::assembly`. The spatial `LocalOperator` trait remains internal to
`eqiora-numerics`; its implementations and consumers are all crate-local.

Reference assembly order and CSR bytes remain unchanged. The ordinary solve
entry points select the reference backend, so existing solutions and golden
tables must be bit-identical. New explicit-assembly entry points and assembly
reports are additive. Trait implementors and exhaustive struct construction
may require 0.x source migration.

`PacketLinearSystem` and `PacketLinearOperator` are additive in-memory Rust
contracts. They add no Semantic Model, artifact wire, digest, provider
registry, or persistence field. Their deletion condition is replacement by a
strictly more general packet action that preserves the same projection and
`LinearOperator` behavior; accelerator-specific storage is not such a
replacement.

## Verification

The first implementation must prove:

1. reference assembly produces the exact pre-migration CSR/rhs fixtures;
2. serial and four-worker assembly are bit-identical for more than one Rayon
   batch and for a cancellation-sensitive shared entry;
3. packet target order cannot change accumulation order;
4. one P1 cell evaluation feeds reduced and full systems, and reactions remain
   identical to the pre-migration oracle;
5. one canonical Poisson revision produces identical fields, reactions,
   assembly structure, CG values, and numerical report fields through direct
   and four-worker assembly;
6. target mismatch, out-of-range DOF, non-finite accumulation, empty work,
   lowest-index packet failure, and final empty CSR rows fail with stable
   diagnostics and expose no partial result; and
7. formatting, MSRV, Clippy, full workspace tests, verification manifests,
   dependency layers, and documentation pass.

The global packet-action extension additionally proves a hand-calculated
nonsymmetric forward/transpose/RHS oracle with skipped rows, fixed columns,
and duplicate scatter; constrained Cartesian Q1 action, transpose, diagonal,
and identity-CG solution agreement with separately assembled CSR storage and
action in dimensions one through three; and fail-closed invalid work, index,
shape, buffer, and finite-value handling. Registered evidence is
[`numerics.global-matrix-free-action`](../verify/numerics/global-matrix-free-action/README.md).

## Research basis

- [PETSc matrix assembly](https://petsc.org/release/manual/mat/) separates
  local dense-block insertion/addition from final storage assembly, recommends
  ownership-aware insertion for distributed matrices, and uses a distinct COO
  path for GPU assembly.
- [Rayon indexed parallel iterators](https://docs.rs/rayon/latest/rayon/iter/trait.IndexedParallelIterator.html)
  preserve logical index identity and ordered collection while allowing task
  scheduling to vary.
- [PETSc `MATSHELL`](https://petsc.org/release/manualpages/Mat/MATSHELL/)
  separates a caller-defined operator action from explicit matrix storage and
  from matrix-dependent preconditioning.
- [MFEM assembly levels](https://mfem.org/howto/assembly_levels/) distinguish
  full, element, partial, and on-the-fly operator representations, while its
  constrained operators keep essential elimination outside the local action.

Eqiora adopts neither library's types. They inform the separation between
local work, constraint projection, global action, placement, accumulation,
and optional finalized storage.

## Security, safety, and governance

The first implementation uses safe Rust. Plans and packets are bounded by the
already validated realization/mesh size; constructors use checked arithmetic
before allocation. Rayon evaluation runs only Eqiora-linked code and cannot
load plugins or arbitrary kernels. Panic does not become accepted assembly
evidence.

A new assembly backend or accumulation policy changes a numerical capability
claim and requires independent structure/value/residual evidence. External
format or device handles may not enter this L2 API.

## Unresolved questions

- Whether repeated nonlinear/time-dependent assembly introduces a versioned
  sparsity-pattern reuse contract.
- Which fast assembly policy (thread-local COO, coloring, row ownership, or a
  backend primitive) wins representative benchmarks.
- How assembly reports link to a future signed evidence bundle without making
  ephemeral target ordinals artifact identity.
- When distributed assembly graduates from owner-routed logical evidence to a
  PETSc adapter or Eqiora MPI implementation.
