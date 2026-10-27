# RFC 0026: Distributed spatial layout and replicated finish

- Status: Accepted
- Authors: Eqiora contributors
- Created: 2026-07-19

## Summary

The first canonical spatial MPI path keeps layout, placement, and numerical
policy orthogonal: every rank deterministically finalizes the same complete
Cartesian system, a distributed layout partitions its algebraic vector, an
MPI adapter solves owned shards only after collective admission, and every
rank gathers and independently reaccepts the complete vector before the
existing method-native finish.

Acceptance of this RFC did not itself claim implementation. The subsequently
registered `numerics.canonical-cartesian-poisson-mpi` case now verifies its
exact one-host, one/two/four-rank Q1 FEM/TPFA slice; the exclusions below,
especially distributed assembly/fields, physical multi-node placement,
scaling, and hybrid rank/thread execution, remain unchanged.

## Motivation

Eqiora already has three relevant boundaries:

- canonical Cartesian scalar-elliptic lowering and Q1 FEM/TPFA realization;
- an opaque finalized spatial system which can hand one immutable
  `LinearSystem` and the sole `SolverPlan` to an execution adapter; and
- transport-neutral ownership, ghost, halo, and distributed-CG contracts with
  an isolated MPI adapter.

Joining these by adding a Poisson entry point to the MPI crate would duplicate
canonical lowering in a transport adapter. Adding `DistributedCpu` to
`Target` would instead conflate where each partition executes with how a
vector is decomposed. Passing a root-only candidate to reconstruction would
also require a second broadcast/failure protocol and would make result shape
depend on rank.

The missing seam is a generic distributed linear system with durable system,
partition, and derived-layout identities. Its communication protocol must
fail collectively before rank-dependent halo patterns can diverge, and its
complete result must cross the same independent host acceptance boundary as a
CUDA-produced candidate.

## Design pass

### Current best formulation

```text
canonical model + Distributed layout + HostCpu per partition
       |
       v
deterministic complete finalization on every rank
       |
       v
complete system + unique-owner partition -> derived shards/layout/halo
       |
       v
fixed-size collective admission -> MPI Jacobi-CG -> ordered all-gather
       |
       v
serial complete-system reacceptance -> existing FEM/FVM finish on every rank
```

The complete system and reconstruction state are replicated in this first
slice. Only the algebraic solve is distributed.

### Rejected formulations

1. **A new distributed target.** Rejected because `VectorLayoutKind` already
   owns decomposition while `Target::HostCpu { threads }` owns the worker
   bound inside each partition. A target is placement, not ownership.
2. **A universal local/remote vector trait.** Rejected because it hides global
   dimension, ownership, halo communication, and complete-vector residency
   behind one superficially uniform action.
3. **Root-only reconstruction.** Rejected for v0. It saves replicated finish
   work but adds root identity, optional results, result broadcast, and a
   collective failure protocol without removing replicated assembly.

The selected formulation has higher replicated memory cost than a genuinely
distributed spatial workflow, but the smallest implementation surface and the
strongest independent oracle. It is the best falsifiable bridge between the
existing contracts.

### Relationship to prior RFCs

- [RFC 0010](0010-execution-backend-contracts.md) remains authoritative for
  the Operator/Numerics/Layout/Execution separation, application-owned MPI
  lifetime, and transport isolation. This RFC composes its existing algebra;
  it extends the already authorized distributed-to-solver vocabulary with a
  complete-CSR projection but does not widen the host `LinearOperator` into a
  distributed vector.
- [RFC 0013](0013-realization-and-run-provenance-wire.md) already reserves
  exact layout/partition digests and workers per partition. This RFC supplies
  closed schemas for those identities without reinterpreting or bumping
  Realization v1 or run-manifest v2.
- [RFC 0023](0023-finalized-spatial-linear-handoff.md) remains authoritative
  for the opaque method-native state and numerical reacceptance boundary. This
  RFC adds a retained vector layout and one distributed producer topology; it
  does not expose reconstruction state to MPI.

## Proposed design

### Orthogonal realization axes

No new `Target` is introduced. The admitted combinations are exact:

| Vector layout | Target | Accepted producer topology | State |
|---|---|---|---|
| `Replicated` | `HostCpu { threads }` | `Host { workers }`, with `workers <= threads` | existing |
| `Replicated` | `CudaGpu { device }` | `Cuda { device }`, exact ordinal | existing |
| `Distributed` | `HostCpu { threads: 1 }` | `Distributed { ranks, workers_per_partition: 1 }` | specified here |
| `Distributed` | `CudaGpu` | none | rejected in this slice |

`FinalizedScalarEllipticCartesianProblem` retains the admitted
`VectorLayoutKind` in addition to its target. Its topology check uses the
table above rather than inferring layout from the producer report. The first
MPI case resolves `HostCpu { threads: 1 }` and uses one worker per partition.
Threaded work inside each rank remains a later, independently admitted slice.

The solver-side `ExecutionTopology::Distributed` gains
`workers_per_partition: NonZeroUsize`. The existing
`ExecutionReport::distributed(adapter, ranks)` constructor remains as the
one-worker constructor and initializes the new field to one. V0 exposes no
constructor which claims a larger per-partition worker count. Adding a field
to the public enum is a pre-alpha Rust API change and requires updating
exhaustive matches, but no stable wire changes. Artifact
`ExecutionTopologyV1::Distributed` already records partitions and workers per
partition; existing `RunManifestV2` validation requires exact equality between
`Target::HostCpu::threads` and `workers_per_partition`, so both are exactly one
in this slice.

### Complete CSR mathematical projection

L2 `eqiora-solver` owns one object-safe complete-CSR projection instead of
making either assembly or distributed storage authoritative:

```text
CsrStorage
    rows / columns
    row_offsets / column_indices / values

CanonicalCsrSystemView::new(storage, right_hand_side, properties)
    captured + validated immutable slices
    Eqiora-owned fixed CSR LinearOperator action
```

`eqiora_assembly::LinearSystem` implements only the object-safe
`CsrStorage` projection. That external implementation point exposes
shape and raw CSR; it has no operator-action or identity method to override.
The RHS is supplied separately when capturing a complete system. Inherent
borrowed actions on `dyn CsrStorage` share row, transpose and adjoint arithmetic
with assembly matrices; implementors cannot override those actions. All input
coordinates must be finite, including structurally unused coordinates.
`CanonicalCsrSystemView::new` captures the matrix and RHS slices once, adds the
realization's property assertion, and validates the complete shape, ordering,
index, and finite-value contract.

The concrete Eqiora-owned view itself implements `LinearOperator` with one
fixed normal CSR row loop over exactly the captured offsets, columns, and
values. It constructs the host `LinearProblem` with itself as the operator and
its captured RHS. Consequently a storage implementation cannot advertise raw
CSR A for identity while executing hidden operator B; there is no virtual
action slot after view construction.

`DistributedLinearSystem::from_complete` and
`LinearSystemEnvelopeV1::from_complete` both consume this view. Thus the host
problem, distributed shards, and durable envelope are projections of one
mathematical source rather than independently assembled lookalikes.

The solver contract also derives a fixed-size
`CanonicalCsrAgreementFingerprintV1` from the view using an Eqiora-owned,
domain-separated binary encoding. It is an in-memory L2 agreement identity,
not an `ArtifactDigest`. L3 `LinearSystemEnvelopeV1` independently encodes
canonical JSON and owns its domain-separated artifact digest. L2 never imports
artifact types, reimplements the artifact JSON encoding, or accepts a digest
which a caller attached to an arbitrary `LinearOperator`.

### Distributed linear system

`DistributedLinearSystem` is an Eqiora-owned L2 algebra contract. It owns:

- one validated `DistributedCsr` derived from a complete, finite, square
  `f64` CSR operator;
- its complete finite right-hand side;
- asserted `LinearOperatorProperties`;
- the validated `Partition`, ordered local layouts, owned-row shards, and halo
  plan contained by that `DistributedCsr`;
- the L2 CSR agreement fingerprint plus typed partition and derived-layout
  agreement identities recomputed during construction; and
- construction of a borrowed `DistributedLinearProblem` whose right-hand side
  is selected in the local layout's explicit owned-index order.

The type does not own mesh, field, canonical model, reconstruction, MPI
communicator, or `SolverPlan`. The plan is supplied to an execution request so
the same distributed system can be evaluated under a separately resolved
policy without acquiring a second solver configuration. It does not own or
depend on `eqiora_assembly::LinearSystem`; adding an `eqiora-distributed ->
eqiora-assembly` edge would reverse the intended composition boundary. The
same `CanonicalCsrSystemView` supplies the complete host `LinearProblem` for
final acceptance; the MPI bridge never accepts an unrelated host operator plus
a caller-asserted identity.

Construction validates all global and local shapes, finite values, strictly
ordered CSR columns, unique ownership, nonempty partitions, and exact
system/partition dimension and scalar agreement. It computes its L2 system
fingerprint from the complete view before the distributed representation drops
complete host row storage. It never accepts caller-supplied ghosts or halo
exchanges; those are derived from sparsity.

`solve_and_replicate` receives the concrete complete view, recomputes its L2
fingerprint at admission, and internally obtains the host `LinearProblem` from
that view. There is no unchecked "attach this digest to this operator"
operation and no arbitrary `LinearProblem` parameter. The bridge rejects a
cross-wired view unless its fingerprint exactly equals the one retained by
`DistributedLinearSystem`. Because both the fingerprint and host action read
the view's captured slices, this proves that both representations project the
same shape, CSR, RHS, and property assertion; equality of dimensions or matrix
values alone is insufficient.

### Collective-safe admission

Before any operator-dependent point-to-point exchange or Krylov collective,
every rank derives one fixed-size, domain-separated v1 admission fingerprint.
It composes three independently typed L2 agreement identities:

1. `CanonicalCsrAgreementFingerprintV1` covers its domain tag, `f64` scalar
   tag, global row and column counts, every CSR row offset, every strictly
   ordered column index, every exact `f64::to_bits` matrix value, the complete
   RHS length and exact value bits, and the asserted
   `LinearOperatorProperties` tag.
2. `PartitionAgreementIdentityV1` covers its domain tag, global scalar and
   dimension, partition count, and the complete owner map assigning every
   global row/unknown.
3. `DistributedLayoutAgreementIdentityV1` covers its domain tag, the first two
   identities, every partition-index-ordered owned/ghost list, and every
   `(owner, receiver)`-ordered halo index list.

The admission fingerprint covers its own protocol/domain tag, those three
fixed-size identities, and every field of the sole `SolverPlan`: algorithm,
preconditioner, reduction, exact tolerance bits, and maximum iterations. This
includes shape, nonzero row boundaries, RHS, properties, owner row partition,
and the complete numerical plan; no one-axis equality can admit a cross-wire.

`CanonicalCsrAgreementFingerprintV1` is the canonical *algebraic agreement*
identity in this protocol, not Semantic Model origin and not the L3 artifact
digest. Two model/Realization histories which produce identical algebra may
share it. Their distinct origin is checked separately by derivation replay
below. Portable counts use checked `u64`; enum tags and integer byte order are
versioned by each L2 identity contract.

The v1 operator-property tags are frozen as `General = 0`,
`SymmetricPositiveDefinite = 1`, and `SymmetricIndefinite = 2`. The final tag
is an additive L2 agreement-domain extension: the fixed General and SPD golden
fingerprints remain byte-identical. It does not retag or widen the separate
distributed linear-system artifact-v1 schema, which fails closed on
`SymmetricIndefinite` until a versioned artifact wire extension is defined.

Admission uses a fixed-size all-gather record containing protocol version,
local validation status, declared partition count, local partition/rank, and
the fingerprint. Once an execution group exists, local preparation produces
either a ready candidate or a stable rejected status; the orchestration passes
that outcome into collective admission instead of applying `?` and returning
on one rank. Host/distributed system-identity comparison is part of this local
status.

The pre-record readiness reduction uses one signed 64-bit protocol word:
`i64::MAX` means ready, while a rejection encodes `rank * 256 + category`.
MPI ranks originate as signed integers and the encoding is checked before the
collective. The signed representation is deliberate: Open MPI and MPICH must
produce the same minimum when one rank rejects and every other rank is ready;
an unsigned maximum sentinel is not part of this protocol. Test launchers add
Open MPI's `--oversubscribe` option only when the selected launcher identifies
itself as Open MPI, so MPICH and other launchers never receive a vendor-only
argument. Pure test-harness classification fixtures cover representative Open
MPI, MPICH/HYDRA, and Intel MPI version output independently of an installed
launcher; multi-rank execution remains a separate transport test.

Because the complete distributed description is replicated, pre-admission
validation scans every shard on every rank, not only the local shard. It checks
all local layouts and halo references, finite RHS values, method/property
admission, and every diagonal required by the selected Jacobi preconditioner.
Allocation bounds needed to enter the first iteration are also prepared before
admission. After the gather, every rank applies the same deterministic checks:

1. all records are locally valid and use the same protocol version;
2. communicator size, declared partition count, and record count agree;
3. every communicator rank names the corresponding `PartitionId` exactly
   once; and
4. every fingerprint is byte-identical.

Any contradiction returns the same stable Eqiora diagnostic on every
participating rank before halo exchange. This prevents system, partition, or
plan drift from producing incompatible blocking communication patterns. It
does not recover from a failed process or an application that does not enter
the collective; those remain properties of the system MPI error policy.

### Collective liveness after admission

Admission agreement alone cannot make a Krylov iteration collective-safe.
After admission, no rank-local numerical or validation operation may use `?`
to return while another rank can reach a halo or reduction.

Every fallible local phase instead produces a fixed-size `PhaseStatusV1`
containing protocol version, phase and iteration indices, ready/rejected state,
and a stable diagnostic code. Immediately before the next communication
boundary, all ranks all-gather that status in partition order. They either all
continue or all return the diagnostic selected by the lowest rejected
partition. Status disagreement is itself a common protocol error. The
sequence covers:

- readiness before each halo exchange;
- local shard action and preconditioner application before the next dot/norm
  collective;
- local Krylov vector updates before the next halo or reduction;
- convergence/report construction before producer-report agreement; and
- gather preparation before the two variable-count gathers.

Each readiness status follows all fallible allocation, shape validation, and
buffer preparation needed by the immediately following communication. No
rank-local return is permitted between a common ready decision and that
communication call.

The communication schedule remains fixed by the admitted plan and iteration
index. A system MPI call which loses a process still follows the MPI
implementation's error policy; Eqiora does not claim that a status collective
can repair transport failure.

### Producer-report agreement

Before owned values are gathered, every rank derives a domain-separated,
fixed-size `ProducerReportSummaryV1` from its accepted local solution report.
The summary covers backend and execution-adapter identities, normal operator
orientation, convergence reason, completed iterations, initial-residual,
backend-reported-residual, producer-verified true-residual, and residual-target
bits, every `SolverPlan` field, and exact
`Distributed { ranks, workers_per_partition: 1 }` topology. An all-gather
requires every summary to be byte-identical. A common vector is not accepted
from reports which disagree on how it was produced or accepted.

### Ordered owner gather

`all_gather_owned` reconstructs one complete vector on every rank without a
balanced or contiguous ownership assumption.

The partition supplies each rank's expected count and ascending global
indices. The sequence is normative:

1. Derive every rank's counts and prefix-sum displacements from the agreed
   partition; check totals and convert every count/displacement to the
   adapter-private `mpi::Count`.
2. Fallibly reserve and initialize both complete receive buffers, plus the
   local explicit-index and value send buffers. No infallible capacity growth
   remains after this phase.
3. Enter one fixed-size readiness preflight carrying allocation, local
   partition, shape, producer-report, and finite-value status. Every rank
   continues or fails together.
4. Invoke the explicit-global-index variable-count all-gather and immediately
   invoke the `f64` value variable-count all-gather. There is no validation,
   allocation, `?`, or rank-local return between the two collectives.
5. Only after both collectives complete, validate rank-order receive blocks
   against the exact owner map and reconstruct global-index order.
6. All-gather the post-validation fixed-size status and only then return the
   complete vector or one common diagnostic.

Post-gather validation rejects an out-of-range, duplicate, missing,
unexpected, or non-finite entry. Global indices are never inferred from counts
or displacements, and balanced/contiguous ownership is never assumed.

This maps directly to the variable-count gather contract in
[`mpi` 0.8.2](https://docs.rs/mpi/0.8.2/mpi/collective/trait.CommunicatorCollectives.html#method.all_gather_varcount_into),
whose count and displacement representation remains private to the MPI
adapter. Large-count collectives are not claimed by v0.

### Solve, replicate, and independently accept

`solve_and_replicate` is generic distributed algebra, not a canonical-Poisson
API. It performs the following ordered phases on every rank:

1. collective admission of system, partition, and complete plan, including
   exact agreement with the supplied canonical view's recomputed L2 identity;
2. construction and solution of that rank's borrowed
   `DistributedLinearProblem` through the status-synchronized MPI Jacobi-CG
   path;
3. domain-separated all-rank agreement on the complete producer-report
   summary and exact `Distributed { ranks, workers_per_partition: 1 }`
   topology;
4. normatively ordered `all_gather_owned` into one complete
   global-index-ordered candidate; and
5. `LinearSolution::accept_with_verifier` against the host `LinearProblem`
   constructed internally from that canonical view using
   `SERIAL_LINEAR_EXECUTION` and its fixed-order inner product.

The resulting backend-neutral `LinearSolution` preserves the MPI backend,
reason, iteration count, recursive residual, and distributed producer report.
Its verification report is independently `Host { workers: 1 }`; its initial
residual, true residual, and target are freshly computed on the complete
system. The existing finalized spatial `finish` then repeats its plan,
topology, and residual checks before method-native reconstruction.

All ranks finish the same complete field and balance in v0. This symmetry is
intentional: it avoids a root-only result type while assembly and method-
native reconstruction are still replicated.

### Typed artifact schemas

Three independent, closed, bounded schemas make the in-memory agreement
replayable. All use canonical JSON, deny unknown fields, use portable `u64`
counts/indices with checked local conversion, reject non-finite numbers, and
have domain-separated SHA-256 identities.

`eqiora.linear-system-envelope/v1` contains:

- explicit `f64` scalar type and nonzero square dimension;
- CSR row offsets, strictly ordered column indices, and finite values;
- complete finite right-hand side; and
- asserted operator properties.

Construction projects `CanonicalCsrSystemView` into L3 canonical JSON; decoding
reconstructs a complete view through the same CSR/RHS/property validation used
by execution. Operator properties remain assertions of the realization; the
decoder does not infer positive definiteness by sampling. This L3 encoder owns
the artifact schema and is not duplicated in `eqiora-solver` or
`eqiora-distributed`.

`eqiora.partition-envelope/v1` contains:

- the `f64` global vector space and nonzero dimension;
- a nonzero partition count; and
- one exact owner partition index for every global index.

Decoding reconstructs `Partition`, including the v1 requirement that every
declared partition owns at least one entry.

`eqiora.distributed-layout-envelope/v1` contains:

- exact linear-system and partition artifact digests;
- local records in partition-index order, each with ascending owned and ghost
  global indices; and
- halo records in `(owner, receiver)` order, each with ascending global
  indices.

Its primary constructor derives layout and halo records from the decoded
system and partition rather than accepting arbitrary records.
`validate_against(system, partition)` rechecks both digests, reconstructs
`DistributedLinearSystem`, and requires every stored local and halo record to
equal the fresh derivation exactly.

Decoder limits independently bound input bytes, nesting, dimension,
partitions, nonzeros, owner entries, local indices, halo records, and aggregate
decoded work before large allocation or reconstruction.

### Artifact content DAG and derivation replay

The dependency graph is acyclic:

```text
RunManifestV2 -> RealizationEnvelopeV1 -> ModelEnvelopeV1
                       | layout
                       v
          DistributedLayoutEnvelopeV1 -> LinearSystemEnvelopeV1
                       | partition
                       v
              PartitionEnvelopeV1

RealizationEnvelopeV1 -> PartitionEnvelopeV1
```

Arrows point from a referencing artifact to a required artifact. They prove
content linkage only. The
distributed-layout envelope references the system and partition, the
Realization references that exact layout and partition, and the run manifest
references the Realization. The linear-system envelope does not reference the
Realization, so deriving all three layout artifacts before constructing the
Realization envelope creates no digest cycle.

Digest linkage alone does not prove that an arbitrary linear-system artifact
was derived from the linked model and Realization. The canonical 2D
verification therefore performs derivation replay: load and validate the
`ModelEnvelopeV1` and `RealizationEnvelopeV1`, re-run canonical lowering and
deterministic complete FEM or FVM finalization, project the resulting
`CanonicalCsrSystemView` into a fresh `LinearSystemEnvelopeV1`, and require
exact canonical bytes and digest equality with the linked system artifact.
CSR/RHS/properties equality is exact here; the later replicated-reference
solution comparison retains its explicit numerical tolerance.

External validation loads the model, system, partition, layout, Realization,
and run as separate artifacts and proves:

1. Model/Realization validation followed by exact system derivation replay;
2. layout-to-system and layout-to-partition digest equality;
3. exact reconstruction of every local layout and halo exchange;
4. Realization `VectorLayoutKind::Distributed` and exact layout/partition
   digest equality;
5. system/partition dimension and scalar agreement;
6. run partition count equal to the partition artifact count;
7. exact one-worker-per-partition run topology and reduction equal to the
   Realization target
   and solver plan under the existing run-manifest v2 rules; and
8. the runtime admission fingerprint recomputed from the linked system,
   partition, and plan.

`RealizationEnvelopeV1` and `RunManifestV2` are not bumped: their existing
distributed digest slots and typed topology already carry these links. The
new envelopes give those previously opaque layout/partition identities closed
schemas. Existing v1/v2 canonical bytes and decoders remain unchanged.

## Compatibility and migration

The three new artifact schemas are additive. Existing model, Realization, and
run artifacts retain byte-for-byte meaning. A distributed Realization that
references opaque layout/partition digests remains decodable, but it cannot
graduate to this RFC's replay claim unless those digests resolve to the new
typed envelopes and pass external validation.

The solver-side `ExecutionTopology` field addition is intentionally made
before 1.0. Existing constructor calls preserve one-worker behavior; direct
exhaustive matches must add the new field. No MPI type enters an Eqiora public
contract or artifact.

## Falsifying verification

`numerics.canonical-cartesian-poisson-mpi` will compile one canonical 2D
manufactured Poisson model and resolve both Q1 FEM and orthogonal TPFA with:

- generated Cartesian mesh;
- `VectorLayoutKind::Distributed`;
- `Target::HostCpu { threads: 1 }`;
- Jacobi-preconditioned CG; and
- `ReductionPolicy::Reproducible`.

The same case runs on one, two, and four ranks. For each method and rank count
it must prove:

- deterministic replicated finalization, one shared complete-CSR projection,
  Eqiora-owned view action equal to its captured raw CSR, and identical L2
  admission fingerprints;
- byte-identical typed artifact round-trip, exact content linkage, and
  Model+Realization re-finalization to byte-identical CSR/RHS/properties system
  artifact and digest;
- rank-local owned-row solve and explicit-index `all_gather_owned`;
- byte-identical all-rank producer-report summaries, followed by an MPI
  producer report plus independent complete-host verifier report;
- fixed-order phase-status agreement through every iteration communication
  boundary and the normative two-gather sequence;
- exact `SolverPlan`, fresh complete-system true-residual acceptance, and
  repeated finalized-handoff acceptance;
- method-native complete field, FEM reaction or FVM facet flux, and global
  balance; and
- agreement with the existing replicated one-worker reference under an
  explicit tolerance, without claiming bit identity across collective trees.

Negative cases must include:

- one-rank system, partition, or plan fingerprint drift before halo exchange;
- host/distributed complete-CSR projection cross-wire;
- a custom storage projection with raw CSR A and an unrelated inherent action
  B, proving that B cannot enter the storage trait and the canonical view,
  fingerprint, shards, envelope, and host residual all use A;
- communicator/partition count or rank/partition identity drift;
- system/partition/layout artifact cross-wiring, forged derived layout, and a
  content-linked system which fails Model+Realization derivation replay;
- a replicated realization carrying an MPI report and a distributed
  realization carrying a host report;
- one-rank producer backend, orientation, reason, iteration, residual, target,
  plan, or topology summary drift;
- count, displacement, global-index, total-length, or allocation overflow;
- local shape, duplicate/missing owner, and non-finite gathered value drift;
- test-only one-rank post-admission local-action and Jacobi-diagonal failure
  injection, both producing the same all-rank diagnostic before the next
  reduction rather than a deadlock;
- a wrong-method vector rejected by the receiving finalized residual; and
- distributed layout with a CUDA target.

The verification must execute through the normal repository runner and name
its exact supported environment. One-host 1/2/4-rank evidence does not inherit
the existing algebra-only physical two-node claim. Canonical multi-node
support graduates only after this exact case runs on distinct physical nodes.

## Exact nonclaims

This RFC does not specify or claim:

- distributed mesh storage, mesh partitioning, assembly, constraint routing,
  method-native reconstruction, or result-field storage;
- non-Cartesian or imported meshes, dimensions other than the planned first 2D
  case, vector/tensor/mixed fields, high order, nonorthogonal FVM, nonlinear
  systems, time integration, differentiation, or hybrid models;
- memory or communication scalability, load balance, graph partitioning,
  NUMA placement, communication overlap, scheduler integration, or more than
  one worker per rank; a general `workers_per_partition` constructor and
  `workers <= HostCpu::threads` distributed admission are deferred;
- root-only, sharded, lazy, durable, or restartable result fields;
- checkpoint/restart, repartitioning, multiple GPUs, GPU-aware MPI, or
  distributed CUDA execution;
- large-count MPI, process-failure recovery, communicator shrink/retry, or
  collective completion after a failed/nonparticipating process; or
- canonical multi-node spatial support until the exact spatial case has
  distinct-physical-node evidence.

## Security, safety, and governance

All public contracts use safe Rust, owned Eqiora values, checked arithmetic,
bounded decoding, and fallible allocation. MPI communicator, datatype,
request, count, displacement, rank, and status values stay inside the L3
adapter. No filesystem path, hostname, communicator handle, or process-local
pointer enters content identity.

External CSR storage projections expose data only. The validated Eqiora view
owns the sole host operator action, so fingerprinted bytes and accepted
residual action cannot be supplied by two independently overridable methods.

Digest equality detects accidental or adversarial content drift but is not
authentication. Signatures and trust policy remain Evidence-layer concerns.
Changing a named envelope, fingerprint domain, owner ordering, halo
derivation, or collective admission sequence requires RFC review.

## Implementation order

1. Add storage-only `CsrStorage`, the fixed-action
   `CanonicalCsrSystemView`, and the distinct L2 agreement fingerprint;
   implement only the storage projection for assembly `LinearSystem`.
2. Add the three typed L3 envelopes, independent content-link replay, and
   Model+Realization system-derivation replay.
3. Add the solver topology worker field and `DistributedLinearSystem` without
   an assembly dependency.
4. Add fixed-size collective admission, all-shard/preconditioner validation,
   per-communication phase status, and failure-injection fixtures.
5. Add producer-report summary agreement and bounded, normatively ordered
   explicit-index `all_gather_owned`.
6. Add generic `solve_and_replicate` with serial-host acceptance.
7. Retain layout in finalized spatial problems and admit only the topology
   table above.
8. Close the 1/2/4-rank FEM/TPFA verification case and only then graduate the
   capability rows.

## Unresolved questions

- Whether a later root-only or sharded result is justified by measured field
  size before genuinely distributed reconstruction exists.
- Whether large-count MPI support should be a separate adapter capability or
  wait for the selected Rust binding to expose an equally bounded safe path.
- Which distributed preconditioner first justifies a new numerical policy
  beyond Jacobi and what durable hierarchy identity it requires.
