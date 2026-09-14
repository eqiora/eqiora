# RFC 0019: Device execution contracts

- Status: Implemented first physical vertical slice
- Authors: Eqiora contributors
- Created: 2026-07-18

## Summary

Eqiora represents device discovery, typed buffer residency, ordered queue
submission, completion, explicit transfer, and phase timing with
backend-neutral L2 contracts. Concrete CUDA contexts, streams, events,
pointers, library handles, descriptors, and workspaces remain inside one L3
adapter. A `CudaGpu` Realization target becomes executable only when that
adapter admits the requested capabilities and produces hardware evidence.

## Motivation

The Semantic Model already separates model-time clocks and hybrid events from
execution placement. The first device implementation must preserve that
boundary. Reusing `Event` for CUDA completion would give one word two
incompatible meanings; hiding transfer and synchronization inside a solver
call would make performance and ownership impossible to audit.

CUDA APIs add lifetime constraints that a flat function cannot express. The
current cuSPARSE Generic API describes the external SpMV workspace as device
memory and requires the same active buffer across repeated calls associated
with preprocessing. The current cudarc API separately owns a `CudaContext`, a
`CudaStream`, and typed `CudaSlice<T>` allocations. Eqiora therefore mirrors
the durable concepts without copying vendor types into its contracts.

## Proposed design

### Orthogonal identities

```text
RuntimeId + DeviceId       discovered placement
BufferId + element type    allocation identity and typed residency
QueueSlot                  logical deployment selection
QueueId + SubmissionId     total order within one materialized command queue
Completion                 fence identity for one submission
TransferPlan               source, destination, direction, shape, bytes
DeviceExecutionTimings     setup / H2D / solve / D2H / verification / total
```

`Completion` is deliberately not a Semantic Model event. `QueueSlot` does not
establish order: an adapter assigns a process-unique materialization identity
to each concrete vendor queue or stream. Ordering is defined only for two
completions on that same complete `QueueId`. Cross-queue dependencies require
an explicit wait or future dependency contract; the L2 layer never invents a
total order.

The shared scalar-storage vocabulary belongs at L0. `ScalarType` therefore
moves from `eqiora-solver` to `eqiora-core` and remains re-exported by the
solver crate for source compatibility. Device buffers use a typed
`DeviceBuffer<T>` seam, while descriptors retain only Eqiora-owned identity,
shape, and residency.

### Crate ownership

```text
eqiora-core          ScalarType
eqiora-device        L2 identity, capability, buffer, queue, transfer, evidence
eqiora-assembly      L2 finalized CSR operator
eqiora-solver        L2 SolverPlan and accepted numerical evidence
eqiora-backend-cuda  L3 cudarc, cuSPARSE, and cuBLAS adapter
```

The device contract does not depend on solver or assembly. The CUDA adapter
depends downward on all required L2 contracts. No CUDA dependency enters the
default facade or Semantic Model graph.

### First executable slice

The adapter accepts a validated nonempty `f64` CSR matrix. It checks shape,
finite values, and lossless `usize` to vendor-index conversion before device
allocation. One selected device must advertise `Float64`, CSR SpMV, dense
level-1 vector primitives, and an asynchronous queue.

The first action uses the cuSPARSE Generic API with CSR descriptors and a
retained external workspace. It uses the deterministic CSR algorithm when a
reproducible action is requested; a backend-native fast choice remains a
separate capability. CPU and GPU results are compared under a declared
floating-point tolerance, never accidental bit identity.

The solver slice consumes the sole `SolverPlan`. CG requires an asserted
symmetric-positive-definite operator; BiCGSTAB admits a general square
operator. Identity uses cuBLAS copy, Jacobi uses a zero-band cuBLAS matrix
action over a checked inverse diagonal, and all Krylov vector algebra remains
resident. cuBLAS host scalar pointer mode is explicit and atomic routines are
disabled. The adapter still advertises only `Fast`: a vendor-native dot/norm
does not become Eqiora's named reproducible expression tree merely because
atomics are disabled.

An accepted solution is copied to the host and its true residual is recomputed
through the independent Eqiora CSR action and fixed-order host norm. The solve
report records two placements: the CUDA execution that produced the values and
the serial-host execution that verified them. Neither is relabelled as the
other.

### Synchronization and evidence

Successful enqueue allocates a monotone `SubmissionId`; its recorded fence
owns the corresponding `Completion`. A transfer completion queue must reside
on at least one device endpoint. Device-to-device plans do not imply peer
access: the adapter must negotiate that separately.

Wall-time phases are measured independently. They need not sum to `total`
because asynchronous work can overlap, but no phase may exceed the observed
end-to-end duration. Setup, H2D, solve/action, D2H, and independent host
verification stay separately visible.

## Alternatives considered

### Add `solve_on_cuda` to the numerical realization

This is short, but it couples method code to device ownership and makes
transfer, queue, workspace, and completion lifetimes implicit. Rejected.

### Expose cudarc types as the public contract

This gives a convenient first adapter but makes a library version and CUDA
runtime model part of Eqiora's stable API, preventing a clean ROCm or other
device adapter. Rejected.

### Make every operator generic over host and device buffers

This would spread device type parameters through the reference CPU and
Semantic Model paths. The first production GPU path is assembled CSR, so the
adapter consumes that explicit lowered artifact instead. Rejected.

### Require phase durations to sum to total

That identity is invalid for overlapping asynchronous transfer and compute.
The contract retains each observation and only requires it to fit within the
end-to-end interval. Rejected.

## Compatibility and migration

The current pre-1.0 surface imports `ScalarType` from its L0 owner,
`eqiora_core`; the displaced solver re-export is removed. No Semantic Model
or artifact wire changes. Adding a device execution topology to accepted run evidence will be
versioned with its run-provenance schema rather than smuggled into opaque
strings.

The CUDA dependency is optional. cudarc 0.18.2 remains exact-pinned because it
is the binding/runtime line covered by the registered device evidence. The
Rust 1.89 workspace can investigate current 0.19.x, but compiler compatibility
does not replace API, ABI, dynamic-library, or physical-device revalidation.
Because the 0.18 generated cuSPARSE loader eagerly resolves unrelated
release-specific symbols, the private FFI boundary loads only the Generic API
functions Eqiora executes and retains the library for all object lifetimes.
Default, MSRV, and all-features builds on a machine without a CUDA device must
compile and run non-hardware tests. Runtime absence returns a typed unsupported
diagnostic; it never silently selects the CPU path. RFC 0059 records the MSRV
amendment without changing this evidence boundary.

## Verification

1. Reject mismatched transfer shapes, host-only copies, identical device
   endpoints, and queue/device mismatch.
2. Prove monotone completion identity on one queue and reject cross-queue
   ordering.
3. Reject invalid or non-losslessly-convertible CSR before allocation.
4. Compare CUDA `f64` CSR action with the independent host action.
5. Solve one SPD case with CG and one nonsymmetric case with BiCGSTAB, then
   independently check each true residual on the host.
6. Record setup, H2D, execution, D2H, verification, and total timing.
7. Run default/MSRV/all-features CI without requiring hardware, and run an
   explicit conformance case on one selected physical NVIDIA device.

## Research basis

- [cudarc 0.18.2](https://docs.rs/cudarc/0.18.2/cudarc/) supplies the
  context, stream, and typed allocation path exercised by the registered
  device evidence. Current
  [0.19.8](https://docs.rs/cudarc/0.19.8/cudarc/) informs the forward API
  direction and fits the workspace MSRV, but compiler compatibility does not
  replace adapter review and fresh physical-device evidence.
- [cuSPARSE Generic API](https://docs.nvidia.com/cuda/cusparse/generic-api/generic-api-functions.html)
  defines CSR SpMV descriptors, external workspace, supported index/scalar
  types, and deterministic versus default algorithms.
- [cuBLAS](https://docs.nvidia.com/cuda/cublas/) defines host/device scalar
  pointer modes, level-1 vector actions, diagonal band actions, atomics mode,
  and the narrower conditions under which results are repeatable.
- [CUDA Toolkit release notes](https://docs.nvidia.com/cuda/cuda-toolkit-release-notes/index.html)
  are checked before implementation for Generic API behavior and known
  workspace constraints.

These libraries own execution mechanics. Eqiora owns admission, numerical
policy, stable diagnostics, and conformance evidence.

## Security, safety, and governance

Workspace crates deny unsafe Rust by default. The CUDA adapter may allow it
only in a private FFI module, with a `SAFETY` argument at every call covering
context binding, pointer provenance, shape, aliasing, descriptor, workspace,
and synchronization lifetime. Untrusted source cannot supply a raw device
pointer, library path, kernel binary, or queue handle.

Adding a capability claim, changing a reproducible algorithm, or widening the
accepted scalar/index space requires executable evidence and review.

## Unresolved questions

- The exact durable run-provenance schema for CUDA driver/library/device
  versions.
- The exact Eqiora-owned device reduction tree required before a CUDA solver
  may advertise `Reproducible` across runtime placements.
- Which memory-pool and pinned-host strategy is justified by scale evidence.
