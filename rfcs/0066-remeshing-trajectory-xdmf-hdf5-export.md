# RFC 0066: Remeshing trajectory XDMF/HDF5 export

- Status: Implemented and verified
- Authors: Eqiora contributors
- Created: 2026-07-22
- Depends on: [RFC 0025](0025-discrete-field-and-import-provenance.md) and
  [RFC 0065](0065-remeshing-correspondence-and-transfer.md)

## Summary

One accepted remeshing-aware spatial trajectory may be projected into a
complete XDMF 3 Temporal Collection and one complete HDF5 file image without
making either external format authoritative for Model, Field, mesh, state, or
trajectory meaning.

## Motivation

RFC 0065 closes one durable V2-to-V3 remesh seam, but its canonical artifact
DAG is deliberately storage independent. General visualization and later ML
Dataset work need one deterministic, bounded, inspectable external
representation of that DAG. Exporting arrays ad hoc would lose Field and
state identity, confuse a MINI bubble coefficient with a cell value, and make
the same-time remesh seam ambiguous.

## Decision

### Projection, not authority

The exact `SpatialTrajectoryEnvelopeV3` and all replayed dependencies remain
authoritative. XDMF supplies presentation metadata; HDF5 supplies primitive
array storage. A versioned storage envelope records their complete raw-byte
identities and the exact canonical artifacts from which every frame and Field
block was derived. Importing those files back into an Eqiora trajectory is a
separate claim.

The first admitted producer is one pure L3 XDMF renderer composed at L4 with
one native HDF5 file-image writer. It receives no filesystem or network
authority. The caller decides whether and where to persist the returned
bytes.

### One remesh seam has one representation

The durable V3 trajectory retains the exact V2 source prefix and begins with a
V3 target state at the same `(step, time)` as the V2 source tip. The external
sequence applies the closed policy
`target-replaces-source-at-remesh`:

1. emit every V2 state before the source tip;
2. omit the V2 source tip;
3. emit the V3 remesh target at that exact coordinate; and
4. emit the positive-duration V3 continuation.

The resulting external frames have contiguous ordinals and strictly
increasing step and time. The exporter never invents an epsilon time and never
emits two representations at one coordinate.

### Content-addressed primitive arrays

The HDF5 image contains fixed-size, contiguous, unfiltered primitive arrays at
canonical paths:

```text
/meshes/<mesh artifact digest>/topology
/geometry/<geometry-state artifact digest>/coordinates
/fields/<discrete-field artifact digest>/values
```

Each unique logical array is created once. Frames reuse its path; aliases and
hard-link presentation names are not created. The first producer uses one
complete in-memory HDF5 file image, canonical object-creation order, disabled
object timestamps, a fixed library compatibility profile, and a recorded
exact native runtime stack. Chunking, filters, extendible datasets, partial
I/O, parallel HDF5, and cross-runtime raw-byte identity are nonclaims.

### Presentation is narrower than storage

Every coefficient block of every selected Field snapshot is stored losslessly
in HDF5. XDMF presents only blocks whose association and basis semantics make
them genuine nodal values in the first profile.

In particular, the MINI cell bubble block is a coefficient of a normalized
cubic basis function. It is not a cell average and must not be emitted as an
XDMF `Center="Cell"` Attribute. The storage envelope marks every block as
either `xdmf-node-attribute` or `hidden`; the first profile permits the former
only for a vertex-associated block. A later genuine cell-valued Field may add
a separately verified presentation profile without reinterpreting bubble
coefficients.

### Exact storage lineage

`XdmfHdf5TrajectoryStorageEnvelopeV1` records only format-specific lineage:

- exact adapter ID and version, plus ordered native runtime components;
- the exact V3 trajectory artifact digest and fixed seam policy;
- complete XDMF and HDF5 raw SHA-256 values and byte counts;
- ordered frame ordinal, step, coherent-SI time, V2/V3 state kind, exact
  spatial-state, mesh, and geometry-state artifact digests;
- each Semantic Field and support Domain identity plus exact snapshot digest;
- each coefficient association, logical DiscreteField digest, exact HDF5
  dataset path, and presentation kind.

Units, value shape, frame, Model, Realization, run, and numerical evidence are
not duplicated: the referenced canonical artifacts already own those facts.
The envelope is a closed canonical DTO with independent byte, nesting, text,
frame, Field, block, and runtime-entry decode limits.

### Layer ownership

```text
eqiora-artifact  exact storage-lineage wire and replay checks
eqiora-io-hdf5   bounded native primitive-array file-image writer
eqiora-io-xml    pure bounded Temporal Collection renderer
eqiora-api       private V2/V3 traversal and authority-bearing composition
eqiora           curated opaque verified export handle
```

No universal trajectory view, artifact registry, filesystem writer, or new
crate is introduced. A shared traversal abstraction may be extracted only
after a second real consumer, expected under ML Dataset work, proves the common
contract.

## Alternatives considered

### Emit both same-time states

This preserves every stored state but gives visualization consumers two
incompatible meshes at one time with no portable replacement meaning. It is
rejected.

### Add an epsilon to the target time

This makes files easy to order by changing physical time. It is mathematically
false and rejected.

### Treat all coefficient blocks as XDMF Attributes

This is superficially uniform but mislabels basis coefficients as sampled
physical values. Lossless storage and truthful presentation are separated
instead.

### Make XDMF/HDF5 the durable trajectory wire

External formats do not retain Eqiora's complete typed identity and replay
obligations. They remain projections of the canonical artifact DAG.

### Introduce a public generic trajectory interface now

V1, V2, and V3 have materially different dependency and transition semantics.
One producer is insufficient evidence for a stable public common abstraction,
so the first normalization seam remains private at L4.

## Compatibility and migration

This RFC changes no existing Model, mesh, Field, geometry-state, spatial-state,
trajectory, or external-import bytes. It adds one new format-specific artifact
schema and additive facade operations. Future storage profiles require new
versioned identities or an explicitly compatible extension; they cannot widen
v1 silently.

## Verification

The registered case must falsify at least:

- stale, substituted, reordered, missing, or duplicate roots, segments, and
  states;
- both same-time representations being emitted, source-tip retention, or an
  invented time offset;
- noncontiguous ordinal, nonincreasing step/time, or Field inventory drift;
- reference coordinates substituted for current coordinates or a source mesh
  used after remeshing;
- missing, substituted, cross-wired, or reordered snapshots and coefficient
  blocks;
- a MINI bubble block exposed as a cell-centered XDMF Attribute;
- HDF5 path, scalar type, shape, value, or raw-byte substitution;
- XDMF metadata or HDF5 file-image substitution;
- every configured resource budget;
- declaration-order sensitivity; and
- repeat generation across a wall-clock delay under the recorded producer
  profile.

The positive case must parse the emitted XML independently, audit the complete
HDF5 image through the accepted native reader, reconstruct every stored array,
and match all storage-envelope references. Visualization-tool compatibility is
useful observation but not semantic proof.

The registered
[`artifacts.xdmf-hdf5-remeshing-trajectory`](../verify/artifacts/xdmf-hdf5-remeshing-trajectory/README.md)
case closes this first profile through the public facade. Layer-specific unit
tests retain the resource-limit, structural substitution, declaration-order,
and presentation-grammar falsifiers; the registered end-to-end case replays
the exact canonical trajectory, independently parses the emitted frame
inventory, audits the complete HDF5 tree, reads the hidden MINI bubble, and
proves wall-clock-separated regeneration under one recorded runtime profile.

## Security and safety

The pure XDMF renderer opens nothing. The native HDF5 writer disables dynamic
filter/plugin loading across its serialized native boundary, uses the Core VFD
without backing storage, preflights rank and value products, and limits the
complete returned file image. Decoding the storage envelope performs no I/O.
Persisting returned bytes is explicit caller authority.

## Nonclaims

- temporal XDMF/HDF5 import or round-trip trajectory reconstruction;
- 3D, high-order, mixed-topology, or multiple-remesh trajectories;
- general cell-centered, face-centered, tensor, complex, or non-f64 Field
  presentation;
- HyperSlab, Function, List, Range, lazy, streaming, incremental, compressed,
  chunked, parallel, SWMR, or remote storage;
- visualization correctness for hidden basis coefficients;
- ALE sensitivity, remesh sensitivity, adjoints, or ML Dataset semantics; and
- bit-identical HDF5 bytes across different native library, ABI, architecture,
  or producer profiles.
