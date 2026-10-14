# XDMF/HDF5 remeshing-trajectory export

Status: verified for the bounded 2D projection below.

This case projects the exact accepted spatial trajectory from
`fsi.remeshing-transfer-2d` into one XDMF 3 temporal Collection and one
complete HDF5 file image. The spatial trajectory and its referenced canonical
artifacts remain authoritative; the external files are a typed, replayable
storage presentation rather than a second trajectory semantics.

The projection replays the complete source prefix and the remesh and
continuation segments before producing bytes. At the zero-model-time seam it
omits the superseded source tip and presents the exact remesh target in
its place. Each frame uses current `GeometryState` coordinates. Vertex
coefficient blocks appear as XDMF Node Attributes, while every logical block
is stored under a content-addressed HDF5 dataset path. In particular, the
Cell-associated MINI bubble remains lossless in HDF5 but is deliberately
hidden from this XDMF presentation profile.

The public integration test independently parses the XML frame inventory,
audits and reads the hidden bubble through the native HDF5 resolver, round
trips the typed storage envelope, and regenerates both files after a two-second
wall-clock separation. Exact byte equality is claimed only within the one
recorded producer/runtime profile.

Run:

```bash
cargo test --locked -p eqiora --features faer,hdf5 --test remeshing_transfer_2d
cargo run --locked -p eqiora-verify -- run \
  --case artifacts.xdmf-hdf5-remeshing-trajectory
```

This evidence does not claim temporal import, arbitrary trajectory shapes,
3D or high-order export, Cell-centered XDMF presentation, compression,
chunking, filters, partial or lazy reads, parallel I/O, filesystem write
authority, production scale, or raw-byte identity across different HDF5
runtimes.

