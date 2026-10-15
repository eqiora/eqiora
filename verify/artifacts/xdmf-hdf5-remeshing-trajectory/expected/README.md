# Expected evidence

The registered target accepts only when all of the following close together:

- every presented frame is derived from a fully replayed current spatial
  state and uses its current geometry;
- the superseded source tip is absent and the exact same-time remesh
  target occupies the seam;
- XDMF contains one Time element per storage frame and no Cell Attribute;
- the canonical storage envelope replays exactly;
- every coefficient block is retained in the HDF5 image, including the hidden
  Cell-associated MINI bubble; and
- wall-clock-separated regeneration produces identical XDMF and HDF5 bytes in
  the recorded native runtime profile.

Substituted metadata/storage, a widened seam, a visible Cell block, or a
missing/changed hidden bubble therefore falsifies the bounded claim.

