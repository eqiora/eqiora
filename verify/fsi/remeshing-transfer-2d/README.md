# Conservative ALE FSI remesh transfer in 2D

Status: verified for the bounded serial-host 2D slice below.

This case closes the first topology-changing continuation slice from [RFC
0065](../../../rfcs/0065-remeshing-correspondence-and-transfer.md). One accepted
moving FSI state crosses a zero-model-time remesh seam between two affine
triangle revisions with different vertex, cell, and interface-facet counts.
The canonical Model and semantic GeometryIdentity do not change.

The source and target meshes share neither an index correspondence nor a
one-to-one tessellation. The target uses independently connected centered
body-wise fans rather than a copied structured-grid connectivity; shared
coordinates remain geometry and never become implicit mesh identity. The
material-solid and current-spatial-fluid common refinements both contain source
cells split across target cells and target cells assembled from multiple source
cells. The old two-facet interface is replaced by a four-facet interface while
retaining every source P1 trace kink. This makes the case a topology-changing
transfer witness, not a renumbering or diagonal-flip example.

## Verified evidence

The Cargo integration target executes the complete vertical slice:

- direct canonical V5 lowering and independent source/target Realization
  resolution;
- one accepted source step whose MINI bubble, nonconstant absolute pressure,
  shared velocity, and material displacement are all nonzero;
- a target interface that refines every source trace breakpoint, while the
  non-nested volume grids still force genuine many-to-many overlap;
- exact replay of each material P1 midpoint displacement followed by `X + d`,
  with every source breakpoint retained and an integer-dyadic proof that the
  exact current source midpoint lies in the target point's binary64 rounding
  cell in both components. The midpoint need not itself be exactly
  representable; one-ULP interpolation/support mutations fail;
- absolute solid-displacement projection in the material chart, followed by
  derivation of target harmonic geometry;
- one coupled velocity projection using current-spatial fluid integration and
  material solid integration, with shared and exterior trace defects recorded
  separately;
- absolute-pressure projection without gauge recentering;
- independent replay of projection residuals, weak incompressibility, total
  momentum, pressure moment, displacement trace, and harmonic geometry;
- independent assembly and solve-report replay under two distinct L/U/P
  computational scale profiles, with the same overlap connectivity and
  physical constraint roles/counts satisfying each profile's dimensionless
  acceptance contracts; their physical Fields are compared in common L/U/P
  units against the case manifest's non-semantic observation bound, without
  claiming bit-identical iteration or coefficients;
- finalization of the transferred state through the ordinary target ALE FSI
  operator and one strictly positive-time backward-Euler step; and
- an immutable source trajectory prefix followed by content-addressed
  remesh and target-continuation segments, including canonical JSON replay.

The seam fails closed for stale or swapped mesh identity, changed semantic or
time-step identity, changed transfer policy, stale Field inventories, a
zero-duration ordinary step, a non-tip source state, or a broken target
predecessor.

Run:

```bash
cargo test --locked -p eqiora --features faer --test remeshing_transfer_2d
cargo run --locked -p eqiora-verify -- run --case fsi.remeshing-transfer-2d
```

This evidence is a deliberately bounded 2D affine-triangle CPU reference. It
does not claim a production remesher, an adaptive estimator, high-order or 3D
transfer, changing semantic geometry, ALE/remesh sensitivity, GPU or MPI
remeshing, durable checkpoint/restart, or process resume.

The registered scale-profile observation bound is one part per billion of the
corresponding characteristic L, U, or P value. It is a regression sensitivity
for this CPU reference case, deliberately separate from the numerical and
physical acceptance contracts and far below the slice's discretization error.

## Projection solver substitution

The same public transfer/continuation fixture runs reference MINRES and BiCGSTAB
with Identity/Reproducible, and Faer BiCGSTAB and SparseLU with Identity/Fast. Equations,
physical scales, mesh revisions, constraints and the original solver controls
remain fixed. Exact provider capability checks require both SPD mass and
symmetric-indefinite velocity projections. The KKT multiplier block has zero
diagonal, so this projection requires Identity; that does not exclude Jacobi
from other problems with an invertible diagonal.

All four paths consume canonical CSR captured from the already assembled dense
coefficients, then independently replay the original raw-matrix residual and
physical constraints. The existing per-matrix sixteen-million coefficient limit,
four-million auxiliary limit and sixty-four-million orthogonalization-work limit
remain unchanged. Raw dense coefficients and two copies of the nonzero values
coexist during capture; these limits do not bound total heap use, indices, RHS
or Krylov storage. No factor retention or universal convergence is promised.
