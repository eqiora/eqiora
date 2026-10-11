# Poisson FEM/FVM convergence evidence

This verified case tests coordinate-dependent spatial expressions and the
realization contracts together.

## Problem

```text
-u'' = pi^2 sin(pi x),  x in (0, 1)
u(0) = u(1) = 0
u_exact(x) = sin(pi x)
```

The Eqiora Language model stores the manufactured source as
`source_scale * math.sin(wave_number * coordinate(0))`. Compilation and semantic
validation produce one dimension-aware expression DAG; spatial lowering
produces one immutable scalar tape evaluated by both methods.

Both methods use the same `LineMesh` revision, four-point Gauss–Legendre cell
quadrature, constraint-aware `CooAssembler`, finalized CSR matrix,
backend-neutral `SolverPlan`, deterministic reference CG backend, and
four-point `L2` error quadrature. CSR implements the same allocation-free
`LinearOperator` contract used by production adapters. Acceptance uses an
independently recomputed true residual, not only the recursive CG residual.

- FEM: continuous linear vertex basis, cell stiffness/load operators.
- FVM: cell-centered unknowns, cell source operator, interior/boundary
  two-point facet flux operators.
- Evidence view: native FEM interpolation and an explicitly declared linear
  FVM reconstruction through boundary and cell-center values. Global balance
  uses FEM endpoint reactions or FVM outward boundary fluxes plus the same
  quadrature-integrated source.

## Results

| Cells | max h | FEM L2 error | FEM order | FVM L2 error | FVM order | FEM balance | FVM balance |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 8 | 1.25000000e-1 | 9.92091980e-3 | — | 6.12166749e-3 | — | 1.41e-16 | 2.83e-16 |
| 16 | 6.25000000e-2 | 2.48650134e-3 | 1.996357 | 1.52556058e-3 | 2.004585 | 0 | 2.12e-16 |
| 32 | 3.12500000e-2 | 6.22017793e-4 | 1.999089 | 3.81087495e-4 | 2.001145 | 2.12e-16 | 7.07e-17 |
| 64 | 1.56250000e-2 | 1.55528985e-4 | 1.999772 | 9.52529716e-5 | 2.000286 | 4.95e-16 | 5.65e-16 |
| 128 | 7.81250000e-3 | 3.88837798e-5 | 1.999943 | 2.38120617e-5 | 2.000072 | 1.13e-15 | 1.84e-15 |

CI requires strictly decreasing errors, every reported order to exceed `1.9`,
and both relative global balance defects to remain below `2e-12`. Reproduce
the numerical table with:

```bash
cargo run -p eqiora-numerics --example poisson_convergence
```

The separate
[`numerics.linear-backends`](../../verify/numerics/linear-backends/README.md)
case solves the same problem through the reference and faer backends. It
verifies that the adapters agree and that true-residual acceptance works; it
does not extend this one-dimensional result.

## Scope

The Cartesian Domain, coordinate operator, expression shape, mesh, geometry
map, reference cell, quadrature, local operator, and assembly contracts all
carry their dimension at runtime, and a five-dimensional tensor-product
quadrature is tested separately.

This case itself is 1D: the `LineMesh` and both numerical realizations. The
spatial expression tape and the scalar elliptic Cartesian lowerer have since
become dimension-independent, but that does not widen this case. Independent 2D
end-to-end evidence is in
[`cartesian-poisson-fem-fvm.md`](cartesian-poisson-fem-fvm.md). Vector and tensor
fields, mixed and high-order elements, unstructured and adaptive meshes, and 3D
realizations need their own evidence.
