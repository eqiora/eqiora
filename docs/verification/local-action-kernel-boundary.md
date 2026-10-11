# Local-action and cross-vendor kernel evidence

- Status: Eqiora local IR and host global packet composition verified; CubeCL graduation rejected
- Date: 2026-07-18
- Design record: [RFC 0020](../../rfcs/0020-local-action-kernel-boundary.md)

## Verified Eqiora boundary

The existing Cartesian Q1 diffusion cell operator lowers to one
shape-homogeneous `LocalLinearActionIr`. The registered
[`numerics.global-matrix-free-action`](../../verify/numerics/global-matrix-free-action/README.md)
case runs in dimensions one, two, and three. For each dimension it:

1. creates a three-cells-per-axis Cartesian mesh and nonzero affine Dirichlet
   boundary;
2. lowers all Q1 cell matrices in canonical cell and local-basis order;
3. gathers one reduced global vector to packed cell-local inputs, using zero
   for eliminated fixed columns;
4. evaluates the ordered reference local action;
5. scatters local outputs in canonical cell order;
6. compares with the host global packet action and separately assembled CSR;
   and
7. verifies packet transpose, diagonal, row action, constraint-aware RHS,
   identity-CG values, CSR true residual, and the affine exact solution.

The action maximum absolute difference is `3e-13`. A separate nonsymmetric
hand oracle falsifies transpose, fixed-column, skipped-row, and duplicate-
scatter mistakes. This verifies the local evaluator and one host reference
composition of the global packet action. It says nothing about a CSR-free
Realization, threading, accelerator residency, memory use, or performance.

## CubeCL 0.10.0 result

The latest evaluated CubeCL release is isolated in its own unpublished Cargo
workspace. Its source compiles the ordered and fast kernel policies without a
device. Before allocation, the adapter requires both `f64` buffer and
arithmetic use from the selected runtime.

CubeCL's common CUDA/HIP C++ type registration omits scalar `f64`. On one
physical NVIDIA RTX 6000 Ada, the CUDA test therefore returns the expected
`MissingF64Capability` before allocation. It produces no device output or
performance numbers, which is the expected outcome when the capability check
fails.

The dependency graph also contains `cubecl-zspace` 0.10.0, whose declared
Rust version is 1.92; Eqiora's production MSRV is 1.89. The experiment uses a
separate toolchain and lockfile, so this does not change production builds.

No physical AMD runner was available. Because the same upstream type registry
already prevents a conforming HIP `f64` launch, ROCm results remain untested.

## Decision

CubeCL remains an experiment and is not a production backend. Revisit after a
supported release provides normal `f64` CUDA/HIP buffers and arithmetic with
an acceptable MSRV. Then run ordered and fast value comparisons on both
vendors and add generated-kernel identity plus transfer-inclusive scale
measurements before graduation.

Vendor CSR execution is unaffected; the device-neutral local-action IR also
remains useful for a different future adapter.

## Commands

```console
cargo run -p eqiora-verify -- run --case numerics.global-matrix-free-action
cargo +stable test --manifest-path experiments/cubecl-local-action/Cargo.toml --locked
CUDA_VISIBLE_DEVICES=0 cargo +stable test \
  --manifest-path experiments/cubecl-local-action/Cargo.toml \
  --locked --features cuda --test conformance -- --ignored
```
