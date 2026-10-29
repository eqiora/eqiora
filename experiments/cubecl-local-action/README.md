# CubeCL local-action experiment

This unpublished workspace evaluates Eqiora's `LocalLinearActionIr` against
CubeCL without adding CubeCL to the production dependency graph.

So far the experiment has failed its graduation check and provides no GPU
support. The historical CubeCL 0.10.0 CUDA check found no ordinary `f64`
buffer/arithmetic capability in its shared CUDA/HIP C++ backend, so Eqiora
rejected the runtime before allocation. The current CubeCL 0.11.3 update has
not been validated on physical hardware. The experiment uses Rust 1.95 or newer because the pinned dependency graph requires it; the
production workspace remains Rust 1.89.

Run the device-independent contract tests with:

```console
cargo +1.95.0 test --manifest-path experiments/cubecl-local-action/Cargo.toml --locked
```

Run the physical capability gates with exactly one visible device:

```console
CUDA_VISIBLE_DEVICES=0 cargo +1.95.0 test \
  --manifest-path experiments/cubecl-local-action/Cargo.toml \
  --locked --features cuda --test conformance -- --ignored

HIP_VISIBLE_DEVICES=0 cargo +1.95.0 test \
  --manifest-path experiments/cubecl-local-action/Cargo.toml \
  --locked --features hip --test conformance -- --ignored
```

The CUDA test currently expects `MissingF64Capability`. The HIP test has not
been run on physical AMD hardware. Do not weaken either case to `f32`; mixed
precision requires a separate Realization and error contract.
