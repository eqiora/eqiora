# Rust API

Use the `eqiora` crate to compile models, configure numerical plans, and run
simulations from Rust.

## Install

[`eqiora 0.1.3`](https://crates.io/crates/eqiora/0.1.3) is available
on crates.io.

In a new Cargo project:

```console
cargo add eqiora@=0.1.3
```

Or add this dependency to `Cargo.toml`:

```toml
[dependencies]
eqiora = "=0.1.3"
```

Building from source requires a Rust toolchain and linker. The minimum supported
Rust version is 1.89. The minimum supported toolchain is checked in CI.

## Build command-line tools from this checkout

The CLI and MCP binaries use the opt-in `cli` feature. Library builds do not
enable it or pull in the CLI argument parser. From the repository root:

```console
cargo install --locked --path crates/eqiora --features cli
```

This installs `eqiora` and `eqiora-mcp`.

## Compile a model

With the registry dependency above, put this in `src/main.rs`, then run
`cargo run`:

```rust
use eqiora::api::ModelDocument;

fn main() {
    let _model = ModelDocument::compile(
        "decay.eqi",
        r#"model decay(parameter rate: 1 / s = 1) {
            state x: 1;
            initial { x = 1; }
            relation flow {
                derivative(x) + rate * x = 0;
            }
        }"#,
    )
    .expect("the model must compile");

    println!("Model compiled successfully.");
}
```

This example compiles the decay model and reports successful compilation. See
the [capability matrix](capability-matrix.md) for available simulation workflows.

## Spatial State coefficients in the source checkout

Use `CommonSpatialPlan::state_fields(&state)` to inspect a common spatial State.
Each item carries the exact Field ID, physical value type, Mesh entities in
coefficient order, and a borrowed coefficient slice with components ordered
within each entity. The Plan rejects States from another state space.
These are finite-dimensional coefficients; physical samples and integrals
still require the admitted reconstruction. Python State fields use this same
native layout. The untyped whole-State `spatial_values()` accessor is private.

`CommonResult::observe` and `observe_at` take a final `Option<usize>` selecting
an exact retained output position. Use `None` for instantaneous Results and
`Some(index)` for a fixed-geometry spatial Trajectory. Samples and integrals share
the admitted nodal reconstruction; `time()` binds to the selected State.
`CommonObservation::output()` retains `(index, time_s, state_identity)` alongside
the Result identity. An output index is not a step ordinal or an arbitrary time,
and sparse outputs do not establish complete time coverage.

Real scalar `CommonTrajectory::observe_time_minimum` and `observe_time_maximum`
use the same `TimeObservationInterpolation` as integrals and return a
`CommonTrajectoryObservation` with original units and `extremum_time_s`.
Closed windows include both accepted event sides; exact computed ties choose the
earliest physical time, then accepted occurrence. These are interpolant extrema,
not continuous-solution bounds, restart States or extrema derivatives.

Spatial temporal reductions use `CommonTrajectory::observe_time_integral` with
`TimeObservationInterpolation::EndpointLinear` and explicit spatial quadrature
by Domain. ODE callers supply an empty spatial rule map. The interval must be
covered by retained accepted steps; endpoint interpolation is of the Observable,
not its State. `CommonTrajectoryObservation::spatial_quadratures` retains the
exact rules. Moving geometry and spatial derivative products are not admitted.

## Optional features

The default feature is `package-filesystem`, which enables filesystem-backed
model-package operations. The Gmsh MSH parser is always available through
`eqiora::io::gmsh`, including with default features disabled. Parsing an existing
mesh needs no external Gmsh executable; automatic mesh generation requires it.

| Feature | Purpose |
| --- | --- |
| `rayon`, `faer` | Optional threaded CPU and linear algebra integrations. |
| `vtu`, `xdmf`, `hdf5` | Optional data-format operations; `hdf5` enables `xdmf` and the native HDF5 dependency. |
| `cad-truck` | Rust-native CAD operations. |
| `diffsol` | Optional adaptive integration backend. |
| `mpi`, `cuda`, `mpi-cuda` | Environment-specific distributed/GPU adapters requiring the corresponding native libraries and hardware. |

Enable only the features needed by the chosen supported path, for example
`eqiora = { version = "=0.1.3", features = ["faer"] }`.

## Compatibility

This is a pre-1.0 API. Later releases may contain breaking changes; pin the exact
version and retain `Cargo.lock` when reproducing a result.
