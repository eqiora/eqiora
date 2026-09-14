# Rust API

Use the `eqiora` crate to compile models, configure numerical plans, and run
simulations from Rust.

## Install

[`eqiora 0.1.0`](https://crates.io/crates/eqiora/0.1.0) is available
on crates.io.

In a new Cargo project:

```console
cargo add eqiora@=0.1.0
```

Or add this dependency to `Cargo.toml`:

```toml
[dependencies]
eqiora = "=0.1.0"
```

Building from source requires a Rust toolchain and linker. The minimum supported
Rust version is 1.89. This release's default-feature installation was tested on
Linux x86-64 with Rust 1.97.1.

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
`eqiora = { version = "=0.1.0", features = ["faer"] }`.

## Compatibility

This is a pre-1.0 API. Later releases may contain breaking changes; pin the exact
version and retain `Cargo.lock` when reproducing a result.
