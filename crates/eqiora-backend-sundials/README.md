# Residual-native IDA adapter

This optional L3 adapter consumes the existing `ImplicitDaeProblem` residual and
paired-JVP contract. It admits BDF with serial dense real-coordinate algebra.
It preserves `Provided` pairs and uses IDA's index-one `IDACalcIC` convention for
`SolveConsistent`. Error tolerances cover every coordinate, including algebraic
coordinates. Requested output samples use native interpolation, with the final
requested time set as the integration stop time.

The exact admitted native release is **SUNDIALS 7.8.0**, built in double precision.
`sundials-sys` 0.6.2 generates bindings from the deployment's matching headers;
the compiled header version and scalar type are checked, and the live native
release is checked before allocating solver objects. Use headers and libraries
from the same trusted installation. No general native ABI compatibility promise
is made. The binding crate's older bundled fallback is rejected at compilation.

`python3 tools/ci/install_sundials.py --prefix <empty-home-backed-prefix>`
reproduces the checksum-pinned native build used by Stable and MSRV CI. It
rejects an existing installation prefix and verifies the upstream release hash
before extracting or executing source.

For the optional native build, set `SUNDIALS_INCLUDE_DIR` and
`SUNDIALS_LIBRARY_DIR` to that installation and provide libclang for bindgen.
The platform dynamic loader must resolve its matching libraries at execution.

```sh
cargo test -p eqiora-backend-sundials --features ida-runtime
cargo clippy -p eqiora-backend-sundials --features ida-runtime --all-targets -- -D warnings
```

Native handles and callback state have one session owner. Callback errors keep
the original diagnostic; panics are caught before returning through C (requires
unwinding panics). Partial initialization and failed solves release acquired
native objects. Genuine native convergence failures remain errors: an accepted
residual-consistent pair does not guarantee convergence for every starting step
or unused algebraic-rate predictor.

This adapter does not yet connect ordinary common Plan/Run or Python execution.
It does not claim events, sensitivities, adjoints, scalable linear providers,
accepted dense-history export, or durable adaptive restart. Those compositions
retain their existing owners under #1563 and #1546.
