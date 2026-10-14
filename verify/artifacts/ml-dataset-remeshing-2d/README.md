# Remeshing-aware ML Dataset

Status: verified for the bounded 2D derivation below.

This case derives one immutable logical ML Dataset directly from the exact
current spatial trajectory in `fsi.remeshing-transfer-2d`. The strict-time
sequence omits the superseded source tip and retains the same-time
remesh target, followed by its ordinary continuation. Its three frames become
singleton training, validation, and test samples.

Fluid velocity is the feature and fluid pressure is the target. Descriptors
retain exact Field, support Domain, coherent-SI dimension, value shape, and
frame meaning. Velocity retains both its active Vertex coefficients and the
Cell-associated MINI bubble. Source and target mesh widths differ, so the CPU
projection returns owned ragged blocks with exact mesh and active-entity
lineage instead of padding or interpolation.

Population mean and standard deviation are fitted per descriptor,
association, and component from the training sample only under the named
ordered-binary64 Welford profile. An independent two-pass oracle over raw
training values checks every statistic, active entity, and normalized scalar.
The test also checks explicit constant-channel scale, canonical materialization
order, declaration-order invariance, bounded decoding and pre-allocation work,
and exact fresh replay. XDMF/HDF5 layout and framework tensors are not inputs
to Dataset identity.

Run:

```bash
cargo test --locked -p eqiora --features faer --test remeshing_transfer_2d
cargo run --locked -p eqiora-verify -- run \
  --case artifacts.ml-dataset-remeshing-2d
```

This evidence does not claim interpolation, padded or dynamically batched
dense tensors, random/cross-validation splits, arbitrary transforms,
framework adapters, training, model registries, GPU/distributed loading, or
production scale.
