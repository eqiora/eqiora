# Model Packages

This directory contains Eqiora Model Packages in source form. They get no
special treatment: no built-in names, search paths, or compiler branches.
Packages maintained by the project and third-party packages resolve through
the same content-addressed store.

## Package layout

`eqiora.toml` is the manifest you write for a local package. Resolution finds
its source directory and generates the `package.json` that is shipped in a
published or vendored package. Dependency identities and the list of source
files are generated from the local dependency closure; you do not write them
by hand.

## Rust API

Preparing a release from a directory takes two calls:

```rust
# // Cargo.toml: eqiora = { features = ["package-filesystem"] }
use eqiora::package::{PackageDirectory, prepare_package_release_v1};

let sources = PackageDirectory::open_ambient(package_root)?.read_sources()?;
let release = prepare_package_release_v1(sources, &exact_dependency_releases)?;
```

`read_sources` reads `package.json` and the files it lists. It does not walk the
directory tree or follow symbolic links that leave the package root. You can
also build the sources in memory. `prepare_package_release_v1` compiles the
package, derives a lock from the dependency releases you pass (the complete
closure), and checks every source against its final namespace before returning
the release. It does not discover packages on disk, take an atomic snapshot of
several files, or publish anything.

If you already hold a sandboxed `cap_std::fs::Dir`, use
`PackageDirectory::try_from_dir` instead of `open_ambient`, so the adapter gets
no access beyond that directory. All three directory adapters on this page need
the `package-filesystem` feature; package identity, in-memory release
preparation, and resolution work without it.

### Compile from a lock

Compiling from an existing lock is read-only. You supply the lock bytes and a
local store; nothing is searched for, installed, or updated:

```rust
use eqiora::package::{
    DirectoryPackageStore, PackagedModelDocument, ResolutionRecordV1,
};

let lock = ResolutionRecordV1::from_json(&lock_bytes)?;
let store = DirectoryPackageStore::try_from_dir(store_root)?;
let model = PackagedModelDocument::compile_locked(
    &store,
    &lock,
    "Main",
)?;
```

Package compilation always produces the current Model format. Persisted Model
v1–v44 bytes are not migrated and are rejected.

`DirectoryPackageStore::open_ambient` opens the store from a path instead of a
directory handle. Either way, the store reads only `<source-bundle-digest>.json`
files (regular files, no symlinks, size-limited), and the resolver re-checks
every package identity and dependency edge it returns.

### Install a release

Installing is separate from reading a store and does not choose dependencies:

```rust
use eqiora::package::{DirectoryPackageInstaller, PackageStageCleanup};

let installer = DirectoryPackageInstaller::try_from_dir(installation_root)?;
let outcome = installer.install(&release)?;
if let PackageStageCleanup::Deferred(kind) = outcome.staging_cleanup() {
    eprintln!("package committed; staging cleanup was deferred: {kind:?}");
}
```

The installer writes the release to a staging file in the same directory (mode
`0600` on Unix), syncs it, and then adds `<source-bundle-digest>.json` without
overwriting anything. If identical content is already present it returns
`AlreadyPresent`; different or invalid content there is an error. The returned
outcome reports when cleanup of the staging file was deferred. Installation is
designed for one release at a time in a store used by a single user. It does
not update the lock or manage shared-store permissions.

## Bundled packages in Python and the CLI

`eqiora.add_bundled_dependency(...)` and `eqiora package add --bundled` add
`Eqiora.Fluid.Incompressible@0.6.0` or `Eqiora.Solid.LinearElasticity@0.6.0`
to a project as ordinary dependencies, writing the manifest, the lock, and the
store entries. `eqiora.vendor_project(...)` copies all resolved dependencies to
an offline store, and `eqiora.open_project(...)` validates such a store without
fetching anything.

## Packages in this directory

Each top-level directory holds one package. Dependencies are selected by
package identity, not by directory.

- [`Eqiora.Electrical.Basic`](Eqiora.Electrical.Basic/): a scalar electrical
  connector and three ideal static components.
- [`Eqiora.Electrical.Circuits`](Eqiora.Electrical.Circuits/): depends on
  `Basic` and provides a closed parallel DC circuit, `ParallelDc`, with three
  scalar parameters.
- [`Eqiora.Electromechanical.DcDrive`](Eqiora.Electromechanical.DcDrive/):
  depends on `Basic`; an ideal linear DC motor, a viscous load, and related
  components for the sampled-drive verification case.
- [`Eqiora.Stochastic.Processes`](Eqiora.Stochastic.Processes/): dimensionless
  Ornstein–Uhlenbeck and geometric Brownian motion components with explicit Ito
  laws, initial conditions and independent Wiener channels. Ordinary time Plans
  choose the numerical method; Runs choose the sample identities.
- [`Eqiora.Solid`](Eqiora.Solid/): the place to start for 2D isotropic linear
  elasticity. Plane-strain and plane-stress models in terms of Young's modulus
  and Poisson's ratio, typed material composition, Lamé-form building blocks,
  and fixed, free, or field-driven displacement and traction boundary
  conditions.
- [`Eqiora.Solid.LinearElasticity`](Eqiora.Solid.LinearElasticity/): the 2D
  small-strain balance law and displacement/traction boundary interface,
  zero-displacement and zero-traction conditions, and first-order
  elastodynamics in 2D and 3D using `Mechanics.Interfaces`.
- [`Eqiora.Mechanics.Interfaces`](Eqiora.Mechanics.Interfaces/): a
  velocity/traction boundary connector and zero-velocity and zero-traction
  conditions in 2D and 3D. Quasistatic solids use their own
  displacement/traction connector instead.
- [`Eqiora.Mechanics.BoundaryLoads`](Eqiora.Mechanics.BoundaryLoads/): depends
  on `Mechanics.Interfaces`; a normal-pressure load driven by a pressure Field.
- [`Eqiora.Fluid`](Eqiora.Fluid/): the place to start for 2D steady Stokes
  flow. A ready-made model, separate balance and interface components, and
  no-slip, traction-free, normal-pressure, inward-normal velocity, and field-driven
  velocity and traction boundary conditions.
- [`Eqiora.Fluid.Incompressible`](Eqiora.Fluid.Incompressible/): depends on
  `Mechanics.Interfaces`; steady 2D and transient 3D incompressible Newtonian
  volume laws, with separate velocity/traction boundary interfaces.
- [`Eqiora.Fluid.InertialStokes`](Eqiora.Fluid.InertialStokes/): the 2D
  incompressible Newtonian law with inertia used by the first fixed-reference
  FSI case. It contains no boundary conditions, time method, or coupling.
- [`org.example.parallel`](org.example.parallel/): an example third-party
  package with an analytic model, used to test offline resolution, source
  normalization, hierarchy expansion, and provenance.
- [`org.example.poisson`](org.example.poisson/): an example third-party scalar
  elliptic model, used to test that package lineage survives into Realization
  and Run artifacts.
- [`org.example.closed_circuit`](org.example.closed_circuit/): depends only on
  `Circuits`, to test that components can be reused transitively without
  reaching into `Basic`.
- [`org.example.dc-motor-control`](org.example.dc-motor-control/): connects the
  DC drive to a proportional controller on a periodic clock.

The `Basic -> Circuits -> org.example.closed_circuit` chain is verified by
[`packages.composed-model-package`](../../../verify/packages/composed-model-package/README.md).
Its intermediate component has no physical boundary ports. Connections across
package boundaries have their own
[`packages.hierarchical-physical-boundary`](../../../verify/packages/hierarchical-physical-boundary/README.md)
case.

## Not supported

Overwriting or deleting store entries, interactive lock updates, project
discovery on disk, CLI authoring, workspace inference, atomic multi-package
transactions, garbage collection of staging files, crash durability of
directory entries, cross-platform runtime verification, registries,
version-range solving, signing, build scripts, Python or Studio package
authoring, dynamic plugins, and execution-provider packages.
