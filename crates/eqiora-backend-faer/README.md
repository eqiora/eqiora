# Pinned faer provider

`eqiora-backend-faer` is an optional host solver adapter. Eqiora owns `SolverPlan`,
operator properties, canonical CSR/RHS, admission, result meaning, and true-residual
acceptance. The adapter owns faer matrices, scratch storage, factorizations, and
error conversion. Its public interface contains Eqiora types; no faer matrix,
workspace, error, or configuration is a Model or artifact type.

The current dependency is crates.io `faer = "=0.24.4"`, from
[the upstream faer project](https://codeberg.org/sarah-quinones/faer).
`Cargo.lock` records checksum
`5ab6df3dd147fe8d702a288b95bcd8fcc499ab572fc80da6828f60cd4d524d67`.
The workspace disables default features and requests `std`, `linalg`, and
`sparse-linalg` (`sparse-linalg` enables `sparse` and `linalg`). Cargo can unify
additional features when another consumer selects them; inspect the actual
profile with `cargo tree --locked -e features -i faer`. Serial execution is an
adapter contract, enforced with explicit `Par::Seq`, independent of feature
unification. The faer source is MIT licensed; this Eqiora adapter inherits the
workspace Apache-2.0 license. Transitive dependencies remain subject to the
workspace `deny.toml` policy. That policy currently records the exact
`RUSTSEC-2024-0436` unmaintained notice for transitive `paste` 1.0.15 through
gemm/pulp; remove the existing exception when the dependency graph permits it.

The adapter inherits the workspace Rust 1.89 MSRV; faer 0.24.4 declares Rust
1.84.0. Building this adapter requires Rust/Cargo and no separate BLAS, C/C++,
GPU, MPI, or native-library installation. The `std` feature includes faer's
CPU kernel dependencies. Platform and unsafe-code effects inside upstream
dependencies remain part of dependency-policy review. The adapter itself
inherits Eqiora's unsafe-code prohibition.

`build.rs` uses `eqiora-build` to project the resolved faer release into
`FAER_VERSION`. `FAER_SOLVER_PROVIDER` records `eqiora.faer`, the compiled adapter
package version, and that exact library inventory. Stable provider ID equality
alone does not establish release equality. The facade's `faer` feature exposes
the adapter; this is explicit numerical policy, absent from Model meaning.

## Bounded continuity evidence

The existing registered case `numerics.linear-backends` runs the `faer_spatial`
integration target. Its `provider_continuity` module resolves one compiled Model
and authenticated mesh through the ordinary scalar TPFA Plan/Result path with:

- Reference CG / Identity / Reproducible / binary64 / serial host;
- faer 0.24.4 CG / Identity / Fast / binary64 / serial host.

Both own the same captured 2-by-2-cell unit-square Poisson system. Independent
integral cell balance gives diagonal 6, neighboring entries -1, RHS 1/4, and
solution 1/16 in each cell. The proof checks each provider against this stencil
and solution, not only against the other provider. Relative tolerance `1e-12`
and absolute tolerance `1e-14` imply residual target `5e-13`; the existing
component error bound is `5e-13`. The stencil eigenvalues 4, 6, 6, 8 independently
bound the solution error by the residual norm divided by 4.

Delegating wrappers capture the actual ordinary backend boundary. Exact Model,
mesh, CSR/RHS, and Field meaning are retained while Plan, Result, and provider
identities change. A real provider's successful solve of `2A` with unchanged RHS
is rejected against the original `A`; its own small reported residual is
insufficient. Same-ID stale adapter/library releases, unknown providers,
unsupported reduction, changed execution capabilities, and an explicit
Reference fallback are falsifiers. Counters and exact report provenance check
the selected path and refusal before solver work.

This proves one steady real SPD TPFA continuity boundary. It establishes no
performance, arbitrary PDE, nonlinear, distributed, accelerator, or general
solver-substitution claim. The existing #126 sparse-LU rational oracle, digest,
tolerances, falsifiers, and serial source guards remain separate and unchanged.

## Updating or replacing the provider

An update is triggered by an upstream release, advisory, license/MSRV/platform
change, changed required features, unavailable provider, or an acceptance or
capability failure. Keep the independent fixture, expected values, tolerances,
and falsifiers fixed while evaluating the replacement; investigate a mismatch
before changing a claim. A replacement must consume the Eqiora-owned contracts
and preserve Eqiora's original-operator acceptance. Stop if it requires foreign
types in a stable artifact, a second solver-policy owner, a Model rewrite,
relaxed tolerances, or silent fallback.

Update the workspace exact pin and lockfile together; review upstream release
notes, source/license/advisories, feature closure, MSRV, and platform effects.
The build-time version projection must match the new resolved release. Re-run
the ordinary continuity positives and mutants, every affected existing
registered case (including #126 when its implementation changes), and matching
optional/MSRV/dependency-policy profiles. Record the exact environment and
provider releases; local evidence does not establish a published-release or
different-platform result. Update the capability matrix at the same bounded
claim boundary.

Focused repository checks include:

```sh
cargo test --locked -p eqiora --test faer_spatial
cargo run --locked -p eqiora-verify -- run --case numerics.linear-backends
cargo xtask check-layers
cargo xtask check-facade
python3 -m unittest discover -s tools/ci/tests -p test_ci_contracts.py -k DependencyIdentityTests
```

The dependency identity tests enforce pin/lock/build-version ownership and the
shared MSRV. Layer and facade checks guard their respective dependency/export
boundaries; the existing sparse-LU test guards explicit serial calls. Public
signature confinement also needs focused source review of adapter exports;
those general checks do not independently analyze every foreign signature.

Removal is localized to this adapter crate and its workspace dependency/pin,
the facade `faer` feature/re-export, explicit faer provider registrations and
consumers, and their claim-local evidence/documentation. Remove displaced
registrations and consumers atomically. The canonical Model, mesh, common
solver contracts, Reference provider, and Eqiora residual owner remain usable
through their existing surfaces; no faer-specific schema migration is needed.
