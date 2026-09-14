# RFC 0059: One production-workspace MSRV contract

- Status: Implemented; existing Diffsol evidence reverified at 0.16.1
- Authors: Eqiora contributors
- Created: 2026-07-21
- Amends: [RFC 0012](0012-python-interop-boundaries.md),
  [RFC 0014](0014-production-time-backend-contracts.md),
  [RFC 0019](0019-device-execution-contracts.md),
  [RFC 0020](0020-local-action-kernel-boundary.md),
  [RFC 0025](0025-discrete-field-and-import-provenance.md),
  [RFC 0027](0027-capability-rooted-package-directory-admission.md), and
  [RFC 0052](0052-cad-semantic-selection.md)

## Summary

Every crate and optional feature in the production Cargo workspace has one
honest Rust support floor. Eqiora raises that floor from Rust 1.85 to 1.89,
checks all targets and features at the declared MSRV, and exact-pins Diffsol
0.16.1 after upstream repaired BDF scratch-memory corruption.

Separately locked applications and unpublished experiments remain distinct
support contracts. Their toolchain versions do not lower or raise the
production workspace's claim.

## Motivation

The workspace declared Rust 1.85 while the public optional
`diffsol-runtime` path resolved nalgebra 0.35 and therefore required Rust 1.89.
The MSRV job checked default features only, so Cargo metadata promised a
configuration that could not compile:

```text
cargo +1.85.0 check --locked \
  -p eqiora-backend-diffsol --features diffsol-runtime
```

Cargo's
[`package.rust-version`](https://doc.rust-lang.org/cargo/reference/rust-version.html)
describes a package, not an individual feature. Calling 1.85 the "core MSRV"
and 1.89 a "feature-specific MSRV" therefore made the published support
contract depend on an informal qualification that downstream resolvers cannot
express.

The dependency audit also found
[Diffsol 0.16.1](https://github.com/martinjrobins/diffsol/releases/tag/v0.16.1).
Its [upstream change set](https://github.com/martinjrobins/diffsol/compare/v0.16.0...v0.16.1)
replaces a BDF order-change scratch `Vec` after a stack-buffer overflow
corrupted the vector's metadata. Eqiora executes BDF for stiff ODE and
constant-mass DAE evidence, so retaining 0.16.0 would knowingly leave the
accepted path on the affected implementation.

## Decision

### Production workspace

The root workspace declares Rust 1.89 through inherited
`package.rust-version`. Its MSRV gate runs:

```text
cargo +1.89.0 check \
  --workspace --all-targets --all-features --locked
```

The gate installs native build dependencies required to compile optional MPI,
CUDA, and FFI adapters. It performs compilation, not physical hardware or
multi-node execution. Those capabilities retain their own registered evidence
and environment gates.

An optional production feature may not require a newer compiler than this
command. If a future adapter cannot fit the workspace floor, Eqiora must do one
of three explicit things before merging it:

1. raise the workspace MSRV with a compatibility decision and complete gate;
2. choose a maintained compatible dependency line and verify it; or
3. move the adapter behind a genuinely separate distribution/workspace
   boundary and remove it from the production facade's feature graph.

Documentation cannot invent a fourth, feature-specific Cargo support
contract.

### Diffsol patch baseline

The optional adapter exact-pins Diffsol 0.16.1. Both upstream `nalgebra` and
`faer` features remain enabled because that release exposes both host matrix
families unconditionally, although Eqiora's first adapter executes through
`NalgebraLU`.

The patch changes backend implementation provenance, not Semantic Model,
Operator IR, TimePlan, or artifact schema meaning. New Diffsol-produced
time-run artifacts name 0.16.1. Existing immutable artifacts that name 0.16.0
retain their bytes and remain historically truthful; replay never rewrites
their backend version.

`DiffsolTimeBackend` owns one compile-time `DIFFSOL_TIME_BACKEND` identity that
atomically binds adapter name and exact release. Every accepted
`TimeExecutionReport` carries that pair, and run-artifact constructors take it
only from the report. A repository-level integration test compares the
adapter-owned release with both the exact root manifest pin and the resolved
lockfile entry. The decoder remains provider-neutral and golden tests preserve
historical 0.16.0 and reference-backend version strings byte for byte.

### Separate workspaces

The CubeCL experiment has an independent manifest and lockfile. It continues
to declare and test its own floor. In particular,
the CubeCL 0.10 investigation remains Rust 1.92 and does not enter the
production dependency graph.

Raising the production MSRV does not automatically graduate newer CUDA, GUI,
or experimental dependencies. A dependency covered by device or numerical
evidence changes only in its own adapter slice with fresh evidence.

## Alternatives considered

### Keep Rust 1.85 and document Diffsol as a higher-MSRV feature

Rejected. Cargo cannot communicate that qualification in
`package.rust-version`, and the exact advertised optional feature already
fails at 1.85.

### Keep Rust 1.85 and move Diffsol to an unpublished experiment

Rejected for the current architecture. Diffsol is an accepted production L3
adapter consumed by the public optional facade and multiple registered time
cases. Moving it would remove a supported execution path rather than repair
its support metadata.

### Downgrade or fork the time backend

Rejected. Diffsol 0.16.0 already requires the higher toolchain and contains the
BDF defect fixed by 0.16.1. Carrying a private numerical fork would add more
maintenance and safety risk than adopting the honest compiler floor.

### Raise the MSRV and update every newly compatible dependency at once

Rejected. Compiler compatibility is not numerical, ABI, device, or provenance
evidence. In particular, cudarc remains at the exact line exercised by current
CUDA observations until a separate adapter upgrade revalidates it.

## Compatibility and migration

- Source API: intentionally tightened. `TimeExecutionReport::new` now requires
  one adapter-owned `TimeBackendIdentity`; time-run manifest constructors no
  longer accept an independent caller-supplied version string. Downstream time
  adapters must bind their name and exact release at execution time.
- Compiler compatibility: production crates now require Rust 1.89 or newer.
- Semantic and wire schemas: unchanged.
- Existing canonical artifact bytes and digests: unchanged and still decoded
  exactly. Newly constructed reference-backend manifests may differ from code
  that previously supplied a descriptive, non-release version string because
  they now record the backend crate's exact release.
- Backend provenance: newly produced Diffsol runs identify 0.16.1; old 0.16.0
  artifacts remain valid records of their original execution.
- Default and optional features: unchanged.
- Studio and experimental workspaces: unchanged.

Eqiora is pre-release and has not promised a long-term public MSRV. This is the
least surprising point to repair the contract before external consumers rely
on the lower value.

## Verification

The change is rejected unless all of the following pass:

- the complete Rust 1.89 all-target/all-feature MSRV command above;
- the stable all-feature workspace test and Clippy gates;
- `time.diffsol-adaptive`, covering Tsitouras, BDF stiff tracking,
  rank-deficient mass-matrix initialization, forward sensitivity, and
  proposal/reset/restart ownership;
- `time.canonical-first-order`, covering canonical explicit and full/singular
  constant-mass lowering through Diffsol; and
- dependency policy and local documentation checks.

The registered cases retain analytic solutions, algebraic constraints,
parameter-sensitivity oracles, unsupported-equation rejection, and exact
backend-version provenance. A successful compile alone is not evidence that
the BDF path remains numerically correct.

## Security, safety, and governance

The upstream memory-corruption fix is treated as a correctness and safety
update even though no RustSec advisory currently names it. The exact pin and
lockfile make the selected implementation auditable. Future relaxation of the
pin must repeat the affected time evidence and review upstream safety notes.

This RFC changes a pre-release toolchain support policy and does not grant an
exception to DCO, dependency review, local verification, or bounded capability
claims.

## Unresolved questions

None for this slice. A future public MSRV longevity policy belongs to release
governance once Eqiora approaches a stable public release.
