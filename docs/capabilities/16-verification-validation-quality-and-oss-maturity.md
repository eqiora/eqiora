# Verification, validation, quality, and OSS maturity

[Capability matrix](../capability-matrix.md) · [legend](../capability-matrix.md#how-to-read-the-matrix)

| Capability | C | X | V | M | Current implementation and planned extensions |
|---|:---:|:---:|:---:|:---:|---|
| Machine-readable verification cases | ✅ | ✅ | ✅ | 🟨 | Case manifests describe reproducible checks. `cargo run -p eqiora-verify -- index` lists the available cases, and `--case` selects a case to run. |
| Repository-derived bounded Cargo authority | ✅ | ✅ | ⬜ | ⬜ | A private Cargo analysis derives workspace manifests, lockfile, packages, targets, and dependencies for a specified commit and added-file overlay. It handles Cargo metadata only; it does not execute builds or select verification cases. [`quality.semantic-impact-cargo-authority`](../../verify/quality/semantic-impact-cargo-authority/README.md). |
| Explicit claim/non-claim boundary | ✅ | ✅ | ✅ | 🟨 | Each case describes the result it checks and the assumptions under which it applies. |
| Deterministic case selection | ✅ | ✅ | ✅ | 🟨 | Select checks by case ID and execution environment. Repeated `--case` options select multiple cases; reports record outcomes and explain excluded cases. |
| Manufactured-solution convergence | ✅ | ✅ | ✅ | ⬜ | Scalar Poisson 1D–3D; broad PDE and element portfolio remains |
| Error of an accepted result against an exact solution | 🟨 | ✅ | 🟨 | ⬜ | Internal Cartesian Poisson checks compute continuum L2 and sampled L-infinity error. The common Result API does not expose a general exact-solution error operation. Other element families and norms remain planned. |
| Conservation/balance evidence | ✅ | ✅ | ✅ | ⬜ | Scalar FEM/FVM balances plus one dimensioned electromechanical `PhysicalSample` power/energy balance with backward-Euler numerical dissipation named separately from physical loss |
| Analytic oracle comparisons | ✅ | ✅ | ✅ | ⬜ | Bounded numerical, time, differentiation, solid, and sampled linear-drive cases, including an independent matrix-exponential trajectory oracle |
| Cross-backend conformance | ✅ | ✅ | ✅ | ⬜ | Reference/faer/Rayon/MPI/CUDA/Diffsol slices; broad matrix remains |
| Restart/replay conformance | ✅ | ✅ | ✅ | ⬜ | Fixed-step semantic lineage; adaptive/distributed paths remain |
| Gradient finite-difference checks | ✅ | ✅ | ✅ | ⬜ | Parameter, shape, time, and hybrid bounded cases |
| Benchmark problem catalog | ✅ | 🟨 | 🟨 | ⬜ | Roadmap exists; most mature application benchmarks remain unimplemented |
| Regression performance baselines | 🟨 | 🟨 | 🟨 | ⬜ | Stable hardware-normalized gates and history remain |
| Numerical tolerance policy | ✅ | ✅ | ✅ | ⬜ | Case-local explicit tolerances; product-wide precision/tolerance guidance remains |
| Property-based testing | 🟨 | 🟨 | 🟨 | ⬜ | Some internal tests; systematic semantic generators and shrinking remain |
| Fuzz testing | 🟨 | 🟨 | 🟨 | ⬜ | Decoders enforce input size limits; continuous fuzzing is not set up |
| Large-model stress testing | ⬜ | ⬜ | ⬜ | ⬜ | Memory, latency, graph, mesh, field, and artifact scale suites remain |
| Long-duration soak testing | ⬜ | ⬜ | ⬜ | ⬜ | Solver, Studio, service, and resource-leak evidence remain |
| Hardware compatibility matrix | 🟨 | 🟨 | 🟨 | ⬜ | Bounded Linux/MPI/CUDA/Python fixtures; supported matrix remains |
| Continuous integration | ✅ | ✅ | 🟨 | 🟨 | Automated pull-request checks cover code and documentation. Site deployment uses a successful main-branch build. Broader environment and scale coverage remains incomplete. |
| Mechanical architecture-layer check | ✅ | ✅ | ✅ | 🟨 | `xtask check-layers` enforces crate boundaries |
| Dependency policy/audit | ✅ | ✅ | ✅ | 🟨 | `cargo-deny` and pinned CI actions; ongoing advisories remain operational work |
| Branch protection and merge gates | ✅ | ✅ | 🟨 | 🟨 | Changes to `main` go through pull requests and required checks, with linear history and protected branch updates. The protected-base guard admits exact architecture reductions and retires a deleted source's line exception only from authenticated commit-bound deletion metadata plus its frozen base-source count; missing content alone is not deletion evidence. The contribution guide describes review and integration. [`quality.ci-routing`](../../verify/quality/ci-routing/README.md). |
| DCO/sign-off policy | ✅ | ✅ | ✅ | 🟨 | Contribution process is documented |
| Code of Conduct | ✅ | ✅ | — | 🟨 | Present |
| Governance and RFC process | ✅ | ✅ | 🟨 | 🟨 | Present; community operation has not yet been exercised at scale |
| Security policy | ✅ | ✅ | — | 🟨 | Present; release response history does not yet exist |
| Contributor guide | ✅ | ✅ | 🟨 | 🟨 | Present; onboarding examples and public feedback remain |
| API semantic-version policy | 🟨 | 🟨 | 🟨 | ⬜ | Pre-1.0 releases may introduce breaking changes; pin versions when reproducing a workflow. Published release bytes remain immutable. |
| Changelog discipline | ✅ | ✅ | 🟨 | 🟨 | Present; automated release linkage remains |
| Release automation | ✅ | ✅ | 🟨 | ⬜ | Python and Rust publication workflows built, checked, staged, and promoted the 0.1.0, 0.1.1, 0.1.2, and 0.1.3 package families. Artifact signatures, automated GitHub release assembly, and broader platform coverage remain planned. [Release topology](../development/ci-topology.md#release-separation). |
| Reproducible builds | 🟨 | 🟨 | 🟨 | ⬜ | Locked dependencies exist; byte-reproducible cross-platform artifacts remain |
| SBOM and signed provenance | 🟨 | 🟨 | 🟨 | ⬜ | Dependency inventory exists; signed release attestations remain |
| crates.io publication | ✅ | ✅ | ⬜ | ⬜ | All 36 crates in the `eqiora 0.1.5` dependency closure are published from annotated tag `v0.1.5` at release commit `f5402e34670c429963f3113c42b16c89a4bf4386`. Registry checksums match all 36 verified archives, and no version is yanked. Other platforms are not verified. [Rust guide](../rust-api.md), [release commit](https://github.com/eqiora/eqiora/commit/f5402e34670c429963f3113c42b16c89a4bf4386), [published crate](https://crates.io/crates/eqiora/0.1.5), [publication CI](https://github.com/eqiora/eqiora/actions/runs/40505348590). |
| PyPI publication | ✅ | ✅ | ⬜ | ⬜ | `eqiora 0.1.5` is published as a source distribution and a Linux x86-64 manylinux CPython abi3 wheel for 3.11–3.14. The accepted manifest SHA-256 is `c244d71cf213c8e45101c867724e7f30a30dc138529f6892116e95599af10faa`; production publishes those exact candidate bytes. Candidate validation and TestPyPI reacquisition passed the supported interpreter and optional JAX/PyTorch profiles. A clean public-PyPI CPython 3.13.14 installation additionally passed the conductivity research example with direct/iterative solvers, changed inputs and mesh refinement, 22 independent research/map checks and ordered ordinary-Run failure/retry history reopening in a second fresh locked environment. Other platforms are not verified. [published distribution](https://pypi.org/project/eqiora/0.1.5/), [candidate replay](https://github.com/eqiora/eqiora/actions/runs/40502623856), [publication CI](https://github.com/eqiora/eqiora/actions/runs/40505348590), [research reproduction](https://github.com/eqiora/eqiora/issues/1708#issuecomment-6442336540). |
| Public repository launch | ✅ | ✅ | 🟨 | 🟨 | The Apache-2.0 repository, governance, protected merge gates, hosted CI, documentation site, public Issue/PR surface, and repeated GitHub/PyPI/crates.io releases through 0.1.3 are live. Custom-domain operation and sustained outside contribution remain |
| Public issue/PR community | 🟨 | 🟨 | 🟨 | ⬜ | Public Issues and pull requests are available; sustained community operation remains |
| User documentation | 🟨 | 🟨 | 🟨 | ⬜ | [Get started](https://eqiora.org/get-started/) installs the published package with uv, then runs and edits a small model. [Learn](https://eqiora.org/learn/) covers mathematical modeling, numerical simulation, heat transfer, fluid mechanics, structural mechanics, circuits and dynamics, and inverse problems, with derivations, exercises, and maintained examples. [Guides](https://eqiora.org/guides/) renders the Python manuals on the site and marks features that need the development build. [Reference](https://eqiora.org/reference/) covers the language, the standard library, and the Python, Rust, and CLI interfaces with runnable examples. Model assumptions and numerical conditions are documented beside each lesson and example; this matrix is the developer index. Advanced domain chapters and broader task coverage are still to come. |
| Tutorial and example suite | 🟨 | 🟨 | 🟨 | ⬜ | Quickstart and fixtures exist; progressive application tutorials remain |
| API reference documentation | ✅ | ✅ | ⬜ | ⬜ | The site includes Rustdoc and a Python reference generated from shipped type stubs. Broader narrative coverage remains in development. [`Python API`](../python/api.md), [`interfaces.python-distribution-candidate`](../../verify/interfaces/python-distribution-candidate/README.md). |
| Physics validation manuals | ⬜ | ⬜ | ⬜ | ⬜ | Each production library needs theory, assumptions, benchmarks, and limitations |
| Support and deprecation lifecycle | ✅ | — | 🟨 | ⬜ | The Python pre-1.0 release policy covers compatibility, deprecation, yanking, security fixes, TestPyPI staging, and promotion to PyPI, and public pre-1.0 releases have followed it. LTS, migration tooling, and end-of-life handling are not defined yet; [`python-release-policy`](../development/python-release-policy.md) |
