# Language and toolchain baselines

Checked: 2026-07-21

This file records implementation policy, not permanent version promises.
Before a language layer is first implemented or materially upgraded, its
official language, packaging, and security documentation must be checked
again. Versions are pinned only when code in that layer exists.

## Rust and Cargo (active)

- Edition: Rust 2024.
- MSRV: Rust 1.89, declared with `package.rust-version` and checked across all
  production workspace targets and features. Cargo has one package-level
  support contract, so an optional feature may not carry an undocumented
  higher MSRV.
- Development/CI: test both MSRV and the latest stable toolchain. The Rust
  project only ships fixes for the latest stable release, while Cargo treats
  `rust-version` as the package support contract.
- Workspace: resolver 3, inherited package metadata/dependencies/lints, one
  lockfile, and `cargo xtask check-layers` for the architectural dependency
  direction.
- Quality gate: `cargo fmt --check`, Clippy on all targets/features with
  warnings denied, tests, dependency policy, and rustdoc.
- Public API: private fields, constructors/builders that validate inputs,
  common traits where semantically valid, sealed kernel traits, and no bare
  string errors.
- `unsafe`: denied by default. Future FFI/backend crates must isolate it in
  dedicated modules and document every safety invariant.

References:

- <https://doc.rust-lang.org/stable/cargo/reference/rust-version.html>
- <https://doc.rust-lang.org/cargo/reference/workspaces.html>
- <https://doc.rust-lang.org/stable/edition-guide/rust-2024/cargo-resolver.html>
- <https://rust-lang.github.io/api-guidelines/checklist.html>
- <https://doc.rust-lang.org/stable/clippy/usage.html>

### Rust CUDA adapter baseline

- Checked 2026-07-21: cudarc 0.19.8 is current. The optional adapter remains
  exact-pinned to cudarc 0.18.2 plus libloading 0.8.9 because that is the
  binding/runtime line covered by Eqiora's committed device evidence. The
  workspace's newer MSRV makes 0.19 admissible to investigate, but does not
  substitute for API review and fresh physical hardware evidence.
- cudarc owns CUDA driver discovery, contexts, streams, and typed allocations.
  cuSPARSE has no safe cudarc wrapper in this line. Its generated dynamic
  loader also resolves unrelated release-specific symbols eagerly, which
  rejects otherwise compatible 12.x libraries. Eqiora's private FFI modules
  load only the CUDA 12 cuSPARSE Generic API and cuBLAS level-1/diagonal-band
  symbols they execute and retain each library for every handle/descriptor
  lifetime. cuBLAS uses explicit host scalar pointer mode and disables atomic
  routines; its reductions remain `Fast`, not Eqiora `Reproducible`.
- `cuda-runtime` remains optional and dynamically loaded. Default and
  all-features CI can compile on hosts without a CUDA device; only an explicit
  ignored hardware gate executes the adapter.
- Every unsafe call is confined to the L3 adapter and documents context,
  pointer, shape, aliasing, descriptor, workspace, and synchronization
  invariants. No vendor pointer or error crosses its public API.

References:

- <https://docs.rs/cudarc/0.18.2/cudarc/>
- <https://docs.rs/cudarc/0.19.8/cudarc/>
- <https://docs.nvidia.com/cuda/cusparse/generic-api/generic-api-functions.html>
- <https://docs.nvidia.com/cuda/cublas/>
- <https://docs.nvidia.com/cuda/cuda-toolkit-release-notes/index.html>

### Cross-vendor kernel experiment baseline

- Checked 2026-07-18: CubeCL 0.10.0 is the latest release and remains
  explicitly alpha. It is pinned only in the unpublished
  `experiments/cubecl-local-action` workspace, not in the production
  dependency graph.
- The experiment requires Rust 1.92 because `cubecl-zspace` 0.10.0 declares
  that minimum; the production workspace retains Rust 1.89. Separate
  manifests and lockfiles make that distinction executable rather than
  aspirational.
- CubeCL's common CUDA/HIP C++ type registry omits ordinary scalar `f64`.
  Eqiora therefore checks `Buffer` and `Arithmetic` capability before
  allocation and does not convert the model to `f32` to make the experiment
  pass.
- The ordered and fast kernels remain test-maintained in CI without a device.
  Physical CUDA currently verifies the expected typed capability rejection;
  physical ROCm and value/performance comparisons remain required before any
  production adoption.

References:

- <https://docs.rs/crate/cubecl/0.10.0>
- <https://github.com/tracel-ai/cubecl/blob/v0.10.0/crates/cubecl-cpp/src/shared/base.rs>
- <https://docs.rs/crate/cubecl-zspace/0.10.0/source/Cargo.toml>

## Python (first vertical slice active)

- Supported initial target: ordinary-GIL CPython 3.11 through 3.14. The first
  distribution family is per-CPython Linux x86-64 with a `manylinux_2_17`
  floor, rebuilt solely from the self-contained source distribution and
  installed outside the checkout. Free-threaded builds and every other wheel
  platform remain independent installed-artifact claims; PyO3's ability to
  compile a target is not an Eqiora support claim.
- Binding baselines checked 2026-09-03: PyO3 0.29.0, rust-numpy 0.29.0, and
  maturin 1.15.0. The facade and binding crate inherit the workspace Rust 1.89
  MSRV.
- The workspace-root PEP 517/518/621 `pyproject.toml` owns the mixed
  Rust/Python maturin layout and derives its version from Cargo. Keeping the
  build root at the complete Cargo workspace makes the sdist self-contained;
  do not introduce `setup.py` or a nested second version source.
- The private `eqiora._eqiora` module depends only on the public Rust facade.
  Python contains ergonomics and bindings, never a second implementation of
  Eqiora semantics.
- Frozen native declarations close into the client-neutral Rust `Module` graph and
  share the parsed-source AST before one typed lowerer. Direct compilation does not
  generate/evaluate source strings, mint final graph IDs during expression assembly, or overload
  symbolic equality/truth as hidden model construction.
- The NumPy C API means the first wheel matrix does not claim `abi3`.
  Long-running Rust work detaches from Python; numerical inner loops never
  invoke Python callbacks.
- Public Python is PEP 561 typed. The installed wheel's handwritten ergonomic
  facade stubs are checked by strict consumers and allowlist-free runtime/stub
  parity with mypy 2.3.0; `_eqiora` remains private and unstubbed. The public
  API reference is generated from those same five stub modules. Control-plane
  artifacts/diagnostics and data-plane NumPy/DLPack buffers have separate
  versioning and ownership contracts.
- NumPy 2.1.0 is the first supported versioned DLPack negotiation baseline,
  exercised exactly on CPython 3.12 alongside the ordinary latest-resolution
  wheel profiles.
  Immutable Result storage is zero-copy only through the enforced read-only
  NumPy path; DLPack exports are independent snapshots until consumer
  mutability can be guaranteed.
- The first PyTorch adapter baseline, checked 2026-07-23, is
  `torch>=2.13,<2.14`, with 2.13.0 tested exactly from an installed wheel.
  It uses `torch.library.custom_op`, `register_fake`, and `register_autograd`;
  backward calls a separate Eqiora VJP custom operator. PyTorch remains an
  optional extra and base `eqiora` never imports it.
- The first JAX adapter baseline, checked 2026-07-23, is exact JAX/JAXLIB
  0.11.0 on CPython 3.13 Linux x86-64. It uses typed CPU XLA FFI for bounded
  first-order eager and `jit` primal/JVP/VJP/gradient execution; wider
  versions, devices, transformations, and export remain separate claims.

References:

- <https://docs.python.org/3.14/>
- <https://packaging.python.org/en/latest/guides/writing-pyproject-toml/>
- <https://packaging.python.org/en/latest/specifications/core-metadata/>
- <https://typing.python.org/en/latest/guides/libraries.html>
- <https://pyo3.rs/main/parallelism.html>
- <https://pyo3.rs/main/building-and-distribution.html>
- <https://www.maturin.rs/>
- <https://docs.rs/numpy/0.29.0/numpy/array/struct.PyArray.html>
- <https://data-apis.org/array-api/latest/design_topics/data_interchange.html>
- <https://docs.pytorch.org/docs/stable/library.html>
- <https://docs.pytorch.org/tutorials/advanced/python_custom_ops.html>
- <https://docs.pytorch.org/tutorials/advanced/python_custom_ops_registrations.html>

## TypeScript and browser Studio (preview active)

- Baselines checked 2026-08-03: Node.js 24.18.1 LTS with npm 11.16.0,
  TypeScript 7.0.2, React/React DOM 19.2.7, React Flow 12.11.2,
  Vite 8.1.5, Vitest 4.1.10, Playwright 1.61.1, axe-core/Playwright
  4.12.1, Biome 2.5.4, and Zod 4.4.3. Studio owns one exact npm lockfile.
- TypeScript 7.0 has no supported programmatic compiler API. Studio therefore
  uses `tsc` only as a command-line type checker and Vite/Biome for build and
  source tooling. A future tool requiring compiler APIs must use the official
  TypeScript 6 compatibility package or wait for a supported TypeScript 7 API;
  it must not depend on TypeScript internals.
- `strict` and explicit safety options are enabled rather than inherited from
  floating compiler defaults. Fixed example projections are runtime validated
  because TypeScript types are erased.
- Packages use ESM. Versioned projection DTOs remain separate from handwritten
  reducer state and React component types.
- The interaction baseline is WCAG 2.2 AA. Every operation is keyboard
  reachable, focus is visible, status is announced semantically, and dragging
  always has a non-pointer alternative. Canvas layout is a view preference,
  never model identity or execution order.

References:

- <https://devblogs.microsoft.com/typescript/announcing-typescript-7-0/>
- <https://www.typescriptlang.org/tsconfig/strict.html>
- <https://nodejs.org/en/about/previous-releases>
- <https://react.dev/>
- <https://reactflow.dev/learn/advanced-use/accessibility>
- <https://playwright.dev/docs/accessibility-testing>
- <https://github.com/dequelabs/axe-core/blob/develop/doc/API.md>
- <https://vite.dev/guide/>
- <https://www.w3.org/TR/WCAG22/>

## Eqiora Language (active)

- The language is specification-first, statically typed, and intentionally
  smaller than a general-purpose language.
- Semantic Kernel constructs are interpreted by `eqiora-sem`; Standard
  Ontology constructs lower to kernel subgraphs before interpretation.
- Compiler optimization never defines meaning. Conformance compares every
  optimized backend with the reference interpreter.
- The v0 frontend uses a byte-lossless lexer, recovering parser, source spans,
  a recursive source AST, and one idempotent formatter. Recursive trees stop
  at the parser boundary; compiler lowering produces the canonical expression
  DAG.
