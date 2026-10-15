# Python JAX typed-FFI differentiation

The optional `eqiora.jax` adapter projects one accepted framework-neutral
`DifferentiableProgram` into JAX without adding a second model or
differentiation semantics. The exact immutable common scalar Plan, its Model,
caller-supplied two-dimensional rectangular Cartesian Mesh, ordered Parameter
identities, and output Field remain static; an ordered tuple of traced scalar
leaves contains only the numerical Parameter point. The verified Plans use Q1 FEM or
cell-centred TPFA with a typed linear solve policy.

Primal, JVP, and VJP execution use current V2 typed XLA FFI targets registered
for CPU with API version 1 and lowered by `jax.ffi.ffi_call` custom-call API
version 4. The targets use the ordinary native ordered map and accepted
first-order product owners. Compiled StableHLO contains Eqiora custom calls,
not `pure_callback` or an XLA Python CPU callback, so neither eager nor `jit`
execution re-enters Python for numerical work.

Each untransformed call accepts exact ordered native-shaped real/complex leaves
and returns an ordered output tuple. This registered spatial profile uses three
scalar `float64` inputs and one real Field output. Private real-coordinate packing
uses only the native signatures; JAX differentiates its real/imaginary projections.
HiJAX batching rules on the solve, primal, JVP and VJP primitives preserve
mapped/shared operands, non-leading axes, output-axis placement, and nested
maps. Static metadata distinguishes point occurrences from derivative seeds:
basis directions reuse accepted points instead of duplicating primal solves.
JAX broadcasting and its transpose preserve sharing along individual axes;
equal numeric coordinates are never inferred to be shared. The ordinary
verified path uses unsharded leaves on one common CPU device; its concrete CPU ordinal follows the
input and is checked through XLA's runtime API. Registration uses
one deterministic key for the complete static program identity and keeps the
native program alive until process exit so a compiled executable can outlive
its temporary Python wrapper. Numerical evaluations and linearizations are not
cached. Point/seed rank is at most 32. Each FFI buffer is at most 64 MiB;
native map and product owners separately apply their 64 MiB additional
retained-numerical-storage limits. These are not peak-memory bounds. Empty
and singleton collections retain their admitted shapes. Any failed member
rejects the dense operation; no partial result or gradient is returned.

The installed-wheel gate uses CPython 3.13 on Linux x86_64 and the exact
JAX/JAXLIB 0.11.0 pair. It exercises:

- Q1 FEM and TPFA FVM primal values against native accepted evaluations;
- eager and `jit` primal, first-order JVP, VJP, and scalar-objective gradient;
- two nested axes, permuted/duplicate points, non-leading input/output axes,
  shared inputs, `vmap(grad)`, and gradients of sum/mean mapped losses;
- `jacfwd`/`jacrev` basis products and their distinct seed-only FFI inputs;
- lowered StableHLO containing only the declared typed custom-call path;
- compiled-executable lifetime after the Python program wrapper is released;
- zero tangent and cotangent actions;
- dtype, rank, static shape, CPU platform, direct and explicitly compiled input
  sharding, finiteness, and unknown-program falsifiers;
- empty/singleton maps, empty seed axes, indexed dense failure under `jit`,
  bounded metadata admission, and constant primitive count as sample count grows;
- explicit rejection of sharding, named collectives, `pmap`, and higher-order
  linearization, plus immutable static program configuration;
- additional ordinary product checks for shaped complex inputs, multiple outputs,
  nonholomorphic real-linear products, native/JAX covector conventions and shared
  complex leaves; these do not widen the registered Q1/TPFA scientific claim;
- exact native ABI layout agreement with the installed JAXLIB header; and
- a base `eqiora` import with no JAX or JAXLIB import.

Registered host evidence obtains this profile from the same complete candidate
and manifest used by the base, typing, and PyTorch cases. Its JAX checks must be
present in that accepted manifest; the focused
`tools/ci/python_jax_gate.py` script remains available for standalone
development but is not a second registered artifact build.

This is an exact in-process first-order host-CPU slice. Direct or explicitly
compiled input sharding, named collectives, `pmap`, and higher-order transformations
are rejected. Explicit output sharding, GPU, TPU, export, serialization,
multiprocessing, and performance are unverified nonclaims. The transformation
seam uses JAX's experimental HiJAX API, so the optional dependency is pinned
exactly rather than presented as a wider compatibility range.

Run the registered installed-wheel evidence with:

```console
cargo run --locked -p eqiora-verify -- run \
  --case interfaces.python-jax-differentiation
```
