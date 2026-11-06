# Ordinary Cartesian Hex elasticity preparation

Status: the archived installed-Python runtime passed one solve and independent
primary/replay checks. Its [source revision](https://github.com/eqiora/eqiora/commit/7612bfbab2cef3d3ce10e349e39a95aa43eaf473)
identifies that runtime; this result does not qualify later producer builds.
The first native Run saved its accepted Result, then its exporter failed on
Cartesian tensor-block shape interpretation. That attempt remains failed. A
separate reviewed recovery harness will decode and observe the saved Result
without another solve; native qualification is still pending recovery review.
The independent mathematics and fixed budgets live in [reference/README.md](reference/README.md).

`model.eqi` authors the displacement, isotropic stress, affine essential data,
five natural tractions, strain energy and three left-face stress integrals.
`production.py` uses public compile, resolve, run, output and observe APIs. It
checks the actual physical corner convention independently of global vertex
storage, selects replayed Field and Observable IDs from the reopened Model,
and binds Model, Mesh, Plan, Result and integral lineage. The stress integrals
are public functionals, not recovered private weak reactions.

`check.py --output DIRECTORY --build-receipt RECEIPT` prepares and persists the
actual Model, Mesh and Plan without solving. The receipt identifies the runtime
source commit and SHA-256 hashes of the wheel, native binary and loaded extension.
It must describe the actual artifact; preparation may investigate another
explicitly named runtime but must never relabel it as this source build.

Only after explicit producer/build review, add `--execute` using a receipt with
`producer_review` equal to `accepted`. Pin one CPU, clear CUDA_VISIBLE_DEVICES,
and set OMP_NUM_THREADS, OPENBLAS_NUM_THREADS, MKL_NUM_THREADS and
RAYON_NUM_THREADS to 1. This runs one PDE solve, immediately saves the canonical
Result, then applies unchanged independent budgets separately to primary and
reopened Result observations. Every attempt uses a new output directory and
preserves failure artifacts.

The full distorted/general-cell work in #1552 remains open. This slice excludes
all geometry, physics, backend and robustness extensions listed in the frozen
reference protocol. Typed geometry and resource-budget rejections precede any
accepted Result; a test must not create an artificial physics whitelist.

The current Cartesian Mesh stores one ordered coordinate list per physical
axis. Its public representation cannot express a warped or non-affine Hex.
The invalid-axis and resource-limit controls exercise actual rejection paths;
they do not establish validity checks for trilinearly distorted geometry. That
work remains with #1552. Altered authored-form controls separately change one
traction component, reverse its sign, omit a contraction axis or change a
physical unit, while leaving the source balance equations in place.

The manufactured continuum displacement is affine and extends smoothly past
the box, so its authored trial regularity is `smooth`. This supports the
classical stress traces in the equations and public face observables. It does
not assert smooth gradients for general discrete Q1 fields. The discrete space
remains Q1; the weak test remains H1 with zero trace only on x=0. An otherwise
identical H1 trial declaration must reject those unsupported classical traces.
The weak form uses the equivalent bilinear strain/divergence identity, because
`isotropic_lift` is not currently an authored weak-form operator; this source
does not extend that authoring capability.


The Rust integration test includes a separately held native producer. Its
scientific test is ignored by ordinary test runs and requires an accepted
receipt binding its source tree, binary, producer, model and reference bytes.
It uses public `run_result`, field blocks and `observe`; saves the actual Result
before observing; then reopens the Result and checks primary and replay exports
with the unchanged independent protocol. The same CPU and thread restrictions
apply. Its root preparation manifest identifies the exact subdirectory holding
the actual Model, Mesh, Plan and Result. No candidate is used as an expectation.


Native coefficient blocks retain one extent per Cartesian axis followed by the
Field component extents. The exporter derives this shape from the actual Mesh
and Field type before flattening all coefficients through the Plan entity map.
The read-only recovery entry verifies the original runtime, source and artifact
hashes, opens the original Plan and Result, and records its own harness identity
separately. It cannot call `run_result`. Recovery never changes the original
failed attempt or uses the Python observations as native expectations.
