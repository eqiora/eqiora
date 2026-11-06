# Ordinary Cartesian Hex elasticity preparation

Status: producer preparation; no numerical candidate has run and this case does
not yet establish qualification. The independent accepted mathematics and fixed
budgets live in [reference/README.md](reference/README.md).

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
