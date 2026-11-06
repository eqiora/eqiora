# Ordinary affine P2 scalar workflow: implementation protocol

This draft integrates the ordinary static scalar P2 path on authenticated affine
triangles and tetrahedra. It depends on the reviewed quadratic space/result
owners, independent [reference](reference/README.md), named polygon/polyhedron
geometry, and authenticated public MSH import. It does not merge those feature
PRs or establish a verified product capability yet.

The first checks construct both approved skew physical domains through public
GeometryGraph APIs, import their exact original Freudenthal simplex inventories
through Mesh.import_msh with explicit metres, and reopen public artifacts. They
perform no PDE solve. Geometry names, physical coordinates, oriented cells,
complete vertex/edge inventories and model/mesh identities must agree.

The product work extends the existing scalar-region discretization and
simplicial assembly owners; it does not add a P2 executor. Static linear scalar
P2 fields use existing global entity/slot numbering, parent-restricted traces,
complete essential interpolation points, natural test functions and complete
Result coefficient blocks. Volume and facet integration consume the same
persisted effective quadrature policy. Unsupported vector, transient, nonlinear
or curved combinations remain typed rejections.

Before any product candidate, a reviewed producer must bind its actual source,
build, Model/Plan/Result and mesh artifacts, full vertex/edge associations and
effective quadrature to the unchanged independent reference. The mixed-boundary
patch uses the original normalized quadratic solution, physical source and
outward natural flux, three Duffy points per axis, and fixed rational residual
and coefficient allowances. Optional native constrained actions must be labelled
as present or absent; reference actions cannot replace a native witness.

The distinct smooth witness requires both dimensions and N=2/4/8, the original
skew meshes, three/five Duffy points per axis in 2D/3D, and the frozen full-vector
L2/H1-seminorm and finite-rate gates. Expectations and allowances are never
adapted to product outputs. Required falsifiers cover missing/shared edge data,
wrong associations/orientation, skipped boundary midpoints, reversed natural
flux, incomplete replay and insufficient quadrature. The reference packet and
its original producer identities remain unchanged.

No product PDE execution is authorized until the implementation, public
roundtrip checks, producer and exact artifact/source hashes have been reviewed.

## Quadrature contracts

A standalone RealizationPlan promises to cover products of the selected basis:
P2 simplex products have degree four, requiring three Duffy points per axis on a
triangle and four on a tetrahedron. That strict validator remains unchanged in
purpose. Its portable projection revalidates that standalone plan.

The portable graph records structural closure and solver compatibility. Its
construction and decoding do not independently establish equation quadrature
adequacy. Removing the former dimensional-graph basis-product restriction is an
intentional correction to that acceptance contract, not a numerical invariance
claim. Moment profiles retain their existing strict checks. An equation-aware
consumer must authenticate the graph against its actual equations.

The ordinary CommonPlan reconstructs the Model, mesh, bound volume forms and all
natural facet requirements on both construction and replay. One persisted rule
covers their maximum requirement and the existing profile accuracy floor. Thus
affine P2 diffusion with constant source admits three tetrahedral Duffy points;
a P2 mass term requires four. Higher source or facet degrees raise the same
persisted rule. A raw graph roundtrip is never sufficient execution admission.
The existing scalar P1 floor remains three, MINI basis products select four in
2D and six in 3D, and tetrahedral moment profiles retain three.

Structural compatibility also preserves existing abstract higher-order field-wise
graphs. It rejects unsupported simplex dimensions but does not promise an ordinary
kernel for every abstract order. The ordinary DiscreteSpace binding independently
rejects orders without implemented kernels; this workflow exposes only P2.

## Initial installed-Python launcher

`check.py` is deliberately limited to the reviewed 2D N=2 quadratic patch. It
accepts either a zero-run preflight or one ordinary run; there are no level,
profile, repetition, warmup or timing switches. The CPU affinity is a resource
envelope of at most four CPUs. The actual Plan remains host-serial with one
worker, and threaded libraries are fixed to one thread. GPUs and multiple ranks
are excluded from this witness.

The launcher requires the reviewed harness revision, native producer revision,
native module digest/build receipt and frozen reference manifest digest. It
persists original source, MSH input, Model, mesh and reopened Plan before running.
After its single ordinary call, it persists the original Result before replay or
field export. Failure keeps those files and the attempted/completed call counters;
it never retries. The complete field then goes through the shared output owner
and the unchanged independent patch gate. No native constrained-action witness
is claimed by this Python launcher.

Use a fresh home-backed output directory and reviewed values for the placeholders:

```
python -I -B check.py --mode preflight --output <fresh-directory> \
  --expected-harness-revision <reviewed-harness-commit> \
  --native-receipt <reviewed-build-receipt> \
  --expected-native-source <actual-native-producer-commit> \
  --expected-native-sha256 <actual-native-module-digest> \
  --expected-reference-sha256 <frozen-reference-manifest-digest>
```

Changing `preflight` to `solve` is a separate scientific execution step. The
source, artifact and launch controls require review before that step. Successful
preflight, pure protocol tests and raw portable graph decode do not establish a
PDE accuracy result.
