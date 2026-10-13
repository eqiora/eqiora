# Realization and run wire verification

This case compiles the canonical Poisson model, resolves typed Realization
policy, serializes `eqiora.realization-envelope/v11`, and links it to
`eqiora.run-manifest/v2`. Both artifacts round-trip to identical canonical
bytes and domain-separated digests.

The `expected/realization.json` literal independently specifies one P1 Field
on one generated Domain, a single SPD system and CG solve, two-point Gauss
quadrature, exact binary tolerances, and one host worker. Fixed IDs 1/2/3 and
an opaque repeated-byte Model digest make this a codec specimen, not a Model
or numerical-result claim. The test removes its final text newline and requires
exact re-encoding. Every numerical choice is embedded once in the portable
graph; displaced scalar, mixed, coupled, and ALE envelopes have no readers.

Linear controls are serialized by the validated `SolverPlan` owner. A separate
positive path independently names the CG, BiCGSTAB, MINRES and sparse-LU wire tags,
resolves each against an explicit matching capability set, and checks exact
solver controls and reduction-policy linkage after artifact replay. This proves
persistence of admitted policy, not numerical execution by a provider.

The falsifying paths pair an unrelated model with a resolution, omit required
distributed layout artifacts, change the executed reduction policy, and alter
the resolved worker count. Every mismatch must fail before a run manifest is
created or accepted against its referenced Realization.

A synthetic CUDA provenance test binds ordinals 0 and 7 to identical portable
Realization bytes and distinct Run digests. It does not initialize a GPU.

Typed loopback/MPI/CUDA provenance vocabulary does not claim those adapters
are executable. Backend support remains gated by RFC 0010 evidence.

Run:

```bash
cargo test -p eqiora-artifact --test realization_run_wire
cargo run -p eqiora-verify -- run --case artifacts.realization-run-wire
```
