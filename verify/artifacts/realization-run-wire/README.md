# Realization and run wire verification

This case compiles the canonical Poisson model, resolves typed Realization
policy, serializes `eqiora.realization-envelope/v1`, and links it to
`eqiora.run-manifest/v2`. Both artifacts round-trip to identical canonical
bytes and domain-separated digests.

The `expected/realization-v1.json` literal fixes the current scalar envelope's
canonical field order and scalar policy bytes. The test removes its final text
newline and requires exact re-encoding; the solver-owner consolidation does not
change this independently fixed specimen. Historical v2/v3 readers are absent.

Linear controls are serialized by the validated `SolverPlan` owner. A separate
positive path independently names the CG, BiCGSTAB, MINRES and sparse-LU wire tags,
resolves each against an explicit matching capability set, and checks exact
solver controls and reduction-policy linkage after artifact replay. This proves
persistence of admitted policy, not numerical execution by a provider.

The falsifying paths pair an unrelated model with a resolution, omit required
distributed layout artifacts, change the executed reduction policy, and alter
the resolved worker count. Every mismatch must fail before a run manifest is
created or accepted against its referenced Realization.

Typed loopback/MPI/CUDA provenance vocabulary does not claim those adapters
are executable. Backend support remains gated by RFC 0010 evidence.

Run:

```bash
cargo test -p eqiora-artifact --test realization_run_wire
cargo run -p eqiora-verify -- run --case artifacts.realization-run-wire
```
