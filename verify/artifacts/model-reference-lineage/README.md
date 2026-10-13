# Typed Model artifact reference lineage

This case compiles three Models in the current artifact epoch, spanning the
spatial, scalar-physical, and field-boundary vocabularies, projects each to one
sealed typed identity contract, and binds each reference to a graph over its actual spatial Field and Domain
in the current Realization and Run v2 artifact chain:

```text
current Model
  -> typed artifact reference
  -> current graph Realization
  -> Run v2
```

Every Model is replayed through the public current-only `ModelDocument` path
before the Realization link is checked. A second test exercises the artifact
owner directly: it encodes and decodes one current Model, replays its immutable
Kernel Program, and proves that the resulting typed reference still validates
the current Model artifact. This native ODE fixture has no spatial Domain and
therefore creates no spatial Realization.

This is identity-lineage evidence. The scalar-physical and field-boundary
Models select their actual dimensionless scalar Field and exact spatial support
for graph identity but are not numerically lowered or executed, and no numerical or physical result is
accepted. Realization and Run retain one typed current reference rather than
an artifact generation selector. The contract performs no historical wire
detection, fallback, schema upgrade, or migration; v1--v7 bytes are negative
specimens owned by the current canonical-identity case.

Run:

```bash
cargo test -p eqiora --test model_artifact_reference_lineage
cargo run -p eqiora-verify -- run --case artifacts.model-reference-lineage
```

See [RFC 0037](../../../rfcs/0037-version-neutral-model-artifact-reference.md)
and [RFC 0083](../../../rfcs/0083-current-model-artifact-epoch.md).
