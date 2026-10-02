# Diagnostic code registry

Diagnostic codes are append-only and are never reused for a different
condition. Public APIs return structured `Diagnostic` values rather than bare
strings.

| Range | Area |
|---|---|
| `EQ00xx` | Infrastructure and feature availability |
| `EQ01xx` | Graph nodes, edges, transactions, and provenance |
| `EQ02xx` | Standard Ontology named subgraphs |
| `EQ03xx` | Semantic definitions, expressions, and clocks |
| `EQ04xx` | Units and dimensions |
| `EQ05xx` | Reference execution and numerical approximation |
| `EQ06xx` | Eqiora Language lexing, parsing, names, and static types |
| `EQ07xx` | Semantic/Operator IR lowering and evaluation |
| `EQ08xx` | Numerical realization and solver kernels |
| `EQ09xx` | Serialized artifacts, manifests, and content identity |

## Assigned codes

| Code | Meaning |
|---|---|
| `EQ0001` | Feature is specified but not implemented |
| `EQ0002` | An internal invariant failed at a public language or process boundary |
| `EQ0101` | Referenced graph node does not exist |
| `EQ0102` | Graph-node identifier already exists |
| `EQ0103` | Erased ID kind does not match the operation |
| `EQ0104` | Edge is not allowed by the kernel schema |
| `EQ0105` | Operation is invalid for the target entity kind |
| `EQ0106` | Optimistic-concurrency precondition failed |
| `EQ0107` | Immutable provenance was targeted for mutation |
| `EQ0201` | Named subgraph violates a structural or schema invariant |
| `EQ0202` | Ontology-view identifier already exists |
| `EQ0203` | Ontology-view identifier does not exist |
| `EQ0204` | A transaction would leave an ontology view with a removed member |
| `EQ0301` | Expression DAG is empty or structurally invalid |
| `EQ0302` | Semantic Kernel definition violates a local invariant |
| `EQ0303` | Expression references a symbol outside the selected model |
| `EQ0304` | Residual expression has incompatible physical dimensions |
| `EQ0305` | ClockDomain exact-time definition is invalid |
| `EQ0401` | Runtime dimension does not match the expected dimension |
| `EQ0501` | Reference-execution time or iteration configuration is invalid |
| `EQ0502` | A required initial value or external input is missing |
| `EQ0503` | An active executable-kernel v0 system is not square; continuous regular balance excludes initial equations |
| `EQ0504` | The reference nonlinear solve did not converge or is singular, including deficient continuous equation incidence |
| `EQ0505` | Expression or solver evaluation produced a non-finite value |
| `EQ0506` | Cooperative execution cancellation was accepted at a safe boundary |
| `EQ0601` | Source text contains an invalid token |
| `EQ0602` | Source text does not satisfy the Eqiora Language grammar |
| `EQ0603` | A source name or static type cannot be resolved |
| `EQ0604` | Typed source cannot be lowered to a graph transaction |
| `EQ0701` | Operator IR is structurally invalid or inconsistent with its source |
| `EQ0702` | Operator IR receives the wrong scalar symbol input count |
| `EQ0703` | Canonical spatial semantics cannot be lowered by the selected realization |
| `EQ0704` | A linearization point, variable binding, tangent, or cotangent is invalid |
| `EQ0705` | A continuous subsystem cannot be lowered to the selected time-equation class |
| `EQ0801` | A numerical grid, coefficient, time step, or state is invalid |
| `EQ0802` | A numerical linear solve failed or produced a non-finite result |
| `EQ0803` | Mesh topology or geometry violates a realization invariant |
| `EQ0804` | A quadrature rule is incompatible with its reference cell or invalid |
| `EQ0805` | A local operator contribution has invalid shape or values |
| `EQ0806` | A local-to-global assembly map or sparse accumulation is invalid |
| `EQ0807` | A realization policy is invalid, contradictory, or unsupported |
| `EQ0808` | External mesh input violates an admitted importer boundary |
| `EQ0809` | A mesh-associated discrete field has invalid association, shape, or values |
| `EQ0810` | External scientific data violates an admitted adapter or resolver boundary |
| `EQ0811` | Accepted scientific data cannot be projected through an export adapter |
| `EQ0901` | A serialized artifact, digest, or run manifest is invalid |
