# Glossary

These terms are used consistently across Eqiora, from a model's mathematical
meaning to the evidence produced when it runs:

```text
meaning → lowered contract → realization → adapter → evidence
```

Capitalized terms name Eqiora concepts; lowercase words keep their ordinary
technical meaning. Terms not listed here are defined where they first appear.

## Easily confused terms

| Term | Distinct from | Boundary |
|---|---|---|
| Semantic Model | Realization | The model states mathematical meaning; Realization selects how to approximate and execute one model revision. |
| Relation | Operator | A Relation is canonical implicit meaning. An Operator is a lowered executable action produced from resolved meaning. |
| ClockDomain | ExecutionSchedule | A ClockDomain controls model-time activation. An ExecutionSchedule assigns deployment resources, priority, or deadlines. |
| hybrid event | Completion | A hybrid event changes model state or mode. A Completion is an execution fence for asynchronous work. |
| Connection | dependency edge | A Connection carries signal or conserving model semantics. A dependency edge records compile, artifact, or execution ordering. |
| projection | canonical state | A projection is a view of one revision. Layout, selection, camera, and panel state do not become model meaning. |
| verification evidence | benchmark result | Evidence tests a specific claim against declared acceptance rules. A benchmark also measures cost in a recorded environment. |
| source provenance | semantic identity | Provenance records where content came from. Formatting and source spans may change without changing model meaning. |
| structural semantic fingerprint | Model artifact identity | The fingerprint compares alpha-normalized graph structure. The artifact identity also covers occurrences, wire format, replay, provenance, and mutation. |

## Modeling

| Term | Meaning |
|---|---|
| Semantic Model | The immutable, canonical mathematical model produced by validated transactions. It excludes solver choices, UI layout, and runtime handles. |
| Semantic Kernel | The closed set of meaning-bearing node kinds: Domain, Representation, Field, Parameter, Port, Relation, Activation, Connection, and ClockDomain. |
| Standard Ontology | Typed named subgraphs built from kernel nodes, such as a Model, Coupling, Scale, Objective, Solver, or EvidenceSet. It is extensible without adding kernel meaning. |
| Relation | An implicit mathematical statement, normally written as a residual equal to zero. ODEs, DAEs, algebraic loops, and acausal networks share this form. |
| Activation | The condition under which a Relation participates: continuous, periodic, event-driven, or guarded. |
| Port | A typed interface point. Signal ports express causal transfer; conserving ports participate in acausal conservation laws. |
| Connection | A typed link between compatible ports. Its kind determines signal or conserving semantics; it never implies task scheduling. |
| connection fragment | One type-checked source statement that contributes to a scalar-physical connection set. It exists only during compilation and is neither a Kernel node nor a final Connection. |
| ClockDomain | Exact model-time activation shared by periodic or synchronous Relations. It is not an RTOS task or wall-clock timer. |
| canonical | Deterministically normalized content with one versioned interpretation. Canonical does not mean optimal or human-preferred. |
| revision | One immutable committed state. Semantic and Realization revisions advance independently and are never interchangeable. |
| transaction | A validated, atomic request to produce a new revision from an explicit base. A rejected transaction leaves the base unchanged. |

## Lowering and numerics

| Term | Meaning |
|---|---|
| lowering | A deterministic transformation from a broader contract to a narrower executable contract, accompanied by the proof or admission facts needed by the next layer. |
| lowered contract | Backend-neutral executable structure after semantic choices are resolved. It contains no UI objects or third-party runtime handles. |
| Operator IR | Owned, versioned intermediate representation for executable scalar or local actions. It is derived from Relations and is not a second model language. |
| residual-native | Operating directly on an implicit residual such as `F(t, y, y_dot) = 0`, without disguising it as a narrower explicit or constant-mass system. |
| LinearizedRelation | The primal residual together with Jacobian actions at one accepted linearization point. It supplies primal, JVP, and VJP operations without differentiating solver iterations. |
| JVP | Jacobian-vector product: the forward action of a linearization on a perturbation direction. |
| VJP | Vector-Jacobian product: the transposed or adjoint action used for reverse sensitivity. |
| local operator | An anonymous entity-local numerical contribution before global numbering, ownership, constraints, and reduction. |
| assembly | The explicit map from validated local contributions into a global algebraic structure under a declared ordering and constraint policy. |
| reference semantics | A small implementation that defines correct behavior. Optimized paths must match it. |

## Realization and execution

| Term | Meaning |
|---|---|
| Realization | The independent layer that selects discretization, solver plan, target, data layout, and deployment schedule for one Semantic Model revision. |
| capability | A typed feature that a complete lowering and backend path actually supports, such as a scalar type, dimension, solver method, layout, target, or reduction policy. |
| admission | The check, before any allocation or execution, that every requirement is supported. An unsupported request is rejected; there is no silent fallback. |
| adapter | An isolated implementation of an Eqiora-owned lowered contract using a third-party library, runtime, file format, or device API. Library types remain behind the adapter. |
| backend | A concrete execution implementation selected only after capability admission, for example the reference path, faer, MPI, CUDA, or Diffsol. |
| ExecutionSchedule | Deployment placement and scheduling policy. It may carry process, thread, priority, or deadline choices but no model-time activation. |
| Completion | Backend-neutral identity for finished asynchronous work on an execution queue. It is not a state-machine event. |
| ordered execution | Execution under a declared operation and reduction order. It does not promise cross-architecture bit identity unless that stronger claim has evidence. |

## Artifacts and evidence

| Term | Meaning |
|---|---|
| artifact | A versioned, bounded, canonical wire object with explicit identity. Filesystem paths and live runtime handles are not artifact identity. |
| digest | A content identity computed over one explicitly versioned canonical domain. A digest is not a compatibility claim or a trust signature. |
| Model artifact reference | The identity of one canonical Model wire without its content: its domain-separated digest, typed Model identity, and semantic revision. Holding a reference does not mean holding the Model. |
| structural semantic fingerprint | A generation-tagged digest of one closed, alpha-normalized Semantic Model graph, used only to compare structure. It is never used as an execution, replay, provenance, or mutation identity. |
| replayable Model artifact | A sealed explicit Model envelope that yields its exact artifact reference and validated immutable Kernel Program together. Replay is neither wire auto-detection nor migration. |
| validated fixed-spatial context | A borrowed, non-serializable runtime value showing that one replayable Model, Realization, geometry, correspondence, and mesh lineage have been checked together, so the expensive replay is not repeated. It is not a general-purpose context artifact. |
| represented physical Field | A Realization-selected Semantic Field that belongs in a physical observation whether its coefficients are algebraic unknowns or reconstructed from an eliminated state relation. Constraint multipliers are not represented physical Fields. |
| Field snapshot | One exact Semantic Field, support, coherent-SI type, frame, and complete canonical coefficient-block inventory over mesh-bound discrete-Field leaves. Storage layout is not Field-snapshot identity. |
| spatial state | The complete represented physical-Field inventory at one accepted model-time coordinate under one exact fixed-spatial lineage. It is an observation, not a solver restart checkpoint. |
| spatial trajectory | An immutable, content-addressed prefix of bounded state-reference segments with exact fixed resources and canonical Field inventory. Publishing an extension creates a new root identity. |
| Dataset view | A typed derived selection of exact trajectory states and Fields. The v1 spatial view is identity-only, unnormalized, unpartitioned, and copies no numerical values. |
| ML Dataset | A distinct immutable interpretation of an exact spatial trajectory: typed feature/target windows, ordered split lineage, and training-only fitted statistics. Its logical artifact stores references rather than values or external layout; bounded materialization returns explicit mesh-local ragged blocks. |
| provenance | Typed records connecting model, lowering, Realization, adapter, environment, and evidence identities. |
| package Run binding | A separate canonical identity edge from one exact package compilation to one model-matched Run manifest. It does not alter either artifact or prove execution. |
| verification case | A reproducible directory under `verify/` containing a machine-readable contract, inputs, expected evidence, and explicit non-claims. |
| evidence | Immutable observations evaluated against declared acceptance rules. The producer and the independent verifier are both recorded. |
| true residual | A residual recomputed from the accepted solution through an independent action, rather than trusting a recursive solver estimate alone. |
| claim | The specific behavior that passing evidence establishes. Behavior outside a claim is unsupported, even if code for it exists. |

## Component and package terms

Some of these terms come from RFCs and may name features that are not fully
implemented. Being listed here does not make a feature supported.

| Term | Meaning |
|---|---|
| component definition | A typed source declaration with a public interface and private internals. Instances elaborate to the existing flat canonical Relation network. |
| instance | A compile-time binding of one component definition at one deterministic path, not a runtime object or kernel node. |
| support slot | A Component declaration for required spatial support: either a volume with an exact ambient dimension or a boundary with an exact parent volume slot. A slot is compiler vocabulary, not a Kernel entity, mesh, or realization. |
| support binding | An occurrence-local mapping from one public support slot to one exact existing Domain. It must preserve support kind, dimension, and boundary parentage and disappears after flattening except for source provenance. |
| Model Package | A versioned, content-addressed unit of reusable model-space declarations. It cannot contain a private semantic engine or execution backend. |
| source bundle | The package manifest with its source and diagnostic material, recorded separately from the semantic content so that formatting or import-alias changes alter provenance without changing meaning. |

Component terms are defined in
[RFC 0021](../rfcs/0021-component-hierarchy-and-instantiation.md) and package
identity in
[RFC 0022](../rfcs/0022-exact-package-identity-and-resolution.md). Local
component elaboration is verified by
[`language.component-elaboration`](../verify/language/component-elaboration/README.md),
and offline packages by
[`packages.offline-model-package`](../verify/packages/offline-model-package/README.md).
There is no registry discovery, version ranges, signatures, trust model, build
scripts, or dynamic plugins.

Scalar-physical connection fragments and boundary partitions are defined by
[RFC 0033](../rfcs/0033-hierarchical-conserving-connection-sets.md). That
contract covers topology only; typed physical-boundary result projection and
replay are defined separately by
[RFC 0036](../rfcs/0036-physical-exposure-projection-artifacts.md).
Occurrence-bound volume/boundary support slots are defined by
[RFC 0034](../rfcs/0034-occurrence-bound-spatial-supports.md) and registered by
[`packages.component-spatial-supports`](../verify/packages/component-spatial-supports/README.md).
They bind Component-local scalar Fields and Relations to existing Domains; they
do not define field-valued Ports, result projections, or numerical transfer.
