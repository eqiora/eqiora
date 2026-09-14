# RFC 0083: One current Model artifact epoch before 1.0

- Status: Implemented
- Authors: Eqiora contributors
- Created: 2026-07-31
- Depends on: [RFC 0037](0037-version-neutral-model-artifact-reference.md),
  [RFC 0054](0054-curated-facade-and-control-plane.md),
  [RFC 0073](0073-structural-semantic-fingerprint.md),
  [RFC 0077](0077-exact-cartesian-domain-edit.md), and
  [RFC 0078](0078-direct-parameter-driven-cartesian-coordinates.md)
- Supersedes: the live multi-generation Model/Transaction compatibility policy
  in RFC 0037, and the stable `eqiora::compatibility` decision and requirement
  for a separate compile/check feature list in every control generation in
  RFC 0054

## Summary

Eqiora will use its pre-1.0 period for one explicit Model artifact epoch reset.
The accepted Model and Model Transaction v8 meanings become the only live
runtime contracts. Their persisted schemas, canonical bytes, digest domains,
decoder limits, and semantic meaning do not change, but product APIs stop
presenting v8 as one selectable member of a historical v1--v8 ladder.

Model and Transaction v1--v7 decoders, encoders, selectors, compatibility
tests, and compatibility-only fixtures are removed. Old bytes reject as
unsupported. No schema identifier is reassigned, no decoder sniffs or retries,
and no migration is inferred from semantic similarity.

This is a bounded breaking compatibility decision, not a general removal of
artifact versions. Every other artifact family keeps its current contract
until a separate decision names it.

## Why now

Direct Parameter-driven Cartesian coordinates and their atomic regeneration
owner close the current Model meaning required by the next CAD work. Keeping
eight Model and Transaction branches would force every new Studio, Python,
CAD, package, and project consumer to carry compatibility policy that the
public alpha has not promised to preserve.

Removing that ladder after those consumers exist would be more expensive and
would reward a temporary architecture with permanent product shape. The reset
therefore lands after RFC 0078 and its regeneration slice, and before the
shared Studio/Python CAD authoring graph.

## Frozen current contract

The implementation retains exactly these persisted identities:

| Contract | Frozen identity |
| --- | --- |
| Model schema | `eqiora.model-envelope/v8` |
| Model Transaction schema | `eqiora.model-transaction-envelope/v8` |
| Canonical encoding | `eqiora.canonical-json/v1` |
| Model meaning | the complete accepted v8 Semantic Kernel vocabulary |
| Transaction meaning | the complete accepted ordered v8 transaction vocabulary |
| Model digest domain | the existing v8 Model schema domain |
| Transaction digest domain | the existing v8 Transaction schema domain |

The independently accepted current-owner fixture in
`crates/eqiora-artifact/tests/current_model_wire_oracle.rs` observes:

| Fixture output | Exact value |
| --- | --- |
| Model canonical byte length | `2347` |
| Model SHA-256 | `e410295337a3a51a271f272e03ae7d7a4b8e7df1b04faf76645bb1e18567e4b3` |
| Transaction canonical byte length | `2646` |
| Transaction SHA-256 | `132168803ac8882f0f35187215d3f2ce44817d03921d6ad95b73a9cac62aa102` |

These values are independently frozen complete canonical Model and Transaction
JSON for the public fixture. Old-schema specimens remain beside them only as a
negative rejection corpus. A semantically equivalent fixture with different
local order is not byte evidence; the implementation does not author, tune, or
relax this oracle.

The same independent lane must re-encode and freeze the shipped
`steady-flow-past-cylinder` Model as current v8. It owns the new complete
canonical bytes, artifact digest, raw resource digest, byte length, and
unversioned packaged resource name used by the Python and Studio registered
cases. It also owns the replacement registered identity oracle
`artifacts.current-model-canonical-identity`; consumers replace the dangling
`model.model-envelope-v7-canonical-identity` reference with that exact case.
The lane proves that the semantic Model identity, revision, and replayed
program are unchanged while the schema-domain artifact identity changes from
v7 to v8. The implementation writer may wire those values but cannot derive
or adjust them. The existing fluid, geometry, mesh, balance, and solver
oracles remain unchanged.

Persisted names retain `v8` because they identify released bytes. Public Rust
and Python owners become unversioned current owners; the suffix is not reused
for future meaning. A future incompatible Model meaning requires a new schema
identifier and a new compatibility decision.

## Removal inventory

The following live Model/Transaction compatibility surfaces are removed
together:

- `ModelEnvelopeV1` through `ModelEnvelopeV7` and
  `ModelTransactionEnvelopeV1` through `ModelTransactionEnvelopeV7`, including
  their public exports, runtime encode/decode branches, and old schema
  admission;
- the public v8 spellings `ModelEnvelopeV8` and
  `ModelTransactionEnvelopeV8`, replaced by one unversioned current
  `ModelEnvelope` and `ModelTransactionEnvelope`;
- `ModelArtifactGeneration`, the generation variants inside
  `AcceptedModelArtifact`, and the versioned Transaction wrapper;
- Rust `ExactModelCodec`, `eqiora::compatibility`, exact-codec constructors,
  capability predicates, codec fields retained by `ModelDocument`, and caller
  generation selection in compile, define, replay, edit, and package paths;
- Python `eqiora.compatibility`, its `ExactModelCodec`, corresponding stubs,
  and exact-generation arguments;
- Studio and control-plane inputs that let a caller choose a historical Model
  generation; and
- compatibility-only v1--v7 wire goldens, route matrices, documentation, and
  capability claims.

There is no deprecated alias or facade for these names. A compatibility facade
would preserve the architecture this reset exists to delete.

Shared implementation code may survive under current, unversioned ownership
when v8 still uses it. In particular, the current implementation hosts v8
encoding in `model_v2.rs` and shares the canonical-encoding constant and wire
DTOs in the original Model/Transaction modules. The owner slice moves or
renames that code under current ownership while preserving its exact behavior;
retaining a private helper does not retain `ModelEnvelopeV2` or its decoder. A
file or helper is not retained merely because deleting it is inconvenient,
nor is current logic rewritten merely because it originated in an older
module. The checked outcome is one current path and no branch that admits or
emits a historical schema.

## Consumer migration inventory

The reset is incomplete until every row reaches the current owner:

| Consumer seam | Required migration |
| --- | --- |
| `eqiora-artifact` | expose one current Model and Transaction owner; retain exact v8 wire and typed replay/reference checks |
| `eqiora-api` and facade | make `ModelDocument` current-only; remove codec state, exact selectors, and the stable compatibility namespace |
| control compile/check | retire control v1 from ordinary dispatch without reinterpreting it; introduce control v2 without caller-selected Model history |
| compiler and package paths | emit the current Transaction and Model only; recompute dependent exact artifact references without changing source/package semantics |
| Rust applications and examples | replace fixed historical codec choices with the current ordinary path |
| Python extension, package, stubs, and tests | expose one ordinary current path and reject old persisted Model bytes |
| Browser Studio request schemas | reference the same current identifiers without a route-local switch |
| spatial authored-field/context projections | change only their Model input to the current typed owner; retain the spatial artifact schemas and replay rules |
| verification and documentation | preserve semantic and scientific assertions, remove compatibility-only claims, and index only the current artifact boundary |

The inventory currently spans `crates/eqiora-artifact`,
`crates/eqiora-api`, `crates/eqiora`, `crates/eqiora-python`,
`bindings/python`, `studio`, `packages`, `examples`, `verify`, `docs`,
`schemas`, `tools/ci`, `tools/docs`, `tools/release`, `tools/xtask`, `api`,
`CHANGELOG.md`, `pyproject.toml`, and affected RFC references. This list is an
ownership boundary, not permission to make unrelated cleanups in those trees.

## Evidence retention rule

Artifact history and scientific evidence are different things.

- Delete a fixture only when its sole claim is that a removed Model or
  Transaction generation remains callable or byte-stable.
- Migrate a fixture when it proves semantic admission, rejection, geometry,
  package, numerical, physical, or lineage behavior independent of the
  removed generation.
- Preserve every scientific expected value, tolerance, balance, convergence
  order, mutant, and falsifier unless a separate oracle review proves it
  wrong.
- Regenerated current artifact digests are implementation outputs, not
  permission to alter an oracle.
- Keep the v8 exact Model/Transaction fixture above and current malformed,
  bounded-decoder, canonical-order, replay, and wrong-schema rejection tests.

Before the reset, `artifacts.model-reference-lineage` and
`interfaces.current-authoring-profile` contained historical compatibility
claims, `interfaces.control-plane-compile-check` was a control-v1 case, and
`interfaces.python-control-plane` retained exact-codec references. The reset
moved each to the current owner and promoted the control-v2 conformance kit
without silently narrowing its claims.

Unknown-protocol falsifiers in Rust unit/integration tests, Studio tests, and
registered control fixtures used the then-future spelling `eqiora.control/v2`.
Promotion moved them to the deliberately invalid non-version spelling
`eqiora.control/unknown-test`; the live control-v2 conformance kit retains that
invalid-spelling rejection.

The packaged cylinder resource is an oracle input, not merely producer output.
`interfaces.python-exact-cylinder-stokes-result` and
`interfaces.studio-exact-cylinder-stokes-demo` must consume the independently
frozen current resource and retain all non-codec scientific and lineage
assertions.

The implementation PR updates `docs/capability-matrix.md`, especially the
rows for deterministic serialization, versioned semantic Model artifacts,
typed Model artifact references, exact package-to-Run lineage, curated Rust
facade, shared application service, current authoring versus exact codecs,
Transaction wire, Python source/native authoring, and Python revision/edit
surfaces. The matrix must describe the implemented current-only boundary, not
the accepted plan. Downstream Realization/Run identity and current-authoring
assertions that remain true must survive.

The existing `model_v8_wire.rs` is decomposed rather than retained verbatim.
Its positive current fixture, canonical permutation checks, and operation-order
checks move to the independent full-byte oracle. Its calls into v1--v7
encoders disappear with those encoders. Literal old-schema specimens form the
separate negative corpus that proves the current decoder rejects historical
bytes without recreating historical runtime branches.

## Separately versioned families retained

This RFC does not collapse or rename these artifact families:

| Family | Retained owners |
| --- | --- |
| CAD and Geometry | CAD design/build evidence, Geometry definition/identity/state/revision association, mesh correspondence, and circular-hole source binding |
| Mesh and import | simplicial Mesh, revision overlap, external-import manifests/observations, resolved arrays, and root registration |
| Realization and execution | Realization v1--v5, Realization references, Run manifests, distributed layouts/systems, and execution provenance |
| Time and restart | time lowering/run, general implicit time, checkpoint, and restart artifacts |
| Spatial data | discrete Fields, spatial states/trajectories and segments, storage chunks, and XDMF/HDF5 trajectory storage |
| Transfer and derived data | remesh transfer/projection, datasets, and ML dataset artifacts |
| Exposure and comparison | physical-exposure artifacts and structural semantic fingerprints |

Their suffixes identify different wire families, not historical Model
compatibility. Any later current-only reset must prove its own ladder,
consumers, exact retained contract, and rejection behavior.

Retaining a family means preserving its schema identity, digest domain,
decoder meaning, and public contract. A Realization, Run, spatial context, or
other exact instance that names a migrated Model digest necessarily receives
new relationally derived bytes and identity; its independently owned semantic
and scientific assertions do not change. Fixtures unrelated to a migrated
Model reference remain byte-identical.

## Control and release behavior

The control protocol remains an independently versioned family.
`eqiora.control/v1` includes caller-selected `modelWire` meaning and therefore
cannot be rewritten in place as current-only. Its schema identity and
historical bytes are not reassigned; ordinary runtime dispatch retires v1 and
rejects it as unsupported.

A new `eqiora.control/v2` compile/check contract removes `modelWire` from the
request. Its response may report the fixed current Model schema as an observed
output fact, but not as caller policy. Rust adapters and Studio consume v2
without a route-local generation switch. Python instead performs its own
language-specific admission and invokes the same `ModelDocument::compile`
operation directly, so it shares compilation meaning without becoming a
control-v2 client. This RFC does not otherwise collapse the control protocol
family.

The Python pre-1.0 release policy permits this immediate removal only because
this RFC fixes the scope, retained bytes, rejection behavior, consumer
inventory, and migration note. This exception does not authorize unrelated
public API removal or silent behavior changes.

### Frozen compile/check v2 contract

Control v2 changes the protocol envelope, not the compile/check operation.
The exact identities are therefore:

| Identity | Value |
| --- | --- |
| protocol | `eqiora.control/v2` |
| command | `model.compile-check/v1` |
| schema file | `crates/eqiora-api/schemas/compile-v2.schema.json` |
| schema `$id` | `urn:eqiora:schema:control:compile-v2` |
| registered case | `interfaces.control-plane-compile-check` |

The request is one closed object with exactly `protocol`, `command`,
`requestId`, `filename`, and `source`, serialized in that order. The response
is one closed object with exactly `protocol`, `command`, `requestId`, and
`outcome`, serialized in that order. `requiredFeatures`, `modelWire`, and every
`model-wire/*` feature disappear from both directions. For this command the
closed protocol and command identities are the complete version-admission
mechanism and supersede RFC 0054's separate per-generation feature-list
requirement. Extending the request vocabulary requires another exact control
contract rather than permissively admitting a new feature spelling.

An accepted outcome has `status` followed by `model`. Its closed Model
descriptor has these fields in order:

1. `schema`, fixed to `eqiora.model-envelope/v8`;
2. `transactionSchema`, fixed to
   `eqiora.model-transaction-envelope/v8`;
3. `digest`, the 64-character lowercase Model artifact digest;
4. `modelId`, the typed Model identity, with length from 1 through 128;
5. `semanticRevision`, a nonnegative integer. The independent witness is
   expected to derive revision 1, but control v2 retains the public revision
   domain admitted by the v1 compile/check schema rather than introducing a
   new positive-only rule.

The two schema values are mandatory observed output facts, not caller policy.
`transactionSchema` discharges RFC 0054's requirement to identify the wire
used for transaction construction. Canonical encoding is implied by the
frozen artifact schemas and is not duplicated as another control wire field.
The schema facts occur only inside an accepted Model descriptor. A rejected
outcome has `status` followed by a nonempty `diagnostics` array and does not
report Model or Transaction schema facts.

The following v1 value shapes retain their field topology and spelling, but
are owned by new v2 Rust DTOs rather than aliases. The numeric response bounds
below are deliberate v2 tightenings rather than inherited v1 admission:

- `requestId` is a string of length 1 through 128 matching
  `^[A-Za-z0-9._:-]+$`;
- `sourceSpan` is a closed `file`, `start`, `end` object in that order;
  `file` has both `maxLength: 4096` and a 4096-byte UTF-8 limit, while
  `start` and `end` are integers from 0 through 4294967295;
- `patch` is a closed object containing one nonempty `summary`; and
- `diagnostic` is a closed object containing `source`, `severity`, `code`,
  `message`, `graphPath`, `span`, and `patch` in that order. Source is
  `control` or `kernel`, severity is `error`, `warning`, or `note`, code
  matches `^[A-Z]{2}[0-9]{4}$`, and message contains 1 through 1048576
  characters and at most 1 MiB of UTF-8. `graphPath` is null or an array of at
  most 256 strings, each containing 1 through 4096 characters and at most
  4096 UTF-8 bytes. `span` and `patch` are required but nullable with the
  shapes above; patch summary contains 1 through 4096 characters and at most
  4096 UTF-8 bytes. A rejected outcome contains at most 1024 diagnostics.

If kernel projection would exceed the message, graph-path, patch-summary,
diagnostic-count, or total response bound, no partial kernel diagnostic is
serialized or truncated. The admitted request receives a rejected v2 response
containing exactly one control-source error with code `EQ0901`, message
`compile/check diagnostics exceed the control v2 response limits`, and null
`graphPath`, `span`, and `patch`.

The v2 schema is derived from this complete section. It does not generally
inherit the v1 schema and in particular must not inherit the v1 schema's
inconsistent `model.schema` v1--v6 enumeration.

The v1 resource policy remains the v2 policy rather than being recomputed from
the smaller envelope: encoded request and response are each bounded by
8 MiB plus 16 KiB; source has both `maxLength: 8388608` and an 8 MiB UTF-8 byte
limit; filename has both `maxLength: 4096` and a 4096-byte UTF-8 limit; and the
request and Model identities are each bounded by 128 characters. A filename
is nonempty and contains no control character. The JSON objects are closed.
Malformed JSON, malformed UTF-8, unknown fields, invalid identifiers, and
exhausted bounds fail with a control-source error diagnostic carrying
`EQ0901`.

Protocol dispatch is bounded and precedes full v2 DTO admission. The
transport-neutral decoder returns either one admitted request or one standalone
`ControlDiagnosticV2`; a pre-admission diagnostic is not a protocol response,
has no request ID, and is never wrapped in a synthetic v2 response. A request
whose `protocol` is `eqiora.control/v1` or `eqiora.control/unknown-test` returns
a control-source error carrying `EQ0001` and never reaches the compiler. The
same rule, including code `EQ0001`, applies to an unknown command after v2
protocol admission. The dispatch prelude admits at most 128 characters and
128 UTF-8 bytes for each of `protocol` and `command`. A longer value returns
`EQ0901` without echoing caller content; its message is required nonempty but
is not frozen. Dispatch never retries or reinterprets the request under another
contract.

A v2 request containing `modelWire`, `requiredFeatures`, or another unknown
member fails the closed v2 DTO with `EQ0901` before compilation. Malformed
input for which no request ID can be admitted likewise returns only the
standalone diagnostic. Once a request is admitted, compilation always produces
the closed v2 response and echoes its request ID; a compilation failure is a
rejected outcome with kernel-source diagnostics and no Model descriptor.

Dispatcher-owned unsupported-protocol and unsupported-command witnesses freeze
the complete diagnostic value, including the stable message. JSON/DTO/resource
admission witnesses and kernel diagnostics freeze source, severity, and code,
and require a nonempty message without freezing parser- or compiler-owned
wording. The empty-source witness retains kernel code `EQ0602`. The v2 schema's
diagnostic definition validates both response diagnostics and the standalone
pre-admission value even though that value is not a top-level response.

Each dispatcher diagnostic has `source = control`, `severity = error`, and
null `graphPath`, `span`, and `patch`. The following are three independent
exact message values; the line feeds separating them are not part of a value:

```text
unsupported control protocol `eqiora.control/v1`; expected `eqiora.control/v2`
unsupported control protocol `eqiora.control/unknown-test`; expected `eqiora.control/v2`
unsupported control command `model.unknown-test`; expected `model.compile-check/v1`
```

The independent oracle owns these exact input witnesses:

- accepted: request ID `shared-accepted-v2`, filename `shared-decay.eqi`, and
  the complete source below;
- rejected source: request ID `shared-rejected-source-v2`, filename
  `empty.eqi`, and an empty source;
- retired protocol: `verify/interfaces/control-plane-compile-check/models/retired-v1.json`,
  the previous accepted v1 request preserved byte-for-byte;
- unknown protocol: the accepted v2 witness with protocol changed to
  `eqiora.control/unknown-test`;
- forbidden selection: the accepted v2 witness plus `modelWire: "v8"`;
- forbidden feature list: the accepted v2 witness plus
  `requiredFeatures: ["model.compile-check/v1", "model-wire/v8"]`;
- unknown command: the accepted v2 witness with command changed to
  `model.unknown-test`; and
- generated resource falsifiers at source byte length 8388609, filename byte
  length 4097, request-ID length 129, and encoded-request length 8404993.

```eqiora
model decay {
  field x: 1 = 1;
  parameter rate: 1 / s = 1;
  relation flow continuous {
    derivative(x) + rate * x = 0;
  }
}
```

The source string includes one final line feed after the closing brace.

Flat source compilation allocates fresh occurrence ULIDs. The Model ULID is
part of exact artifact identity, so independent compilations of this source
correctly produce different `modelId` and `digest` values; RFC 0073 and
RFC 0078 explicitly make equality of those values a nonclaim. The control
oracle therefore must not freeze either value from one producer run.

Instead, the positive oracle freezes a relation already owned by RFC 0073:
two independent control invocations and one ordinary compilation of this exact
source must have pairwise-distinct exact Model identities and digests, revision
1, and equal generation-v2 `StructuralSemanticFingerprint` values. Generation
v2 alpha-normalizes occurrence ULIDs and completely covers this witness's
scalar meaning. RFC 0078 separately makes current coordinate sources consumers
of the same structural comparison boundary.

One in-process `execute_compile_v2` invocation returns a
`CompileControlExecutionV2` containing both the closed response and the exact
optional `ModelDocument` that produced it. The response's digest, Model
identity, and revision must equal that same value's artifact reference; they
must never be compared with a second compilation. The fingerprint relation may
use independently compiled documents because it is occurrence-invariant.

The accepted response is also checked for the exact schema and transaction
schema constants, a valid 64-character lowercase digest, a nonempty Model
identity within its bound, and revision 1. The oracle does not freeze one
fingerprint digest or fresh occurrence identity: RFC 0073's registered evidence
owns the fingerprint algorithm, while this application-surface oracle owns the
precommitted equality/inequality relations and response linkage.

The expected contract has schema
`eqiora.verify.control-plane-compile-check/v2`. It names every request file,
the accepted schema facts, revision, identity-shape predicates, structural
fingerprint equality and exact-identity inequality relations, response-to-
document linkage, expected outcome, diagnostic source, severity and code
for every rejection, whether the exact message or only nonemptiness is frozen,
the generated resource boundaries, and the forbidden response fields
`source`, `mesh`, `fields`, and `trajectory`.

The promoted case retains the previous accepted v1 request byte-for-byte at
`verify/interfaces/control-plane-compile-check/models/retired-v1.json` and the
previous v1 schema byte-for-byte at
`verify/interfaces/control-plane-compile-check/expected/historical/compile-v1.schema.json`.
The reset deleted the former live v1 schema, the v1 rejected-source and
unsupported-protocol specimens, and the v1 expected contract as
compatibility-only. Historical copies are not generated, registered, packaged,
or dispatched.

The case ID remains `interfaces.control-plane-compile-check`, with scope
`client-neutral-bounded-compile-check-control-v2`, reference kind
`shared-wire-fixtures-and-ordinary-compiler-admission`, and conformance kit
`control-compile-check-v2`. Its claim boundary replaces `model_wires` with
`model_owner = "current"` and
`model_schema = "eqiora.model-envelope/v8"`; every other boolean non-claim is
retained. All five existing capability entries carry forward. In
`fail-closed-control-negotiation`, negotiation now means closed protocol and
command admission; it does not imply a feature list.

The independent control-oracle lane wrote only the former staging tree
`verify/interfaces/control-plane-compile-check/oracle/v2/**`; it did not edit
the live schema, v1 fixtures, manifest, implementation, capability matrix, or
registries. The atomic implementation promoted that frozen schema to
`crates/eqiora-api/schemas/compile-v2.schema.json` and the frozen v2 fixtures byte-for-byte
into the live case, then removed the staging tree.

### Relational identity transition

Changing the Model digest changes a downstream artifact only when that exact
artifact embeds the Model reference. It does not authorize rewriting an
unrelated separately versioned golden. In particular, the Realization v1--v3
goldens under `artifacts.realization-run-wire` retain their historical opaque
Model references and exact bytes.

Before the implementation writer migrates a checked-in relational identity,
another provider lineage owns the registered oracle
`artifacts.current-model-relational-identity-transition`. A read-only pass
first searches the complete repository for checked-in Model references. Its
write pass commits the complete classification as evidence and independently
derives every permitted identity-only delta. The following are required
classification targets, not an exhaustive inventory:

- fixed-topology ALE 3D accepted trajectory;
- packaged DC motor controller identities;
- composed, offline, and typed-execution package identities;
- Realization v4 wire evidence, classified as a retained separate-family
  golden rather than a current-Model identity delta;
- canonical Cartesian Poisson CUDA and MPI recorded evidence, classified for
  historical bridging rather than identity delta;
- fixed-reference CUDA and distributed MPI FSI recorded evidence, likewise
  classified for historical bridging;
- fixed-reference FSI spatial trajectory;
- geometry-to-Model and typed Model-reference lineage; and
- agent-authored Model change evidence.

The classification is by producer semantics, not by a closed fixture list.
Every Model-bearing fixture found by the repository search belongs to exactly
one of these classes:

- Deterministic hierarchy elaboration, package compilation, or replay receives
  independently precommitted complete current Model bytes and every permitted
  downstream digest. Before the reset, the oracle captures the accepted
  deterministic producer through its already-live current encoder, then
  independently checks canonical JSON, schema-domain hashing, and every
  artifact-reference edge from those captured bytes. The reset writer cannot
  regenerate or select those literals.
- Flat compilation that deliberately allocates fresh occurrence identities
  receives no occurrence-dependent literal. Its oracle freezes identity shape,
  exact-identity inequality, semantic-fingerprint equality, and same-execution
  reference linkage, as in the control-v2 witness above.
- A recorded execution from an environment not reproduced by the default gate
  remains historical observation evidence and receives the semantic bridge
  below, not a relabelled current Run.
- A retained separately versioned artifact-family golden keeps its exact bytes
  and treats a retired Model reference as opaque unless a later transition for
  that artifact family says otherwise.

The composed-package, offline-package, typed-execution, packaged DC-motor, and
fixed-topology ALE fixtures are in the deterministic class, not the flat-fresh
class. Their current Model bytes and downstream identity literals are frozen by
the transition oracle before implementation. Same-execution linkage assertions
are added without replacing their existing cross-run reproducibility claims.

For every class, arrays, coordinates, fields, time values, source and package
identities, tolerances, balances, convergence evidence, and scientific
falsifiers are immutable inputs. A permitted identity-only delta may change
only the current Model digest and an identity computed from an artifact that
embeds that reference.

The Realization v4 golden remains exact historical evidence for its separately
versioned artifact family, like the v1--v3 goldens above. Its embedded Model
digest becomes an opaque historical reference: the reset verifies the
Realization bytes and digest without decoding the retired Model artifact and
does not re-encode the golden against an arbitrary current Model. A separate
future Realization-family transition would be required to replace it.

The checked-in canonical Cartesian Poisson CUDA and fixed-reference CUDA FSI
Model/Realization/Run bundles are recorded accelerator observations. Any
checked-in canonical Cartesian Poisson MPI or distributed MPI FSI execution
bundle is the same class: the default gate does not reproduce an optional
accelerator or distributed environment. Their bytes remain unchanged as
historical evidence and are not members of the identity-delta class. A new
current Run may replace one only after fresh execution in its stated
environment. Until then, the ordinary numerical path consumes the current
Model owner while the case narrows its claim to the historical observation
plus a structural semantic bridge.

That bridge is exact evidence, not prose: before historical decoders are
removed, the independent oracle records the historical artifact raw hash,
artifact digest, and a freshly observed
`SemanticFingerprintGeneration::V2` value. It constructs a current Model
artifact from the same decoded semantic program through the current owner and
records the same fields. The generation-v2 values are produced by the already
verified RFC 0073 owner; this
transition case owns their equality relation, not an independent derivation of
the fingerprint byte projection. RFC 0078 establishes that current coordinate
sources participate in this comparison. The fingerprints must agree while the
schema-domain artifact digests differ. Source identity is deliberately not a
bridge field: the CUDA FSI bundle has no checked-in source, and source-identity
construction is not owned by this transition oracle.
After the reset, runtime verification hashes the untouched historical bytes
without admitting them through a product decoder and replays only the current
artifact. Capability and case wording must say that the old Run observed the
historical artifact and that current semantic equivalence is independently
bridged; it must not say that the current Run was observed.

The historical side's generation-v2 value is derived fresh in this transition
pass and is not inferred from an earlier generation-v1 value. The relational
oracle classifies the control-v2 accepted Model facts but excludes them from
identity-delta ownership because the control oracle owns their shape and
structural relation. Its
writable allowlist is only
`verify/artifacts/current-model-relational-identity-transition/**` and
`crates/eqiora-artifact/tests/current_model_relational_identity_transition.rs`
together with its private support module under
`crates/eqiora-artifact/tests/current_model_relational_identity_transition/`.
Candidate replacement bytes and target paths live under that oracle case;
the oracle did not edit target consumer cases, capability matrix, roadmap,
crate roots, or registries. It returned those registration deltas to the
integrator, and the implementation wired only the precommitted values.
For every classified fixture, the oracle also returns any required claim-
wording delta. The implementation writer updates prose colocated with its
owned test, while the integrator alone edits case manifests and the capability
matrix. In particular, the CUDA/MPI wording narrows to historical observation
plus the semantic bridge; deterministic package and ALE claims retain their
cross-run reproducibility rather than narrowing to same-execution linkage.

The implementation records the break under the `Changed` heading of
`CHANGELOG.md` `[Unreleased]`. That entry names the removed Rust
`ExactModelCodec` and
`eqiora::compatibility` surfaces, Python `eqiora.compatibility`, Model and
Transaction v1--v7 runtime support, and control v1 dispatch. It names the
ordinary current replacements: Rust `ModelDocument::{compile, define,
replay}`, unversioned current Model/Transaction artifact owners, Python
`eqiora.compile`, `Model.define`, and `eqiora.replay`, plus control v2. It also
states that old Model/Transaction bytes and control-v1 requests reject and
that no automatic migration is provided. Release notes derived from the
changelog retain those facts.

## Implementation ownership and order

1. Merge this decision and inventory before implementation.
2. A different provider lineage independently freezes the complete canonical
   v8 Model/Transaction fixture, the v1--v7 negative rejection corpus, and the
   current packaged cylinder Model resource. Implementation does not begin
   until the reference fixture agrees with the candidate lengths and digests
   above.
3. One writer owns the invariant-bearing Model/Transaction seam across the
   artifact owner, API, facade, and every compiling consumer. The accepted
   change must not leave a second compatibility path.
4. Migrate Python, Studio, packages, demonstrations, and registered evidence
   through that same current owner. Work may be parallelized only after its
   consumed contract and writable paths are disjoint.
5. Remove historical modules and ratchet public-surface, file-size, and
   architecture counts down. Do not add a debt entry or raise a ceiling as an
   ordinary step of the reset.
6. Consider other artifact families separately after this reset is accepted.

If a mergeable intermediate revision would require a public compatibility
alias, broaden the owner slice instead of introducing the alias.

## Required falsifiers

- bytes with Model or Transaction schema v1--v7 reject before semantic use;
- control v1 rejects as unsupported and cannot be interpreted as control v2;
- replacing an old schema string with `v8` cannot make old bytes admissible;
- unknown Model/Transaction schemas reject without sniffing, retry, fallback,
  or migration;
- the current v8 fixture retains the exact bytes and digests frozen above;
- no Rust, Python, Studio, control, package, example, or verification caller
  can select a historical Model codec;
- Rust, Studio, and registered control-v2 evidence reject one deliberately
  invalid non-version protocol spelling;
- the release note names every removed public route, its current replacement,
  old-byte/control-v1 rejection, and the absence of automatic migration;
- no accepted consumer remains pinned to a removed envelope;
- changing only a removed generation does not delete or relax an independent
  scientific oracle;
- Realization, Run, Geometry, Mesh, spatial, and other retained artifact
  families keep their schema identities and meanings; only instances that
  name a migrated Model acquire relationally derived new identities; and
- public-surface and architecture counts do not rise to accommodate the reset.

## Alternatives

### Keep v1--v8 behind a compatibility namespace

Rejected. That is the present architecture and continues the audit and
fan-out cost.

### Auto-detect the schema from JSON

Rejected. Sniffing turns malformed or substituted bytes into a decoder search
and makes rejection behavior depend on ordering.

### Rewrite v8 as a new v1

Rejected. Reusing an old schema identifier would reinterpret persisted bytes
and destroy content identity.

### Remove every version suffix in the repository

Rejected. Unrelated artifact families have independent meanings and
compatibility histories.

### Keep aliases for one prerelease

Rejected for this bounded epoch. An alias would keep the generation ladder in
the exact period when new CAD and project consumers would begin depending on
it.

## Nonclaims

This RFC does not promise 1.0 stability, automatic migration, an offline
migrator, cross-schema semantic equality, a universal artifact envelope, or a
reset of any non-Model artifact family.

It does not change Semantic Kernel meaning, physical equations, numerical
methods, scientific tolerances, Geometry meaning, mesh correspondence,
execution plans, Result meaning, or visualization. It does not authorize
deleting evidence whose claim survives the wire reset.
