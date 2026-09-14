# RFC 0016: Studio as an accessible canonical projection

- Status: Native shell retired; browser projection retained
- Authors: Eqiora contributors
- Created: 2026-07-18

## Summary

Eqiora Studio is an accessible browser projection over fixed, runtime-validated
example identities. It demonstrates coordinated semantic views and local
workspace interaction without compiling, executing, or persisting canonical
Eqiora content. Diagram positions, camera, selection, and panel arrangement are
UI/workspace state and never acquire mathematical meaning.

The first implementation used a Tauri 2 shell, a semantic-DOM React frontend,
and React Flow for the relation view. Rust remains the only authority for
compilation, validation, graph commit, and execution. The WebView is an
untrusted client behind versioned runtime-validated IPC DTOs and least-
privilege Tauri capabilities.

The native shell was retired before 0.1.0 because its GTK3 dependency closure
could not satisfy the repository security policy. The retained browser surface
is a presentation prototype over fixed preview identities. The native runtime,
IPC, execution, authored-CAD, export, and persistence decisions below remain a
historical design record rather than current capability claims.

## Motivation

A technically correct engineering platform can still be unusable when users
must infer hidden state, diagnose failures from a console, or manipulate a
diagram through precise pointer gestures. Conversely, a visually polished
editor can damage the architecture if its local node objects become a second
model or if canvas order silently defines execution.

Studio must make Eqiora's distinctions visible:

- canonical model meaning versus view/layout state;
- semantic ClockDomain versus execution schedule;
- signal versus conserving connection;
- model revision versus Realization and run evidence;
- requested operation versus accepted atomic transaction; and
- unsupported capability versus failed execution.

The design therefore begins with information and trust boundaries, not a
palette of widgets.

## Proposed design

### Ownership path

```text
canonical revision + typed views
             ↓ projection
      Studio view DTO v5
             ↓ render
  outline / diagram / source / inspector / evidence
             ↓ user intent
      versioned edit or run intent
             ↓ validate + commit in Rust
new canonical revision or structured diagnostics
```

The implementation layers are:

```text
studio/src                 React presentation + pure UI reducer
       │ runtime-validated versioned DTOs
       ▼
studio/src-tauri           least-privilege IPC adapter + session handles
       │ public facade only
       ▼
eqiora::api                shared compile/commit/artifact/run application path
       ▼
Semantic Model / Realization / Artifact / Evidence
```

`eqiora-api` is not Studio-specific. It owns the common application sequence
previously at risk of duplication across bindings: source compile, bounded
transaction wire reconstruction, atomic graph commit, canonical artifact
reconstruction, and owned reference results. Python and Studio both consume
this path through the public `eqiora` facade.

### Canonical and workspace state

Canonical state contains typed Semantic Kernel definitions and graph edges.
Realization and run state contain typed policy, capability decisions, content
identities, and evidence. Studio workspace state contains only presentation:

- node positions, collapsed groups, viewport and camera;
- selection, active view, open panels, and split sizes;
- source-editor cursor/folds and command history; and
- recent local locations and transient import progress.

Workspace state keys canonical nodes by stable ID and records the model digest
it was arranged against. Missing nodes are pruned; new nodes receive a
deterministic initial layout. Layout never changes a model digest and no
backend observes it. The local `eqiora.studio.workspace/v1` envelope is a
separate, bounded schema with exact digest matching and finite coordinates;
unknown versions and malformed storage fail closed. Writes are coalesced while
dragging. It is not inserted into model-envelope v1 and is not yet claimed as
a portable project artifact.

### Editing and concurrency

UI components emit semantic intents such as `set_parameter`, `connect_ports`,
or `move_view_node`. They do not mutate shared model objects. Model intents are
translated to the versioned transaction wire and include their base revision
or digest. A stale base fails with a structured precondition diagnostic; the
frontend does not replay an edit onto a changed model without showing the
conflict.

Preview, validation, atomic commit, revision creation, and undo/redo are
distinct states. Undo creates or selects revision lineage; it does not patch a
Rust object backward in place. View-only intents update the pure frontend
reducer and never invoke the model service.

The first vertical slice compiles a complete source document and visualizes
its accepted projection. The third slice proves one deliberately narrow typed
edit: a finite scalar value on a `Field` or `Parameter`, expressed in coherent
SI units without changing its physical dimension. Preview returns the exact
shared `eqiora.model-transaction-envelope/v1` through `v4` identity selected
by the immutable document and requires both the base graph revision and
previous typed quantity. Commit replays that plan only through the same exact
codec and returns a same-codec child revision;
neither React nor the Tauri adapter can manufacture a replacement projection.

The frontend retains a bounded revision lineage for navigation. Undo and redo
select immutable documents; they do not synthesize inverse operations or
mutate a Rust object backward. Committing after undo creates a branch from the
selected revision and discards the abandoned forward UI path. The source
editor remains visibly labelled as the source basis after a transaction child
is created. Recompiling that text deliberately starts a new source-derived
lineage; the current slice does not pretend to rewrite arbitrary source while
preserving formatting and comments.

A source textarea is not presented as a complete language editor until
incremental syntax/LSP work provides robust spans, recovery, and navigation.

### Execution and data

Run submission is asynchronous relative to the WebView and UI thread. The
Tauri adapter clones immutable inputs and performs Rust work on a blocking
worker. Numerical inner loops never call JavaScript. Completed first-slice
series cross IPC as owned finite arrays. The first two-dimensional scalar
Field uses a separate bounded chunked data-plane contract described below;
wider mesh/Field and shared-buffer transport remain future contracts. CPU/GPU
transfer is never an animation side effect or implicit renderer behavior.

Controlled execution is an application contract below Studio. The semantic
interpreter observes the accepted state after initial consistency and any
zero-time tick, then after each nonterminal accepted continuous, periodic, or
event step. It never observes inside expression evaluation, Newton iteration,
event localization, or an atomic activation commit. An observer returns only
`Continue` or `Cancel`. Cancellation produces a typed terminal boundary with
model time and accepted-step count; it constructs neither a partial trajectory
nor successful-run evidence. The existing uninterrupted API delegates to this
path and is verified to return the same result.

Bridge v5 gives every run a UUID, permits one active run per native Studio
session, and routes cancellation only to the exact active identity. The
adapter owns a Rust atomic cancellation flag and a coalescing observer. The
first accepted boundary is emitted, later presentation progress is emitted at
most once per 100 ms, and cancellation forces a final boundary emission.
Progress crosses a Tauri channel and is advisory: it may be dropped or
coalesced. Completed, cancelled, and diagnostic outcomes return through the
typed command response and are not inferred from channel closure. This is a
cooperative safe-point contract, not solver preemption or a real-time latency
promise.

The second slice resolves one `ReferenceRunPlan` in the shared Rust
application service before submission. The plan owns a versioned exact-float
key, adapter/version, host placement, integration and nonlinear methods,
tolerances, event controls, and safety limits. Studio bridge v5 previews that
plan and must replay its key immediately before execution; stale or forged
keys fail before starting work. A successful result alone carries typed
elapsed/output evidence. Its acceptance is explicitly `SemanticOracle` with
no independent optimized producer, rather than presenting the reference
interpreter as if it had a second verifier. This is reference-run admission,
not yet the general spatial Realization editor or device capability surface.

Bridge v5 retains the typed value-plan preview and exact-key commit alongside
the run-plan path. Native active-lineage retention is bounded to 32 immutable
documents and UI navigation to 24 lineage entries; both prune the abandoned
forward branch after a new commit. Those are resource and UX limits, not
semantic history retention promises. A missing document, target
drift, non-finite scalar, wrong entity kind, changed value/revision, forged
key, or result-lineage mismatch fails closed with structured diagnostics;
`ST0006` is reserved for Studio value-plan replay failure.

Result visualization is a bounded projection. Large series are reduced to a
fixed SVG point budget by preserving the minimum and maximum of deterministic
ordered buckets, including both endpoints. The screen-reader table has its
own fixed row budget and states the represented and total counts. Full owned
result arrays remain data, not DOM or renderer objects.

### Spatial Realization capability slice

Bridge v5 introduces the first general-architecture Realization interaction
through one deliberately narrow executable consumer. A canonical scalar
elliptic model advertises an applicable workflow only after the shared spatial
lowering derives its dimension, `f64` scalar requirement, and replicated
layout. The frontend then collects method, cells per axis, and host workers as
an independent Realization intent. Editing that intent increments a
Realization revision; it never creates a canonical model revision.

The shared `eqiora-api` application service previews before numerical
allocation. It checks the powered cell and field counts against a fixed
250,000-entity boundary, constructs either FEM with continuous
Q1/Gauss-Legendre policy or FVM with cell-constant/centroid policy, resolves
identity-preconditioned reproducible CG and host placement through the typed
Realization capability, and returns a content-addressed plan tied to model digest and
`RealizationEnvelopeV1` identity. The run request carries the exact plan key;
native code reconstructs and compares the plan before mesh allocation.

Host parallelism is captured once per native Studio session from Rust's
available-parallelism estimate, bounded to 64. The wire calls this a
`studio-session-budget`: it is a local admission value, not a physical-core,
affinity, exclusive-capacity, NUMA, or cluster claim. One worker resolves the
serial adapter; a larger admitted request resolves a run-owned Rayon pool.
Adapter and worker count are cross-checked at both protocol boundaries.

The run result remains a control-plane projection. It contains field
location/count/minimum/maximum, assembly packet/target counts, producer and
verifier execution topology, reported and independently recomputed true
residual with its target, continuous boundary/source balance, elapsed time,
and full model/Realization/plan lineage. It contains no mesh coordinates,
connectivity, or Field values. The bounded assembly and solve have no accepted
observer boundary yet; the UI uses an indeterminate native progress element
and does not expose cancellation or a percentage.

### Bounded two-dimensional scalar Field data plane

The accepted `ScalarEllipticRunPlan` also owns an immutable Cartesian Field
projection before numerical allocation: exact Field and Domain IDs, optional
presentation alias, value dimension, coherent-SI bounds, association, logical
shape, and last-axis-fastest order. Completed FEM/FVM values and summary must
match that projection before native publication.

Field transfer is not an automatic consequence of run completion. An explicit
**View field** action opens `eqiora.studio.field-view/v1` with the exact Model
digest, canonical UUID-v4 run identity, and content-addressed plan key. The
native adapter retains at most two complete accepted host arrays and emits raw
little-endian `f64` chunks of exactly 4,096 values except for the final chunk.
The descriptor fixes `uniform-cartesian-2d`,
`row-major-last-axis-fastest`, value count, extrema, and chunk count.

The client rejects descriptor identity/shape drift, missing or reordered
chunks, incorrect byte length, non-finite values, or final count/range drift.
It publishes no renderable value array until the complete transfer is
validated. The resulting Cartesian raster, semantic table, and inspector share
one keyboard/pointer selection. This is a bounded presentation of generated
Cartesian scalar data, not a durable Field artifact, mesh transport,
unstructured visualization, zero-copy path, or general renderer contract.

### Security boundary

The packaged frontend consists only of local assets under a restrictive
content-security policy. Remote navigation and remote script execution are
disabled. The main WebView receives only the Tauri permissions required by
the implemented commands; shell, unrestricted filesystem, arbitrary URL,
clipboard, and process execution capabilities are absent by default.

IPC input is untrusted despite TypeScript types. Rust validates byte/count
limits, numeric finiteness, protocol version, base identity, and operation
shape. The frontend independently validates every response at runtime before
state transition. Tauri or frontend error strings never replace Eqiora's
structured diagnostic codes, source spans, graph paths, and patches.

Filename values in compile requests are diagnostic labels, not implicit file
authority. A later open/save dialog receives a narrowly scoped path capability
and passes bytes to the same document operation.

### UX and accessibility contract

Studio targets WCAG 2.2 AA even as a desktop application. In particular:

- every canvas action has keyboard and simple click/tap alternatives;
- diagram nodes and edges have semantic labels, focus order, visible focus,
  selection announcements, and automatic focus visibility;
- view movement is available from the inspector, and future connect/delete
  operations must be available from an inspector or command palette, never
  only by dragging;
- status is not conveyed by color alone, target sizes and contrast are
  measured, and reduced-motion preferences disable nonessential motion;
- focus is restored after dialogs, errors focus their source span or graph
  path, and no panel creates a keyboard trap;
- resizing and zoom preserve usable source, inspector, and diagnostic views;
- all strings, units, shortcut labels, and accessibility announcements are
  localizable data rather than text embedded in canvas drawing code.

The default workspace favors comprehension over maximum density: model
outline on the left, the selected semantic projection in the center, a typed
inspector on the right, and collapsible diagnostics/results below. The command
palette provides the same operations independent of spatial memory. Empty,
loading, validation, stale, unsupported, running, completed, cancelled, and
failed states each have explicit copy and recovery actions.

Source spans remain kernel-owned UTF-8 byte offsets. Studio converts them to
DOM UTF-16 selection indices in one tested projection function and records the
input source associated with each diagnostic. A span is actionable only while
that source exactly matches the current editor. This prevents an apparently
helpful navigation action from selecting the wrong text after an intervening
edit.

Run fields retain editable text, including temporarily empty input, separately
from validated numeric requests. Only finite, strictly positive ASCII decimal
or scientific values within the bridge step bound can produce a run request.
Both the presentation layer and the bridge validate independently; an invalid
request produces a structured `ST0002` envelope rather than an unhandled
promise rejection. A completed result retains its validated configuration;
semantically equivalent spelling such as `0.1` and `1e-1` remains current,
while a changed numeric value is visibly labelled as preceding evidence.

Run lifecycle state is a discriminated union distinct from retained result
data. Only monotone progress with the exact request and run identities can
advance the visible run. Requesting cancellation immediately changes the
action copy but leaves the accepted-boundary progress visible. A cancelled or
failed successor never replaces the last completed result; that result remains
visible with its original digest/configuration and explicit stale labelling.
The run panel keeps progress and its safe-point action simultaneously visible,
uses a native progress element and polite status region, and preserves a
24-pixel target for the collapsible numerical contract.

### Frontend state and code rules

The application core is a pure reducer over discriminated unions. It holds
domain DTOs separately from ephemeral interaction state. Components receive
narrow selectors and commands rather than one mutable global store. React
Flow node/edge arrays are derived projections; they are not authoritative
model storage, and drag events update only layout state.

Runtime wire schemas are defined once with Zod and inferred into TypeScript.
Generated schemas, protocol adapters, pure state, view derivation, components,
and visual tokens occupy separate modules. Exhaustive switches fail the type
check when a protocol state or kernel kind is added. CSS uses documented
design tokens, native semantic controls, logical properties, visible focus,
and no color-only selectors.

### Closed application registry

The current application shell has one compiled-in registry for exactly three
workflows: Relations, scalar elliptic, and the bounded CAD box. This is a
closed application inventory, not runtime plugin discovery or a second source
of model meaning. Applicability comes only from accepted projections:

- Relations requires an accepted canonical document;
- scalar elliptic requires the workflow advertised by canonical spatial
  lowering; and
- CAD requires an accepted CAD projection whose Model digest exactly matches
  the current document.

The same registry owns command identity, workflow scope, navigation,
workspace selection, shortcut and focus targets, disabled-reason keys,
projection budgets, and required semantic alternatives. Header controls and
the command palette consume the same resolved command availability. A stale or
unsupported CAD projection returns the application to Relations without
changing Model, Realization, run, or persisted workspace identity.

Projection limits reuse the versioned Studio, Field-view, and CAD protocol
constants instead of restating numeric policy in components. The scalar
spatial control projection remains summary-only. Its separate explicit Field
projection declares an owned-host copy, fixed chunk and value budgets, and a
semantic-table alternative rather than bypassing the registry.

Registry-owned labels, descriptions, and remediation reasons are resolved
through a typed presentation catalog. Message keys and fallback English are
application presentation, never canonical identity, a diagnostic code, or an
IPC field. The boundary is intentionally narrow; it does not claim that all
Studio or kernel text is localized.

Secondary example fixtures are command-palette actions rather than permanent
header controls. This keeps the primary document/workspace actions legible at
the supported shell size while retaining a keyboard-searchable path to every
example.

### Initial implementation baselines

Baselines checked against official sources on 2026-07-18:

- Tauri 2.11.5 / Tauri API 2.11.1;
- React and React DOM 19.2.7;
- React Flow 12.11.2;
- TypeScript 7.0.2 and Vite 8.1.5;
- Vitest 4.1.10 and Playwright 1.61.1;
- axe-core/Playwright 4.12.1;
- Biome 2.5.4 and Zod 4.4.3; and
- Node.js 24.18.0 LTS with an exact npm lockfile.

TypeScript 7.0 has no programmatic compiler API. The first Studio package does
not need one: `tsc` performs type checking, Vite handles builds, and Biome
provides formatting/linting without typescript-eslint. A future tool that
requires the compiler API must use the official TypeScript 6 compatibility
package or wait for the TypeScript 7.1 API; it must not reach into private
compiler internals.

These are implementation pins, not Studio protocol promises. Upgrades require
the same type, unit, browser-interaction, accessibility, screenshot, and Tauri
build evidence.

## Alternatives considered

### Slint-native primary UI

Slint provides attractive Rust integration, a declarative language, and a
small native runtime. Its current browser target renders through a canvas and
documents that screen-reader support and ordinary DOM/CSS behavior are absent.
That weakens a browser-reusable, accessible OSS editing surface. Slint remains
a reasonable future embedded/HMI projection, not the primary Studio client.

### Custom egui/wgpu editor

Immediate-mode and native GPU rendering are useful for dense visualization,
but making them the primary editor would require Eqiora to own text, focus,
assistive technology, drag alternatives, and platform behavior before working
on engineering interactions. A dedicated wgpu renderer may later serve mesh
and field views behind the renderer adapter boundary.

### Browser-only service first

A browser deployment is desirable, but requiring a server for the initial
local engineering workflow adds authentication, storage, and remote-execution
semantics before the client protocol is stable. The DOM frontend remains
browser-compatible and can later use a network adapter implementing the same
versioned operations.

### Mirror Rust enums into React state

Rejected. It exposes crate refactors, permits invalid intermediate objects,
and makes React state a second model implementation. Projection DTOs are
purpose-built, versioned, and reconstructed from an accepted model.

### Make diagram order execution order

Rejected. Layout is presentation. When ordering has model meaning, it must be
a typed canonical relation/activation contract and appear consistently in
source, Python, and every view.

## Compatibility and migration

Studio protocol, workspace layout, canonical transaction, model artifact, and
run artifact are separate compatibility surfaces. A Studio release may change
layout storage or component structure without changing model bytes. Unknown
protocol or workspace versions fail closed. Accepted model/artifact v1 bytes
are not reinterpreted to accommodate UI state.

The TypeScript and Tauri packages live outside the core Rust workspace so
platform GUI dependencies do not widen the computational core's default build
or MSRV. Their CI is an additional gate, not a replacement for core gates.

## Verification

The implemented slices prove:

1. source crosses the shared Rust application service and versioned
   transaction wire before an accepted model projection is returned;
2. malformed source returns structured diagnostics and focuses a meaningful
   source location;
3. relation/field/parameter projection is deterministic for one model digest;
4. moving a diagram node changes layout only, not canonical identity;
5. compile and reference run do not block the WebView/UI thread;
6. field-local result series render with their real independent time axes;
7. keyboard and click-only users can select every node and invoke every
   implemented action without dragging;
8. automated accessibility checks find no serious/critical violations in the
   primary states; and
9. formatting, type checks, reducer/protocol/component tests, production
   frontend build, Rust checks, and platform smoke builds pass from lockfiles.
10. an accepted native plan key is replayed before reference execution and a
    mismatched key fails without silently selecting another plan;
11. completed evidence retains producer, placement, numerical controls,
    acceptance meaning, output counts, elapsed time, and model lineage;
12. the command palette exposes compile, accepted run, reflow, and focus
    navigation without canvas location or pointer memory, using a native modal
    focus boundary with explicit disabled reasons;
13. workspace-only positions round-trip through a distinct bounded v1 schema,
    reconcile against the current canonical IDs, and cannot modify model
    bytes; and
14. chart and semantic-table projections remain within fixed DOM/SVG budgets
    while retaining ordered extrema and endpoints;
15. a finite `Field`/`Parameter` scalar preview exposes the exact shared
    transaction identity while retaining dimension, target, base digest,
    revision, and previous-value preconditions;
16. exact-key commit creates an immutable child projection and leaves the base
    document unchanged, while stale, forged, no-op, non-finite, and wrong-kind
    edits fail closed;
17. undo/redo navigates retained revisions, committing after undo creates a
    branch, and selecting another node or editing input suppresses obsolete
    asynchronous preview responses; and
18. source-basis and current canonical-child identities remain visibly
    distinct, and value editing is available from both the inspector and the
    keyboard-searchable command path;
19. controlled completion preserves the uninterrupted reference result, while
    a test observer cancels exactly after accepted step three and receives no
    partial trajectory;
20. Studio accepts monotone progress only for the exact request/run identity,
    routes cancellation only to that active UUID, and rejects another
    concurrent native run;
21. progress remains an advisory coalesced channel while completed and
    cancelled outcomes are typed terminal command responses; and
22. cancelling a successor run preserves the preceding completed evidence,
    exposes the accepted cancellation boundary, admits no partial result, and
    keeps progress and the safe-point action unobstructed;
23. canonical spatial lowering alone determines workflow applicability and
    dimensional requirements, while method/cells/workers remain a separate
    revisioned Realization intent;
24. capability preview bounds powered mesh/field counts before allocation,
    produces one coherent FEM or FVM policy, and rejects worker-budget or
    adapter contradictions;
25. spatial submission replays the exact model, Realization revision, method,
    resolution, workers, and content plan key before execution; and
26. completed spatial evidence matches its previewed Field shape, passes an
    independently recomputed true-residual target and continuous balance, and
    crosses the control IPC without bulk mesh/Field arrays or fabricated
    progress;
27. the closed workflow and command registries contain unique exhaustive
    identities, and applicability depends only on accepted projection state
    plus exact CAD Model identity;
28. navigation, header actions, and the command palette consume one resolved
    availability and disabled reason, while hidden-workflow focus commands
    cannot execute; and
29. every registered workflow declares an existing protocol-owned projection
    budget and keyboard-accessible semantic alternative, and every
    registry-owned message key resolves through the fallback catalog;
30. the application-owned scalar Field projection binds exact
    Model/run/Realization identity, Field/Domain meaning, coherent-SI bounds,
    association, logical shape, and canonical value order before allocation;
31. Field transfer starts only after explicit intent, is bounded to fixed
    little-endian chunks from a two-entry accepted-host cache, and publishes no
    partial array; and
32. foreign identity, descriptor drift, missing/reordered/incorrect/non-finite
    chunks, or final count/range drift fails before rendering, while the
    accessible table, raster, and inspector preserve one selection.

The accessibility gate uses the WCAG 2.0, 2.1, and 2.2 A/AA axe tags and
rejects serious or critical findings in primary, field-validation, and
structured-diagnostic states without exclusions or disabled rules. Separate
Playwright cases verify tab order, visible focus, exact source selection,
non-drag graph movement, minimum-shell containment, and stale-result labelling.
Automated checks are intentionally not claimed as complete WCAG conformance;
manual keyboard and inclusive user testing remain release responsibilities.
The executable commands and falsifying cases are collected in
[`docs/verification/studio-capability-evidence.md`](../docs/verification/studio-capability-evidence.md).
The controlled-run boundary, exact-identity routing, and lifecycle UI have a
separate case in
[`docs/verification/studio-run-lifecycle.md`](../docs/verification/studio-run-lifecycle.md).
The first spatial Realization, local placement, exact replay, and evidence path
is specified independently in
[`docs/verification/studio-realization-capability.md`](../docs/verification/studio-realization-capability.md).

## Nonclaims

The current slices do not claim connection/topology editing, vector/tensor or
array value editing, unit conversion at the UI boundary, portable revision
history, source round-trip rewriting, imported/adaptive/high-order/mixed
spatial Realizations, a general solver or device editor,
hard real-time cancellation latency, solver-inner-loop preemption, pause/resume
or restart from a cancelled boundary, concurrent or remote runs, arbitrary
plugins or remote content, collaborative editing, production source LSP, a
general Simulink-equivalent component palette, model-reference/variant UX,
mesh rendering, unstructured/nonuniform or vector/tensor Field visualization,
production 3D volume visualization, durable Field artifacts, LOD/production
scale, or zero-copy device arrays. Python async/cancellation remains a separate
adapter gate even though
it can reuse the application-level controlled-run contract. Each remaining
surface requires an executable consumer and its own interaction, security,
and verification evidence.
