# Eqiora language server

`eqiora-language-server` is an editor-independent LSP preview backed by
Eqiora's compiler-owned editor analysis service.

For released binaries and tested Vim/Neovim configuration, follow the
[editor setup guide](site/src/content/docs/guides/editors.mdx). The same guide is
[published on the documentation site](https://eqiora.org/guides/editors/).

To build the development server from a checkout:

```console
cargo install --locked --path crates/eqiora-language-server
eqiora-language-server --version
```

The preview uses standard UTF-16 LSP positions and full-document synchronization.
Each document has a 16 MiB analysis limit. The server publishes ordered
parser/compiler diagnostics after open and accepted newer changes, clears
diagnostics on close, and serves whole-document formatting, nested document
symbols, folding ranges, Markdown declaration hover, and definition locations.
Resolved references can be found across open modules and exact local packages.
Go to Definition also resolves owned Model/Component Field/Parameter/Port/Clock value references and
public Ports of Model direct children to exact declaration names, including imported
sources. Find References accepts their terminal value-reference tokens or exact
declaration names and returns source-qualified identifier spans in file/offset
order, optionally including the declaration once. An admitted unused declaration
returns an empty list. Successfully prepared owned Component declarations are
supported without a Model instance, including private body locals. Multiple spellings such
as `a.p` and `b.p` can refer to the same source declaration across Models and files;
these results describe declaration provenance, not physical occurrence identity.
Prepared signature Field requirements also retain their own declaration and value
references, without projecting occurrence-specific types or supports from bindings.
Caller binding labels are not value references.
Nested binder scopes and families, private or deeper child members, record members and alias declaration targets remain unsupported. Qualifiers, units,
comments and notation contents are not value references. Invalid or recovering
snapshots grant no navigation; unsaved versions remain authoritative. The bounded
reference results do not support rename.
Lifecycle events are emitted as one JSON object per line on stderr, leaving
stdout exclusively for LSP framing.

## Documentation while editing

Hover, completion (`textDocument/completion`) and signature help
(`textDocument/signatureHelp`) share descriptions of the currently supported
scalar mathematics, integer conversion, spatial and time operators. Hover and
completion also explain common language constructs (`model`, `component`,
`relation`, `parameter`, `state`, `initial`, `import`, conditionals and connectors)
and mathematical types (`integer`, `bool`, `complex`, `array`, `vector`, `tensor`
and the operator-local generic classes). Construct help includes syntax and
meaning: a Relation is simultaneous equations; an array axis is not a spatial axis.
These are documented keyword/type candidates, not callable signatures or a claim
that every specialized grammar child has help. For example,
`math.sqrt` explains its dimension rule, value domain and derivative restriction.
Completion after `math.` replaces the complete qualified name, and call help
tracks the active argument through nested calls and array expressions, including
unfinished calls. Constants such as `math.pi` have documentation but no call signature.

Authored declarations use the existing `///` comments; no new language syntax is
needed. Document each signature entry immediately before its declaration:

```eqiora
/// Scale a dimensionless value.
operator scale(
  /// Value to scale.
  input x: 1,
  /// Multiplicative factor.
  input factor: 1
): 1 = x * factor;
```

The native editor API supplies documented `EditorSymbol` values for all three
features; the language server projects them into LSP responses.

Declaration hover displays admitted `@{...}` notation through the existing plain
label renderer, with explicit script delimiters and no TeX execution.
`EditorDefinition::notation()` retains the compiler-selected top-level declaration's
validated notation and source range, including imported declarations.
`EditorSymbol::notation()` also retains notation on exact authored declarations,
including local fields, parameters and public interface members. Authored hover
shows the label at the current declaration-name token, or at the terminal name
of a compiler-recorded value reference whose declaration file and range match.
Keyword, unit and qualifier tokens cannot borrow a supported declaration's
hover. Recovery can retain
notation on a declaration without granting typed facts; nested binder scopes,
recovering references and unsupported declaration scopes omit the label.
Missing or rejected notation never supplies another declaration's or snapshot's
label. Occurrence qualification, inferred styling and activation are not added by
this notation projection.

For prepared owned Model/Component Fields, Parameters and Ports, and Model public
child Ports, hover adds known
scalar domain, physical dimension, shape/frame, outer channel-array rank, field
or Port role, activation and spatial support to the authored declaration. These facts come from the same
compiler scope used by completion and are attached only at the exact declaration
name or terminal value-reference token, with a matching declaration file and range.
Named activation retains an unknown occurrence
identity; spatial support identity is definition-local. Aliases,
arbitrary expression inference and nested binder scopes retain authored detail
without claiming inferred types. Hover queries perform no elaboration or solve.
Preparation resolves Model static Parameter defaults and Let expressions through
the existing compiler owner, so a declared `array<1,n>` with `n=3` retains
shape `[3]` and channel-array rank `1` across hover, outline and supported
completion/navigation. Required free values and borrowed clocks remain symbolic;
this does not grant aliases declaration-navigation targets or infer missing extents.
If static resolution fails during editing, completion retains its existing
literal-type advice without using a partially resolved static map.
Component preparation instead requires successful unspecialized support, symbolic
Parameter/property/Let, Field-interface and declaration-scope preparation. It reuses
ordinary compiler binding without child expansion or instance specialization; failed
preparation publishes no Component type or declaration target. Required symbolic
scalar Parameters can retain their type, but unknown extents, frames, supports and
clock identities are not borrowed from an instantiated Component.

Owned Model/Component periodic Clock hover adds the exact reduced period and phase in
coherent seconds from the existing compiler time conversion. For example,
`periodic(100[ms], phase=50[ms])` displays period `1/10 s` and phase `1/20 s`.
The same facts appear at value uses such as `period(tick)` and at exact authored
activation names on Fields, signal Ports, Relations and `let` assertions. The
ordinary parser retains these token ranges on the declaration AST; source-free
factory nodes have no activation-name range. Activation meaning and equality remain
independent of this location metadata.
Equal schedules remain distinct declarations. This does not assign occurrence
identity or infer borrowed or child Clock schedules; Event
metadata remains outside this slice. Clock definition/reference navigation uses
the exact owned Model/Component declaration or prepared signature Clock requirement
for retained value occurrences such as `period(tick)` and these activation names.
A required Clock has a source declaration even when its schedule is unknown.
Caller binding labels are not value uses; instance schedules are never projected
onto the requirement. An unused admitted declaration differs from an unsupported query.

Document-symbol details reuse the same prepared compiler facts for exact owned
Model/Component Field, Parameter and Port declarations and owned periodic Clocks, including channel-array
rank and exact local schedules. Unsupported
declarations keep their lexical kind. Invalid or incomplete workspace snapshots
clear earlier typed details while retaining the recovered outline. Cancelled and
stale preparation cannot publish those facts. Nested-binder inference, borrowed
Fields and exact named activation identities remain outside
this projection.

Hover in a validated workspace includes the declaration's origin. Canonical
hover shows its complete compilation namespace and source label; supported
authored declaration/value hover shows the namespace, canonical module and source
path. Identical exports from different exact packages or modules remain
distinguishable after import aliases change. Namespace segments retain their
boundaries as escaped literals; the editor does not interpret opaque segments as
version or digest fields. Invalid/recovering workspaces and contextless document
assistance omit exact origin. All origin text stays inside a source-safe code fence.

Local completion includes declarations in the cursor's enclosing lexical scope,
including unfinished Model/Component bodies. Declaration, type and expression
positions select different candidate families; sibling locals are never offered.
Hover also describes local members. Operator, Component and Model signature help
includes argument declarations and their documentation, and named arguments select
the matching formal even when written out of declaration order. Imported calls use the current graph's public declaration signatures and
documentation, including recovered unsaved source. Authored
prose retains the language's bounded Markdown rendering; comments and notation
islands do not trigger code assistance.

Import completion walks canonical module segments from the supplied workspace and
its exact direct dependencies. An imported alias exposes public declarations;
instance and connector completion exposes declared interface members, not private
bodies. Named-argument completion includes remaining required/defaulted bindings,
their types and documentation. It inserts `name = `, or only the name when an equals
sign already follows, and ignores commas in nested expressions. Unknown targets
do not fabricate signatures.

Invalid unsaved package source can recover names from the admitted disk graph with
the current text overlaid. This recovery does not publish resolved references or
a compilable Model. Completion performs no fetch, install, lock or store writes;
unavailable dependencies are not invented.

For simple name/path references in Model or owned Component Parameter initializers,
Model instance named bindings and scalar connection endpoints, a prepared compiler scope
ranks compatible candidates before unknown and incompatible candidates. Details
explain the available type, physical dimension, shape, nominal identity or
endpoint role using the compiler's existing rules. Equal dimensions do not make
distinct nominal types or physical Connectors interchangeable. Completion inserts
names only; it does not insert conversions or physical adapters.

This initial ranking covers continuous non-spatial signal connections and scalar
physical endpoints. Clock/support identities, dependent types that cannot be
resolved, arithmetic operands, unsupported Component contexts and incomplete analysis retain
ordinary name completion without claiming compatibility. General Component field
suggestions keep authored source detail; only supported Parameter defaults gain
contextual type ranking. Ranking is advisory and
does not establish the validity of every connection in a Model. The immutable
workspace analysis prepares the contracts with cancellation checkpoints; each
completion query performs no parsing, elaboration, network access or solve.

Automatic imports, overload selection and member inference through arbitrary
expressions remain outside this surface. Completion and signature help are
independent of the optional inspection protocol and are advertised as ordinary
LSP capabilities.

Files opened under the same initialization workspace folder are analyzed as one
module graph. When that folder contains `eqiora.toml`, the server reads its explicit
local candidate sources, including unopened files, without writing a lock or store.
An existing lock fixes the selected versions and authored requests during analysis;
changing a request requires an explicit project update. Open model sources override their disk content until they are
closed, so hover and definition navigation stay current after full-document
changes. Workspace analysis runs on one background worker, coalesces pending
edits, and prevents superseded results from publishing diagnostics. An editor
request waiting for the current snapshot can be cancelled through
`$/cancelRequest`. Partial edits are planned next.

Watch notifications for Eqiora sources, `eqiora.toml` and `eqiora.lock`, or saving an
open Eqiora document, retry package discovery within the already initialized root.
A manifest missing or invalid at startup can recover after correction. Native
package admission still owns the source-path map; current open buffers are applied
only to admitted paths before a snapshot publishes. Failed admission keeps current
open-source assistance without canonical package facts. Events outside initialized
roots do not discover projects; changing the workspace roots requires a restart.

## Rich model inspection

The [official VS Code extension](https://github.com/nkiyohara/eqiora-vscode) consumes
`capabilities.experimental.eqioraInspection = 1`. Other clients can send the same
`eqiora/inspect` request:

```json
{"textDocument":{"uri":"file:///workspace/main.eqi"},"model":"Decay","fingerprint":false}
```

`model` is optional and defaults to the first Model declared in the selected file.
The read-only response contains the exact open-document `version`, declared
`models`, selected `model`, compiled `nodes` and `edges`, and rendered `equations`.
Each equation retains generated `latex`, `plain`, `speech`, an explicit `fallback`
flag, and referenced entity IDs. Node locations are standard LSP locations. Clients
must discard responses for older document versions and render fallback text as text.
No source text is interpreted as executable TeX or HTML.

Compilation uses the retained resolved editor graph, including unsaved sources and
exact local-package dependencies. Required unbound model parameters and invalid
models return `errors` without a partial model. Limits are 4096 entities, 16384
edges and 4 MiB per response. Graph kind strings are presentation labels, not an
alternate semantic vocabulary or editable artifact.

`fingerprint: true` requests Eqiora's existing bounded structural semantic
fingerprint. Unsupported vocabulary or exhausted comparison limits leave
`fingerprint` null with an error. Equal rendered equations do not substitute for
that comparison. Inspection neither constructs a numerical Plan nor runs a solve.

### Validated numerical Plan inspection

Servers advertising `capabilities.experimental.eqioraPlanInspection = 1` accept
an optional `plan` string on `eqiora/inspect`: the exact canonical UTF-8 contents
of a `.eqplan` artifact written through the ordinary Python Plan API. The limit
is 2 MiB. Do not parse and reserialize the artifact before sending it.

The existing native Plan decoder validates schema, identity, provider versions
and numerical admission without running a solve. Invalid, noncanonical or
unsupported artifacts reject the request. The response's `plan` projection
includes its identity, Model revision/digest, Geometry and Mesh digests, backend,
and canonical metadata with embedded binary roots omitted. `matchesSelectedModel`
compares the artifact's exact Model digest to `selectedModelDigest`; a false value
means the validated artifact belongs to a different Model and must be displayed
as such. This comparison is independent of the structural fingerprint.

Clients must clear stale projections on edits and revalidate against the new
selected Model. A server without this capability cannot validate an attachment.
The first adapter uses the same Faer and Diffsol providers as Python, so other
provider identities or versions may be rejected. Complex, spectral and modal
Result projections remain unavailable; clients must not infer phase, power or
probability from this Plan metadata.
