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
Go to Definition also resolves simple value references to fields and parameters
declared in the same Model and file, using the compiler's prepared scope and
exact declaration name ranges. Unsaved document versions are respected. Invalid
snapshots, nested binder scopes, Component locals, aliases and qualified members
do not gain local navigation. Find References accepts either a same-Model field
or parameter declaration or one of its supported value references, optionally
including the declaration. It returns exact identifier ranges in source order,
including an empty list for an unused declaration. Its results are limited to
simple value references outside nested binder scopes: qualified names, Component
locals, aliases and cross-file local references remain unsupported. These bounded
results do not support rename.
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

For prepared Model fields, parameters and public child Ports, hover adds known
scalar domain, physical dimension, shape/frame, field or Port role, activation
and spatial support to the authored declaration. These facts come from the same
compiler scope used by completion and are attached only when the resolved
declaration's file and range match. Named activation retains an unknown occurrence
identity; spatial support identity is definition-local. Component bodies, aliases,
arbitrary expression inference and nested binder scopes retain authored detail
without claiming inferred types. Hover queries perform no elaboration or solve.

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

For simple name/path references in Model parameter initializers, Component named
parameter bindings and scalar connection endpoints, a prepared compiler scope
ranks compatible candidates before unknown and incompatible candidates. Details
explain the available type, physical dimension, shape, nominal identity or
endpoint role using the compiler's existing rules. Equal dimensions do not make
distinct nominal types or physical Connectors interchangeable. Completion inserts
names only; it does not insert conversions or physical adapters.

This initial ranking covers continuous non-spatial signal connections and scalar
physical endpoints. Clock/support identities, dependent types that cannot be
resolved, arithmetic operands, Component bodies and incomplete analysis retain
ordinary name completion without claiming compatibility. Ranking is advisory and
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
