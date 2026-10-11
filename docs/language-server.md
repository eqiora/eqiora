# Eqiora language server

`eqiora-language-server` is a preview LSP server that works with any editor. It uses the same
editor analysis as the Eqiora compiler.

For released binaries and tested Vim/Neovim configurations, follow the
[editor setup guide](site/src/content/docs/guides/editors.mdx), which is also
[published on the documentation site](https://eqiora.org/guides/editors/).

To build the development server from a checkout:

```console
cargo install --locked --path crates/eqiora-language-server
eqiora-language-server --version
```

Lifecycle events are written to stderr as one JSON object per line; stdout carries only LSP
messages.

## Document synchronization

The server uses standard UTF-16 LSP positions and incremental synchronization. All edits in one
notification, whether ordered range edits or whole-document replacements, are applied together
to the current unsaved text, each range against the text left by the edits before it. The
deprecated `rangeLength` field is ignored, and columns past the end of a line clamp to the line
end. An edit with a reversed range, a nonexistent line, a position inside a surrogate pair, or a
stale version leaves the text and version unchanged. Parsing and analysis always rebuild from
the final full text.

A repeated `didOpen` for a URI that is already open keeps its current text and version.
`didClose` followed by `didOpen` starts a new document lifetime, whose version numbers may start
again. Each document is limited to 16 MiB.

## Diagnostics, formatting, and outline

Parser and compiler diagnostics are published in order when a document opens and after each
accepted change, and cleared when it closes. The server also provides whole-document
formatting, folding ranges, and document symbols.

Clients that support hierarchical document symbols receive a nested outline with selection
ranges and details; other clients receive a flat list with source locations and container
names. Without a declared symbol-kind value set, symbols use the original LSP kind range;
declaring a value set enables the extended kinds. In the nested outline each item spans its
whole declaration, and selecting it highlights the declaration's name. Names are located from
the parser's header tokens, including keywords used as names, notation, and comments, so they
survive error recovery, but a recovered position never enables navigation. Import aliases and
other entries without a locatable name select their full range.

The outline also lists Events, `indexset` and `observable` declarations, Component `let`
declarations, and Model Relation families, each with its documentation text. For owned Model
and Component Fields, Parameters, and Ports, and for owned periodic Clocks, symbol details show
the compiler's type information, including channel-array rank and local schedules; other
declarations show only their kind. When the workspace is invalid or incomplete, typed details
are cleared and the recovered outline remains. Cancelled or outdated analysis never publishes
them.

## Hover, completion, and signature help

Hover, completion (`textDocument/completion`), and signature help (`textDocument/signatureHelp`)
share documentation for the built-in scalar mathematics, integer conversion, and spatial and
time operators. For example, `math.sqrt` explains its dimension rule, value domain, and
derivative restriction; constants such as `math.pi` are documented but have no call signature.
Hover and completion also explain common constructs (`model`, `component`, `relation`,
`parameter`, `state`, `initial`, `import`, conditionals, and connectors) and types (`integer`,
`bool`, `complex`, `array`, `vector`, `tensor`, and the operator-local generic classes), with
their syntax and meaning: for instance, that a Relation is a set of simultaneous equations and
that an array axis is not a spatial axis. Not every grammar construct has help. Completion after
`math.` replaces the whole qualified name, and signature help tracks the active argument through
nested calls and array expressions, including calls that are not finished.

Document your own declarations with `///` comments; no new syntax is needed. Put each comment
immediately before the declaration or signature entry it describes:

```eqiora
/// Scale a dimensionless value.
operator scale(
  /// Value to scale.
  input x: 1,
  /// Multiplicative factor.
  input factor: 1
): 1 = x * factor;
```

The native editor API returns documented `EditorSymbol` values for all three features, and the
language server converts them into LSP responses.

**Types in hover.** For owned Model and Component Fields, Parameters, and Ports, and for public
Ports of a Model's direct children, hover adds the scalar domain, physical dimension, shape and
frame, channel-array rank, Field or Port role, activation, and spatial support. This information
comes from the same compiler scope as completion and appears only on the declaration name or on
the last name of a value reference that resolves to it. Named activations do not identify a
particular occurrence, and spatial support identities are local to the definition. Model static
Parameter defaults and `let` expressions are resolved, so `array<1,n>` with `n=3` shows shape
`[3]` and rank 1 in hover, the outline, completion, and navigation. Required free values and
borrowed clocks stay symbolic, and missing extents are never guessed. If static resolution fails
while you type, completion falls back to its literal-type advice. A Component is typed only if
its unspecialized declaration can be prepared without instantiating it; otherwise it gets no
type or definition target. Extents, frames, supports, and clock identities are never taken from
an instance. Aliases, arbitrary expressions, and nested binder scopes show only their source
text. Hover never elaborates or solves the model.

For owned periodic Clocks, hover shows the reduced period and phase in seconds:
`periodic(100[ms], phase=50[ms])` displays period `1/10 s` and phase `1/20 s`. The same
information appears at uses such as `period(tick)` and at activation names on Fields, signal
Ports, Relations, and `let` assertions. Clocks with equal schedules are still distinct
declarations. Schedules of borrowed or child Clocks are not inferred, and Event guard types and
event times are not computed.

**Origin.** In a valid workspace, hover shows where a declaration comes from: its compilation
namespace and source label, or, for authored declarations and values, the namespace, module,
and source path. Identical exports from different packages or modules therefore stay
distinguishable when import aliases change. Namespace segments are shown as escaped literals and
are not interpreted as versions or digests. Invalid or recovering workspaces omit the origin.
Origin text is always shown in a code block.

**Notation.** Hover displays a declaration's `@{...}` notation with the plain label renderer,
using explicit script delimiters and never executing TeX. `EditorDefinition::notation()` returns
the validated notation and source range of the top-level declaration the compiler selected,
including imported declarations, and `EditorSymbol::notation()` does the same for authored
declarations, including local fields, parameters, and public interface members. The label
appears on the declaration name, or on the last name of a value reference whose declaration
matches; keywords, units, and qualifiers never pick up a declaration's label. A recovered
declaration can keep its notation without any typed information, while nested binder scopes,
recovering references, and unsupported scopes show no label. A missing or rejected notation
never falls back to another declaration's label.

**Local completion.** Completion offers the declarations in the enclosing scope, including in
unfinished Model and Component bodies. Declaration, type, and expression positions get different
candidates, and locals from sibling scopes are never offered. Hover also describes local
members. Signature help for operators, Components, and Models lists the arguments with their
documentation, and a named argument selects the matching parameter even when written out of
order. Calls to imported declarations use their public signatures and documentation, including
unsaved source. Documentation is rendered as the language's limited Markdown; comments and
notation never trigger code assistance.

**Imports and named arguments.** Import completion walks module names in the workspace and its
direct dependencies. An import alias exposes public declarations, and instances and connectors
expose their declared interface members, not their private bodies. Named-argument completion
lists the remaining required and defaulted parameters with their types and documentation; it
inserts `name = `, or just the name if `=` already follows, and ignores commas in nested
expressions. Unknown call targets get no invented signature. When an unsaved package file is
invalid, names are recovered from the valid version on disk with the current text overlaid, but
no resolved references or compilable Model are published. Completion never fetches, installs,
locks, or writes to the package store, and never invents unavailable dependencies.

**Type-aware ranking.** For simple names in Model and owned Component Parameter initializers,
Model instance named bindings, and scalar connection endpoints, compatible candidates are listed
before unknown and incompatible ones. Their details explain the type, physical dimension, shape,
nominal identity, or endpoint role using the compiler's rules; equal dimensions do not make
different nominal types or physical connectors interchangeable. Ranking covers continuous
non-spatial signal connections and scalar physical endpoints. Elsewhere, including clock and
support identities, unresolved dependent types, arithmetic operands, unsupported Component
contexts, and incomplete analysis, completion offers plain names. Completion inserts only names,
never conversions or adapters, and ranking is a hint: it does not check that every connection in
the Model is valid. Contracts are prepared once per workspace analysis, with cancellation
checks, so a completion request does no parsing, elaboration, network access, or solving.
Automatic imports, overload selection, and member inference through arbitrary expressions are
not supported. Completion and signature help are ordinary LSP capabilities and do not depend on
the inspection protocol described below.

## Definitions, references, and highlights

Go to Definition resolves references across open modules and local packages. It works from
references to owned Model and Component Fields, Parameters, Ports, Clocks, and Events, and from
public Ports of a Model's direct children, and it lands on the declaration name, including in
imported sources. Clock navigation also covers prepared signature Clock requirements, uses such
as `period(tick)`, and activation names; a required Clock has a declaration even when its
schedule is unknown, and instance schedules are never attached to it. Prepared signature Field
requirements have their own declarations and references, without types or supports taken from
bindings. Components that prepare successfully are supported without a Model instance,
including private body locals.

Find References accepts the last name of a value reference or a declaration name and returns
identifier ranges in file and offset order, optionally including the declaration once. An
unused declaration returns an empty list, which is different from an unsupported query. Because
`a.p` and `b.p` can refer to the same declaration across Models and files, results identify
declarations rather than physical instances, and rename is not supported. Caller binding labels,
qualifiers, units, comments, and notation are not value references. Nested binder scopes and
families, private or deeper child members, record members, and alias targets are not supported.
Invalid or recovering snapshots give no navigation, and unsaved text always takes precedence.

`textDocument/documentHighlight` uses the same resolved references, including the declaration
when it is in the current file, and returns ranges from that file only, ordered and
deduplicated. Qualified type names keep their complete reference range. The highlight kind is
always `Text`, because equations have no read/write or assignment direction. Comments, units,
and unrelated declarations with the same spelling are not highlighted, and unsupported queries
or invalid analysis return nothing. Highlights follow the same rules as Find References for
unsaved versions, waiting for the workspace, and cancellation.

## Workspaces and packages

Files open under the same client-declared workspace folder are analyzed as one module graph. If
the folder contains `eqiora.toml`, the server also reads the project's local sources, including
files that are not open, without writing a lock or package store. An existing lock fixes the
selected versions and requests during analysis; changing a request needs an explicit project
update. Open documents take precedence over their files on disk until they are closed, so hover
and navigation stay current as you edit. Analysis runs on one background worker that coalesces
pending edits, and outdated results never publish diagnostics. A request waiting for the current
snapshot can be cancelled with `$/cancelRequest`.

File-watch notifications for Eqiora sources, `eqiora.toml`, and `eqiora.lock`, or saving an open
Eqiora document, rerun package discovery within the current root, so a manifest that was missing
or invalid at startup is picked up once it is fixed. The source-path map still comes from the
native package checks, and open buffers are applied only to paths that pass them; if the checks
fail, open documents keep their assistance without package information. File events outside the
current roots are ignored.

Standard `workspace/didChangeWorkspaceFolders` notifications add or remove roots without a
restart. Earlier analyses are cancelled and all open documents are regrouped, keeping their
unsaved text and versions. Removed roots lose their package maps, and new roots go through the
normal package checks. A change that leaves the folder set the same does nothing. Clients that
opt into file-watch registration receive it when the first folder becomes available, even if
the session started without folders. Clients that do not report folder changes need a server
restart when their roots change.

## Rich model inspection

The [official VS Code extension](https://github.com/nkiyohara/eqiora-vscode) uses
`capabilities.experimental.eqioraInspection = 1`. Other clients can send the same
`eqiora/inspect` request:

```json
{"textDocument":{"uri":"file:///workspace/main.eqi"},"model":"Decay","fingerprint":false}
```

`model` is optional and defaults to the first Model declared in the file. The read-only
response contains the open document's `version`, the declared `models`, the selected `model`,
the compiled `nodes` and `edges`, and the rendered `equations`. Each equation has generated
`latex`, `plain`, and `speech` forms, a `fallback` flag, and the IDs of the entities it
references. Node locations are standard LSP locations. Clients must discard responses for older
document versions and show fallback text as plain text; no source text is ever executed as TeX
or HTML.

Compilation uses the editor's resolved graph, including unsaved sources and local package
dependencies. If a required model parameter is unbound or the model is invalid, the response
contains `errors` and no partial model. Responses are limited to 4096 entities, 16384 edges, and
4 MiB. Graph kind strings are display labels, not a separate semantic vocabulary or an editable
format.

`fingerprint: true` requests the structural semantic fingerprint. If the model uses unsupported
vocabulary or exceeds the comparison limits, `fingerprint` is null and an error is returned.
Equal rendered equations are no substitute for this comparison. Inspection never builds a
numerical Plan or runs a solve.

### Validated numerical Plan inspection

Servers advertising `capabilities.experimental.eqioraPlanInspection = 1` accept an optional
`plan` string on `eqiora/inspect`: the exact UTF-8 contents of a `.eqplan` artifact written by
the Python Plan API, up to 2 MiB. Send the file as it is; do not parse and reserialize it.

The native Plan decoder validates the schema, identity, provider versions, and numerical
settings without running a solve, and rejects invalid, noncanonical, or unsupported artifacts.
The response's `plan` field contains the Plan identity, Model revision and digest, Geometry and
Mesh digests, backend, and metadata, without the embedded binary data. `matchesSelectedModel`
compares the artifact's Model digest with `selectedModelDigest`; if it is false, the Plan
belongs to a different Model and must be shown as such. This check is separate from the
structural fingerprint.

Clients must clear stale Plan information after edits and validate again against the newly
selected Model. A server without this capability cannot validate an attached Plan. The adapter
uses the same Faer and Diffsol providers as Python, so Plans from other providers or provider
versions may be rejected. Complex, spectral, and modal Result projections are not available;
do not infer phase, power, or probability from Plan metadata.
