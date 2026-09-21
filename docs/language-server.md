# Eqiora language server

`eqiora-language-server` is an editor-independent LSP preview backed by
Eqiora's compiler-owned editor analysis service.

Install it from a checkout:

```console
cargo install --locked --path crates/eqiora-language-server
eqiora-language-server --version
```

Configure an LSP client to start `eqiora-language-server` over stdio for `.eqi`
files. For example, Neovim 0.11 can start it from `ftplugin/eqiora.lua`:

```lua
vim.lsp.start({
  name = "eqiora",
  cmd = { "eqiora-language-server" },
  root_dir = vim.fs.root(0, { "eqiora.toml", ".git" }) or vim.fn.getcwd(),
})
```

The preview uses standard UTF-16 LSP positions and full-document synchronization.
Each document has a 16 MiB analysis limit. The server publishes ordered
parser/compiler diagnostics after open and accepted newer changes, clears
diagnostics on close, and serves whole-document formatting, nested document
symbols, folding ranges, Markdown declaration hover, and definition locations.
Resolved references can be found across open modules and exact local packages.
Lifecycle events are emitted as one JSON object per line on stderr, leaving
stdout exclusively for LSP framing.

## Documentation while editing

Hover, completion (`textDocument/completion`) and signature help
(`textDocument/signatureHelp`) share descriptions of the currently supported
scalar mathematics, integer conversion, spatial and time operators. For example,
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

Local completion includes declarations in the cursor's enclosing lexical scope.
Hover also describes local members. Operator, Component and Model signature help
includes argument declarations and their documentation, and named arguments select
the matching formal even when written out of declaration order. Compiler-resolved
imported calls use their exact declaration's signature and documentation. Authored
prose retains the language's bounded Markdown rendering; comments and notation
islands do not trigger code assistance.

This is a bounded editing preview: suggestions are not filtered by inferred argument
type or expected physical dimension. Import-member discovery, automatic imports,
overload selection, and imported-call recovery in an unresolved workspace remain
outside this surface. Completion and signature help are independent of the optional
inspection protocol and are advertised as ordinary LSP capabilities.

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
