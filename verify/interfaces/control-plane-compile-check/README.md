# Compile/check control-plane verification

This case fixes one small application contract across Rust and Python. Browser
Studio checks the current request schema separately without becoming a compile
client. The already-public
`ModelDocument::compile(&str, &str) -> Result<ModelDocument, Vec<Diagnostic>>`
operation is the only owner of compilation meaning. A closed
`eqiora.control/v2` request adapts that operation by selecting the
`model.compile-check/v1` command and supplies only its request identity,
filename, and source. The control adapter invokes the operation once and then
projects its unchanged response plus the optional document from that same
invocation; callers cannot select a Model wire or feature list. Request IDs,
protocol identity, response limits, and overflow substitution remain control
policy and do not enter the operation.

The focused product tests check these boundaries:

- accepted source produces only a typed current-Model identity descriptor,
  linked to the document from that same execution;
- syntactically rejected source produces a kernel diagnostic and no Model;
- retired or unsupported protocol and command identities are rejected by the
  bounded dispatch prelude before DTO admission or compilation;
- retired `modelWire` and `requiredFeatures` members are rejected by the
  closed v2 DTO; and
- request and diagnostic resource bounds fail closed without reflecting
  oversized caller content or publishing partial diagnostics.

Compiling the accepted source twice through control and once through
`ModelDocument::compile` produces equal canonical occurrence identities, artifact
digests and structural fingerprints. Each accepted response is also checked
against the document returned by its own execution.

The Python adapter is a separate consumer of the same operation. After its
own filename/source admission it invokes `ModelDocument::compile` directly on
a detached native path, without constructing a control request or importing a
control DTO. Independently accepted Python, control-v2, and direct
compilations of the same source have equal canonical Model IDs, artifact digests
and current-generation structural fingerprints. Ordinary rejected compilation is
normalized across all three paths; control-envelope overflow behavior is not
a cross-adapter claim.

The product tests compare returned Model schema identities with the current
public control schema. They do not pin whole request files, source hashes, or
copies of retired schemas.

Client adapters may choose native function calls or Python objects, but they
consume this meaning rather than recreating it. Responses
must not contain source, meshes, Fields, or trajectories. Preview, execution,
cancellation, artifact inspection/diff, remote transport, and bulk data remain
separate slices defined by
[RFC 0054](../../../rfcs/0054-curated-facade-and-control-plane.md).
No CLI, MCP, HTTP, stdio, remote execution, preview, run, cancellation, or
data-plane operation is added by this evidence. This case makes no Studio
capability or workflow claim.

Run:

```bash
cargo test --locked -p eqiora --test control_plane_compile_check
cargo run --locked -p eqiora-verify -- run --case interfaces.control-plane-compile-check
```
