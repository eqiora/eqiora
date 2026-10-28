# Bounded local CLI compile/check

These product tests exercise one host-Unix application boundary: the built `eqiora`
binary accepts `eqiora check <MODEL_PATH>`, selects one bounded UTF-8 regular-
file input, invokes the existing transport-neutral compile operation once,
and emits either the returned document's structural fingerprint or
independently normalized ordinary diagnostics.

The integration oracle freezes the two help snapshots, every exit and fixed
message, path and source bounds, direct/symlink selected-input behavior,
payload containment, exact diagnostic escaping, and atomic overflow
substitution. A private injected `FnOnce` witness observes zero operation
executions before admission and exactly one after admission while preserving
the exact filename, source, and returned result. Two isolated subprocess
witnesses panic before the operation and during accepted projection and
require the single fixed internal-failure result with no partial payload.

The oracle also injects two distinct accepted return documents and one rich
rejection, contains an argument-supplier panic, and exercises the production
reader through a deterministic abstract `Read` witness. Exact maximum input
reaches the operation, reported oversize touches no reader, a growing stream
is capped at 8,388,609 observed bytes, and read failure maps to the frozen
availability result. No open count, kernel object identity, stable-path
snapshot, allocator behavior, or worklist lifetime is claimed.

A compact source check names the three compile/check owner files, the private
command-route entry, the sole accepted compile authority with its exact
function-pointer signature, forbidden cross-layer authorities, and absence of
an externally public item. Execution cardinality is not inferred from source.
Other command adapters, including stationary studies, are outside this case;
private directory membership is not part of its claim.
The selected-input regression checks distinct rate-2 selected and rate-3 decoy
files through the built executable and rejects substitution through the private
operation seam. The existing injected-call tests cover operation cardinality
without rewriting and rebuilding production source.

The suite uses Cargo's already-built executable. It does not rebuild or install
a second copy, require a clean Git checkout, or validate Cargo installation.
The repository architecture, facade, and layer checks remain separate.

This case does not claim compilation artifacts, execution, solving, packages,
published distribution, stdin, multiple files, JSON output, remote operation,
Windows behavior, hostile filesystem-race consistency or snapshot identity,
detached operation work, multiple user processes, Python, Studio, optional
backends, performance, rendering, or scientific correctness. Help and the
accepted fingerprint are public command output; Model occurrence IDs,
artifact bytes and digests, raw source, OS errors, panic data, ephemeral
counter bytes, and adapter internals are not.
