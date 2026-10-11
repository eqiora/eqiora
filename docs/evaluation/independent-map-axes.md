# Independent evaluation axes

An evaluation map runs one differentiable program at many input points, independently of one
another. `EvaluationMapPlan` executes complete input points through the existing Q1/TPFA
differentiable programs, and `EvaluationMapProducts` forms JVP and VJP products over nested
grids of points and seeds. Both use the same private indexing code as the arithmetic reference,
[`axes.rs`](../../crates/eqiora-api/src/evaluation_map/axes.rs). The map only arranges and
indexes dense buffers; it has no evaluator, callback registry, or solver of its own.

## What a map reuses

Every member of a map uses the same program signature.
[`DifferentiableProgramIdentity`](../../crates/eqiora-api/src/differentiation.rs) fixes the
Model, Plan, ordered Parameter inputs, selected output Field, flat dimensions, scalar precision,
device, derivative contract, and solver. `DifferentiableParameterPoint` holds one complete,
ordered input point, and an accepted `DifferentiableEvaluation` holds its primal solution and
linearization. A map cannot create, replace, or infer an accepted evaluation.

The map does not copy metadata. Dimensions, real or complex domain, component shape, and frame
come from `ParameterDef`/`FieldDef` and `ValueType`; support, activation, and Representation stay
on the semantic graph; the Plan keeps the Mesh, spaces, placement, and numerical layout; output
Field and coefficient associations stay with the Result, as does any basis identity. If metadata
is missing or the signature is unsupported, the map is rejected; matching flat lengths are not
enough. The differentiable program currently supports only real binary64 scalar Q1/TPFA output
on the host CPU. The indexing reference can carry a complex `ValueType` but performs no complex
evaluation.

A batch axis is a collection axis, separate from the mathematical type. `ValueType::array`
describes a component array and cannot represent an empty batch. Batch axes never change field
support, clocks, physical interactions, or component bases.

## Axes and composition

At each map level, every input leaf is either mapped along one axis or shared. All mapped axes
at a level have the same extent; singletons are not broadcast and nothing is truncated. A shared
value is used unchanged at that level. If every input is shared, the extent must be given
explicitly, and an explicit extent must agree with any mapped inputs.

Axis positions refer to the complete dense buffer. Removing the map axes while keeping the
order of the others must give back the unbatched leaf shape exactly. For example, two channels
mapped at position 1 have buffer shape `[2, N]`, and the two channels are still components of
one member. One physical axis cannot serve two map levels.

Outputs have a fixed, ordered leaf structure, supplied by the signature before execution. Each
output states where every map axis is inserted into its unbatched shape, and an output never
drops a map axis because its values happen to agree. In this reference, buffers are contiguous
row-major binary64; inserting an axis changes the indexing, not what the elements mean. Other
layouts must be supported explicitly by the execution layer.

Mapped inputs at the same level are zipped. A Cartesian product is written as a nested map: at
the outer level map `a` and share `b`; at the inner level share `a` and map `b`. Coordinates run
from outermost to innermost, and the last one varies fastest. Two-level buffers name both axis
positions in the complete shape, so removing an outer axis cannot shift the meaning of an inner
position. Nesting adds no communication, collectives, recurrence, or interaction between
members.

## Member identity

A member is identified by its position in the request, not by its numerical point. Duplicates,
signed-zero bits, and the requested order are all preserved. Permuting the inputs and their
sample IDs permutes the outputs and their associations in the same way. The map never sorts,
deduplicates, or requires a default anchor point.

An optional stable sample ID selects a stochastic path, and only through a supported sampler.
Repeating an ID requests the same path on purpose; independent replicates need distinct IDs.
A sample ID is not the member's position, its input point, or an evaluation identity, and an
equal ID does not permit reuse at a different point or under a different sampling law, Plan, or
environment. The reference only carries caller-supplied IDs next to the indices; it does not
generate paths or evaluation IDs.

All members of a dense map must have the same program and Plan signature, including Mesh
identity and output structure and association. A different mesh with the same coefficient
count, ragged leaf lengths, a different leaf structure, or outputs that belong to another
program are rejected. Grouping or heterogeneous collections would need their own interface;
padding is not a map operation. The reference checks dense buffer shapes and keeps a real
program identity in its signature tests; accepting each member is still the job of the
evaluation layer.

## Empty maps and resource limits

Extents of zero and one are allowed. Because the output signature, leaf types, and shapes are
known before any member runs, a zero extent returns correctly typed empty buffers and performs
no evaluations. A mean, or any reduction that needs a nonempty domain, must reject an empty
collection itself. Shared values are still validated when the extent is zero. There is no
required anchor and no minimum of two members.

Before allocating member or output buffers, the map checks rank, axis positions, duplicate axes,
every shape product, addressable strides, input lengths, the total number of output values, and
the estimated bytes against explicit limits. A zero factor in one product does not excuse
overflow in another stride. The reference takes its limits on rank and on retained binary64
values and bytes from the caller; they are not a fixed maximum, and they cover only these
buffers, not total solver memory. A solve map must add its own estimate of per-member state,
history, and execution resources before allocating.

## Native execution

The tests use `f(a, b) = (2*a + b, a - 3*b)` with exactly representable integer values.
For zipped `(a,b) = [(2,10), (1,20), (2,10)]`, the two output leaves are
`[14,22,14]` and `[-28,-59,-28]`. Sharing `b=10` gives `[14,12,14]` and `[-28,-29,-28]`.
The product of `a=[1,2]` and `b=[10,20]`, with `a` outermost, gives
`[12,22,14,24]` and `[-29,-59,-28,-58]`. These hand-derived values tell zip and product apart
and expose mistakes in ordering, duplicates, or accidental sharing. Separate focused tests cover
non-leading component axes, nested axes, static empty outputs, shape, axis, and extent errors,
aggregate limits, metadata preservation, sample association, and foreign program signatures.

The indexing reference lives in the `eqiora-api` unit tests. The native `EvaluationMapPlan`
takes one shared `Arc<DifferentiableProgram>`, a slice of complete input slices in request
order, and an `EvaluationMapExecutionPolicy`. Its `execute` returns either a
`CompleteEvaluationMap` or an `EvaluationMapTerminalReport`. The policy sets the chunk size, a
cap on host workers, and `Retain` or `Recompute`. Each worker runs an independent
single-threaded reference solve; there is no nested solver pool, and the order in which members
finish never changes the order of the result.

With `Retain`, `members()` returns the evaluations in order and `evaluation(index)` borrows one.
With `Recompute`, those buffers are released and `members()` returns `None`, but
`receipt(index)` still holds each member's complete original acceptance record and `len()` still
reports the full membership. `execute_with_delivery(cancel, callback)` passes borrowed
evaluations to the callback in request order at chunk boundaries; any copies the callback keeps
count against the caller's own storage. The callback may receive accepted members of a chunk
that later fails, so only a complete return guarantees that every member succeeded. Delivery
never reduces the results or filters out failures.

In a terminal report, `occurrence(index)` tells whether a position was accepted (with its state
retained or released), failed, was cancelled, or never started. Accepted members need not form
a prefix: workers that were already dispatched finish normally after another member fails. Each
failure keeps its original diagnostics. `stopped_index()` is the smallest failed or cancelled
index, not a count of accepted members. Cancellation is checked before dispatch, never inside a
solve. An empty map, or one whose final member completes, counts as complete; a member that
fails in flight is reported as a failure even if cancellation also arrives. Nothing is retried
automatically.

For an already compiled program with three selected inputs in the declared order:

```rust
use std::sync::Arc;
use eqiora::api::{EvaluationMapExecutionPolicy, EvaluationMapPlan};

let program = Arc::new(program);
let p1 = [2.0, 0.75, 0.5];
let p2 = [3.0, 1.25, -0.25];
let policy = EvaluationMapExecutionPolicy::retained(64 * 1024 * 1024);
let plan = EvaluationMapPlan::new(program, &[&p2, &p1, &p2], policy)?;
match plan.execute() {
    Ok(complete) => assert_eq!(complete.len(), 3),
    Err(report) => eprintln!("Stopped at {}: {:?}", report.stopped_index(), report.diagnostics()),
}
```

For parallel execution that releases buffers, pass `EvaluationMapExecutionPolicy::new` a chunk
size (for example 8), `Target::HostCpu { threads: 2 }` (the thread count is a `NonZeroUsize`),
`EvaluationMapRetention::Recompute`, and a storage limit. `recompute(index)` re-evaluates only
that frozen point and must reproduce the original receipt exactly; it cannot turn a terminal
report into a complete map. `from_samples` takes already-frozen `SampledParameterPoint` records,
so no sampler runs during execution or recomputation.

`estimated_storage_bytes()` reports the native storage estimate. It counts every point, indexed
outcome, and receipt (including output fingerprints); the state and right-hand-side vectors the
policy keeps; a conservative dense upper bound on CSR nonzeros; output and residual Parameter
Jacobians; output-state associations; and frozen sample metadata, channels, and ordering
buffers. Accepted receipts share one immutable deployment binding, and temporary worker
bindings are charged by their encoded size rather than a heap bound. Products are checked before
points are copied or members reserved. The estimate excludes the shared Program, allocator
overhead, diagnostic strings, OS thread stacks, copies made by the callback, and transient solver
workspace, so it is not a limit on process or peak memory. State and output sizes come from the
Program, and the estimate does not assume that the sparsity pattern stays the same at a new
point.

The earlier single-Parameter Study API, which sorted, deduplicated, and anchored points at a
default, has been removed. Its two registered cases now use ordered complete points and inspect
partially accepted maps, with independent pointwise evaluation still serving as the reference:

```bash
mise run affected -- \
  --case differentiation.bounded-parameter-study \
  --case differentiation.bounded-parameter-study-private
```

## First-order products

`EvaluationMapProducts` borrows a `CompleteEvaluationMap`; a terminal report cannot be used for
derivatives. You choose the shared Parameter IDs explicitly, in the Program's own coordinates.
Their values must be identical at every member, and equal values are never treated as sharing.
The remaining Parameters are mapped in the Program's order.

The product view takes `point_shape`, `seed_shape`, and the positions of the point axes in the
combined grid. The point shape flattens to the original list of requests; the seed shape
enumerates independent first-order products. For example, point `[3]`, seed `[2]`, and
point-axis position `[1]` give a seed-major `[2,3]` grid, while position `[0]` gives point-major
`[3,2]`. Several point and seed axes can be nested and interleaved. These axes index records;
they are not Field component axes or derivative orders. Values stay real and in coherent SI
units, paired with the Program's inputs and outputs.

```rust
use eqiora::api::EvaluationMapProducts;

// `complete` has three accepted points sharing this exact input coordinate.
let shared = [complete.plan().program_identity().inputs()[0]];
let products = EvaluationMapProducts::new(&complete, &shared, &[3], &[2], &[1], 1 << 30)?;
// Shared tangents: seed axes then shared coordinates.
// Mapped tangents: combined point/seed axes then remaining coordinates.
let jvp = products.jvp(&shared_tangents, &mapped_tangents)?;
let vjp = products.vjp(&output_cotangents)?;
assert_eq!(jvp.products().len(), 6);
```

For `y_i=f(s,x_i)`, the JVP is `Ds_i ds + Dx_i dx_i`. The VJP keeps each member's mapped
covector and **sums** the shared contributions over points, separately for each seed. The sum
follows the original request order, whatever the axis placement. Overflow is an error, and
regrouping the sum is not guaranteed to give bit-identical results. For an arithmetic mean,
apply a separate outer reduction.

Each result keeps the ordinary JVP/VJP record, the complete primal output, and the point and
Plan it came from. Retained maps reuse their primal. A `Recompute` map re-evaluates one frozen
point per product action, checks it against the original receipt, uses it for all seeds, and
then releases it. Malformed or non-finite inputs are rejected before any action runs; if an
action, the evidence check, or a finite sum fails, no partial derivatives are returned. Empty
point axes give empty records and zero shared sums; zero seed axes perform no derivative
actions. Only real, first-order implicit products are available: product results cannot be
differentiated again, and complex or profile adapters are handled separately.

An explicit byte limit applies to the numerical buffers of one complete JVP and one complete
VJP, including copied point coordinates, duplicated primal payloads, full and mapped covectors,
and shared sums. It does not cover maps already retained, other metadata, caller buffers, solver
scratch space, or peak process memory. Ranks, extents, strides, and products are checked before
allocation. The [mapped-product case](../../verify/differentiation/mapped-products/README.md)
derives the shared chain rule independently and compares Q1/TPFA products with ordinary retained
evaluations. When changing this code, run it with `--case differentiation.mapped-products`
together with the two ordered-map cases above.

The Python and JAX dense adapters use serial retained execution; their CPU/f64 batching rules
are documented with those adapters. This scheduler adds no GPU, MPI, or remote execution, no new
stochastic law, persistence, higher derivatives, or speedup guarantee, and no source-language
syntax.
