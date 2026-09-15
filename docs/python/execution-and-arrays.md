# Execution, diagnostics, and arrays

## One run lifecycle

Continue with `decay.eqi` and the environment from [Get started](/get-started/).
Run these blocks from the folder containing that file. Submit the model when
you want a handle to inspect or cancel; use `eqiora.run` when you simply want to
wait for its result:

```python
import eqiora

model = eqiora.compile(path="decay.eqi")
field = model.field("x")
plan = eqiora.resolve(
    model,
    temporal=eqiora.time.Tsitouras45(
        initial_step_s=0.01,
        relative_tolerance=1.0e-9,
        absolute_tolerances={field: 1.0e-11},
    ),
)
state = eqiora.State.initial(plan)
run = eqiora.submit(
    plan,
    state=state,
    until_s=1.0,
    output_times_s=(1.0,),
)
print(run.status, run.progress)
result = run.result()

# The blocking convenience uses the same lifecycle.
same_kind_of_result = eqiora.run(
    plan,
    state=state,
    until_s=1.0,
    output_times_s=(1.0,),
)
```

`RunStatus` records a finite accepted history from creation through one
terminal state. `progress` is an execution-family-specific coalesced snapshot,
not a percentage or event log. Repeated `result()` calls return the same
immutable Python result object.

## Phase-level profiling

Pass `profile=True` to `run` or `submit` to collect process-local execution
telemetry without changing the Model, Plan, numerical result, or persisted
Result artifact:

```python
result = eqiora.run(
    plan,
    state=eqiora.State.initial(plan),
    until_s=1.0,
    output_times_s=(1.0,),
    profile=True,
)
print(result.profile.summary())
for event in result.profile.events:
    print(event.path, event.fields)
```

The summary names total inclusive time, self time, calls, and inclusive mean per
call. Child time is subtracted once from its parent's self time, so rows must not
be added to recover a run total. Each `ProfilePhase` exposes the same values as
`inclusive_seconds`, `self_seconds`, `calls`, and `mean_seconds`.

Aggregation uses the nested phase path together with semantic identity fields,
available through `ProfilePhase.fields`. Occurrence observations such as step,
time, iteration, and residual remain on `ProfileEvent` and do not split repeated
calls. Phase identities without occurrence fields retain one representative
event; their complete count and timing remain on `ProfilePhase`, so repeated
local evaluations do not make event storage grow with every cell. Backend
resolution and discretization preparation have distinct setup
roles; initial linearization and line-search trials have distinct assembly roles.
Transient work nests under `run/solve/time_step`. Faer SparseLU distinguishes
symbolic factorization, numeric factorization, and backsolve. A
Result decoded from bytes has no profile because telemetry is deliberately not
part of artifact or semantic identity. Leave profiling disabled for ordinary
runs; library crates emit spans while subscriber configuration and presentation
stay at the application boundary.

Awaiting does not introduce another native runtime:

```python
async def simulate(plan):
    run = eqiora.submit(
        plan,
        state=eqiora.State.initial(plan),
        until_s=1.0,
        output_times_s=(1.0,),
    )
    try:
        return await run
    finally:
        if not run.done:
            run.cancel()
```

Cancelling the surrounding asyncio task and dropping a Run do not implicitly
cancel native work. Call `run.cancel()` explicitly. Cancellation is
cooperative at execution boundaries and never exposes a partial result. A
request after the last cancellable boundary may still complete.

Long native waits release the ordinary CPython GIL only after inputs are
owned. Solver iterations do not call Python. Free-threaded Python and
subinterpreter shutdown remain separate capabilities.

## Structured failures

Eqiora model and execution failures derive from `EqioraError`. Stable
subclasses distinguish validation, compatibility, capability, execution,
cancellation, and internal failures. Every Eqiora-raised error retains
structured diagnostics:

```python
try:
    result = eqiora.run(
        plan,
        state=eqiora.State.initial(plan),
        until_s=-1.0,
        output_times_s=(-1.0,),
    )
except eqiora.EqioraError as error:
    print(error.category)
    for diagnostic in error.diagnostics:
        print(diagnostic.code, diagnostic.severity, diagnostic.message)
        print(diagnostic.graph_path, diagnostic.source_span)
```

Python call-shape mistakes remain ordinary `TypeError` rather than fabricated
model diagnostics. Guarded native boundaries sanitize unwinding Rust panics as
`InternalError`; process abort and memory exhaustion are not recoverable
claims.

## NumPy ownership

An `Array` owns a dense, native-endian, rank-one CPU `float64` allocation.
Inspecting descriptors does not import NumPy.

```python
array = result.series(field).values
view = array.numpy(copy=False)  # `None` has the same meaning
writable = array.numpy(copy=True)

assert not view.flags.writeable
assert writable.flags.writeable
```

The first no-copy projection transfers the native allocation once into an
opaque owner. The resulting C-contiguous NumPy array is irreversibly read-only
and remains alive independently of the Result and Array handles. `copy=True`
returns an independent writable allocation.

## DLPack

Eqiora exports an independent versioned CPU snapshot:

```python
import numpy as np

snapshot = np.from_dlpack(array)
```

The snapshot never aliases result arrays. Legacy capsule requests,
non-CPU transfers, non-`None` streams, and `copy=False` fail closed because
consumer enforcement of DLPack's advisory read-only flag is not universal.

Differentiable-program inputs may arrive from a complete CPU:0 DLPack
producer. Eqiora requests a no-transfer view, validates dtype, rank, length,
byte order, alignment, and contiguity, then copies the input before native execution.

## Fixed arrays in execution sessions

`Model.execution_session` accepts invariant real or exact
integer channel arrays through its typed input tables. Pass one complete tuple
per tick. This model accumulates two input channels:

```python
sampled = eqiora.compile(source="""
public model Accumulator(
  clock tick: periodic,
  input drive: array<1, 2> at tick,
  output total: array<1, 2> at tick
) {
  state memory: array<1, 2> at tick;
  initial { pre(memory) = [0, 0]; }
  relation update at tick {
    next(memory) = [pre(memory)[0] + drive[0], pre(memory)[1] + drive[1]];
    total = next(memory);
  }
}
""", entry="Accumulator", bindings={"tick": eqiora.ClockDomain(period_s=1)})
session = sampled.execution_session(
    end_time_s=2, max_step_s=0.1,
    inputs={"drive": ("tick", [(3, 4), (5, 6), (7, 8)])},
)
session.advance_ticks(1)
resumed = sampled.resume_execution(session.checkpoint())
resumed.advance_ticks(2)
print(resumed.output("total", 2))  # (Fraction(2, 1), (15.0, 18.0))
```

Array State values and accepted output values are complete nested tuples; exact
integer components remain Python integers, including values above `2**53`.
A failed tick commits neither a partial array nor another State update. Restart
preserves the accepted clock position and previously absent or present outputs.
The immutable Model fixes every extent. Scalar broadcasting, partial indexed
writes, spatial tensors and complex execution are unsupported in execution sessions.
Input and retained output limits count scalar components, including nested arrays.
