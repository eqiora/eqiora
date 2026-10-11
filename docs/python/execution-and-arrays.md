# Execution, diagnostics, and arrays

## Running a model

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
        absolute_tolerances={(field, 0, 0, False): 1.0e-11},
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

`RunStatus` records each state the run passes through, from creation to a single
final state. `progress` is the most recent progress snapshot; its contents depend
on the kind of execution, and it is neither a percentage nor an event log. Calling
`result()` again returns the same immutable object.

## Phase-level profiling

Pass `profile=True` to `run` or `submit` to record where the time goes. Profiling
does not change the Model, Plan, numerical result, or saved Result:

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

For each phase, the summary gives the total inclusive time, the self time, the
number of calls, and the mean inclusive time per call. Self time excludes time
spent in child phases, so adding up the rows does not give the run's total. Each
`ProfilePhase` has the same values as `inclusive_seconds`, `self_seconds`, `calls`,
and `mean_seconds`.

Phases are grouped by their nested path and their identifying fields, available as
`ProfilePhase.fields`. Per-occurrence values such as step, time, iteration, and
residual are recorded on each `ProfileEvent` and do not split a phase into
separate rows. For phases without such values, only one representative event is
kept while `ProfilePhase` holds the full count and timing, so storage does not
grow with the number of cells. Setup is split into backend resolution and
discretization preparation, and assembly into initial linearization and
line-search trials. Transient work is nested under `run/solve/time_step`, and
Faer SparseLU reports symbolic factorization, numeric factorization, and back
substitution separately. A Result loaded from bytes has no profile, because timing
data is not part of the saved artifact. Leave profiling off for ordinary runs.

A submitted run can also be awaited from asyncio, using the same native runtime:

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

Cancelling the asyncio task or dropping the Run does not stop native work; call
`run.cancel()`. Cancellation takes effect at the next checkpoint during execution
and never produces a partial result. A run that has already passed its last
checkpoint may still complete.

Native execution releases the GIL once it has copied its inputs, and solver
iterations never call back into Python. This applies to standard CPython with the
GIL; free-threaded builds and subinterpreter shutdown are not covered.

## Structured failures

Every model or execution error raised by Eqiora derives from `EqioraError`, with
subclasses for validation, compatibility, capability, execution, cancellation, and
internal failures. Each one carries structured diagnostics:

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

Calling a function with the wrong arguments raises an ordinary `TypeError`. A Rust
panic inside Eqiora is reported as `InternalError`, but a process abort or running
out of memory cannot be recovered from.

## NumPy ownership

An `Array` holds a dense, native-endian, rank-one `float64` buffer in CPU memory.
Reading its shape and type does not import NumPy.

```python
array = result.series(field).values
view = array.numpy(copy=False)  # `None` has the same meaning
writable = array.numpy(copy=True)

assert not view.flags.writeable
assert writable.flags.writeable
```

With `copy=False`, the buffer is handed to NumPy without copying. The resulting
C-contiguous array is permanently read-only and stays valid after the Result and
Array are gone. `copy=True` returns a separate, writable copy.

## DLPack

DLPack consumers receive a copy of the data on the CPU:

```python
import numpy as np

snapshot = np.from_dlpack(array)
```

The copy never shares memory with the result. Requests for a legacy capsule, a
non-CPU device, a stream other than `None`, or `copy=False` raise an error,
because not every DLPack consumer respects the read-only flag.

Inputs to a differentiable program can come from any CPU:0 DLPack producer.
Eqiora checks the dtype, rank, length, byte order, alignment, and contiguity, then
copies the input before native execution.

## Fixed arrays in execution sessions

`Model.execution_session` accepts fixed-size arrays of real or integer values as
inputs, one complete tuple per tick. This model accumulates two input channels:

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

Array values in States and outputs are nested tuples, and integer components stay
Python integers, even above `2**53`. A failed tick leaves the State unchanged. A
resumed session continues from the same clock position, with the same outputs
present or absent. The Model fixes every array size. Execution sessions do not
support scalar broadcasting, writing part of an array, spatial tensors, or complex
values. Limits on inputs and stored outputs count individual scalar components.

For ordinary ODE runs, `state.value(field)` returns the whole declared value: real
scalars as `float`, complex scalars as `complex`, and arrays as nested immutable
tuples with the declared axes. `derivative_order=` selects a stored higher-order
derivative; asking for one that is not stored is an error. Saving and restarting a
State keeps these values intact. `result.series(field, component=...,
imaginary=...)` selects one real coordinate as a time series.
