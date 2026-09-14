# Differentiation and framework adapters

## Evaluate a point and its derivatives

Use `eqiora.diff` to select the parameters and output field to differentiate:

```python
import numpy as np

program = eqiora.diff.compile(
    plan,
    inputs=(model.parameter("source"),),
    output=plan.capability.fields[0],
)

evaluation = program.evaluate(np.array([1.5], dtype=np.float64))
primal = evaluation.primal()
jvp = evaluation.jvp(np.array([1.0], dtype=np.float64))
vjp = evaluation.vjp(
    np.ones(program.output_shape, dtype=np.float64)
)
```

Each evaluation stores its input point and linearization. Evaluating a new
point leaves the Model and Plan unchanged. Parameters that you did not select
keep their model values.
Retaining one evaluation while evaluating another point cannot retarget its
primal, JVP, or VJP.

Python differentiation supports rectangular 2D Cartesian meshes, Q1 FEM or
TPFA FVM, and a serial CPU `float64` solve. Point values, tangents, and
cotangents must be rank-one CPU arrays with the layout described in
[Execution, diagnostics, and arrays](execution-and-arrays.md).

Each program selects one output field and computes first derivatives.

## Native ordered batches

The batch API is included in `0.1.0a13`. Install the release in a clean
environment:

```console
uv venv --python 3.13 .venv
uv pip install --python .venv/bin/python eqiora==0.1.0a13
uv run --no-project --python .venv/bin/python your_batch.py
```

Plan a batch without solving, then execute it once in native code:

```python
batch = program.map(np.array([[1.5], [0.5], [1.5]], dtype=np.float64))
print(batch.input_shape, batch.output_shape)
result = batch.execute()
if isinstance(result, eqiora.CompleteEvaluationMap):
    states = result.primal()  # read-only [3, output components]
    tangents = result.jvp(np.ones((3, 1), dtype=np.float64)).tangent
    cotangents = result.vjp(np.ones(batch.output_shape, dtype=np.float64))
else:
    print(result.stopped_index, result.statuses, result.diagnostics)
```

The trailing input axis follows `program.input_ids`; preceding axes are the
row-major point grid. Equal points remain separate occurrences in request
order. A rank-one input is one point, and zero extents preserve output metadata
without executing a solver. `batch.points` exposes the frozen complete inputs;
`batch.occurrence_coordinates(i)` maps a flat occurrence back to its grid.

Share selected coordinates explicitly across the whole grid. For a program
whose ordered inputs are `(source, diffusion, boundary_offset)`:

```python
batch = program.map(
    np.array([[3.0, 0.0], [1.0, 0.2], [3.0, 0.0]], dtype=np.float64),
    shared_inputs=(model.parameter("diffusion"),),
    shared=np.array([2.0], dtype=np.float64),
)
result = batch.execute()
if isinstance(result, eqiora.CompleteEvaluationMap):
    jvp = result.jvp(
        np.ones((3, 2), dtype=np.float64),
        shared=np.array([0.1], dtype=np.float64),
    )
    vjp = result.vjp(np.ones(batch.output_shape, dtype=np.float64))
    # Shared covectors sum over all point occurrences; mapped ones stay separate.
    print(vjp.shared_cotangents, vjp.mapped_cotangents)
```

Mapped coordinates are the remaining inputs in Program order, not the order
of a Python dictionary. Shared values follow `shared_inputs`, which must be
distinct references from the exact Program's Model. Sharing along only some
point axes is unsupported; supply the exact shape without broadcasting.

JVP and VJP reuse accepted native linearizations. Optional `seed_shape` and
`point_axes` place point axes inside a nested product grid. For point shape
`(2, 4)`, `seed_shape=(3,)` and `point_axes=(0, 2)` mean `(2, 3, 4)`;
mapped tangent and output-cotangent arrays append their coordinate extent.
Shared tangents use `seed_shape + (shared_count,)`. Products retain the Plan,
axis metadata and per-member results. Rank is bounded to 32 point/seed axes;
`retained_bytes_limit` and `numerical_bytes_limit` bound native retained
numerical storage, not process peak memory.

NumPy, Eqiora Array and CPU DLPack inputs use the existing exact `float64`
ownership rules. NumPy/DLPack tensors must be aligned, native-endian and
C-contiguous. Inputs are copied before execution releases the GIL, so later
mutations cannot retarget a Plan or product. Eqiora Array is rank one; use its
read-only NumPy view and explicit reshape when a point grid is desired.

`eqiora.EvaluationMapCancellation()` can be passed to `batch.execute` and
cancelled from another Python thread. Native execution polls it only between
occurrences. A failed or cancelled prefix exposes accepted `member(i)` values
and exact status/diagnostics, but has no complete `primal`, `jvp` or `vjp`.
Importing and using batches requires neither JAX nor PyTorch.

## PyTorch

Install the optional adapter and bind outside the compiled function:

```console
uv venv --python 3.13 .venv
uv pip install --python .venv/bin/python ".[torch]"
```

```python
import torch
import eqiora.torch as eqtorch

torch_program = eqtorch.bind(program)
theta = torch.tensor(
    [1.5],
    dtype=torch.float64,
    requires_grad=True,
)
state = torch_program(theta)
state.square().sum().backward()

compiled_objective = torch.compile(
    lambda point: torch_program(point).square().sum(),
    fullgraph=True,
)
```

The current adapter declares PyTorch `>=2.13,<2.14` and verifies 2.13.0. It
registers a functional project-namespaced custom operator, a metadata-only fake
implementation, and a first-order autograd rule whose backward invokes
Eqiora's accepted VJP through a second custom operator.

Inputs are exact rank-one contiguous CPU:0 `float64` tensors. The adapter
mutates no input and returns a fresh versioned DLPack snapshot rather than an
alias of native evidence. Static programs are retained process-locally because
autograd and compiled graphs may outlive a temporary wrapper; mutable
evaluations and derivatives are not cached.

The current adapter supports in-process `torch.compile(fullgraph=True)` with
first-order gradients. Double backward, `vmap`, AMP, CUDA, `torch.export`, and
AOT packaging are not yet supported.

## JAX

The optional JAX adapter uses native typed FFI:

```console
uv venv --python 3.13 .venv
uv pip install --python .venv/bin/python ".[jax]"
```

```python
import jax
import jax.numpy as jnp
import eqiora.jax as eqjax

jax.config.update("jax_enable_x64", True)
jax_program = eqjax.bind(program)
theta = jnp.array([1.5], dtype=jnp.float64)
direction = jnp.array([0.25], dtype=jnp.float64)

state = jax.jit(jax_program)(theta)
_, tangent = jax.jvp(
    jax_program,
    (theta,),
    (direction,),
)
gradient = jax.grad(
    lambda point: jnp.sum(jax_program(point) ** 2)
)(theta)

# Request order and repeated points are preserved by the native map.
points = jnp.stack((theta + 0.1, theta, theta + 0.1))
states = jax.jit(jax.vmap(jax_program))(points)
point_gradients = jax.jit(jax.vmap(jax.grad(
    lambda point: jnp.mean(jax_program(point))
)))(points)
```

This adapter requires Python 3.12 or newer and the exact JAX/JAXLIB 0.11.0
pair. Separate primal, JVP, and VJP typed FFI targets keep compiled numerical
execution free of Python host callbacks and do not differentiate solver
iterations.

Each call takes one rank-one host-CPU `float64` point. Use `vmap` to compose
mapped and shared arguments, non-leading `in_axes`, `out_axes`, and nested
batches. `jit`, first-order JVP/VJP, `vmap(grad(...))`, gradients of summed or
averaged mapped losses, and `jacfwd`/`jacrev` use the native accepted
map and products. Broadcasting a shared input sums its reverse contributions;
only an explicit mean divides by the collection size. Any failed member rejects
the dense operation with an occurrence diagnostic.

Program identity, shapes, dtype, layout, and CPU platform stay static while
numeric points vary. Point and derivative-seed axes remain distinct, with at
most 32 combined axes. Each FFI buffer has a 64 MiB limit; the native map and
products each also have a separate 64 MiB retained-numerical-storage limit,
not a peak-memory guarantee. Empty and singleton batches retain their shapes.
Named collective axes, sharding, `pmap`, higher-order derivatives, accelerators,
and export remain unsupported; batching makes no speedup claim.

Importing base `eqiora` imports neither optional framework.
