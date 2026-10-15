# Differentiation and framework adapters

Use the environment and `eqiora-source` checkout from [Get started](/get-started/).
The [inverse problems textbook](/learn/inverse-problems/) explains how a change
in a parameter changes an observation. Here we work directly with the solved
field and its derivatives.

## Prepare a Poisson problem

This example uses the unit square, a sinusoidal source, and a constant boundary
value. Run the following blocks in order, from the folder containing `.venv`
and `eqiora-source`:

```python
from pathlib import Path

import eqiora
import numpy as np

graph = eqiora.geometry.GeometryGraph()
rectangle = graph.rectangle(x_bounds=(0.0, 1.0), y_bounds=(0.0, 1.0))
boundaries = ("x_lower", "x_upper", "y_lower", "y_upper")
geometry = graph.build(
    rectangle,
    named_topology={
        "square": rectangle.region,
        **dict(zip(boundaries, rectangle.boundaries, strict=True)),
    },
)
mesh = eqiora.meshing.generate(eqiora.meshing.resolve(
    geometry, eqiora.meshing.CartesianMesher(cells=(8, 8)),
))
model = eqiora.compile(
    path=Path("eqiora-source/examples/inverse-poisson.eqi"),
    entry="InversePoisson",
    geometry=geometry,
    bindings={
        "square": geometry.selection("square"),
        **{
            name: (geometry.selection(name), geometry.selection("square"))
            for name in boundaries
        },
        "diffusion": 1.0,
        "wave_number": np.pi,
        "source_scale": 2.0 * np.pi**2,
        "boundary_offset": 0.0,
    },
)
plan = eqiora.resolve(
    model,
    mesh=mesh,
    spatial=eqiora.fem.Q1(),
    solve=eqiora.solve.Linear(
        algorithm=eqiora.solve.LinearSolver.BiConjugateGradientStabilized,
        preconditioner=eqiora.solve.Preconditioner.Identity,
        reduction=eqiora.solve.Reduction.Reproducible,
        provider=eqiora.solve.SolverProvider.reference(),
        relative_tolerance=1.0e-10,
        absolute_tolerance=1.0e-12,
        maximum_iterations=10_000,
    ),
)
```

Open `eqiora-source/examples/inverse-poisson.eqi` to inspect the equations and
boundary conditions. `source_scale` multiplies the spatial source pattern;
`diffusion` controls the diffusion coefficient.

## Evaluate a point and its derivatives

Use `eqiora.diff` to select the parameters and output field to differentiate:

```python
import numpy as np

program = eqiora.diff.compile(
    plan,
    inputs=(model.parameter("source_scale"),),
    outputs=(plan.capability.fields[0],),
)

evaluation = program.evaluate((1.5,))
primal = evaluation.primal()
jvp = evaluation.jvp((1.0,))
vjp = evaluation.vjp(
    (np.ones(program.outputs[0].shape, dtype=np.float64),)
)
```

`primal.output[0]` contains the solved field values.
`jvp.tangent[0]` contains their directional change for a unit increase in
`source_scale`. `vjp.input_cotangent[0]` contains the derivative of the sum
of field values, because the output cotangent above is all ones. JVP means
Jacobian–vector product; VJP is the reverse product, useful for a scalar loss
with many input parameters.

Each evaluation keeps its own input point and linearization, so evaluating a new
point changes neither the Model and Plan nor the primal, JVP, or VJP of an earlier
evaluation. Parameters you did not select keep their model values.

Point values, selected outputs, tangents and input cotangents are ordered tuples of read-only NumPy leaves.
Each leaf keeps its native real/complex dtype and shape; a scalar has shape `()`.
Use `program.inputs` and `program.outputs` for the exact native signatures,
including mathematical types, units and spatial coefficient associations.
For example, `outputs=(model.field("z"), model.observable("loss"))` returns
`field, loss = evaluation.primal().output` without concatenating their values.
Total JVP inputs follow `program.inputs`; VJP seeds follow `program.outputs`.
Their results use output and input leaves respectively. Complex covectors pair as
`Re(sum(conj(c) * t))`; base signatures do not assign primal units to dual values.
Dense map projections and residual/unknown partial actions still use real coordinates.

Spatial differentiation currently supports rectangular 2D Cartesian meshes,
Q1 FEM or TPFA FVM, one selected Field and a serial CPU `float64` solve.
Finite affine and nonlinear Plans select ordered Fields and Observables through
the same accepted-point API. Array leaves use the CPU transport described in
[Execution, diagnostics, and arrays](execution-and-arrays.md).

## Finite nonlinear points and partial actions

A strict-interior finite nonlinear Plan from the
[constraints guide](../language/constraints.md#strict-interior-nonlinear-execution)
works with the same API. Pass its initial State and a scalar instantaneous
Observable:

```python
program = eqiora.diff.compile(
    plan, inputs=(model.parameter("p"),),
    outputs=(model.observable("output"),), state=state,
)
point = program.evaluate((4.0,))
zero = np.array([0.0], dtype=np.float64)
one = np.array([1.0], dtype=np.float64)
partial = point.output_partial_jvp(zero, one)
reduced = point.jvp((1.0,))
```

For `R(w,p)=w*w-p`, positive `w`, and `O(w,p)=w+p`, this point has
`w=2` and `O=6`. Holding `w` fixed gives `partial=1`; solving the implicit
sensitivity gives `reduced.tangent[0]=1.25`. At `p=9` the corresponding total
is `7/6`. `residual_jvp(dw, dp)` applies `R_w dw + R_p dp`, while
`output_partial_jvp(dw, dp)` applies `O_w dw + O_p dp`. Their VJP counterparts
return `(unknown_cotangent, parameter_cotangent)` without an implicit solve.
Unknown coordinates follow `point.accepted_unknowns`; Parameter coordinates
follow the ordered `inputs` selection.

As above, each evaluation keeps its own solution and partial derivatives. The
seed, the strict-interior conditions, and the residual and regularity checks are
applied again at every point. `evidence.primal_solve` is a `NonlinearSolveSummary`,
even when no Newton update was needed, and `derivative_solve` describes the
linear sensitivity solve, normal or transposed. This path does not support ordered
maps, global tracking of a solution branch, or higher implicit derivatives.

## Native ordered batches

Batches need no extra package. `program.map` prepares a batch without solving
anything; `execute` then solves every point in native code:

```python
batch = program.map((np.array([1.5, 0.5, 1.5], dtype=np.float64),), point_shape=(3,))
print(batch.point_shape, program.outputs)
result = batch.execute()
if isinstance(result, eqiora.CompleteEvaluationMap):
    states, = result.primal()  # read-only [3, *intrinsic output shape]
    tangents, = result.jvp((np.ones(3, dtype=np.float64),)).tangent
    cotangents = result.vjp((np.ones_like(states),))
else:
    print(result.stopped_index, result.statuses, result.diagnostics)
```

Inputs are ordered leaves following `program.inputs`. Each leaf has shape
`point_shape + signature.shape`; real and complex values retain their native dtype.
Repeated points are solved separately in row-major order. `point_shape=()` means
one point, and a zero extent returns correctly shaped output without running a
solver. `batch.points` holds the complete input leaves,
and `batch.occurrence_coordinates(i)` maps a flat index back to grid coordinates.

To hold some inputs at one value across the whole grid, pass them as shared
inputs. For a program whose inputs are `(source_scale, diffusion, boundary_offset)`:

```python
shared_program = eqiora.diff.compile(
    plan,
    inputs=(
        model.parameter("source_scale"),
        model.parameter("diffusion"),
        model.parameter("boundary_offset"),
    ),
    outputs=(plan.capability.fields[0],),
)
batch = shared_program.map(
    (np.array([3.0, 1.0, 3.0]), np.array([0.0, 0.2, 0.0])),
    point_shape=(3,),
    shared_inputs=(model.parameter("diffusion"),),
    shared=(2.0,),
)
result = batch.execute()
if isinstance(result, eqiora.CompleteEvaluationMap):
    jvp = result.jvp(
        (np.ones(3), np.ones(3)),
        shared=(0.1,),
    )
    vjp = result.vjp(tuple(np.ones_like(leaf) for leaf in result.primal()))
    # Shared covectors sum over all point occurrences; mapped ones stay separate.
    # Ordered immutable leaves; collection axes precede each intrinsic leaf shape.
    print(vjp.shared_cotangents, vjp.mapped_cotangents)
```

The mapped leaves are the remaining inputs, in Program order. Shared values
follow the order of `shared_inputs`, whose entries must be distinct and belong to
the Program's Model. An input is shared across all point axes or none, and every
array must have its full shape; there is no broadcasting.

JVP and VJP reuse the linearizations computed by `execute`. The optional
`seed_shape` and `point_axes` arguments place the point axes inside a larger grid
of seeds: for point shape `(2, 4)`, `seed_shape=(3,)` and `point_axes=(0, 2)` give
`(2, 3, 4)`. Mapped tangent and output-cotangent leaves have this grid followed
by their intrinsic shape; shared tangent leaves have `seed_shape` followed by
their intrinsic shape. JVP `shape` describes the grid only. The products
keep the Plan, the axis metadata, and the per-point results. At most 32 point and
seed axes are allowed. `retained_bytes_limit` and `numerical_bytes_limit` cap the
numerical data Eqiora keeps, not the process's peak memory.

Each leaf may be a scalar, NumPy array, Eqiora Array, or CPU DLPack tensor,
with the exact signature dtype (`float64` or `complex128`).
NumPy and DLPack data must be aligned, native-endian, and C-contiguous. Inputs are
copied before the GIL is released, so changing them afterwards has no effect. An
Eqiora Array is rank one; to use it as a grid, take its read-only NumPy view and
reshape it.

You can pass `eqiora.EvaluationMapCancellation()` to `batch.execute` and cancel it
from another Python thread; it is checked between points. If a batch fails or is
cancelled, `member(i)` returns the points completed so far, with their statuses
and diagnostics, but the batch has no `primal`, `jvp`, or `vjp`.

## PyTorch

From the same working folder, install the optional adapter. Bind the program
outside the function you compile:

```console
uv add "eqiora[torch]"
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

The adapter requires PyTorch `>=2.14,<2.15`. Backpropagation uses the VJP of
the solved equations.

Inputs must be rank-one, contiguous `float64` tensors on CPU:0. The adapter never
modifies its input and returns a new tensor rather than a view of stored results.
Programs are kept alive for the rest of the process, because autograd and compiled
graphs can outlive the wrapper; evaluations and derivatives are not cached.
PyTorch and JAX share the native registration owner. The exact initial State is
part of program identity, so distinct nonlinear seeds retain their own solutions
and derivatives.

The current adapter supports in-process `torch.compile(fullgraph=True)` with
first-order gradients. Double backward, `vmap`, AMP, CUDA, `torch.export`, and
AOT packaging are not yet supported.

## JAX

Install the optional JAX adapter into the same environment:

```console
uv add "eqiora[jax]"
```

```python
import jax
import jax.numpy as jnp
import eqiora.jax as eqjax

jax.config.update("jax_enable_x64", True)
jax_program = eqjax.bind(program)
theta = (jnp.asarray(1.5, dtype=jnp.float64),)
direction = (jnp.asarray(0.25, dtype=jnp.float64),)

state, = jax.jit(jax_program)(theta)
_, (tangent,) = jax.jvp(
    jax_program,
    (theta,),
    (direction,),
)
gradient = jax.grad(
    lambda point: jnp.sum(jax_program(point)[0] ** 2)
)(theta)

# Request order and repeated points are preserved by the native map.
points = (jnp.stack((theta[0] + 0.1, theta[0], theta[0] + 0.1)),)
states, = jax.jit(jax.vmap(jax_program))(points)
point_gradients = jax.jit(jax.vmap(jax.grad(
    lambda point: jnp.mean(jax_program(point)[0])
)))(points)
```

This adapter requires Python 3.12 or newer and exactly JAX and jaxlib 0.11.0. The
primal, JVP, and VJP run as separate typed FFI calls, so compiled code never calls
back into Python, and derivatives come from the solved equations rather than from
differentiating solver iterations.

Each call takes an ordered tuple or list of JAX leaves and returns an ordered
tuple of outputs. `jax_program.inputs` and `jax_program.outputs` are the native
signatures: each leaf retains its exact intrinsic shape and `float64` or
`complex128` dtype. Scalars are zero-dimensional arrays with an explicit dtype.
Concrete leaves must reside together on one CPU device. Use `vmap` for
mapped and shared arguments, non-leading `in_axes`, `out_axes`, and nested
batches. `jit`, first-order JVP and VJP, `vmap(grad(...))`, gradients of summed or
averaged losses over a batch, and `jacfwd`/`jacrev` all run through the native
batch and its products. When a shared input is broadcast, its reverse
contributions are summed; they are divided by the batch size only if you take an
explicit mean. If any point fails, the whole operation fails with a diagnostic
naming that point.

For complex leaves, the adapter follows [JAX's cotangent convention](https://docs.jax.dev/en/latest/301/cookbook.html#complex-numbers-and-differentiation):
`Re(sum(cotangent * tangent))`. Native Eqiora covectors instead use
`Re(sum(conj(cotangent) * tangent))`. JAX differentiates the private real/imaginary
projections around the native real-linear action, so JAX VJPs and gradients
retain the standard JAX representation. Conjugate complex covectors when comparing
them with native results. No holomorphic assumption is made.

The program, shapes, dtype, layout, and CPU platform are fixed; only the numerical
values change between calls. Point and seed axes are kept separate, with at most
32 in total. Each FFI buffer is limited to 64 MiB, and the native batch and its
products each keep at most 64 MiB of numerical data; neither limit bounds peak
memory. Empty and single-point batches keep their shapes. Named collective axes,
sharding, `pmap`, higher-order derivatives, accelerators, and export are not
supported, and batching is not guaranteed to be faster than a loop.

Importing `eqiora` does not import PyTorch or JAX.

For finite Newton Plans, the residual products use the Plan's normalized
equations: each residual and its derivatives with respect to unknowns and
Parameters are divided by the same physical residual scale. The reduced output JVP
and VJP use the real inner product for complex coordinates and do not depend on
that normalization.
