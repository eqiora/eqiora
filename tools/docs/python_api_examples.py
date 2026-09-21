"""Reader examples included by the existing Python reference generator."""

WORKFLOW = '''## Compile and run

Follow [Get started](/get-started/) to install the Python package from the same
source as these pages. This example needs only Eqiora and its NumPy dependency.

```python
import eqiora

model = eqiora.compile(source="""
model decay(parameter rate: 1 / s = 1) {
    state x: 1;
    initial { x = 1; }
    relation flow { derivative(x) + rate * x = 0; }
}
""", filename="decay.eqi")
x = model.field("x")
plan = eqiora.resolve(
    model,
    temporal=eqiora.time.Tsitouras45(
        initial_step_s=0.01,
        relative_tolerance=1e-9,
        absolute_tolerances={x: 1e-11},
    ),
)
result = eqiora.run(
    plan,
    state=eqiora.State.initial(plan),
    until_s=1.0,
    output_times_s=(0.25, 0.5, 1.0),
)
series = result.series(x)
for time_s, value in series:
    print(f"{time_s:.2f} s: {value:.6f}")
```

The three values are approximately `0.778801`, `0.606531`, and `0.367879`:
`x` is dimensionless, time is in seconds, and the equation gives exponential
decay. `compile` defines the mathematics, `resolve` chooses its numerical method,
and `run` computes the requested observations.

## Inspect, edit, and save

Continue with the objects above:

```python
print(model.parameter("rate").value)
print(model.render_equations("flow", profile="plain")[0].text)

faster = model.commit(model.preview_value_edit("rate", 2.0))
print(model.parameter("rate").value, faster.parameter("rate").value)

import numpy as np
np.savetxt(
    "decay.csv",
    np.column_stack((series.time.numpy(), series.values.numpy())),
    delimiter=",", header="time_s,x", comments="",
)
```

The original rate remains `1.0`; the new Model has rate `2.0`. Resolve the new
Model to simulate it. The CSV contains the observations from the original run.
Equation rendering provides readable mathematics without reconstructing it
from numerical output.

For spatial models, continue with [geometry](/reference/python/geometry/),
[meshing](/reference/python/meshing/), and the complete
[modeling examples](/guides/modeling/). For gradients, see
[differentiation](/guides/differentiation/).
'''

EXAMPLES = {
    "eqiora": '''## Example: model, plan, and result

The [complete Python example](/reference/python/#compile-and-run) creates
`model`, `plan`, `result`, and `x`. Continue with those objects:

```python
print(model.field("x") == x)
print(model.parameter("rate").value)
print(model.render_equations("flow", profile="plain")[0].text)
values = result.series(x).values.numpy()
print(values[-1])
```

This selects the declared field and parameter by name, displays the equation,
and reads the final dimensionless value (about `0.367879`). Result arrays are
read-only; use `values.copy()` when you need a writable NumPy array.
''',
    "lang": '''## Example: write equations in Python

```python
import eqiora
from eqiora import lang as q

source = eqiora.Module("example")
component = source.model("Balance")
x = component.field("x", value_type=eqiora.ValueType.real(),
                    role=eqiora.FieldRole.Variable)
component.relation("balance", q.equation(x, 2))
print(source.to_eqi())
model = eqiora.compile(source=source)
```

`to_eqi()` prints the model declaration and `x = 2` relation. Compilation
uses the same language as a handwritten `.eqi` file. Build the declarations
before emitting or compiling the Module. See [modeling](/guides/modeling/)
for parameters, components, imports, and spatial equations.
''',
    "units": '''## Example: dimensional quantities

```python
import eqiora
from eqiora import lang as q, units as u

source = eqiora.Module("units_example")
component = source.model("Speed")
speed = component.field(
    "speed", role=eqiora.FieldRole.Variable,
    value_type=eqiora.ValueType.real(eqiora.Dimension(length=1, time=-1)),
)
component.relation("given", q.equation(speed, q.quantity(3, u.m / u.s)))
print(source.to_eqi())
model = eqiora.compile(source=source)
```

`Dimension` declares the field's physical type; `quantity` attaches an input
unit to a value. The equation specifies 3 metres per second. Replacing the
right-hand unit with `u.s` produces a dimensional error during compilation.
''',
    "geometry": '''## Example: name a rectangle and its edges

```python
from eqiora import geometry as g

graph = g.GeometryGraph()
rectangle = graph.rectangle(x_bounds=(0.0, 2.0), y_bounds=(0.0, 1.0))
geometry = graph.build(rectangle, named_topology={
    "body": rectangle.region,
    "left": rectangle.boundaries[0],
    "right": rectangle.boundaries[1],
    "walls": rectangle.boundaries[2:],
})
print(geometry.selection_dimension("body"))
print(geometry.selection_dimension("left"))
```

Coordinates are in metres. `body` has dimension 2; `left` has dimension 1.
The boundary order is left, right, bottom, top. Bind these selections to model
supports, or continue with [meshing](/reference/python/meshing/#example-mesh-the-rectangle).
''',
    "meshing": '''## Example: mesh the rectangle

Continue with `geometry` from the [geometry example](/reference/python/geometry/):

```python
from eqiora import meshing

mesh_plan = meshing.resolve(geometry, meshing.CartesianMesher(cells=(8, 4)))
mesh = meshing.generate(mesh_plan)
print(mesh.dimension, mesh.cell_count, mesh.vertex_count)
print(mesh.coordinates.shape)
```

The rectangle becomes 32 cells with 45 vertices in two dimensions.
`coordinates` contains the vertex positions. Use this same mesh when resolving
a spatial model bound to this geometry; see [modeling](/guides/modeling/).
''',
    "fem": '''## Example: select element spaces

```python
from eqiora import fem

scalar_method = fem.Q1()
flow_method = fem.MiniP1()
print(scalar_method.space)
print(flow_method.velocity_space, flow_method.pressure_space)
```

`Q1` selects continuous tensor-product elements for Cartesian scalar and
elasticity problems. `MiniP1` selects the mixed velocity/pressure spaces for
triangular Stokes problems. Pass the matching policy as `spatial=` to
`eqiora.resolve`; complete setups are in [modeling](/guides/modeling/).
''',
    "fvm": '''## Example: select a cell-centred method

```python
from eqiora import fvm

scalar_method = fvm.CellCenteredTpfa()
flow_method = fvm.CellCentered()
print(scalar_method.space)
print(flow_method.velocity_space, flow_method.pressure_space)
```

TPFA uses two-point face fluxes on orthogonal meshes. The flow method stores
velocity and pressure at cell centres. Supply the appropriate policy through
`eqiora.resolve(..., spatial=...)`; see [modeling](/guides/modeling/) for the
model, geometry, and boundary conditions that accompany each method.
''',
    "formulation": '''## Example: choose a conservative form

With `model` and `mesh` from a scalar elliptic setup in
[modeling](/guides/modeling/):

```python
import eqiora

plan = eqiora.resolve(
    model, mesh=mesh,
    spatial=eqiora.fvm.CellCenteredTpfa(),
    formulation=eqiora.formulation.IntegralConservative,
)
```

This pairs a conservative mathematical form with a finite-volume policy.
Use `PrimalGalerkin` with scalar Q1 finite elements and `MixedGalerkin` with
mixed velocity/pressure elements. The formulation and spatial method must fit
the model together.
''',
    "solve": '''## Example: configure linear and nonlinear tolerances

```python
from eqiora import solve

linear = solve.Linear(objective=solve.Robust, relative_tolerance=1e-10, absolute_tolerance=1e-12,
                      maximum_iterations=1000)
nonlinear = solve.Newton(linear=linear, absolute_tolerance=1e-10,
                         maximum_iterations=20)
print(nonlinear.maximum_iterations)
```

Pass `linear` or `nonlinear` as `solve=` in `eqiora.resolve`. Newton uses the
nested linear policy for each update. A smaller tolerance requests a more
accurate algebraic solve; mesh refinement and time-step accuracy are separate
choices. See [modeling](/guides/modeling/) for full solver setups.
''',
    "time": '''## Example: control ODE error

Use `model` and `x` from the [decay example](/reference/python/#compile-and-run):

```python
import eqiora

method = eqiora.time.Tsitouras45(
    initial_step_s=0.01,
    relative_tolerance=1e-9,
    absolute_tolerances={x: 1e-11},
)
plan = eqiora.resolve(model, temporal=method)
```

The initial step is in seconds. Each absolute tolerance is bound to an exact
Model field and expressed in that field's SI units; `x` is dimensionless here.
`output_times_s` in `run` chooses observation times independently of the
adaptive internal steps.
''',
    "viewer": '''## Example: view a geometry and mesh

In a notebook, continue with `geometry` and `mesh` from the
[meshing example](/reference/python/meshing/):

```python
from eqiora.viewer import View

view = View().add(geometry).add(mesh)
view.show()
```

The scene overlays the exact geometry and its mesh. Use `view.close()` when
you no longer need it. Adding a field output changes the display without
changing the model or numerical result. For Colab, first follow
[notebook preparation](/reference/python/colab/).
''',
    "colab": '''## Example: prepare the notebook viewer

After installing Eqiora in a Google Colab runtime:

```python
from eqiora.colab import prepare

prepare()
```

Run this before [showing a View](/reference/python/viewer/). It checks the
runtime and sets up the notebook transport. In a local Jupyter notebook,
use `View` directly.
''',
}

# Specialized examples explicitly continue a complete setup in the guide.
EXAMPLES.update({
    "fluid": '''## Example: choose reference scales

```python
from eqiora.fluid import IncompressibleScaling

scaling = IncompressibleScaling(length_m=0.1, velocity_m_per_s=1.0)
print(scaling.length_m, scaling.velocity_m_per_s)
```

For the exact-cylinder flow setup in [modeling](/guides/modeling/), pass this
as `scaling=` to `eqiora.resolve`. These reference values scale the numerical
system; inlet speed, viscosity, and other physical values belong in the model.
''',
    "solid": '''## Example: read forces after an elastic solve

Continue with the linear-elasticity `result` from
[modeling](/guides/modeling/):

```python
from eqiora.solid import linear_elasticity_evidence

forces = linear_elasticity_evidence(result)
print(forces.integrated_body_force)
print(forces.constrained_reaction)
```

Each pair gives the x and y components. For a static body loaded only by a
body force, the constrained reaction balances the integrated body force.
Keep the model's two-dimensional force convention when interpreting them.
''',
    "fsi": '''## Example: inspect a coupled result

Run the fixed-reference fluid–structure example in [modeling](/guides/modeling/)
to create `result`, then inspect its coupled output:

```python
from eqiora.fsi import evidence

coupling = evidence(result)
for state in coupling.states:
    print(state.next_kinetic_energy_j_per_m, state.next_elastic_energy_j_per_m)
```

Each row reports kinetic and elastic energy per unit depth, in joules per
metre, for one coupled state. Use the original result to plot displacement
and flow on their respective domains.
''',
    "trajectory": '''## Example: inspect spatial observations

A transient spatial run from [modeling](/guides/modeling/) returns a Result
whose `trajectory` carries spatial observations:

```python
trajectory = result.trajectory
for state in trajectory.states:
    print(state.time_s)
print(trajectory.coordinates.shape, trajectory.cells.shape)
```

The times are in seconds; the array shapes describe vertices and cells.
Keep the trajectory with the Model and its field references. It provides the
spatial data used by [Matplotlib](/reference/python/matplotlib/) and derived
field operations below; an ODE's scalar time series is read with `result.series`.
''',
    "matplotlib": '''## Example: save a spatial field plot

Run `uv add "eqiora[matplotlib]"` in the project environment described in
[Get started](/get-started/). Continue with a scalar spatial `result` and its
Model field `field` from [modeling](/guides/modeling/):

```python
from eqiora.matplotlib import plot_scalar_field

figure = plot_scalar_field(result, field=field)
figure.savefig("field.png", dpi=150)
```

The color scale shows the field's values on its mesh. For a displacement
field, use `plot_deformed_field(result, field=field, scale=10.0)`; `scale`
magnifies the display without changing the computed displacement.
''',
    "diff": '''## Example: evaluate a parameter derivative

Continue with the scalar Poisson `model` and `plan` from
[the differentiation guide](/guides/differentiation/):

```python
import numpy as np
import eqiora.diff

program = eqiora.diff.compile(
    plan, inputs=(model.parameter("source"),),
    output=plan.capability.fields[0],
)
evaluation = program.evaluate(np.array([1.5], dtype=np.float64))
values = evaluation.primal()
directional_derivative = evaluation.jvp(np.array([1.0], dtype=np.float64))
```

`values` holds the solution at source value 1.5. The JVP gives its change per
unit increase of that parameter. A VJP instead propagates output weights back
to parameter coordinates. The original model and plan remain unchanged.
''',
    "torch": '''## Example: differentiate a scalar objective

Install the `torch` extra, then create `program` with the
[differentiation example](/reference/python/diff/):

```python
import torch
from eqiora.torch import bind

operator = bind(program)
parameters = torch.tensor([1.5], dtype=torch.float64, requires_grad=True)
values = operator(parameters)
values.square().sum().backward()
print(parameters.grad)
```

The gradient is the derivative of the sum of squared output values with
respect to the selected parameter. Inputs are contiguous, one-dimensional
CPU `float64` tensors. See [framework adapters](/guides/differentiation/) for
installation and compiled execution.
''',
    "jax": '''## Example: differentiate a scalar objective

Install the `jax` extra, then create `program` with the
[differentiation example](/reference/python/diff/):

```python
import jax
import jax.numpy as jnp
from eqiora.jax import bind

jax.config.update("jax_enable_x64", True)
operator = bind(program)
parameters = jnp.array([1.5], dtype=jnp.float64)
gradient = jax.grad(lambda p: jnp.square(operator(p)).sum())(parameters)
print(gradient)
```

This differentiates the sum of squared output values. Keep the parameter
array on the CPU with `float64` enabled. See
[framework adapters](/guides/differentiation/) for JIT and batching examples.
''',
})


def module_example(slug: str) -> list[str]:
    return [EXAMPLES[slug].strip(), "", "## API", ""]
