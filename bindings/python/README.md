# Eqiora

Define mathematical models, run them from Python, and inspect the results.
Eqiora lets you write equations, run them synchronously or asynchronously, read
results as NumPy or DLPack arrays, differentiate them with PyTorch or JAX, and
plot them with Matplotlib or in a notebook viewer.

This is release `0.1.3`. The [Gallery](https://eqiora.org/gallery/) has complete
worked examples.

## Install

Eqiora `0.1.3` supports ordinary-GIL CPython 3.11–3.14 on
manylinux x86-64:

```console
uv init --python 3.13 first-eqiora-model
cd first-eqiora-model
uv add eqiora
```

In an existing uv project, just run `uv add eqiora`. A matching prebuilt wheel
needs no Rust compiler. To select this release explicitly, use `uv add eqiora==0.1.3`.

Automatic meshing, used by the cylinder examples, requires Gmsh 4.15.2. On
Linux, install it with:

```console
sudo apt-get install libglu1-mesa
uv add "eqiora[gmsh]"
```

Gmsh is an optional extra because its wheel needs a newer Linux than the base
`manylinux_2_17` package.

Install plotting or first-order framework adapters as needed:

```console
uv add "eqiora[torch]"
uv add "eqiora[jax]"
uv add "eqiora[matplotlib]"
```

The cylinder pressure example needs both: `uv add "eqiora[gmsh,matplotlib]"`.

Run scripts with `uv run python your_script.py`; uv manages and locks the
project environment. Some features in the development guides may be newer than
this release.

The base package pins `anywidget==0.11.0` and bundles the viewer's JavaScript
and CSS, so notebook views work without downloading anything at display time.
Importing `eqiora` does not load anywidget, ipywidgets, or traitlets. The
PyTorch extra requires `torch>=2.14,<2.15` and is tested with PyTorch 2.14.0.
The release candidate is tested with JAX/JAXLIB 0.11.0 and Matplotlib 3.11.1 on
CPython 3.13. The JAX extra requires Python 3.12 or newer.

## Run a model

[Get started](https://eqiora.org/get-started/) walks through a complete decay
example with the published package. Lessons that need features newer than
`0.1.3` say so and link to the [development-build instructions](https://eqiora.org/get-started/#development-build).

A spatial workflow has five steps:

1. Define geometry and name its regions and boundaries.
2. Choose a mesher and generate the mesh.
3. Compile the equations with that geometry and the model parameters.
4. Resolve the model, mesh, elements, and solver settings into a Plan.
5. Run the Plan and select the fields to inspect.

The [modeling guide](https://eqiora.org/guides/modeling/) contains complete
examples for steady cylinder flow, mixed-boundary linear elasticity, and
fixed-reference fluid–structure interaction. It also explains how to compile
and inspect locked model packages.

For cylinder flow, `GeometryGraph` subtracts a circle from a rectangle, with
coordinates and radii in metres. Gmsh fills the 2D geometry with linear
triangles and raises an error if Gmsh is missing, the wrong version, or produces
invalid output. The target size guides the mesher; individual edges can be
longer or shorter.

## Read and plot results

Select fields from the Result with the field handles from the Plan. Field
values are in the same order as the mesh coordinates. Check the units, and
compare values with a prediction before you trust a plot.

For example, after running the steady cylinder model:

```python
import eqiora.matplotlib as eqplot

pressure = result.snapshots[0]
figure = eqplot.plot_scalar_field(result, field=pressure.field)
figure.savefig("exact-cylinder-pressure.png")
```

Pressure is in pascals. The plotting helper handles scalar fields on vertices or
cells. The [Gallery](https://eqiora.org/gallery/) explains how to read the
pressure, displacement, and vorticity plots in its examples.

In a notebook, display geometry, mesh, and a scalar field:

```python
view = eqiora.View().add(geometry).add(mesh).add(pressure)
view.show()
view.close()  # Release the widget when finished.
```

The viewer shows planar geometry, 2D triangle or quadrilateral meshes, named
edge and face selections, and scalar fields on vertices or cells. A field must be
on the mesh shown in the same view. Outside a notebook, the view prints a text
summary instead.

## Structured diagnostics

Errors carry a stable category and a list of structured diagnostics:

```python
try:
    eqiora.run(
        plan,
        state=eqiora.State.initial(plan),
        until_s=-1.0,
        output_times_s=(-1.0,),
    )
except eqiora.EqioraError as error:
    print(error.category)
    for diagnostic in error.diagnostics:
        print(diagnostic.code, diagnostic.severity, diagnostic.message)
```

Validation, compatibility, capability, execution, cancellation, and internal
errors each have their own subclass. Calling a function with the wrong arguments
still raises an ordinary `TypeError`.

## NumPy arrays and copies

Eqiora `Array` values own dense, rank-one CPU `float64` storage:

```python
array = result["state"].values
view = array.numpy(copy=False)
writable = array.numpy(copy=True)

assert not view.flags.writeable
assert writable.flags.writeable
```

`copy=False` and `copy=None` both return a read-only NumPy view that stays valid
as long as you hold it; if Eqiora cannot provide one, it raises an error rather
than copying silently. `copy=True` returns an independent, writable copy. DLPack
exports are always independent copies. DLPack inputs to a differentiable program
are checked for dtype, shape, alignment, byte order, and contiguity, and copied
before execution. See
[Execution, diagnostics, and arrays](https://eqiora.org/guides/execution-and-arrays/)
for details.

## Await, progress, and cancellation

Use `eqiora.run(plan)` to wait synchronously, or submit a run and await it:

```python
async def simulate(plan):
    run = eqiora.submit(
        plan,
        state=eqiora.State.initial(plan),
        until_s=10.0,
        output_times_s=(10.0,),
    )
    try:
        print(run.status, run.progress)
        return await run
    finally:
        if not run.done:
            run.cancel()
```

Cancelling the surrounding asyncio task or dropping a `Run` does not stop the
native work; call `run.cancel()`. Cancellation takes effect at the next point
where execution checks for it, and a cancelled run returns no partial Result.

To see where time goes, pass `profile=True` to `run` or `submit`.
`result.profile` then holds a hierarchical timing summary with phase and solver
events. Profiles cover the current process only and are not saved with Results.

## PyTorch and JAX

Compile a differentiable program by choosing a Plan's parameters and output
field, then bind it with `eqiora.torch.bind(program)` or
`eqiora.jax.bind(program)`. PyTorch uses the vector–Jacobian product during
backpropagation; JAX supports evaluation and first-order JVPs and VJPs.

Both adapters support first-order differentiation on CPU with ordered
`float64`/`complex128` leaves in exact native shapes and an ordered output tuple.
PyTorch supports ordinary autograd and native mapped first-order transforms;
see the guide for compilation and higher-order boundaries. The native Program
owns solver and output admission. Registered spatial evidence covers scalar
elliptic Q1 FEM and TPFA FVM on rectangular 2D Cartesian meshes; ordinary JAX
composition tests also cover finite complex fields with real loss and constraint
outputs. Move data between devices yourself, and enable 64-bit JAX with
`jax.config.update("jax_enable_x64", True)`.

Follow [Differentiation and framework adapters](https://eqiora.org/guides/differentiation/)
for complete setup, input shapes, and framework examples.

## Compatibility

`0.1.3` is a pre-1.0 release, so Python APIs and saved-file formats may change;
the release notes describe each change and how to migrate. Fixes to a published
package are always released under a new version.

## Links

- [Documentation](https://eqiora.org)
- [Guides](https://eqiora.org/guides/)
- [Reference](https://eqiora.org/reference/)
- [Source](https://github.com/nkiyohara/eqiora)
- [Issue tracker](https://github.com/nkiyohara/eqiora/issues)
- [Security policy](https://github.com/nkiyohara/eqiora/security/policy)
- [Apache-2.0 license](https://github.com/nkiyohara/eqiora/blob/main/LICENSE)
