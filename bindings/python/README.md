# Eqiora

Define mathematical models, run them from Python, and inspect their results.
Eqiora provides equation authoring, synchronous and asynchronous execution,
NumPy and DLPack arrays, first-order differentiation with PyTorch and JAX,
and optional Matplotlib plots and notebook views.

**Alpha — `0.1.0a11`.** See [Capabilities](https://eqiora.org/capabilities/)
for available models, methods, and platforms.

## Install

Eqiora `0.1.0a11` supports ordinary-GIL CPython 3.11–3.14 on
manylinux x86-64:

```console
uv venv --python 3.13 .venv
uv pip install --python .venv/bin/python eqiora==0.1.0a11
```

Automatic exact-cylinder meshing requires Gmsh 4.15.2. The conventional Linux
installation is:

```console
sudo apt-get install libglu1-mesa
uv pip install --python .venv/bin/python "eqiora[gmsh]==0.1.0a11"
```

The Gmsh extra is separate so the base `manylinux_2_17` package keeps its
compatibility floor; the current Gmsh wheel has a newer Linux floor.

Install plotting, notebook viewing, or first-order framework adapters as needed:

```console
uv pip install --python .venv/bin/python "eqiora[torch]==0.1.0a11"
uv pip install --python .venv/bin/python "eqiora[jax]==0.1.0a11"
uv pip install --python .venv/bin/python "eqiora[matplotlib]==0.1.0a11"
uv pip install --python .venv/bin/python "eqiora[viewer]==0.1.0a11"
```

The exact-cylinder pressure example combines the mesher and plot adapter:
`uv pip install --python .venv/bin/python "eqiora[gmsh,matplotlib]==0.1.0a11"`.

Run scripts with `uv run --no-project --python .venv/bin/python your_script.py`
to use this environment explicitly. Current-source features described in the
development guides may not yet be included in this published release.

The base package imports none of these optional libraries. The viewer extra
pins `anywidget==0.11.0`; its JavaScript and CSS are already carried inside the
Eqiora wheel, so the host does not fetch renderer assets at display time. The PyTorch extra
declares `torch>=2.13,<2.14`; the tested version is PyTorch 2.13.0.
JAX/JAXLIB 0.11.0 and Matplotlib 3.11.1 were also tested on CPython 3.13.
The JAX extra requires Python 3.12 or newer.

## Run a model

Start with [Get started](https://eqiora.org/get-started/) for a complete decay
example. The guides follow the current source revision; use their source-install
instructions when trying features newer than `0.1.0a11`.

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

For cylinder flow, `GeometryGraph` subtracts a circle from a rectangle.
Coordinates and radii use metres. Gmsh generates linear triangles for this
2D geometry; missing Gmsh, an incompatible version, or invalid output raises
an error. The mesh target size controls meshing and does not guarantee every
edge length.

## Read and plot results

Select fields from the Result using the field handles supplied by the Plan.
Field values and mesh coordinates use matching order. Check the units and
compare the values with a prediction before interpreting a plot.

For example, after running the steady cylinder model:

```python
import eqiora.matplotlib as eqplot

pressure = result.snapshots[0]
figure = eqplot.plot_scalar_field(result, field=pressure.field)
figure.savefig("exact-cylinder-pressure.png")
```

The pressure scale is in pascals. The plotting helper supports scalar vertex
and cell fields. The [Gallery](https://eqiora.org/gallery/) explains how to
read pressure, displacement, and transient vorticity plots from worked examples.

In a notebook with the viewer extra, display geometry, mesh, and a scalar field:

```python
view = eqiora.View().add(geometry).add(mesh).add(pressure)
view.show()
view.close()  # Release the widget when finished.
```

The viewer supports planar geometry, 2D triangular or quadrilateral meshes,
named edge/face selections, and scalar vertex/cell fields. A field must use
the same mesh displayed in the view. Without the viewer extra or a rich
notebook host, the view provides a text representation.

## Structured diagnostics

Failures expose stable categories and structured diagnostics:

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
failures have distinct subclasses. Ordinary Python call-shape errors remain
`TypeError`.

## NumPy arrays and copies

Eqiora `Array` values own dense, rank-one CPU `float64` storage:

```python
array = result["state"].values
view = array.numpy(copy=False)
writable = array.numpy(copy=True)

assert not view.flags.writeable
assert writable.flags.writeable
```

`copy=False` and `copy=None` return the same lifetime-safe, read-only NumPy
projection. If that contract cannot be honored, Eqiora fails instead of
copying silently. `copy=True` returns an independent writable allocation.
DLPack exports copy data into independent versioned CPU snapshots.
Differentiable-program DLPack inputs are checked for dtype, shape, alignment,
byte order, and contiguity, then copied before execution. Read the details in
[Execution, diagnostics, and arrays](https://eqiora.org/guides/execution-and-arrays/).

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

Cancelling the surrounding asyncio task or dropping a `Run` does not
implicitly cancel native work. Call `run.cancel()` explicitly. Cancellation
is cooperative: it takes effect when execution reaches a cancellation point.
A cancelled run does not publish a partial Result.

Pass `profile=True` to `run` or `submit` when investigating runtime cost. The
returned `result.profile` provides a hierarchical timing summary and structured
phase/solver events. Profiling is process-local telemetry and is omitted from
persisted Result artifacts.

## PyTorch and JAX

Compile a differentiable program by selecting the parameters and output field
of a Plan, then bind it with `eqiora.torch.bind(program)` or
`eqiora.jax.bind(program)`. PyTorch uses the vector–Jacobian product during
backpropagation; JAX provides primal evaluation and first-order JVP/VJP.

These adapters use real, rank-one CPU `float64` inputs and rectangular 2D
Cartesian meshes with scalar-elliptic Q1 FEM or TPFA FVM. Enable JAX 64-bit
mode with `jax.config.update("jax_enable_x64", True)`. Device transfers must
be explicit, and differentiation is first order.

Follow [Differentiation and framework adapters](https://eqiora.org/guides/differentiation/)
for complete setup, input shapes, and framework examples.

## Compatibility

`0.1.0a11` is an alpha prerelease. Python APIs and saved-file formats may change
before 1.0; release notes describe changes and migrations. Corrections to a
published package receive a new version.

## Links

- [Documentation](https://eqiora.org)
- [Guides](https://eqiora.org/guides/)
- [Reference](https://eqiora.org/reference/)
- [Source](https://github.com/nkiyohara/eqiora)
- [Issue tracker](https://github.com/nkiyohara/eqiora/issues)
- [Security policy](https://github.com/nkiyohara/eqiora/security/policy)
- [Apache-2.0 license](https://github.com/nkiyohara/eqiora/blob/main/LICENSE)
