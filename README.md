<p align="center">
  <a href="https://eqiora.org"><img src="docs/site/src/assets/brand/eqiora-mark.svg" width="88" height="88" alt="Eqiora"></a>
</p>

<h1 align="center">Eqiora</h1>

<p align="center">
  <strong>Any physics. One language.</strong><br>
  Model and couple physical systems with readable mathematics—and the freedom to choose or build your own numerical methods.<br>
  An open-source computational physics platform. Python for exploration. Rust at the core.
</p>

<p align="center">
  <a href="https://pypi.org/project/eqiora/"><img src="https://img.shields.io/pypi/v/eqiora?include_prereleases&amp;style=flat-square&amp;logo=pypi&amp;logoColor=white" alt="PyPI version"></a>
  <a href="https://docs.rs/eqiora/0.1.0-alpha.13/eqiora/"><img src="https://img.shields.io/crates/v/eqiora?style=flat-square&amp;logo=rust" alt="crates.io version"></a>
  <a href="https://github.com/nkiyohara/eqiora/releases"><img src="https://img.shields.io/github/v/release/nkiyohara/eqiora?include_prereleases&amp;sort=semver&amp;style=flat-square&amp;logo=github" alt="Latest release including alphas"></a>
  <a href="https://pypi.org/project/eqiora/"><img src="https://img.shields.io/pypi/pyversions/eqiora?style=flat-square&amp;logo=python&amp;logoColor=white" alt="Supported Python versions"></a>
  <a href="docs/rust-api.md"><img src="https://img.shields.io/crates/msrv/eqiora?style=flat-square&amp;logo=rust&amp;label=Rust" alt="Minimum Rust version"></a>
</p>

<p align="center">
  <a href="https://github.com/nkiyohara/eqiora/actions/workflows/ci.yml"><img src="https://img.shields.io/github/actions/workflow/status/nkiyohara/eqiora/ci.yml?event=pull_request&amp;style=flat-square&amp;label=PR%20CI&amp;logo=githubactions&amp;logoColor=white" alt="Pull request CI"></a>
  <a href="https://github.com/nkiyohara/eqiora/actions/workflows/pages.yml"><img src="https://img.shields.io/github/actions/workflow/status/nkiyohara/eqiora/pages.yml?branch=main&amp;event=push&amp;style=flat-square&amp;label=docs%20build&amp;logo=githubactions&amp;logoColor=white" alt="Documentation build"></a>
  <a href="https://eqiora.org"><img src="https://img.shields.io/badge/docs-eqiora.org-17417e?style=flat-square" alt="Documentation at eqiora.org"></a>
  <a href="https://eqiora.org/capabilities/"><img src="https://img.shields.io/badge/status-alpha-orange?style=flat-square" alt="Development status: alpha"></a>
  <a href="LICENSE"><img src="https://img.shields.io/pypi/l/eqiora?style=flat-square" alt="Apache License 2.0"></a>
</p>

<p align="center">
  <a href="#-get-started-with-uv"><strong>Get started</strong></a> ·
  <a href="https://eqiora.org/gallery/"><strong>Explore simulations</strong></a> ·
  <a href="https://eqiora.org/reference/">API reference</a> ·
  <a href="docs/roadmap.md">Roadmap</a> ·
  <a href="CONTRIBUTING.md">Contribute</a>
</p>

---

## 🔬 See the physics

**Pressure around a circular obstacle.** This steady-Stokes example starts with
an exact geometry and a mathematical model, then produces a pressure field,
boundary forces, and fluxes with Python.

[![Fine-mesh exact-cylinder steady-Stokes pressure field](docs/site/src/assets/gallery/exact-cylinder-pressure-presentation.png)](https://eqiora.org/gallery/exact-cylinder-steady-stokes/)

| Explore | Inside the walkthrough |
| --- | --- |
| 🌊 [Flow past a cylinder](https://eqiora.org/gallery/exact-cylinder-steady-stokes/) | Exact geometry, Gmsh meshing, steady Stokes, pressure and boundary observables. |
| 🧱 [Linear elasticity](https://eqiora.org/gallery/mixed-boundary-elasticity/) | A constrained solid, mixed boundary conditions, and a displacement field. |
| 🎞️ [Transient flow startup](https://eqiora.org/gallery/transient-cylinder-startup/) | A ten-step startup demonstration with vorticity and force outputs. |

## ✨ Why Eqiora?

Eqiora is being built around a simple kind of freedom: describe the physics you
care about, combine the models you need, and use the numerical methods that fit
the problem.

- **🌐 One language across physics.** Bring fields, equations, physical
  connections, continuous dynamics, and discrete events into a common
  mathematical model.
- **🔗 Coupling belongs in the model.** Make interactions explicit, from
  connected components to strongly coupled systems whose unknowns must be
  solved together.
- **🛠️ Space for your own numerical methods.** Separate what the equations
  mean from how they are solved—the foundation for changing discretizations,
  integrating a custom solver, or developing a new method.
- **🧮 Mathematics you can read.** Express quantities, units, equations, and
  boundary conditions in Eqiora's domain-specific language (`.eqi`). Keep the
  assumptions visible to the people who read, review, and extend a model.
- **🧩 Models that grow with your work.** Reuse components and constitutive
  laws, compose them into larger systems, and replace individual parts as your
  research or application evolves.
- **🐍 A natural home for computational experiments.** Use Python to build
  geometry, mesh, run simulations, inspect NumPy field data, and make plots.
  Connect the same workflow to your experiment scripts and analysis tools.
- **🦀 Native execution, one shared core.** Python and Rust applications share
  the same mathematical model and execution foundations. Numerical backends
  live alongside that model, keeping physical definitions separate from
  implementation choices.
- **🔎 Calculations you can inspect.** Type and dimension checks catch model
  inconsistencies; explicit plans and typed outputs connect each result to its
  equations, geometry, mesh, and solver settings. Follow examples into their
  source and checks.

## 🚀 Get started with uv

With [uv](https://docs.astral.sh/uv/getting-started/installation/) installed,
create a project with meshing and plotting support:

```console
uv init --python ">=3.11,<3.15" eqiora-demo
cd eqiora-demo
uv add "eqiora[gmsh,matplotlib]==0.1.0a13"
```

The published wheels support **ordinary-GIL CPython 3.11–3.14 on Linux x86-64
(manylinux)**. Gmsh also needs the system OpenGL runtime; see the
[installation guide](https://eqiora.org/get-started/) for setup details.

Save the example below as `cylinder.py`, then run:

```console
uv run cylinder.py
```

It prints pressure, cylinder force, and net flux, then saves `pressure.png`.

<details>
<summary><strong>🐍 Show the complete cylinder-flow example</strong></summary>

```python
from importlib.resources import files

import eqiora
import eqiora.matplotlib as eqplot

graph = eqiora.geometry.GeometryGraph()
rectangle = graph.rectangle(x_bounds=(0.0, 2.2), y_bounds=(0.0, 0.41))
circle = graph.circle(center=(0.2, 0.2), radius=0.05)
fluid = graph.subtract(rectangle, circle)
geometry = graph.build(fluid, named_topology={
    "fluid": fluid.region,
    "inlet": rectangle.boundaries[0],
    "outlet": rectangle.boundaries[1],
    "walls": rectangle.boundaries[2:4],
    "cylinder": circle.boundaries[0],
})
mesh_request = eqiora.meshing.GmshMesher(
    maximum_boundary_error=1e-4,
    maximum_target_size=0.025,
    minimum_mean_ratio=1e-5,
    maximum_boundary_facets=50,
)
mesh_plan = eqiora.meshing.resolve(geometry, mesh_request)
mesh = eqiora.meshing.generate(mesh_plan)

model = eqiora.compile(
    path=files(eqiora).joinpath("examples", "steady-flow-past-cylinder.eqi"),
    geometry=geometry,
    parameters={
        "dynamic_viscosity": 1.0e-3,
        "zero_pressure": 0.0,
        "inlet_speed": 0.3,
        "channel_height": geometry.bounds[1][1] - geometry.bounds[1][0],
    },
)
linear = eqiora.solve.Linear(
    relative_tolerance=1e-6,
    absolute_tolerance=1e-13,
    maximum_iterations=10_000,
)
plan = eqiora.resolve(
    model, mesh=mesh, spatial=eqiora.fem.MiniP1(), solve=linear, scaling=None,
)
result = eqiora.run(plan)
pressure = result.output(plan.capability.pressure)
pressure_values = pressure.values("vertex")
cylinder_force = result.boundary_force(geometry.selection("cylinder"))
inlet_flux = result.boundary_flux(geometry.selection("inlet"))
outlet_flux = result.boundary_flux(geometry.selection("outlet"))

print(result.solve)
print("pressure", min(pressure_values), max(pressure_values), "Pa")
print("cylinder force on fluid", cylinder_force.on_domain, "N/m")
print("net flux", inlet_flux.value + outlet_flux.value, "m^2/s")

figure = eqplot.plot_scalar_field(result, field=plan.capability.pressure)
figure.savefig("pressure.png", dpi=180)
```

</details>

Read the [step-by-step walkthrough](https://eqiora.org/gallery/exact-cylinder-steady-stokes/)
alongside the [full example script](examples/python/exact_cylinder_stokes.py).

## 🧩 From a model to a result

The model describes the mathematics. A resolved **Plan** records the numerical
choices. A **Result** carries the outputs and diagnostics of that run.

```text
Equations + Geometry   →   Model   →   Plan   →   Result
                          compile     resolve    run
                                      ↑
                              Mesh · Method · Solver
```

This separation keeps the physical model readable while you experiment with
discretizations and solver policies. Fields and diagnostics remain connected to
the choices that produced them.
Explore the [architecture](docs/architecture.md) for how the pieces fit.

## 🦀 Use from Rust

Add the published facade to a Cargo project:

```console
cargo add eqiora@=0.1.0-alpha.13
```

Start with the [Rust guide](docs/rust-api.md) for model compilation and optional
backends, or browse the [API docs](https://docs.rs/eqiora/0.1.0-alpha.13/eqiora/).

## 🛠️ More ways to work

| Interface | What to reach for |
| --- | --- |
| 🐍 **Python** | The primary simulation API: author, compile, mesh, solve, and plot. [Reference →](https://eqiora.org/reference/python/) |
| 🦀 **Rust** | Embed Eqiora through the `eqiora` Rust crate. [Guide →](docs/rust-api.md) |
| ⌨️ **CLI & MCP** | Check a local `.eqi` file with `eqiora check`, or connect agents to the compile/check tool in `eqiora-mcp`. [Build the tools →](docs/rust-api.md#build-command-line-tools-from-this-checkout) |
| 📝 **Editor preview** | Diagnostics, formatting, hover, and cross-module navigation through LSP. Currently installed from a source checkout. [Setup →](docs/language-server.md) |

## 🌱 Growing in the open

Eqiora is **alpha research software**. The current release covers focused paths
through hybrid execution, scalar FEM/FVM, fluid flow, elasticity, implicit
differentiation, and selected CPU/CUDA/MPI adapters. Consult the
[capability matrix](docs/capability-matrix.md) for each method and environment,
and the [benchmarks](docs/benchmarks.md) for reproduced results.

Pre-1.0 APIs evolve without compatibility shims. Eqiora is not certified for
safety-critical or production engineering decisions.

## 🤝 Build with us

Useful bug reports, clearer examples, numerical methods, and improvements to the
developer experience are all welcome. Start with the [contributing guide](CONTRIBUTING.md),
explore the [roadmap](docs/roadmap.md), or [join an issue discussion](https://github.com/nkiyohara/eqiora/issues).

Eqiora is developed in public under the
[Apache License 2.0](LICENSE), with
[DCO sign-off](CONTRIBUTING.md#developer-certificate-of-origin) on contributions.

[Security](SECURITY.md) · [Governance](GOVERNANCE.md) ·
[Release policy](docs/development/python-release-policy.md)
