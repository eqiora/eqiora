#!/usr/bin/env python3
"""Run public quick starts against an installed Eqiora distribution."""

from __future__ import annotations

import argparse
import importlib.metadata
import sys


POISSON = """
public component ReleaseSmokePoisson(
  support square: volume(ambient_dimension = 2),
  support x_lower: boundary(parent = square),
  support x_upper: boundary(parent = square),
  support y_lower: boundary(parent = square),
  support y_upper: boundary(parent = square),
  parameter diffusion: 1,
  parameter wave_number: 1 / m,
  parameter source_scale: 1 / m ^ 2,
  parameter boundary_offset: 1,
) {
  variable potential: 1 on square;
  relation balance on square {
    -div(diffusion * grad(potential))
      - source_scale * math.sin(wave_number * coordinate(0))
        * math.sin(wave_number * coordinate(1)) = 0;
  }
  relation x_lower_value on x_lower { trace(potential) - boundary_offset = 0; }
  relation x_upper_value on x_upper { trace(potential) - boundary_offset = 0; }
  relation y_lower_value on y_lower { trace(potential) - boundary_offset = 0; }
  relation y_upper_value on y_upper { trace(potential) - boundary_offset = 0; }
}
"""

DECAY = """
model decay(parameter rate: 1 / s = 1) {
  state x: 1;
  initial { x = 1; }
  relation flow {
    derivative(x) + rate * x = 0;
  }
}
"""


def decay_model(eqiora):
    return eqiora.compile(source=DECAY, filename="release-smoke-decay.eqi")


def decay_plan(eqiora, model):
    field = model.field(model.field_ids[0])
    plan = eqiora.resolve(
        model,
        temporal=eqiora.time.Tsitouras45(
            initial_step_s=0.01,
            relative_tolerance=1.0e-9,
            absolute_tolerances={field: 1.0e-11},
        ),
    )
    return plan, field


def rectangle_scene(eqiora):
    graph = eqiora.geometry.GeometryGraph()
    rectangle = graph.rectangle(x_bounds=(0.0, 1.0), y_bounds=(0.0, 1.0))
    geometry = graph.build(
        rectangle,
        named_topology={
            "square": rectangle.region,
            "x_lower": rectangle.boundaries[0],
            "x_upper": rectangle.boundaries[1],
            "y_lower": rectangle.boundaries[2],
            "y_upper": rectangle.boundaries[3],
        },
    )
    mesh_plan = eqiora.meshing.resolve(
        geometry, eqiora.meshing.CartesianMesher(cells=(4, 4))
    )
    mesh = eqiora.meshing.generate(mesh_plan)
    return geometry, mesh


def differentiable_program(eqiora):
    geometry, mesh = rectangle_scene(eqiora)
    model = eqiora.compile(
        source=POISSON,
        entry="ReleaseSmokePoisson",
        geometry=geometry,
        bindings={
            "square": geometry.selection("square"),
            **{
                name: (geometry.selection(name), geometry.selection("square"))
                for name in ("x_lower", "x_upper", "y_lower", "y_upper")
            },
            "diffusion": 1.0,
            "wave_number": 3.141592653589793,
            "source_scale": 19.739208802178716,
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
    return eqiora.diff.compile(
        plan,
        inputs=(
            model.parameter("source_scale"),
            model.parameter("diffusion"),
            model.parameter("boundary_offset"),
        ),
        output=plan.capability.fields[0],
    )


def base_smoke(expected_version: str) -> None:
    before = set(sys.modules)
    import eqiora

    assert importlib.metadata.version("eqiora") == expected_version
    assert eqiora.__version__ == expected_version
    assert not (
        {
            "torch",
            "jax",
            "jaxlib",
            "matplotlib",
            "gmsh",
            "anywidget",
            "ipywidgets",
            "traitlets",
        }
        & (set(sys.modules) - before)
    )

    geometry, mesh = rectangle_scene(eqiora)
    view = eqiora.View().add(geometry).add(mesh)
    bundle = view._repr_mimebundle_()
    data = bundle[0] if isinstance(bundle, tuple) else bundle
    assert "application/vnd.jupyter.widget-view+json" in data
    view.close()

    model = decay_model(eqiora)
    plan, field = decay_plan(eqiora, model)
    result = eqiora.run(
        plan,
        state=eqiora.State.initial(plan),
        until_s=1.0,
        output_times_s=(1.0,),
    )
    series = result.series(field)
    time = series.time.numpy(copy=False)
    values = series.values.numpy(copy=False)
    assert time.shape == values.shape
    assert time[-1] == 1.0
    assert 0.0 < values[-1] < 1.0
    assert not values.flags.writeable
    owned = series.values.numpy(copy=True)
    assert owned.flags.writeable
    assert owned.base is None

    try:
        eqiora.run(
            plan,
            state=eqiora.State.initial(plan),
            until_s=-1.0,
            output_times_s=(-1.0,),
        )
    except eqiora.EqioraError as error:
        assert error.category
        assert error.diagnostics
        assert all(diagnostic.code for diagnostic in error.diagnostics)
    else:
        raise AssertionError("invalid execution should expose structured diagnostics")


def torch_smoke(expected_version: str) -> None:
    base_smoke(expected_version)
    import eqiora
    import eqiora.torch as eqtorch
    import torch

    solve = eqtorch.bind(differentiable_program(eqiora))
    point = torch.tensor([17.0, 1.2, 0.1], dtype=torch.float64, requires_grad=True)
    output = solve(point)
    output.square().sum().backward()
    assert output.device.type == "cpu"
    assert point.grad is not None
    assert torch.isfinite(point.grad).all()


def jax_smoke(expected_version: str) -> None:
    base_smoke(expected_version)
    import eqiora
    import eqiora.jax as eqjax
    import jax
    import jax.numpy as jnp

    jax.config.update("jax_enable_x64", True)
    solve = eqjax.bind(differentiable_program(eqiora))
    point = jnp.array([17.0, 1.2, 0.1], dtype=jnp.float64)
    output = jax.jit(solve)(point)
    gradient = jax.grad(lambda value: jnp.sum(solve(value) ** 2))(point)
    output.block_until_ready()
    gradient.block_until_ready()
    assert output.devices() == {jax.devices("cpu")[0]}
    assert bool(jnp.all(jnp.isfinite(gradient)))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--expected-version", required=True)
    parser.add_argument("--profile", choices=("base", "torch", "jax"), required=True)
    arguments = parser.parse_args()
    {"base": base_smoke, "torch": torch_smoke, "jax": jax_smoke}[arguments.profile](
        arguments.expected_version
    )
    print(
        f"installed Eqiora {arguments.expected_version} {arguments.profile} smoke passed"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
