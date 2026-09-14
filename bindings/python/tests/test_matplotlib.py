from __future__ import annotations

import gc
import io
import os
import struct
import sys
from importlib.resources import files
from pathlib import Path

import numpy as np
import pytest

import eqiora


assert "matplotlib" not in sys.modules
matplotlib = pytest.importorskip("matplotlib")
matplotlib.use("Agg", force=True)

import matplotlib.image as image  # noqa: E402
from matplotlib.axes import Axes  # noqa: E402
from matplotlib.figure import Figure  # noqa: E402

import eqiora.matplotlib as eqplot  # noqa: E402


EXPECTED_MATPLOTLIB_VERSION = os.environ.get("EQIORA_TEST_MATPLOTLIB_VERSION")
if EXPECTED_MATPLOTLIB_VERSION is not None:
    assert matplotlib.__version__ == EXPECTED_MATPLOTLIB_VERSION


def cylinder():
    graph = eqiora.geometry.GeometryGraph()
    rectangle = graph.rectangle(x_bounds=(0.0, 2.2), y_bounds=(0.0, 0.41))
    circle = graph.circle(center=(0.2, 0.2), radius=0.05)
    fluid = graph.subtract(rectangle, circle)
    geometry = graph.build(
        fluid,
        named_topology={
            "fluid": fluid.region,
            "inlet": rectangle.boundaries[0],
            "outlet": rectangle.boundaries[1],
            "walls": rectangle.boundaries[2:],
            "cylinder": circle.boundaries[0],
        },
    )
    request = eqiora.meshing.GmshMesher(
            maximum_boundary_error=1.0e-4,
            minimum_mean_ratio=1.0e-5,
            maximum_boundary_facets=50,
        )
    mesh_plan = eqiora.meshing.resolve(geometry, request)
    mesh = eqiora.meshing.generate(mesh_plan)
    model = eqiora.compile(
        path=files(eqiora).joinpath('examples', 'steady-flow-past-cylinder.eqi'),
        geometry=geometry,
        entry='SteadyFlowPastCylinder',
        bindings={
            'fluid': geometry.selection('fluid'),
            'inlet': (geometry.selection('inlet'), geometry.selection('fluid')),
            'outlet': (geometry.selection('outlet'), geometry.selection('fluid')),
            'walls': (geometry.selection('walls'), geometry.selection('fluid')),
            'cylinder': (geometry.selection('cylinder'), geometry.selection('fluid')),
            'dynamic_viscosity': 0.001,
            'zero_pressure': 0.0,
            'inlet_speed': 0.3,
            'channel_height': geometry.bounds[1][1] - geometry.bounds[1][0],
        },
    )
    plan = eqiora.resolve(
        model,
        mesh=mesh,
        spatial=eqiora.fem.MiniP1(),
        solve=eqiora.solve.Linear(
            algorithm=eqiora.solve.LinearSolver.SparseLu,
            preconditioner=eqiora.solve.Preconditioner.Identity,
            reduction=eqiora.solve.Reduction.Fast,
            provider=eqiora.solve.SolverProvider.faer(),
            relative_tolerance=1.0e-6,
            absolute_tolerance=1.0e-13,
            maximum_iterations=10_000,
        ),
        scaling=None,
    )
    return geometry, mesh, plan, eqiora.run(plan)


def elasticity() -> tuple[eqiora.Plan, eqiora.Result]:
    graph = eqiora.geometry.GeometryGraph()
    rectangle = graph.rectangle(x_bounds=(0.0, 1.0), y_bounds=(0.0, 1.0))
    geometry = graph.build(
        rectangle,
        named_topology={
            "body": rectangle.region,
            "x_lower": rectangle.boundaries[0],
            "x_upper": rectangle.boundaries[1],
            "y_lower": rectangle.boundaries[2],
            "y_upper": rectangle.boundaries[3],
        },
    )
    request = eqiora.meshing.CartesianMesher(cells=(16, 16))
    mesh_plan = eqiora.meshing.resolve(geometry, request)
    mesh = eqiora.meshing.generate(mesh_plan)
    model = eqiora.compile(
        path=files(eqiora).joinpath('examples', 'mixed-boundary-elasticity.eqi'),
        geometry=geometry,
        entry='MixedBoundaryElasticity2d',
        bindings={
            'body': geometry.selection('body'),
            'x_lower': (geometry.selection('x_lower'), geometry.selection('body')),
            'x_upper': (geometry.selection('x_upper'), geometry.selection('body')),
            'y_lower': (geometry.selection('y_lower'), geometry.selection('body')),
            'y_upper': (geometry.selection('y_upper'), geometry.selection('body')),
            'mu': 3.0,
            'lambda': 0.0,
            'length_scale': 1.0,
        },
    )
    plan = eqiora.resolve(
        model,
        mesh=mesh,
        spatial=eqiora.fem.Q1(),
        solve=eqiora.solve.Linear(
            algorithm=eqiora.solve.LinearSolver.ConjugateGradient,
            preconditioner=eqiora.solve.Preconditioner.Identity,
            reduction=eqiora.solve.Reduction.Reproducible,
            provider=eqiora.solve.SolverProvider.reference(),
            relative_tolerance=1.0e-10,
            absolute_tolerance=1.0e-12,
            maximum_iterations=10_000,
        ),
    )
    return plan, eqiora.run(plan)


@pytest.fixture(scope="module")
def cylinder_case():
    return cylinder()


@pytest.fixture(scope="module")
def scalar(cylinder_case) -> tuple[eqiora.Plan, eqiora.Result]:
    _, _, plan, result = cylinder_case
    return plan, result


@pytest.fixture(scope="module")
def transient_vorticity(cylinder_case):
    geometry, mesh, steady_plan, steady_result = cylinder_case
    model = eqiora.compile(
        path=files(eqiora).joinpath('examples', 'transient-flow-past-cylinder.eqi'),
        geometry=geometry,
        entry='TransientFlowPastCylinder',
        bindings={
            'fluid': geometry.selection('fluid'),
            'inlet': (geometry.selection('inlet'), geometry.selection('fluid')),
            'outlet': (geometry.selection('outlet'), geometry.selection('fluid')),
            'walls': (geometry.selection('walls'), geometry.selection('fluid')),
            'cylinder': (geometry.selection('cylinder'), geometry.selection('fluid')),
            'density': 1.0,
            'dynamic_viscosity': 0.001,
            'zero_pressure': 0.0,
            'inlet_speed': 0.3,
            'channel_height': geometry.bounds[1][1] - geometry.bounds[1][0],
        },
    )
    linear = eqiora.solve.Linear(
        algorithm=eqiora.solve.LinearSolver.SparseLu,
        preconditioner=eqiora.solve.Preconditioner.Identity,
        reduction=eqiora.solve.Reduction.Fast,
        provider=eqiora.solve.SolverProvider.faer(),
        relative_tolerance=1.0e-6,
        absolute_tolerance=1.0e-9,
        maximum_iterations=20_000,
    )
    plan = eqiora.resolve(
        model,
        mesh=mesh,
        spatial=eqiora.fem.MiniP1(),
        temporal=eqiora.time.BackwardEuler(0.0001),
        solve=eqiora.solve.Newton(linear=linear),
        scaling=eqiora.fluid.IncompressibleScaling(
            length_m=0.41,
            velocity_m_per_s=0.3,
            pressure_pa=0.09,
        ),
    )
    steady_velocity = steady_result.output(steady_plan.capability.velocity)
    steady_pressure = steady_result.output(steady_plan.capability.pressure)
    state = eqiora.State.initial(
        plan,
        time_s=0.0,
        fields=(
            eqiora.InitialField(
                plan.capability.velocity,
                vertex_values=np.asarray(steady_velocity.values("vertex")).reshape(
                    mesh.vertex_count, 2
                ),
                cell_values=np.asarray(steady_velocity.values("cell-bubble")).reshape(
                    mesh.cell_count, 2
                ),
            ),
            eqiora.InitialField(
                plan.capability.pressure,
                vertex_values=np.asarray(steady_pressure.values("vertex")),
            ),
        ),
    )
    result = eqiora.run(plan, state=state, steps=1, output_steps=(1,))
    wake_state = result.trajectory.state(1)
    return (
        plan,
        result,
        wake_state.curl(plan.capability.velocity),
        state.curl(plan.capability.velocity),
    )


@pytest.fixture(scope="module")
def structural() -> tuple[eqiora.Plan, eqiora.Result]:
    return elasticity()


def test_scalar_field_uses_exact_plan_field_output(
    scalar: tuple[eqiora.Plan, eqiora.Result],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    plan, result = scalar
    field = plan.capability.pressure
    assert field is not None
    output = result.output(field)
    expected_coordinates = output.mesh.coordinates.copy()
    expected_cells = output.mesh.cells.copy()
    expected_values = output.values("vertex").numpy(copy=False).copy()
    observed: dict[str, object] = {}
    original = Axes.tripcolor
    original_colorbar = Figure.colorbar

    def capture(axes: Axes, *args: object, **kwargs: object):
        observed["x"] = np.asarray(args[0]).copy()
        observed["y"] = np.asarray(args[1]).copy()
        observed["values"] = np.asarray(args[2]).copy()
        observed["cells"] = np.asarray(kwargs["triangles"]).copy()
        return original(axes, *args, **kwargs)

    monkeypatch.setattr(Axes, "tripcolor", capture)

    def capture_colorbar(figure: Figure, *args: object, **kwargs: object):
        observed["colorbar_axes"] = kwargs["ax"]
        assert "cax" not in kwargs
        return original_colorbar(figure, *args, **kwargs)

    monkeypatch.setattr(Figure, "colorbar", capture_colorbar)
    figure = eqplot.plot_scalar_field(result, field=field)
    np.testing.assert_array_equal(observed["x"], expected_coordinates[:, 0])
    np.testing.assert_array_equal(observed["y"], expected_coordinates[:, 1])
    np.testing.assert_array_equal(observed["cells"], expected_cells)
    np.testing.assert_array_equal(observed["values"], expected_values)
    assert observed["colorbar_axes"] is figure.axes[0]
    assert figure.axes[1].get_ylabel() == "Pressure [Pa]"


def test_cell_scalar_uses_exact_derived_snapshot_and_diverging_scale(
    transient_vorticity,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _, result, vorticity, initial_vorticity = transient_vorticity
    trajectory = result.trajectory
    support = vorticity.support_indices("cell")
    expected_cells = trajectory.cells[support]
    expected_values = vorticity.values("cell")
    observed: dict[str, object] = {}
    original = Axes.tripcolor

    def capture(axes: Axes, *args: object, **kwargs: object):
        observed["values"] = np.asarray(kwargs["facecolors"]).copy()
        observed["cells"] = np.asarray(kwargs["triangles"]).copy()
        observed["shading"] = kwargs["shading"]
        observed["cmap"] = kwargs["cmap"]
        observed["vmin"] = kwargs["vmin"]
        observed["vmax"] = kwargs["vmax"]
        return original(axes, *args, **kwargs)

    monkeypatch.setattr(Axes, "tripcolor", capture)
    figure = eqplot.plot_scalar_field(
        trajectory,
        step=1,
        field=vorticity,
    )

    np.testing.assert_array_equal(observed["cells"], expected_cells)
    np.testing.assert_array_equal(observed["values"], expected_values)
    assert observed["shading"] == "flat"
    assert observed["cmap"] == "coolwarm"
    assert observed["vmin"] == -observed["vmax"]
    assert figure.axes[1].get_ylabel() == "Vorticity [s^-1]"
    assert "step 1" in figure.axes[0].get_title()

    with pytest.raises(ValueError, match="different accepted State"):
        eqplot.plot_scalar_field(trajectory, step=1, field=initial_vorticity)
    with pytest.raises(TypeError, match="requires a Trajectory"):
        eqplot.plot_scalar_field(result, field=vorticity)


def test_deformed_field_uses_exact_plan_field_output(
    structural: tuple[eqiora.Plan, eqiora.Result],
) -> None:
    plan, result = structural
    field = plan.capability.displacement
    output = result.output(field)
    figure = eqplot.plot_deformed_field(result, field=field, scale=2.0)
    assert output.coefficient_count("vertex") == 289
    assert output.value_shape == (2,)
    assert output.mesh.cells.shape == (256, 4)
    assert len(figure.axes) == 1
    assert "scale 2" in figure.axes[0].get_title()


@pytest.mark.parametrize("scale", [-1.0, float("inf"), float("nan")])
def test_deformed_field_rejects_invalid_scale_before_rendering(
    structural: tuple[eqiora.Plan, eqiora.Result], scale: float
) -> None:
    plan, result = structural
    with pytest.raises(ValueError, match="finite and nonnegative"):
        eqplot.plot_deformed_field(result, field=plan.capability.displacement, scale=scale)


def test_figures_are_headless_caller_owned_and_nonblank(
    scalar: tuple[eqiora.Plan, eqiora.Result],
    structural: tuple[eqiora.Plan, eqiora.Result],
    tmp_path: Path,
) -> None:
    scalar_plan, scalar_result = scalar
    structural_plan, structural_result = structural
    assert scalar_plan.capability.pressure is not None
    figures = (
        eqplot.plot_scalar_field(scalar_result, field=scalar_plan.capability.pressure),
        eqplot.plot_deformed_field(
            structural_result, field=structural_plan.capability.displacement, scale=1.0
        ),
    )
    del scalar_result, structural_result
    gc.collect()
    for index, figure in enumerate(figures):
        encoded = io.BytesIO()
        figure.savefig(encoded, format="png")
        payload = encoded.getvalue()
        destination = tmp_path / f"field-{index}.png"
        figure.savefig(destination)
        assert payload.startswith(b"\x89PNG\r\n\x1a\n")
        width, height = struct.unpack(">II", payload[16:24])
        assert width > 0 and height > 0
        encoded.seek(0)
        pixels = image.imread(encoded, format="png")
        assert np.ptp(pixels[..., :3]) > 0.0
        assert destination.read_bytes().startswith(b"\x89PNG\r\n\x1a\n")


def test_displaced_plotting_compatibility_is_absent() -> None:
    assert not hasattr(eqplot, "plot_displacement")
    assert "plot_displacement" not in eqplot.__all__
