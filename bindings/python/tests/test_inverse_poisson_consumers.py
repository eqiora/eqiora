"""Installed saved Scalar examples against an independently derived Q1 solution."""
from __future__ import annotations

from pathlib import Path
import subprocess
import sys

import eqiora
import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[3]
IMPORTED = Path("examples/inverse-poisson.eqi")
HANDWRITTEN = Path(
    "docs/site/src/content/docs/learn/inverse-problems/_examples/inverse-poisson-handwritten.eqi"
)
SCRIPT = Path("examples/python/learn_inverse_poisson.py")


def _unit_source_response(coordinates: np.ndarray, cells: int) -> np.ndarray:
    # Independent uniform-square Q1 calculation, with numeric coordinates in m:
    # K has interior diagonal 8/3 and all eight neighbours -1/3. For
    # sin(i*theta)*sin(j*theta), theta=pi/N, its eigenvalue is
    # lambda=(8-4*cos(theta)-4*cos(theta)^2)/3.
    # The maintained Q1 policy uses tensor two-point Gauss for this sine load
    # (the non-polynomial source adds no polynomial-exactness requirement).
    # Integrating the two adjacent 1D hats gives h*Q*sin(i*theta), where
    # Q=sum_{t=(1+/-1/sqrt(3))/2}(1-t)*cos(theta*t). Thus unit-source
    # nodal response is h^2*Q^2/lambda times the sine mode. No Eqiora
    # output, assembled matrix, quadrature object or solver is used here.
    theta = np.pi / cells
    points = (1.0 + np.array([-1.0, 1.0]) / np.sqrt(3.0)) / 2.0
    q = np.sum((1.0 - points) * np.cos(theta * points))
    eigenvalue = (8.0 - 4.0 * np.cos(theta) - 4.0 * np.cos(theta) ** 2) / 3.0
    response = q**2 / (cells**2 * eigenvalue) * np.prod(
        np.sin(np.pi * coordinates), axis=1
    )
    boundary = np.any((coordinates == 0.0) | (coordinates == 1.0), axis=1)
    response[boundary] = 0.0
    return response


def _model_and_plan(path: Path, cells: int):
    graph = eqiora.geometry.GeometryGraph()
    rectangle = graph.rectangle(x_bounds=(0.0, 1.0), y_bounds=(0.0, 1.0))
    names = ("x_lower", "x_upper", "y_lower", "y_upper")
    geometry = graph.build(rectangle, named_topology={
        "square": rectangle.region,
        **dict(zip(names, rectangle.boundaries, strict=True)),
    })
    mesh = eqiora.meshing.generate(eqiora.meshing.resolve(
        geometry, eqiora.meshing.CartesianMesher(cells=(cells, cells))
    ))
    model = eqiora.compile(
        path=ROOT / path, geometry=geometry, entry="InversePoisson",
        bindings={
            "square": geometry.selection("square"),
            **{name: (geometry.selection(name), geometry.selection("square")) for name in names},
            "diffusion": 1.0, "wave_number": np.pi,
            "source_scale": 2.0 * np.pi**2, "boundary_offset": 0.0,
        },
    )
    plan = eqiora.resolve(
        model, mesh=mesh, spatial=eqiora.fem.Q1(),
        solve=eqiora.solve.Linear(
            algorithm=eqiora.solve.LinearSolver.BiConjugateGradientStabilized,
            preconditioner=eqiora.solve.Preconditioner.Identity,
            reduction=eqiora.solve.Reduction.Reproducible,
            provider=eqiora.solve.SolverProvider.reference(),
            relative_tolerance=1e-12, absolute_tolerance=1e-14,
            maximum_iterations=10_000,
        ),
    )
    return model, plan


@pytest.mark.parametrize("path", [IMPORTED, HANDWRITTEN], ids=["imported", "handwritten"])
def test_saved_inverse_poisson_run_and_parameter_actions_match_analytic_q1(path: Path) -> None:
    cells = 12
    model, plan = _model_and_plan(path, cells)
    response = _unit_source_response(np.asarray(plan.mesh.coordinates), cells)
    field = plan.capability.fields[0]
    nominal = np.array([2.0 * np.pi**2, 1.0, 0.0])
    result = eqiora.run(plan)
    # Reuse the established independent analytic field/product tolerance from
    # test_spatial_observable_differentiation (2e-10, zero relative tolerance).
    np.testing.assert_allclose(
        result.output(field).values("vertex"), nominal[0] * response, rtol=0, atol=2e-10
    )
    assert (model.package_compilation_digest is not None) == (path == IMPORTED)
    assert plan.package_compilation_digest == model.package_compilation_digest
    execution = eqiora.submit(plan)
    assert execution.package_compilation_digest == model.package_compilation_digest
    np.testing.assert_allclose(execution.result().output(field).values("vertex"),
                               nominal[0] * response, rtol=0, atol=2e-10)
    program = eqiora.diff.compile(
        plan, inputs=[model.parameter(name) for name in
                      ("source_scale", "diffusion", "boundary_offset")], outputs=(field,),
    )
    coordinates = np.asarray(plan.mesh.coordinates)
    weights = np.full(len(response), 1.0 / cells**2)
    for axis in (0, 1):
        weights[(coordinates[:, axis] == 0.0) | (coordinates[:, axis] == 1.0)] *= 0.5
    direction = np.array([0.7, -0.2, 0.3])
    for source, diffusion, boundary in (nominal, [0.8 * nominal[0], 1.2, 0.15]):
        point = program.evaluate(tuple(np.asarray(value) for value in (source, diffusion, boundary)))
        # u=b+(s/c)*response, hence du/ds=response/c,
        # du/dc=-s*response/c^2 and du/db=1, including boundary nodes.
        derivatives = np.array([response / diffusion,
                                -source * response / diffusion**2, np.ones_like(response)])
        np.testing.assert_allclose(point.primal().output[0],
                                   boundary + source * response / diffusion, rtol=0, atol=2e-10)
        for index in range(3):
            seed = np.eye(3)[index]
            actual = point.jvp(tuple(np.asarray(value) for value in seed)).tangent[0]
            np.testing.assert_allclose(actual, derivatives[index], rtol=0, atol=2e-10)
        np.testing.assert_allclose(point.jvp(tuple(np.asarray(value) for value in direction)).tangent[0],
                                   direction @ derivatives, rtol=0, atol=2e-10)
        actual_gradient = np.asarray(point.vjp((weights,)).input_cotangent).reshape(3)
        np.testing.assert_allclose(actual_gradient, derivatives @ weights, rtol=0, atol=2e-10)


@pytest.mark.parametrize("path", [IMPORTED, HANDWRITTEN], ids=["imported", "handwritten"])
def test_saved_inverse_poisson_learning_script_runs_against_installed_api(path: Path) -> None:
    # Script recovery is a workflow witness, not the independent scientific
    # oracle; the preceding test owns the analytic primal/derivative checks.
    completed = subprocess.run(
        [sys.executable, str(ROOT / SCRIPT), str(ROOT / path), "--cells", "12"],
        check=True, capture_output=True, text=True, cwd=ROOT,
    )
    assert "area-mean gradient:" in completed.stdout
    assert "forward/reverse pairing difference:" in completed.stdout
