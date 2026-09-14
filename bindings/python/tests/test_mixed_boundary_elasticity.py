"""Installed-wheel contract for the common structural Plan and observation."""

from __future__ import annotations

import subprocess
import sys
from importlib.resources import files
from pathlib import Path

import numpy as np
import pytest

import eqiora
from _signature_bindings import support_bindings


REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
PYTHON_DEMO = REPOSITORY_ROOT / "examples" / "python" / "mixed_boundary_elasticity.py"


def geometry_and_mesh() -> tuple[eqiora.geometry.Geometry, eqiora.meshing.Mesh]:
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
    plan = eqiora.meshing.resolve(geometry, request)
    return geometry, eqiora.meshing.generate(plan)


def accepted() -> tuple[eqiora.Model, eqiora.Plan, eqiora.Result]:
    geometry, mesh = geometry_and_mesh()
    model = eqiora.compile(path=files(eqiora).joinpath('examples', 'mixed-boundary-elasticity.eqi'), geometry=geometry, entry='MixedBoundaryElasticity2d', bindings={**support_bindings(geometry, ['body'], [('x_lower', 'body'), ('x_upper', 'body'), ('y_lower', 'body'), ('y_upper', 'body')]), **{'mu': 3.0, 'lambda': 0.0, 'length_scale': 1.0}})
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
    return model, plan, eqiora.run(plan)


def test_common_plan_result_and_observation_close_exact_lineage() -> None:
    model, plan, result = accepted()
    displacement = plan.capability.displacement
    output = result.output(displacement)
    evidence = eqiora.solid.linear_elasticity_evidence(result)

    assert result.model_digest == model.digest == plan.model_digest
    assert result.plan_key == plan.identity == evidence.plan_key
    assert output.field == displacement
    assert output.mesh is plan.mesh
    assert output.coefficient_count("vertex") == 289
    assert output.value_shape == (2,)
    assert output.dimension == (0, 1, 0, 0, 0, 0, 0)
    values = output.values("vertex").numpy(copy=False).reshape(289, 2)
    assert values.shape == (289, 2)
    assert not values.flags.writeable
    assert np.isfinite(values).all()

    assert evidence.exact_bounds == ((0.0, 1.0), (0.0, 1.0))
    assert evidence.assembly_packets > 0
    assert evidence.assembly_targets > 0
    assert np.isfinite(evidence.constrained_reaction).all()
    assert np.isfinite(evidence.integrated_body_force).all()
    assert evidence.solve.true_residual_norm <= evidence.solve.residual_target
    np.testing.assert_allclose(
        np.asarray(evidence.constrained_reaction)
        + np.asarray(evidence.integrated_body_force),
        np.zeros(2),
        rtol=0.0,
        atol=1.0e-10,
    )


@pytest.mark.parametrize("planned", [False, True])
def test_solver_inventory_survives_plan_and_result_replay(planned: bool) -> None:
    model, manual, _ = accepted()
    plan = eqiora.resolve(
        model,
        mesh=manual.mesh,
        spatial=eqiora.fem.Q1(),
        solve=eqiora.solve.Linear(
            objective=eqiora.solve.Robust,
            relative_tolerance=1.0e-10,
            absolute_tolerance=1.0e-12,
            maximum_iterations=10_000,
        ),
    ) if planned else manual
    replay = eqiora.Plan.from_bytes(plan.to_bytes())
    assert replay.identity == plan.identity
    result = eqiora.run(replay)
    restored = eqiora.Result.from_bytes(replay, result.to_bytes())
    # Independent equilibrium solution: -2*mu*u_xx=2*mu,
    # u(0)=0, u_x(1)=0; the transverse displacement vanishes (lambda=0).
    x = replay.mesh.coordinates[:, 0]
    expected = np.column_stack((x - x*x/2.0, np.zeros_like(x)))
    for accepted_result in (result, restored):
        values = accepted_result.output(replay.capability.displacement).values("vertex").numpy(copy=False)
        np.testing.assert_allclose(values.reshape(-1, 2), expected, rtol=0.0, atol=1.0e-9)
        evidence = eqiora.solid.linear_elasticity_evidence(accepted_result)
        np.testing.assert_allclose(evidence.integrated_body_force, [6.0, 0.0], rtol=0.0, atol=1.0e-10)
        np.testing.assert_allclose(evidence.constrained_reaction, [-6.0, 0.0], rtol=0.0, atol=1.0e-9)


def test_root_plan_rejects_foreign_model_field_and_observation() -> None:
    model, plan, result = accepted()
    foreign_geometry, foreign_mesh = geometry_and_mesh()
    foreign = eqiora.compile(path=files(eqiora).joinpath('examples', 'mixed-boundary-elasticity.eqi'), geometry=foreign_geometry, entry='MixedBoundaryElasticity2d', bindings={**support_bindings(foreign_geometry, ['body'], [('x_lower', 'body'), ('x_upper', 'body'), ('y_lower', 'body'), ('y_upper', 'body')]), **{'mu': 4.0, 'lambda': 0.0, 'length_scale': 1.0}})
    foreign_plan = eqiora.resolve(
        foreign,
        mesh=foreign_mesh,
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
    assert foreign_plan.identity != plan.identity
    with pytest.raises(ValueError, match="different exact Model"):
        result.output(foreign_plan.capability.displacement)
    with pytest.raises(eqiora.ValidationError):
        eqiora.resolve(
            model,
            mesh=plan.mesh,
            spatial=eqiora.fem.MiniP1(),
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


def test_displaced_structural_lifecycle_is_absent() -> None:
    for name in (
        "LinearElasticity",
        "LinearElasticityPlan",
        "MixedBoundaryElasticityResult",
        "resolve",
        "solve_mixed_boundary_elasticity",
    ):
        assert not hasattr(eqiora.solid, name)
        assert name not in eqiora.solid.__all__


def test_checked_in_python_demo_runs_with_packaged_component_resource() -> None:
    completed = subprocess.run(
        [sys.executable, str(PYTHON_DEMO)],
        cwd=REPOSITORY_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    assert "constrained reaction" in completed.stdout
    assert "integrated body force" in completed.stdout
