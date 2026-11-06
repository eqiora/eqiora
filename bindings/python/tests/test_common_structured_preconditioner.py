"""Installed Python wiring for the ordinary packaged Stokes block composition.

This reuses the existing uniform-translation solution and its 2e-10 tolerance.
It checks the public execution and persistence path, not new scientific evidence.
"""

import numpy as np
import pytest

import eqiora as q
from test_exact_cylinder_stokes_result import IMPORTED_TRANSLATION_SOURCE


@pytest.mark.parametrize("ranked", [False, True], ids=["manual", "ranked"])
def test_packaged_stokes_executes_and_replays_complete_block_policy(ranked):
    graph = q.geometry.GeometryGraph()
    rectangle = graph.rectangle(x_bounds=(0.0, 1.0), y_bounds=(0.0, 1.0))
    # Reuse the existing all-Dirichlet translation fixture on four straight sides.
    names = ("inlet", "outlet", "walls", "cylinder")
    geometry = graph.build(rectangle, named_topology={
        "body": rectangle.region,
        **dict(zip(names, rectangle.boundaries, strict=True)),
    })
    mesh = q.meshing.generate(q.meshing.resolve(
        geometry, q.meshing.AffineTriangleMesher(cells=(2, 2)),
    ))
    speed = 0.7
    model = q.compile(
        source=IMPORTED_TRANSLATION_SOURCE, entry="Translation", geometry=geometry,
        bindings={"speed": speed, "body": geometry.selection("body"), **{
            name: geometry.selection(name) for name in names
        }},
    )
    velocity = model.field("definition.velocity")
    pressure = model.field("definition.pressure")
    provider = q.solve.SolverProvider.faer()
    factor = q.solve.FixedFactorization(provider=provider)
    groups = ([q.solve.AlgebraicBlock.field(velocity)], [
        q.solve.AlgebraicBlock.field(pressure),
        q.solve.AlgebraicBlock.constraint_multiplier(pressure),
    ])
    policy = q.solve.Preconditioner.upper_diagonal_schur(
        groups=groups, a_factorization=factor, schur_factorization=factor,
        maximum_entries=100_000, maximum_products=10_000_000,
    )
    selection = {"objective": q.solve.Robust} if ranked else {
        "algorithm": q.solve.LinearSolver.BiConjugateGradientStabilized,
        "provider": provider, "reduction": q.solve.Reduction.Fast,
    }
    request = q.solve.Linear(
        **selection, preconditioner=policy, relative_tolerance=1e-11,
        absolute_tolerance=1e-13, maximum_iterations=10_000,
    )
    plan = q.resolve(model, mesh=mesh, spatial=q.fem.MiniP1(), solve=request,
                     scaling=q.fluid.IncompressibleScaling(
                         length_m=1.0, velocity_m_per_s=1.0, pressure_pa=1.0))
    plan_bytes = plan.to_bytes()
    assert q.Plan.from_bytes(plan_bytes).to_bytes() == plan_bytes
    live_result = q.run(plan)
    assert plan.to_bytes() == plan_bytes
    replayed_plan = q.Plan.from_bytes(plan_bytes)
    assert replayed_plan.to_bytes() == plan_bytes
    for candidate_plan in (plan, replayed_plan):
        assert candidate_plan.requested_solve.preconditioner == policy
        assert candidate_plan.solve.preconditioner == policy
        assert candidate_plan.solve.provider == provider
        assert candidate_plan.capability.pressure_gauge == q.fluid.PressureGauge2d.ZeroIntegral
        config = candidate_plan.solve.preconditioner.upper_diagonal_schur_config
        assert config.groups == groups
        assert config.a_factorization.provider == config.schur_factorization.provider == provider
        assert config.a_factorization.algorithm == q.solve.LinearSolver.SparseLu
        assert config.maximum_entries == 100_000
        assert config.maximum_products == 10_000_000
        result = live_result if candidate_plan is plan else q.run(candidate_plan)
        result_bytes = result.to_bytes()
        replayed_result = q.Result.from_bytes(candidate_plan, result_bytes)
        assert replayed_result.to_bytes() == result_bytes
        for candidate in (result, replayed_result):
            evidence = q.fluid.steady_stokes_evidence(candidate)
            assert evidence.solve.preconditioner == policy
            assert evidence.solve.backend == provider.id
            assert evidence.solve.completed_iterations > 0
            assert evidence.solve.true_residual_norm <= evidence.solve.residual_target
            assert abs(candidate.gauge_multiplier) <= 2e-10
            vertices = np.asarray(candidate.output(velocity).values("vertex")).reshape(-1, 2)
            np.testing.assert_allclose(vertices[:, 0], speed, atol=2e-10, rtol=0)
            np.testing.assert_allclose(vertices[:, 1], 0, atol=2e-10, rtol=0)
            np.testing.assert_allclose(candidate.output(velocity).values("cell-bubble"),
                                       0, atol=2e-10, rtol=0)
            np.testing.assert_allclose(candidate.output(pressure).values("vertex"),
                                       0, atol=2e-10, rtol=0)
