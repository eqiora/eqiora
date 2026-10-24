"""Ordinary BDF consumes the same constrained spatial Model and physical State."""

import eqiora as q
import numpy as np
import pytest

from test_spatial_constrained_midpoint import _problem


@pytest.mark.parametrize("complex_field", [False, True])
@pytest.mark.parametrize(
    "spatial,mixed_rows",
    [(q.fem.Q1, False), (q.fem.Q1, True), (q.fvm.CellCenteredTpfa, False)],
)
def test_bdf_constrained_physical_outputs_and_replay(spatial, mixed_rows, complex_field):
    baseline, fields, initial_u = _problem(spatial, complex_field, mixed_rows)
    policy = q.time.Bdf(
        initial_step_s=1e-4,
        relative_tolerance=1e-9,
        absolute_tolerances={
            (field, 0, 0, imaginary): 1e-11
            for field in fields
            for imaginary in ([False, True] if complex_field else [False])
        },
    )
    plan = q.resolve(
        baseline.model, mesh=baseline.mesh, spatial=spatial(),
        solve=baseline.requested_solve, temporal=policy,
    )
    plan = q.Plan.from_bytes(plan.to_bytes())
    initial = q.State.initial(plan)
    before = initial.to_bytes()
    times = (0.07, 0.19, 0.3)
    dense = q.run(plan, state=initial, until_s=0.3, output_times_s=times)
    sparse = q.run(plan, state=initial, until_s=0.3, output_times_s=(0.3,))
    association = "vertex" if spatial is q.fem.Q1 else "cell"
    # Original equations give u=u0+t and z=|u0+t|², irrespective of
    # the invertible mixing of their residual rows or the output schedule.
    for time, state in zip(times, dense.trajectory.states, strict=True):
        for field, expected in zip(
            fields, (initial_u + time, abs(initial_u + time) ** 2), strict=True
        ):
            np.testing.assert_allclose(
                state.field(field).values(association), expected, rtol=0, atol=1e-6
            )
        assert q.State.from_bytes(plan, state.to_bytes()).to_bytes() == state.to_bytes()
    assert dense.trajectory.states[-1].to_bytes() == sparse.trajectory.states[-1].to_bytes()
    assert initial.to_bytes() == before
    assert q.Result.from_bytes(plan, dense.to_bytes()).to_bytes() == dense.to_bytes()
    trajectory = dense.trajectory
    assert q.trajectory.Trajectory.from_bytes(plan, trajectory.to_bytes()).to_bytes() == trajectory.to_bytes()
