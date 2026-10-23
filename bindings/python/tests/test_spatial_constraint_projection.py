"""Nonuniform weak constraints and independently diagonalized Galerkin dynamics."""

import eqiora as q
import numpy as np
import pytest
from numpy.polynomial import polynomial as poly
from test_spatial_constrained_midpoint import _problem


def _initial(plan, fields, values, nodal):
    key = "vertex_values" if nodal else "cell_values"
    return q.State.initial(
        plan,
        fields=tuple(
            q.InitialField(field, **{key: value})
            for field, value in zip(fields, values, strict=True)
        ),
    )


def _weak_constraint(u, z):
    """Integrate each hat times z-|u|² as a polynomial, without quadrature."""
    dx = 1 / (len(u) - 1)
    residual = np.zeros_like(u)
    for cell in range(len(u) - 1):
        local_u = [u[cell], u[cell + 1] - u[cell]]
        local_z = [z[cell], z[cell + 1] - z[cell]]
        balance = poly.polysub(local_z, poly.polymul(local_u, np.conj(local_u)))
        for local, hat in enumerate(([1, -1], [0, 1])):
            coefficients = poly.polymul(hat, balance)
            residual[cell + local] += dx * sum(
                value / (degree + 1) for degree, value in enumerate(coefficients)
            )
    return residual


@pytest.mark.parametrize("complex_field", [False, True])
@pytest.mark.parametrize(
    "spatial,mixed_rows",
    [(q.fem.Q1, False), (q.fem.Q1, True), (q.fvm.CellCenteredTpfa, False)],
)
def test_nonuniform_constraint_projection(spatial, mixed_rows, complex_field):
    plan, fields, c = _problem(
        spatial, complex_field, mixed_rows, authored_initial=False
    )
    nodal = spatial is q.fem.Q1
    association = "vertex" if nodal else "cell"
    x = np.array([0.0, 0.5, 1.0] if nodal else [0.25, 0.75])
    u0 = c * (1 + x)
    # On each reference cell, interpolation of |a+b*x|² exceeds the quadratic
    # by |b|²*dx²*xi*(1-xi). Both hat moments vanish after subtracting
    # |b|²*dx²/6. TPFA fields are cell constants and need no Q1 projection.
    offset = abs(c) ** 2 * 0.5**2 / 6 if nodal else 0.0
    z0 = np.asarray(np.abs(u0) ** 2 - offset, dtype=u0.dtype)
    initial = _initial(plan, fields, (u0, z0), nodal)
    original = initial.to_bytes()
    times = (0.05, 0.1, 0.15, 0.2, 0.3)
    dense = q.run(plan, state=initial, until_s=0.3, output_times_s=times)
    sparse = q.run(plan, state=initial, until_s=0.3, output_times_s=(0.3,))
    assert len(dense.trajectory.states) == len(times)
    for time, state in zip(times, dense.trajectory.states, strict=True):
        u, z = [np.asarray(state.field(field).values(association)) for field in fields]
        np.testing.assert_allclose(u, u0 + time, rtol=0, atol=1e-10)
        np.testing.assert_allclose(
            z, np.abs(u0 + time) ** 2 - offset, rtol=0, atol=1e-10
        )
        if nodal:
            np.testing.assert_allclose(_weak_constraint(u, z), 0, rtol=0, atol=1e-10)
            # Pointwise nodal squaring leaves |b|²*dx³/12 per local hat row.
            wrong = _weak_constraint(u, np.abs(u) ** 2)
            assert np.min(np.abs(wrong)) > offset / 8
        else:
            np.testing.assert_allclose(
                0.5 * (z - np.abs(u) ** 2), 0, rtol=0, atol=1e-10
            )
    assert (
        dense.trajectory.states[-1].to_bytes()
        == sparse.trajectory.states[-1].to_bytes()
    )
    assert initial.to_bytes() == original
    assert q.Result.from_bytes(plan, dense.to_bytes()).to_bytes() == dense.to_bytes()


def _q1_nodal_from_gauss(values):
    mean = (values[0] + values[1]) / 2
    half_difference = np.sqrt(3) * (values[1] - values[0]) / 2
    return np.array([mean - half_difference, mean + half_difference])


@pytest.mark.parametrize("complex_field", [False, True])
@pytest.mark.parametrize(
    "spatial,mixed_rows",
    [(q.fem.Q1, False), (q.fem.Q1, True), (q.fvm.CellCenteredTpfa, False)],
)
def test_nonuniform_feedback_midpoint_recurrence_and_refinement(
    spatial, complex_field, mixed_rows
):
    # For one Q1 cell, every weak integrand has degree <=3. Two-point Gauss
    # integration is exact and its test-value matrix is invertible. Therefore
    # M*u'=-M*z, M*z=load(|u|²) diagonalize at the Gauss points as v'=-|v|².
    # v=a+ib has constant b and a'=-a²-b². This derivation uses no assembly data.
    # For TPFA the same ODE holds directly for the two cell constants.
    nodal = spatial is q.fem.Q1
    association = "vertex" if nodal else "cell"
    physical = _q1_nodal_from_gauss if nodal else np.asarray
    c = 1 + 2j if complex_field else 1.0
    profile = (
        [1.5 - 1 / (2 * np.sqrt(3)), 1.5 + 1 / (2 * np.sqrt(3))]
        if nodal
        else [1.25, 1.75]
    )
    v0 = c * np.array(profile)
    end = 1 / 8
    b = np.imag(v0)
    exact = (
        b * np.tan(np.arctan(np.real(v0) / b) - b * end) + 1j * b
        if complex_field
        else v0 / (1 + v0 * end)
    )
    errors = []
    for h in (1 / 32, 1 / 64, 1 / 128):
        plan, fields, _ = _problem(
            spatial,
            complex_field,
            mixed_rows,
            authored_initial=False,
            feedback=True,
            cells=1 if nodal else 2,
            step_s=h,
        )
        initial_u = physical(v0)
        initial_z = np.asarray(physical(np.abs(v0) ** 2), dtype=v0.dtype)
        initial = _initial(plan, fields, (initial_u, initial_z), nodal)
        before = initial.to_bytes()
        times = tuple(h * i for i in range(1, round(end / h) + 1))
        result = q.run(plan, state=initial, until_s=end, output_times_s=times)
        assert len(result.trajectory.states) == len(times)
        v = v0.copy()
        for state in result.trajectory.states:
            # r=a_old-h*(r²+b²)/2; choose the root continuous as h tends to 0.
            r = (2 * np.real(v) - h * b * b) / (
                1 + np.sqrt(1 + 2 * h * np.real(v) - h * h * b * b)
            )
            v = 2 * r - np.real(v) + (1j * b if complex_field else 0)
            actual = np.asarray(state.field(fields[0]).values(association))
            z = np.asarray(state.field(fields[1]).values(association))
            np.testing.assert_allclose(actual, physical(v), rtol=0, atol=1e-10)
            np.testing.assert_allclose(z, physical(np.abs(v) ** 2), rtol=0, atol=1e-10)
            residual = (
                _weak_constraint(actual, z)
                if nodal
                else 0.5 * (z - np.abs(actual) ** 2)
            )
            np.testing.assert_allclose(residual, 0, rtol=0, atol=1e-10)
        errors.append(np.max(np.abs(actual - physical(exact))))
        assert initial.to_bytes() == before
        assert (
            q.Result.from_bytes(plan, result.to_bytes()).to_bytes() == result.to_bytes()
        )
    # Smooth nonsingular Riccati trajectories have midpoint global error O(h²).
    # The broad factor-four band is fixed independently of the measured errors;
    # the exact discrete recurrence above is the stronger method falsifier.
    ratios = np.array(errors[:-1]) / errors[1:]
    assert np.all((ratios > 3.5) & (ratios < 4.5))
