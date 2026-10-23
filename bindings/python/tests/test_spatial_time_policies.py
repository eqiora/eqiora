"""Time policies compose with the admitted equations and spatial coordinates."""

import math

import numpy as np
import pytest
import eqiora as q

from test_spatial_time_domains import _resolve


STEP = 1 / 16
HORIZON = 4 * STEP


def _temporal(method, field, complex_field):
    if method == "backward_euler":
        return q.time.BackwardEuler(STEP)
    tolerances = {
        (field, 0, 0, imaginary): 1e-12
        for imaginary in ((False, True) if complex_field else (False,))
    }
    if method == "midpoint":
        return q.time.ImplicitMidpoint(
            step_s=STEP, relative_tolerance=1e-10, absolute_tolerances=tolerances,
        )
    return q.time.Tsitouras45(
        initial_step_s=STEP, relative_tolerance=1e-10, absolute_tolerances=tolerances,
    )


def _amplitude(method, rate):
    if method == "backward_euler":
        return (1 / (1 + STEP * rate)) ** 4
    if method == "midpoint":
        return ((1 - STEP * rate / 2) / (1 + STEP * rate / 2)) ** 4
    return math.exp(-rate * HORIZON)


@pytest.mark.parametrize("rate", [8, 12])
@pytest.mark.parametrize("complex_field", [False, True])
@pytest.mark.parametrize("method", ["midpoint", "adaptive"])
def test_finite_decay_control(rate, complex_field, method):
    scalar = "complex<1>" if complex_field else "1"
    datum = "math.complex(1,2)" if complex_field else "1"
    model = q.compile(source=f"""model Decay() {{
        state u:{scalar}; initial {{u={datum};}}
        relation flow {{derivative(u)=-{rate}[1/s]*u;}}
    }}""")
    field = model.field("u")
    plan = q.resolve(model, temporal=_temporal(method, field, complex_field), solve=(
        q.solve.Linear(objective=q.solve.Robust, relative_tolerance=1e-12,
                       absolute_tolerance=1e-14, maximum_iterations=100)
        if method == "midpoint" else None
    ))
    plan = q.Plan.from_bytes(plan.to_bytes())
    initial = q.State.initial(plan)
    before = initial.to_bytes()
    result = q.run(plan, state=initial, until_s=HORIZON, output_times_s=(HORIZON,))
    amplitude = _amplitude(method, rate)
    for imaginary in ((False, True) if complex_field else (False,)):
        expected = (2 if imaginary else 1) * amplitude
        assert list(result.series(field, imaginary=imaginary))[-1][1] == pytest.approx(
            expected, rel=0, abs=2e-9,
        )
    assert q.Result.from_bytes(plan, result.to_bytes()).to_bytes() == result.to_bytes()
    assert initial.to_bytes() == before


@pytest.mark.parametrize("spatial", [q.fem.Q1, q.fvm.CellCenteredTpfa])
@pytest.mark.parametrize("complex_field", [False, True])
@pytest.mark.parametrize("method", ["backward_euler", "midpoint", "adaptive"])
def test_spatial_decay_uses_the_same_temporal_policy(spatial, complex_field, method):
    scalar = "complex<1>" if complex_field else "1"
    baseline = _resolve(f"""public component Heat(
        support body:volume(ambient_dimension=1), support left:boundary(parent=body),
        support right:boundary(parent=body)) {{
        state u:{scalar} on body in smooth;
        relation balance on body {{1[s/m^2]*derivative(u)-div(grad(u))=0;}}
        relation left_value on left {{trace(u)=0;}}
        relation right_value on right {{trace(u)=0;}}
    }}""", False, step=STEP, spatial=spatial)
    field = baseline.fields[0]
    plan = q.resolve(
        baseline.model, mesh=baseline.mesh, spatial=spatial(),
        solve=baseline.requested_solve, temporal=_temporal(method, field, complex_field),
    )
    plan = q.Plan.from_bytes(plan.to_bytes())
    multiplier = 1 + 2j if complex_field else 1.
    if spatial is q.fem.Q1:
        # Exact two-cell Q1 free row: M=1/3, K=4, so lambda=12/s.
        rate, association = 12, "vertex"
        values = multiplier * np.array([0., 1., 0.])
        prescribed = q.InitialField(field, vertex_values=values)
    else:
        # M=I/2, K=[[6,-2],[-2,6]]; (1,1) has lambda=8/s.
        rate, association = 8, "cell"
        values = multiplier * np.ones(2)
        prescribed = q.InitialField(field, cell_values=values)
    initial = q.State.initial(plan, fields=(prescribed,))
    before = initial.to_bytes()
    initial = q.State.from_bytes(plan, before)
    observed = []
    for dense_output in (False, True):
        if method == "backward_euler":
            result = q.run(plan, state=initial, steps=4,
                           output_steps=(1, 2, 3, 4) if dense_output else (4,))
        else:
            result = q.run(plan, state=initial, until_s=HORIZON,
                           output_times_s=(STEP, 2 * STEP, 3 * STEP, HORIZON)
                           if dense_output else (HORIZON,))
        final = result.trajectory.states[-1]
        assert final.time_s == HORIZON
        assert final.step == (4 if method == "backward_euler" else None)
        actual = np.asarray(final.fields[0].values(association))
        np.testing.assert_allclose(actual, values * _amplitude(method, rate), rtol=0, atol=2e-9)
        observed.append(actual)
        assert q.Result.from_bytes(plan, result.to_bytes()).to_bytes() == result.to_bytes()
        trajectory = result.trajectory
        assert q.trajectory.Trajectory.from_bytes(plan, trajectory.to_bytes()).to_bytes() == trajectory.to_bytes()
        assert initial.to_bytes() == before
    np.testing.assert_allclose(observed[0], observed[1], rtol=0, atol=2e-9)
    if method == "midpoint":
        # Half-step observations use the accepted linear collocation polynomial.
        sampled = q.run(plan, state=initial, until_s=HORIZON,
                        output_times_s=(STEP / 2, 2 * STEP, HORIZON))
        factor = (1 - STEP * rate / 2) / (1 + STEP * rate / 2)
        middle = sampled.trajectory.states[0]
        assert middle.time_s == STEP / 2
        assert middle.step is None
        np.testing.assert_allclose(
            np.asarray(middle.fields[0].values(association)),
            values * (1 + factor) / 2, rtol=0, atol=2e-9,
        )
        restart = q.State.from_bytes(plan, sampled.trajectory.states[1].to_bytes())
        assert restart.step is None
        resumed = q.run(plan, state=restart, until_s=HORIZON, output_times_s=(HORIZON,))
        np.testing.assert_allclose(
            np.asarray(resumed.trajectory.states[-1].fields[0].values(association)),
            observed[0], rtol=0, atol=2e-9,
        )
