import math

import pytest

import eqiora as q


@pytest.mark.parametrize("method", ["midpoint", "tsitouras"])
def test_quadratic_time_window_preserves_units_history_and_replay(method):
    model = q.compile(source="""model Ramp() {
        state x:m;
        initial { x=0[m]; }
        parameter rate:m/s=2;
        relation flow { derivative(x)=rate; }
        observable square:m^2=x*x;
    }""")
    field = model.field("x")
    tolerances = {(field, 0, 0, False): 1e-13}
    temporal = (
        q.time.ImplicitMidpoint(step_s=0.25, relative_tolerance=1e-11, absolute_tolerances=tolerances)
        if method == "midpoint"
        else q.time.Tsitouras45(initial_step_s=0.25, relative_tolerance=1e-11, absolute_tolerances=tolerances)
    )
    plan = q.resolve(model, temporal=temporal)
    result = q.run(plan, state=q.State.initial(plan), until_s=1.0, output_times_s=(0.25,))
    dense = q.run(plan, state=q.State.initial(plan), until_s=1.0, output_times_s=(0.125, 0.5, 0.875, 1.0))
    observable = model.observable("square")
    rule = q.time.TimeFunctionalQuadrature.AcceptedStepSimpson
    before = result.to_bytes()
    replay = q.Result.from_bytes(q.Plan.from_bytes(plan.to_bytes()), before)
    for interval in [(0.125, 0.875), (0.0, 0.125), (0.875, 1.0), (0.0, 1.0)]:
        # x=2t; independently integrate x^2 to 4*(b^3-a^3)/3 m^2*s.
        expected = 4 * (interval[1] ** 3 - interval[0] ** 3) / 3
        value = result.observe_time_integral(observable, quadrature=rule, interval_s=interval)
        assert math.isclose(value.value, expected, abs_tol=2e-12, rel_tol=0)
        assert value.value_type == q.ValueType.real(q.Dimension(length=2, time=1))
        assert value.interval_s == interval
        assert value.value == dense.observe_time_integral(observable, quadrature=rule, interval_s=interval).value
        restored = replay.observe_time_integral(observable, quadrature=rule, interval_s=interval)
        assert restored.value == value.value
        assert restored.trajectory_identity == value.trajectory_identity
    if method == "midpoint":
        for interval, expected in [(None, 11/8), ((0.125, 0.875), 59/64)]:
            value = result.observe_time_integral(observable,
                quadrature=q.time.TimeFunctionalQuadrature.AcceptedStepTrapezoid, interval_s=interval)
            assert value.value == pytest.approx(expected, abs=2e-12, rel=0)
            assert value.spatial_quadratures == {}
    for interval in [(-0.1, 0.5), (0.5, 1.1), (0.5, 0.5), (0.7, 0.2), (math.nan, 1.0), (0.0, math.inf)]:
        with pytest.raises((ValueError, q.ValidationError)):
            result.observe_time_integral(observable, quadrature=rule, interval_s=interval)
    assert result.to_bytes() == before


@pytest.mark.parametrize("rule", [q.time.TimeFunctionalQuadrature.AcceptedStepSimpson,
                                  q.time.TimeFunctionalQuadrature.AcceptedStepTrapezoid])
def test_time_window_event_integral_and_parameter_variation(rule):
    model = q.compile(source="""model Reset() {
        state x:1;
        initial { x=0; }
        parameter threshold:1=0.4;
        relation flow { derivative(x)=1[1/s]; }
        event hit=crossing(x-threshold,direction=rising);
        relation reset at hit { next(x)=0; }
        observable sample:1=x;
    }""")
    field, threshold = model.field("x"), model.parameter("threshold")
    events = q.time.EventPolicy(max_events=1, guard_tolerances=(q.time.GuardTolerance(model.activation("hit"), 1e-10, q.Dimension()),))
    forward = q.time.ForwardSensitivity(relative_tolerance=1e-10, absolute_tolerances=(
        q.time.SensitivityTolerance((field, 0, 0, False), threshold, 1e-12, q.Dimension()),
    ))
    plan = q.resolve(model, temporal=q.time.Tsitouras45(
        initial_step_s=1e-3, relative_tolerance=1e-11,
        absolute_tolerances={(field, 0, 0, False): 1e-13},
        events=events, forward_sensitivities=forward,
    ))
    result = q.run(plan, state=q.State.initial(plan), until_s=0.7, output_times_s=(0.2,))
    observable = model.observable("sample")
    direction = {threshold: (q.Dimension(), 1.0)}
    # x=t before p=.4 and x=t-p afterwards. The derivative includes the
    # jump term only when the moving event is inside the fixed window.
    for interval, integral, derivative in [((0.1, 0.3), 0.04, 0.0), ((0.1, 0.6), 0.095, 0.2), ((0.5, 0.6), 0.015, -0.1)]:
        value = result.observe_time_integral(observable, quadrature=rule, interval_s=interval)
        tangent = result.observe_time_integral_parameter_jvp(observable, direction, quadrature=rule, interval_s=interval)
        assert math.isclose(value.value, integral, abs_tol=1e-9, rel_tol=0)
        assert math.isclose(tangent.value, derivative, abs_tol=1e-9, rel_tol=0)
        assert tangent.interval_s == interval
