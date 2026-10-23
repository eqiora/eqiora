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
        observable bowl:m^2=(x-0.75[m])*(x-0.75[m]);
        observable plateau:m=7[m];
    }""")
    field = model.field("x")
    tolerances = {(field, 0, 0, False): 1e-13}
    temporal = (
        q.time.ImplicitMidpoint(step_s=0.25, relative_tolerance=1e-11, absolute_tolerances=tolerances)
        if method == "midpoint"
        else q.time.Tsitouras45(initial_step_s=0.25, relative_tolerance=1e-11, absolute_tolerances=tolerances)
    )
    plan = q.resolve(model, temporal=temporal, solve=(q.solve.Linear(
        objective=q.solve.Robust, relative_tolerance=1e-12,
        absolute_tolerance=1e-14, maximum_iterations=100,
    ) if method == "midpoint" else None))
    result = q.run(plan, state=q.State.initial(plan), until_s=1.0, output_times_s=(0.25,))
    dense = q.run(plan, state=q.State.initial(plan), until_s=1.0, output_times_s=(0.125, 0.5, 0.875, 1.0))
    observable = model.observable("square")
    rule = q.time.TimeObservationInterpolation.NativeMidpointQuadratic
    before = result.to_bytes()
    replay = q.Result.from_bytes(q.Plan.from_bytes(plan.to_bytes()), before)
    for interval in [(0.125, 0.875), (0.0, 0.125), (0.875, 1.0), (0.0, 1.0)]:
        # x=2t; independently integrate x^2 to 4*(b^3-a^3)/3 m^2*s.
        expected = 4 * (interval[1] ** 3 - interval[0] ** 3) / 3
        value = result.observe_time_integral(observable, interpolation=rule, interval_s=interval)
        assert math.isclose(value.value, expected, abs_tol=2e-12, rel_tol=0)
        assert value.value_type == q.ValueType.real(q.Dimension(length=2, time=1))
        assert value.interval_s == interval
        assert value.value == dense.observe_time_integral(observable, interpolation=rule, interval_s=interval).value
        restored = replay.observe_time_integral(observable, interpolation=rule, interval_s=interval)
        assert restored.value == value.value
        assert restored.trajectory_identity == value.trajectory_identity
    if method == "midpoint":
        for interval, expected in [(None, 11/8), ((0.125, 0.875), 59/64)]:
            value = result.observe_time_integral(observable,
                interpolation=q.time.TimeObservationInterpolation.EndpointLinear, interval_s=interval)
            assert value.value == pytest.approx(expected, abs=2e-12, rel=0)
            assert value.spatial_quadratures == {}
    for trajectory in (result, dense, replay):
        bowl = model.observable("bowl")
        minimum = trajectory.observe_time_minimum(bowl, interpolation=rule)
        maximum = trajectory.observe_time_maximum(bowl, interpolation=rule)
        assert minimum.value == pytest.approx(0., abs=2e-12, rel=0)
        assert minimum.extremum_time_s == pytest.approx(0.375, abs=2e-12, rel=0)
        assert maximum.value == pytest.approx(25/16, abs=2e-12, rel=0)
        assert maximum.extremum_time_s == 1.0
        assert minimum.value_type == q.ValueType.real(q.Dimension(length=2))
        assert minimum.evaluation_kind == "time-minimum"
        assert maximum.evaluation_kind == "time-maximum"
        assert not minimum.parameter_jvp
        assert minimum.interpolation == rule
        for operation in (trajectory.observe_time_minimum, trajectory.observe_time_maximum):
            plateau = operation(model.observable("plateau"), interpolation=rule, interval_s=(0.13,0.87))
            assert plateau.value == 7.0
            assert plateau.extremum_time_s == 0.13
        if method == "midpoint":
            linear = trajectory.observe_time_minimum(bowl, interpolation=q.time.TimeObservationInterpolation.EndpointLinear)
            assert linear.value == 1/16
            assert linear.extremum_time_s == 0.25
    for interval in [(-0.1, 0.5), (0.5, 1.1), (0.5, 0.5), (0.7, 0.2), (math.nan, 1.0), (0.0, math.inf)]:
        with pytest.raises((ValueError, q.ValidationError)):
            result.observe_time_integral(observable, interpolation=rule, interval_s=interval)
    assert result.to_bytes() == before


@pytest.mark.parametrize("rule", [q.time.TimeObservationInterpolation.NativeMidpointQuadratic,
                                  q.time.TimeObservationInterpolation.EndpointLinear])
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
    for operation, expected, time in ((result.observe_time_minimum, 0., 0.4), (result.observe_time_maximum, 0.4, 0.4)):
        extreme = operation(observable, interpolation=rule, interval_s=(0.1,0.6))
        assert extreme.value == pytest.approx(expected, abs=1e-9, rel=0)
        assert extreme.extremum_time_s == pytest.approx(time, abs=1e-9, rel=0)
    direction = {threshold: (q.Dimension(), 1.0)}
    # x=t before p=.4 and x=t-p afterwards. The derivative includes the
    # jump term only when the moving event is inside the fixed window.
    for interval, integral, derivative in [((0.1, 0.3), 0.04, 0.0), ((0.1, 0.6), 0.095, 0.2), ((0.5, 0.6), 0.015, -0.1)]:
        value = result.observe_time_integral(observable, interpolation=rule, interval_s=interval)
        tangent = result.observe_time_integral_parameter_jvp(observable, direction, interpolation=rule, interval_s=interval)
        assert math.isclose(value.value, integral, abs_tol=1e-9, rel_tol=0)
        assert math.isclose(tangent.value, derivative, abs_tol=1e-9, rel_tol=0)
        assert tangent.interval_s == interval
