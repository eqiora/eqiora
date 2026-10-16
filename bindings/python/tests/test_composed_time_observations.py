"""Composition retains demand and structural dependence across Observable boundaries."""
import pytest

import eqiora as q


def test_composed_time_values_products_and_inactive_dependencies():
    model = q.compile(source="""model Ramp() {
        parameter rate:m/s=2;
        state x:m;
        initial { x=0[m]; }
        relation motion { derivative(x)=rate; }
        observable position:m=x;
        observable velocity:m/s=derivative(x);
        observable twice:m/s=velocity+velocity;
        observable clock:1=time()/1[s];
        observable singular:m=1[m]/(clock-0.5);
        observable selected:m=if clock <= 0.5 then position else singular;
        observable active:m=if rate <= 2[m/s] then position else singular;
        observable flat:1=(rate/1[m/s]-2)^2;
        observable stationary:m=if flat <= 0 then position else singular;
    }""")
    field, rate = model.field("x"), model.parameter("rate")
    plan = q.resolve(model, temporal=q.time.Tsitouras45(
        initial_step_s=0.125, relative_tolerance=1e-11,
        absolute_tolerances={(field, 0, 0, False): 1e-13},
        forward_sensitivities=q.time.ForwardSensitivity(
            relative_tolerance=1e-10,
            absolute_tolerances=(q.time.SensitivityTolerance(
                (field, 0, 0, False), rate, 1e-12, q.Dimension(time=1),
            ),),
        ),
    ))
    result = q.run(plan, state=q.State.initial(plan), until_s=1., output_times_s=(0.25, 0.5))
    encoded = result.to_bytes()
    direction = {rate: (q.Dimension(length=1, time=-1), 1.)}
    for candidate in (result, q.Result.from_bytes(q.Plan.from_bytes(plan.to_bytes()), encoded)):
        for index, time in enumerate((0.25, 0.5)):
            # x=rate*t; changing rate by 1 m/s changes x by t metres.
            selected = model.observable("selected")
            value = candidate.observe(selected, output_index=index)
            tangent = candidate.observe_parameter_jvp(selected, direction, output_index=index)
            assert value.value == pytest.approx(2*time, abs=2e-12, rel=0)
            assert tangent.value == pytest.approx(time, abs=2e-12, rel=0)
            assert value.value_type == tangent.value_type == q.ValueType.real(q.Dimension(length=1))
        twice = model.observable("twice")
        assert candidate.observe(twice, output_index=1).value == pytest.approx(4., abs=2e-12)
        assert candidate.observe_parameter_jvp(twice, direction, output_index=1).value == pytest.approx(2., abs=2e-12)
        assert candidate.observe_terminal(twice).value == pytest.approx(4., abs=2e-12)
        assert candidate.observe_terminal_parameter_jvp(twice, direction).value == pytest.approx(2., abs=2e-12)
        rule = q.time.TimeObservationInterpolation.NativeMidpointQuadratic
        integral = candidate.observe_time_integral(twice, interpolation=rule)
        product = candidate.observe_time_integral_parameter_jvp(twice, direction, interpolation=rule)
        assert integral.value == pytest.approx(4., abs=2e-12)
        assert product.value == pytest.approx(2., abs=2e-12)
        assert integral.value_type == product.value_type == q.ValueType.real(q.Dimension(length=1))
        with pytest.raises(q.ExecutionError, match="NonFinite"):
            candidate.observe(model.observable("singular"), output_index=1)
        # An active predicate tie must reject even when its first product is zero.
        for name in ("active", "stationary"):
            with pytest.raises((q.ValidationError, q.ExecutionError), match="comparison boundary"):
                candidate.observe_parameter_jvp(model.observable(name), direction, output_index=1)
        assert candidate.to_bytes() == encoded
