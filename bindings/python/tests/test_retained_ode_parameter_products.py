"""Fixed-time observation derivatives consume native retained output products."""
import json
import math
import pytest
import eqiora as q


def test_retained_decay_products_preserve_direct_rates_units_and_replay():
    model = q.compile(source="""model Decay() {
        state x:m; initial {x=3[m];}
        parameter rate:1/s=2;
        relation flow {derivative(x)=-rate*x;}
        observable sample:m=x;
        observable weighted:m/s=rate*x;
        observable speed:m/s=derivative(x);
        observable clock:s=time();
    }""")
    field, rate = model.field("x"), model.parameter("rate")
    forward = q.time.ForwardSensitivity(relative_tolerance=1e-10, absolute_tolerances=(
        q.time.SensitivityTolerance((field, 0, 0, False), rate, 1e-12, q.Dimension(length=1, time=1)),
    ))
    plan = q.resolve(model, temporal=q.time.Tsitouras45(initial_step_s=1e-3,
        relative_tolerance=1e-11, absolute_tolerances={(field, 0, 0, False): 1e-13},
        forward_sensitivities=forward))
    times = (0.125, 0.3, 0.875)
    result = q.run(plan, state=q.State.initial(plan), until_s=1., output_times_s=times)
    encoded = result.to_bytes()
    replay = q.Result.from_bytes(q.Plan.from_bytes(plan.to_bytes()), encoded)
    direction = {rate: (q.Dimension(time=-1), 1.)}
    for candidate in (result, replay):
        for index, time in enumerate(times):
            primal = 3 * math.exp(-2*time)
            for name, expected, dimension in (
                ("sample", -time*primal, q.Dimension(length=1)),
                ("weighted", (1-2*time)*primal, q.Dimension(length=1, time=-1)),
                ("speed", (2*time-1)*primal, q.Dimension(length=1, time=-1)),
                ("clock", 0., q.Dimension(time=1)),
            ):
                product = candidate.observe_parameter_jvp(model.observable(name), direction, output_index=index)
                assert product.value == pytest.approx(expected, abs=1e-8, rel=0)
                assert product.value_type == q.ValueType.real(dimension)
                assert product.evaluation_kind == "parameter-jvp"
                assert product.output == (index, time, q.State.from_result(plan, candidate, time_s=time).digest)
                assert product.result_identity == json.loads(encoded)["identity"]
        with pytest.raises(q.ValidationError):
            candidate.observe_parameter_jvp(model.observable("sample"), direction, output_index=len(times))
        with pytest.raises((ValueError, q.ValidationError)):
            candidate.observe_parameter_jvp(model.observable("sample"), {rate: (q.Dimension(), 1.)}, output_index=0)
        assert candidate.to_bytes() == encoded
    with pytest.raises(q.ValidationError):
        q.Result.from_bytes(plan, encoded.replace(b"eqiora.common-result/v26", b"eqiora.common-result/v20"))


@pytest.mark.parametrize("value,guard,gradient", [
    ("0.4", "x-threshold", 1.),
    ("0", "x-(0.4+threshold*threshold)", 0.),
])
def test_event_output_requires_a_stable_side_even_with_zero_event_time_gradient(value, guard, gradient):
    model = q.compile(source=f"""model Reset() {{
        state x:1; initial {{x=0;}}
        parameter threshold:1={value};
        relation flow {{derivative(x)=1[1/s];}}
        event hit=crossing({guard},direction=rising);
        relation reset at hit {{next(x)=0;}}
        observable sample:1=x;
    }}""")
    field, threshold = model.field("x"), model.parameter("threshold")
    events = q.time.EventPolicy(max_events=1, guard_tolerances=(
        q.time.GuardTolerance(model.activation("hit"), 1e-10, q.Dimension()),
    ))
    forward = q.time.ForwardSensitivity(relative_tolerance=1e-10, absolute_tolerances=(
        q.time.SensitivityTolerance((field, 0, 0, False), threshold, 1e-12, q.Dimension()),
    ))
    plan = q.resolve(model, temporal=q.time.Tsitouras45(
        initial_step_s=1e-3, relative_tolerance=1e-11,
        absolute_tolerances={(field, 0, 0, False): 1e-13},
        events=events, forward_sensitivities=forward,
    ))
    sample = model.observable("sample")
    direction = {threshold: (q.Dimension(), 1.)}
    result = q.run(plan, state=q.State.initial(plan), until_s=0.7, output_times_s=(0.2, 0.6))
    for index, expected in ((0, 0.), (1, -gradient)):
        product = result.observe_parameter_jvp(sample, direction, output_index=index)
        assert product.value == pytest.approx(expected, abs=1e-9, rel=0)
    event_time = result.observe_time_maximum(sample,
        interpolation=q.time.TimeObservationInterpolation.NativeMidpointQuadratic,
        interval_s=(0.1, 0.6)).extremum_time_s
    sampled = q.run(plan, state=q.State.initial(plan), until_s=0.7, output_times_s=(event_time,))
    encoded = sampled.to_bytes()
    products = json.loads(encoded)["content"]["payload"]["parameter_sensitivity"]
    assert products["event_time_gradients"][0][0] == pytest.approx(gradient, abs=1e-12, rel=0)
    # For the second guard tau(p)=.4+p^2 at p=0, tau'(0)=0. Nonetheless
    # x(.4,0)=0 after reset while lim[p->0,p!=0] x(.4,p)=.4 before reset.
    # A zero first-order event-time product cannot certify this point derivative.
    for candidate in (sampled, q.Result.from_bytes(plan, encoded)):
        assert candidate.observe(sample, output_index=0).value == 0.
        with pytest.raises(q.ValidationError, match="event"):
            candidate.observe_parameter_jvp(sample, direction, output_index=0)
        assert candidate.observe_parameter_jvp(sample, {}, output_index=0).value == 0.
        assert candidate.to_bytes() == encoded
