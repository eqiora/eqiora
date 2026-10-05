"""Complex and real oscillators use the same exact Plan/State/Run lifecycle."""
import math
import pytest
import eqiora

SOURCE = """
model Oscillators() {
    parameter omega:1/s=1;
    state z:array<complex<1>,2>;
    state u:1;
    state v:1;
    initial { z=[math.complex(1,0),math.complex(2,0)]; u=3; v=0; }
    relation flow {
        derivative(z)=math.complex(0,1)*omega*z;
        derivative(u)=-omega*v;
        derivative(v)=omega*u;
    }
}
"""
@pytest.mark.parametrize("complex_parameter", [False, True])
def test_midpoint_complex_plan_samples_and_accepted_restart(complex_parameter):
    source = SOURCE
    if complex_parameter:
        source = source.replace("parameter omega:1/s=1;", "parameter omega:1/s=1; parameter rotation:array<complex<1/s>,2>=[math.complex(0,1),math.complex(0,1)];")
        source = source.replace("derivative(z)=math.complex(0,1)*omega*z;", "derivative(z)[0]=rotation[0]*z[0]; derivative(z)[1]=rotation[1]*z[1];")
    model = eqiora.compile(source=source)
    z, u, v = (model.field(name) for name in ("z", "u", "v"))
    tolerances = {(z, 0, component, imaginary): 1e-14
                  for component in range(2) for imaginary in (False, True)}
    tolerances.update({(u, 0, 0, False): 1e-14, (v, 0, 0, False): 1e-14})
    temporal = eqiora.time.ImplicitMidpoint(step_s=0.05, relative_tolerance=1e-12,
                                          absolute_tolerances=tolerances)
    plan = eqiora.resolve(model, temporal=temporal)
    plan = eqiora.Plan.from_bytes(plan.to_bytes())
    assert plan.capability.scalar_type == "f64"
    assert plan.capability.value_representation == "real-coordinates"
    assert isinstance(plan.temporal, eqiora.time.ImplicitMidpoint)
    assert plan.temporal.step_s == 0.05
    assert plan.temporal.absolute_tolerances == tolerances
    state = eqiora.State.initial(plan)
    state = eqiora.State.from_bytes(plan, state.to_bytes())
    assert state.value(z) == (1+0j, 2+0j)
    assert isinstance(state.value(z), tuple)
    assert all(isinstance(component, complex) for component in state.value(z))
    assert state.value(u) == 3.0
    with pytest.raises(KeyError):
        state.value(z, derivative_order=1)
    result = eqiora.run(plan, state=state, until_s=1.0, output_times_s=(0.025, 0.375, 0.5, 1.0))
    angle = 40 * math.atan(0.025)
    for component in range(2):
        for imaginary in (False, True):
            series = result.series(z, component=component, imaginary=imaginary)
            assert series.component == component and series.imaginary is imaginary
            expected = (component + 1) * (math.sin(angle) if imaginary else math.cos(angle))
            assert math.isclose(list(series)[-1][1], expected, rel_tol=0, abs_tol=1e-11)
    assert math.isclose(list(result.series(u))[-1][1], 3*math.cos(angle), abs_tol=1e-11)
    prefix = eqiora.run(plan, state=state, until_s=0.5, output_times_s=(0.5,))
    restart = eqiora.State.from_result(plan, prefix, time_s=0.5)
    restart = eqiora.State.from_bytes(plan, restart.to_bytes())
    half_angle = 20 * math.atan(0.025)
    for component, value in enumerate(restart.value(z)):
        expected = (component + 1) * complex(math.cos(half_angle), math.sin(half_angle))
        assert abs(value - expected) < 1e-11
    resumed = eqiora.run(plan, state=restart, until_s=1.0, output_times_s=(1.0,))
    for imaginary in (False, True):
        assert math.isclose(list(resumed.series(z, imaginary=imaginary))[-1][1],
                            list(result.series(z, imaginary=imaginary))[-1][1],
                            rel_tol=0, abs_tol=1e-11)
