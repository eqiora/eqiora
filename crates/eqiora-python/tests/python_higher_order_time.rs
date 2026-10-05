use pyo3::ffi::c_str;
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyDictMethods, PyModule};
use std::path::Path;

#[test]
fn higher_order_python_solve_preserves_derivative_coordinates_and_replay() -> PyResult<()> {
    Python::initialize();
    Python::attach(|py| {
        let module = public_module(py)?;
        let locals = PyDict::new(py);
        locals.set_item("eqiora", &module)?;
        py.run(c_str!(r#"
import math
complex_model = eqiora.compile(source="model ComplexState() { state z:array<complex<m>,2>; initial { z=[math.complex(1,2)*1[m],math.complex(3,4)*1[m]]; } relation flow { derivative(z)=math.complex(0,1)*1[1/s]*z; } }")
z = complex_model.field("z")
complex_plan = eqiora.resolve(complex_model, temporal=eqiora.time.ImplicitMidpoint(step_s=0.05, relative_tolerance=1e-12, absolute_tolerances={(z,0,c,i):1e-14 for c in range(2) for i in (False,True)}).with_conserved_norm([z],target=30.0,tolerance=1e-10,dimension=eqiora.Dimension(length=2)))
complex_plan = eqiora.Plan.from_bytes(complex_plan.to_bytes())
complex_state = eqiora.State.initial(complex_plan)
assert complex_state.value(z) == (1+2j, 3+4j)
complex_state = eqiora.State.from_bytes(complex_plan, complex_state.to_bytes())
assert complex_state.value(z) == (1+2j, 3+4j)
complex_result = eqiora.run(complex_plan, state=complex_state, until_s=0.5, output_times_s=(0.5,))
complex_state = eqiora.State.from_result(complex_plan, complex_result, time_s=0.5)
angle = 20*math.atan(0.025)
rotation = complex(math.cos(angle),math.sin(angle))
for actual, seed in zip(complex_state.value(z), (1+2j, 3+4j)):
    assert abs(actual-seed*rotation) < 1e-11
model = eqiora.compile(source="model Oscillator() { parameter k:1/s^2=4; state x:m; initial { x=1[m]; derivative(x)=2[m/s]; } relation motion { derivative(derivative(x))+k*x=0[m/s^2]; } observable velocity:m/s=derivative(x); observable acceleration:m/s^2=derivative(derivative(x)); observable squared_acceleration:m^2/s^2=derivative(derivative(x*x)); }")
field = model.field("x")
temporal = eqiora.time.Tsitouras45(initial_step_s=0.001, relative_tolerance=1e-10, absolute_tolerances={(field, 1, 0, False):2e-12, (field, 0, 0, False):1e-12})
assert temporal.absolute_tolerances == {(field, 0, 0, False):1e-12, (field, 1, 0, False):2e-12}
plan = eqiora.resolve(model, temporal=temporal)
assert plan.fields == (field,)
restored_plan = eqiora.Plan.from_bytes(plan.to_bytes())
assert restored_plan.temporal.absolute_tolerances == temporal.absolute_tolerances
for inspected in (plan, restored_plan):
    form = inspected.formulation
    assert form.effective == eqiora.FormulationKind.FirstOrderEvolution
    assert form.requested == eqiora.FormulationSelectionMode.Automatic
    assert form.state_coordinates == ((field, 0, 0, False), (field, 1, 0, False))
    assert form.source_relation_id is not None
    assert form.source_relation_id == plan.formulation.source_relation_id
    assert "time.derive.v1.companion-equations" in form.rule_ids
initial = eqiora.State.initial(restored_plan)
assert initial.state_coordinates == ((field, 0, 0, False), (field, 1, 0, False))
assert initial.value(field) == 1.
assert initial.value(field, derivative_order=1) == 2.
restored_initial = eqiora.State.from_bytes(restored_plan, initial.to_bytes())
assert restored_initial.value(field, derivative_order=1) == 2.
result = eqiora.run(restored_plan, state=restored_initial, until_s=1., output_times_s=(0.25, 0.5, 1.))
restored_result = eqiora.Result.from_bytes(restored_plan, result.to_bytes())
assert list(restored_result.series(field, derivative_order=1)) == list(result.series(field, derivative_order=1))
x = result.series(field)
v = result.series(field, derivative_order=1)
assert x.field == v.field == field
assert x.derivative_order == 0 and v.derivative_order == 1
assert x.dimension == (0, 1, 0, 0, 0, 0, 0)
assert v.dimension == (0, 1, -1, 0, 0, 0, 0)
# Exact solution of x''+4*x=0 with x(0)=1, x'(0)=2.
for time, value in x:
    assert abs(value - math.cos(2*time) - math.sin(2*time)) < 1e-8
for time, value in v:
    assert abs(value - 2*math.cos(2*time) + 2*math.sin(2*time)) < 2e-8
# The authored first-order system has identical physical initial data and controls.
first_order = eqiora.compile(source="model FirstOrder() { parameter k:1/s^2=4; state x:m; state v:m/s; initial { x=1[m]; v=2[m/s]; } relation motion { derivative(x)=v; derivative(v)=-k*x; } }")
authored_x, authored_v = first_order.field("x"), first_order.field("v")
authored_plan = eqiora.resolve(first_order, temporal=eqiora.time.Tsitouras45(initial_step_s=0.001, relative_tolerance=1e-10, absolute_tolerances={(authored_x, 0, 0, False):1e-12, (authored_v, 0, 0, False):2e-12}))
authored_result = eqiora.run(authored_plan, state=eqiora.State.initial(authored_plan), until_s=1., output_times_s=(0.25, 0.5, 1.))
for normalized, authored in ((x, authored_result.series(authored_x)), (v, authored_result.series(authored_v))):
    normalized_points, authored_points = list(normalized), list(authored)
    assert len(normalized_points) == len(authored_points)
    for (time, value), (authored_time, authored_value) in zip(normalized_points, authored_points):
        assert time == authored_time
        assert abs(value - authored_value) < 2e-8
# Restart retains both coordinates at the explicit nonzero boundary.
restart = eqiora.State.from_result(restored_plan, restored_result, time_s=0.5)
assert restart.time_s == 0.5
restart = eqiora.State.from_bytes(restored_plan, restart.to_bytes())
resumed = eqiora.run(restored_plan, state=restart, until_s=1., output_times_s=(1.,))
for order, expected in ((0, math.cos(2)+math.sin(2)), (1, 2*math.cos(2)-2*math.sin(2))):
    assert abs(list(resumed.series(field, derivative_order=order))[-1][1] - expected) < 2e-8
# Lower derivatives are stored coordinates; highest derivatives come from the flow.
x1 = math.cos(2) + math.sin(2)
v1 = 2*math.cos(2) - 2*math.sin(2)
for name, expected in (("velocity", v1), ("acceleration", -4*x1), ("squared_acceleration", 2*v1*v1-8*x1*x1)):
    assert abs(result.observe_terminal(model.observable(name)).value - expected) < 1e-7
    assert restored_result.observe_terminal(model.observable(name)).value == result.observe_terminal(model.observable(name)).value
for select in (lambda: initial.value(field, derivative_order=2), lambda: result.series(field, derivative_order=2)):
    try:
        select()
    except KeyError:
        pass
    else:
        raise AssertionError("unrepresented derivative order must not select displacement")
# A third-order equation uses the same execution and result path.
jerk = eqiora.compile(source="model Jerk() { state q:m; initial { q=0[m]; derivative(q)=0[m/s]; derivative(derivative(q))=0[m/s^2]; } relation motion { derivative(derivative(derivative(q)))=6[m/s^3]; } }")
q = jerk.field("q")
jerk_plan = eqiora.resolve(jerk, temporal=eqiora.time.Tsitouras45(initial_step_s=0.001, relative_tolerance=1e-10, absolute_tolerances={(q, k, 0, False):1e-12 for k in range(3)}))
jerk_result = eqiora.run(jerk_plan, state=eqiora.State.initial(jerk_plan), until_s=1., output_times_s=(0.25, 0.5, 1.))
for order, exact in enumerate((lambda t: t**3, lambda t: 3*t*t, lambda t: 6*t)):
    for time, value in jerk_result.series(q, derivative_order=order):
        assert abs(value - exact(time)) < 1e-9

# Explicit time must keep its enclosing timeline across a nonzero restart.
# q''=t, q(0)=q'(0)=0 gives q=t^3/6, q'=t^2/2.
timed = eqiora.compile(source="model Timed() { state q:m; initial { q=0[m]; derivative(q)=0[m/s]; } relation motion { derivative(derivative(q))=time()*1[m/s^3]; } }")
q = timed.field("q")
timed_plan = eqiora.resolve(timed, temporal=eqiora.time.Tsitouras45(initial_step_s=0.001, relative_tolerance=1e-10, absolute_tolerances={(q, order, 0, False):1e-12 for order in (0, 1)}))
prefix = eqiora.run(timed_plan, state=eqiora.State.initial(timed_plan), until_s=0.5, output_times_s=(0.5,))
restart = eqiora.State.from_result(timed_plan, prefix, time_s=0.5)
assert abs(restart.value(q) - 1/48) < 1e-9
assert abs(restart.value(q, derivative_order=1) - 1/8) < 1e-9
suffix = eqiora.run(timed_plan, state=eqiora.State.from_bytes(timed_plan, restart.to_bytes()), until_s=1., output_times_s=(1.,))
for order, expected in ((0, 1/6), (1, 1/2)):
    assert abs(list(suffix.series(q, derivative_order=order))[-1][1] - expected) < 1e-9

# Fresh initialization binds time too; sensitivities start at this chosen instant.
fresh = eqiora.compile(source="model Fresh() { parameter p:m/s^3=1; state q:m; initial { time()*q=time()*time()*time()*time()*1[m/s^3]/6; derivative(q)=time()*time()*1[m/s^3]/2; } relation flow { derivative(derivative(q))=p*time(); } observable position:m=q; observable velocity:m/s=derivative(q); }")
q, p = fresh.field("q"), fresh.parameter("p")
control = eqiora.time.ForwardSensitivity(relative_tolerance=1e-10, absolute_tolerances=(eqiora.time.SensitivityTolerance((q, 0, 0, False), p, 1e-12, eqiora.Dimension(time=3)), eqiora.time.SensitivityTolerance((q, 1, 0, False), p, 1e-12, eqiora.Dimension(time=2))))
fresh_plan = eqiora.resolve(fresh, temporal=eqiora.time.Tsitouras45(initial_step_s=0.001, relative_tolerance=1e-10, absolute_tolerances={(q, order, 0, False):1e-12 for order in (0, 1)}, forward_sensitivities=control))
# The initial position is underdetermined at zero but regular at t=2.
# Plan admission must not secretly solve these equations at zero.
try:
    eqiora.State.initial(fresh_plan, time_s=0.)
except eqiora.ValidationError:
    pass
else:
    raise AssertionError("singular zero-time initialization must not invent a position")
fresh_state = eqiora.State.initial(fresh_plan, time_s=2.)
assert fresh_state.time_s == 2.
assert abs(fresh_state.value(q)-8/6) < 1e-9
assert abs(fresh_state.value(q, derivative_order=1)-2) < 1e-9
fresh_state = eqiora.State.from_bytes(fresh_plan, fresh_state.to_bytes())
fresh_result = eqiora.run(fresh_plan, state=fresh_state, until_s=3., output_times_s=(3.,))
fresh_result = eqiora.Result.from_bytes(fresh_plan, fresh_result.to_bytes())
direction = {p: (eqiora.Dimension(length=1, time=-3), 1.)}
# Integrating u and (3-u)*u over [2,3] gives 5/2 and 7/6.
for name, value, tangent in (("position", 27/6, 7/6), ("velocity", 9/2, 5/2)):
    observable = fresh.observable(name)
    assert abs(fresh_result.observe_terminal(observable).value-value) < 1e-9
    assert abs(fresh_result.observe_terminal_parameter_jvp(observable, direction).value-tangent) < 1e-9

# q=1+p*t, so D(q²)=2p(1+p*t). At p=2,T=1 its value is 12,
# its p-directional derivative is 10, integral is 8 and integral derivative is 6.
linear = eqiora.compile(source="model Linear() { parameter p:m/s=2; state q:m; initial { q=1[m]; } relation flow { derivative(q)=p; } observable rate:m^2/s=derivative(q*q); observable unsupplied:m/s^2=derivative(derivative(q)); }")
q = linear.field("q")
p = linear.parameter("p")
control = eqiora.time.ForwardSensitivity(relative_tolerance=1e-10, absolute_tolerances=(eqiora.time.SensitivityTolerance((q, 0, 0, False), p, 1e-12, eqiora.Dimension(time=1)),))
linear_plan = eqiora.resolve(linear, temporal=eqiora.time.Tsitouras45(initial_step_s=0.001, relative_tolerance=1e-10, absolute_tolerances={(q, 0, 0, False):1e-12}, forward_sensitivities=control))
linear_result = eqiora.run(linear_plan, state=eqiora.State.initial(linear_plan), until_s=1., output_times_s=(0.5,))
rate = linear.observable("rate")
rule = eqiora.time.TimeFunctionalQuadrature.AcceptedStepSimpson
assert abs(linear_result.observe_terminal(rate).value - 12.) < 1e-9
assert abs(linear_result.observe_time_integral(rate, quadrature=rule).value - 8.) < 1e-9
direction = {p: (eqiora.Dimension(length=1, time=-1), 1.)}
assert abs(linear_result.observe_terminal_parameter_jvp(rate, direction).value - 10.) < 1e-8
assert abs(linear_result.observe_time_integral_parameter_jvp(rate, direction, quadrature=rule).value - 6.) < 1e-8

try:
    linear_result.observe_terminal(linear.observable("unsupplied"))
except eqiora.ValidationError as error:
    assert "derivative order is not supplied" in str(error)
else:
    raise AssertionError("an Observable must not invent a higher derivative state")

"#), Some(&locals), Some(&locals))
    })
}

fn public_module(py: Python<'_>) -> PyResult<Bound<'_, PyModule>> {
    let native = pyo3::wrap_pymodule!(_eqiora::_eqiora)(py);
    let package_directory = Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../../bindings/python/python/eqiora")
        .canonicalize()?;
    let locals = PyDict::new(py);
    locals.set_item("native", native.bind(py))?;
    locals.set_item("package_directory", package_directory.to_string_lossy())?;
    py.run(
        c_str!(
            r#"
import importlib.util
import pathlib
import sys

package_path = pathlib.Path(package_directory)
spec = importlib.util.spec_from_file_location(
    "eqiora",
    package_path / "__init__.py",
    submodule_search_locations=[str(package_path)],
)
assert spec is not None and spec.loader is not None
package = importlib.util.module_from_spec(spec)
sys.modules["eqiora"] = package
sys.modules["eqiora._eqiora"] = native
spec.loader.exec_module(package)
"#
        ),
        None,
        Some(&locals),
    )?;
    Ok(locals
        .get_item("package")?
        .expect("the package loader must bind eqiora")
        .cast_into::<PyModule>()?)
}
