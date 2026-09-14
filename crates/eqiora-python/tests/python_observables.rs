use pyo3::ffi::c_str;
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyModule};

// The fixture replaces sys.modules; imports can release the GIL mid-initialization.
static MODULE_FIXTURE: std::sync::Mutex<()> = std::sync::Mutex::new(());

#[test]
fn python_result_observations_retain_types_rules_and_exact_state_lineage() -> PyResult<()> {
    let _fixture = MODULE_FIXTURE.lock().expect("Python package fixture lock");
    Python::initialize();
    Python::attach(|py| {
        let locals = PyDict::new(py);
        locals.set_item("eqiora", public_module(py)?)?;
        py.run(c_str!(r#"
import math
linear = eqiora.solve.Linear(algorithm=eqiora.solve.LinearSolver.BiConjugateGradientStabilized, preconditioner=eqiora.solve.Preconditioner.Identity, reduction=eqiora.solve.Reduction.Reproducible, provider=eqiora.solve.SolverProvider.reference(), relative_tolerance=1e-12, absolute_tolerance=1e-14, maximum_iterations=100)
finite = eqiora.compile(source="model M() { domain P = scalar_physical(across voltage: 1, through current: 1); port a: P; port b: P; connect a, b; relation voltage { a.voltage=2; } relation ground { b.current=0; } observable twice: 1=a.voltage+a.voltage; }")
output = finite.observable("twice")
assert isinstance(output, eqiora.ObservableRef)
assert finite.observable(output.id) == output
assert eqiora.Model.from_bytes(finite.to_bytes()).observable(output.id) == output
finite_linear = eqiora.solve.Linear(algorithm=eqiora.solve.LinearSolver.SparseLu, preconditioner=eqiora.solve.Preconditioner.Identity, reduction=eqiora.solve.Reduction.Fast, provider=eqiora.solve.SolverProvider.faer(), relative_tolerance=1e-12, absolute_tolerance=1e-14, maximum_iterations=100)
plan = eqiora.resolve(finite, solve=finite_linear)
assert isinstance(plan.capability, eqiora.solve.AlgebraicPlanView)
assert plan.capability.unknown_count == 4
result = eqiora.run(plan, state=eqiora.State.initial(plan))
observed = result.observe(output)
assert math.isclose(observed.value, 4.0, abs_tol=1e-12)
assert observed.value_type == eqiora.ValueType.real(eqiora.Dimension())
assert observed.evaluation_kind == "value"
assert observed.observable_id == output.id
assert observed.quadrature is observed.quadrature_points is None
assert eqiora.Result.from_bytes(plan, result.to_bytes()).observe(output).result_identity == observed.result_identity
for invalid in (lambda: finite.observable("a"), lambda: result.observe("twice"),
                lambda: result.observe_terminal(output),
                lambda: result.observe_time_integral(output, quadrature=eqiora.time.TimeFunctionalQuadrature.AcceptedStepSimpson),
                lambda: result.observe(output, quadrature_points=2)):
    try:
        invalid()
    except (ValueError, TypeError):
        pass
    else:
        raise AssertionError("invalid Observable selection or rule was admitted")
foreign = eqiora.compile(source="model M() { domain P = scalar_physical(across voltage: 1, through current: 1); port a: P; port b: P; connect a, b; relation voltage { a.voltage=3; } relation ground { b.current=0; } observable twice: 1=a.voltage+a.voltage; }")
try:
    result.observe(foreign.observable("twice"))
except ValueError:
    pass
else:
    raise AssertionError("foreign ObservableRef was admitted")

graph = eqiora.geometry.GeometryGraph()
interval = graph.interval(bounds=(0.0, 1.0))
geometry = graph.build(interval, named_topology={"body": interval.region, "left": interval.boundaries[0], "right": interval.boundaries[1]})
source = """
public component Heat(support body: volume(ambient_dimension=1), support left: boundary(parent=body), support right: boundary(parent=body)) {
  variable temperature: K on body;
  parameter capacity: J/(K*m)=3;
  relation balance on body { -div(grad(temperature))=12[K/m^2]; }
  relation left_value on left { trace(temperature)=300[K]; }
  relation right_value on right { trace(temperature)=300[K]; }
  observable energy: J=integral(capacity*(temperature-300[K]), measure(body));
  observable endpoint: K=integral(trace(temperature), measure(left));
}
"""
model = eqiora.compile(source=source, geometry=geometry, entry="Heat", bindings={"body": geometry.selection("body"), "left": (geometry.selection("left"), geometry.selection("body")), "right": (geometry.selection("right"), geometry.selection("body"))})
mesh = eqiora.meshing.generate(eqiora.meshing.resolve(geometry, eqiora.meshing.CartesianMesher(cells=(4,))))
plan = eqiora.resolve(model, mesh=mesh, spatial=eqiora.fem.Q1(), solve=linear)
result = eqiora.run(plan)
energy, endpoint = model.observable("definition.energy"), model.observable("definition.endpoint")
value = result.observe(energy, quadrature_points=2)
# Q1 hats integrate nodal T-300=(0,9/8,3/2,9/8,0) to 15/16; capacity is 3.
assert math.isclose(value.value, 45/16, abs_tol=1e-9)
assert value.quadrature == "GaussLegendre" and value.quadrature_dimension == 1
assert value.quadrature_points == 2
boundary = result.observe(endpoint, quadrature_points=1)
assert math.isclose(boundary.value, 300, abs_tol=1e-10)
assert boundary.quadrature == "Point" and boundary.quadrature_dimension == 0
field = model.field("definition.temperature")
tangent = result.observable_state_tangent({field: (eqiora.Dimension(temperature=1), [2.0]*5)})
jvp = result.observe_state_jvp(energy, tangent, quadrature_points=2)
assert math.isclose(jvp.value, 6.0, abs_tol=1e-10)
assert jvp.evaluation_kind == "state-jvp" and jvp.value_type == value.value_type
assert jvp.result_identity == tangent.result_identity == value.result_identity
for invalid in (lambda: result.observe(energy), lambda: result.observe(endpoint, quadrature_points=2),
                lambda: result.observable_state_tangent({field: (eqiora.Dimension(), [2.0]*5)}),
                lambda: result.observable_state_tangent({field: (eqiora.Dimension(temperature=1), [2.0]*4)}),
                lambda: result.observable_state_tangent({field: (eqiora.Dimension(temperature=1), [float("nan")]*5)})):
    try:
        invalid()
    except (ValueError, eqiora.ValidationError):
        pass
    else:
        raise AssertionError("invalid spatial observation policy or tangent was admitted")
other_plan = eqiora.resolve(model, mesh=mesh, spatial=eqiora.fem.Q1(), solve=eqiora.solve.Linear(algorithm=eqiora.solve.LinearSolver.BiConjugateGradientStabilized, preconditioner=eqiora.solve.Preconditioner.Identity, reduction=eqiora.solve.Reduction.Reproducible, provider=eqiora.solve.SolverProvider.reference(), relative_tolerance=1e-10, absolute_tolerance=1e-12, maximum_iterations=100))
other_result = eqiora.run(other_plan)
try:
    other_result.observe_state_jvp(energy, tangent, quadrature_points=2)
except eqiora.ValidationError:
    pass
else:
    raise AssertionError("stale Result tangent was admitted")
"#), Some(&locals), Some(&locals))
    })
}

#[test]
fn python_time_functionals_use_accepted_history_and_exact_model_lineage() -> PyResult<()> {
    let _fixture = MODULE_FIXTURE.lock().expect("Python package fixture lock");
    Python::initialize();
    Python::attach(|py| {
        let locals = PyDict::new(py);
        locals.set_item("eqiora", public_module(py)?)?;
        py.run(c_str!(r#"
import base64
import json
import math
source = "model Decay() { state x: 1; initial { x=3; } parameter rate: 1/s=2; relation flow { derivative(x)=-rate*x; } observable sample: 1=x; }"
model = eqiora.compile(source=source)
field = model.field('x')
observable = model.observable('sample')
plan = eqiora.resolve(model, temporal=eqiora.time.Tsitouras45(initial_step_s=1e-3, relative_tolerance=1e-11, absolute_tolerances={field: 1e-13}))
rule = eqiora.time.TimeFunctionalQuadrature.AcceptedStepSimpson
sparse = eqiora.run(plan, state=eqiora.State.initial(plan), until_s=1.0, output_times_s=(0.25,))
dense = eqiora.run(plan, state=eqiora.State.initial(plan), until_s=1.0, output_times_s=tuple(i/10 for i in range(1, 10)))
terminal = sparse.observe_terminal(observable)
integral = sparse.observe_time_integral(observable, quadrature=rule)
assert isinstance(terminal, eqiora.TrajectoryObservation)
assert terminal.evaluation_kind == 'terminal' and terminal.quadrature is None
assert terminal.interval_s == (1.0, 1.0)
assert terminal.endpoint_convention == 'terminal-after-events'
assert integral.interval_s == (0.0, 1.0)
assert integral.endpoint_convention == 'fixed-interval-dt'
assert integral.evaluation_kind == 'time-integral' and integral.quadrature == rule
# Independent solution x(t)=3exp(-2t), J=3(1-exp(-2))/2 over [0,1].
assert math.isclose(terminal.value, 3*math.exp(-2), abs_tol=1e-9, rel_tol=0)
assert math.isclose(integral.value, 1.5*(1-math.exp(-2)), abs_tol=1e-8, rel_tol=0)
assert terminal.value_type == eqiora.ValueType.real(eqiora.Dimension())
assert integral.value_type == eqiora.ValueType.real(eqiora.Dimension(time=1))
assert integral.observable_id == terminal.observable_id == observable.id
assert terminal.result_identity == integral.result_identity == json.loads(sparse.to_bytes())['identity']
trajectory_wire = json.loads(base64.b64decode(json.loads(sparse.to_bytes())["content"]["payload"]["trajectory_base64"]))
assert integral.trajectory_identity == terminal.trajectory_identity == trajectory_wire["identity"]
assert dense.observe_terminal(observable).value == terminal.value
assert dense.observe_time_integral(observable, quadrature=rule).value == integral.value
assert dense.observe_time_integral(observable, quadrature=rule).result_identity != integral.result_identity
replayed_plan = eqiora.Plan.from_bytes(plan.to_bytes())
replayed = eqiora.Result.from_bytes(replayed_plan, sparse.to_bytes())
receipt = replayed.observe_time_integral(observable, quadrature=rule)
assert receipt.value == integral.value
assert receipt.result_identity == integral.result_identity
assert receipt.trajectory_identity == integral.trajectory_identity
foreign = eqiora.compile(source=source.replace('1/s=2', '1/s=3')).observable('sample')
for invalid in (lambda: sparse.observe_terminal(foreign),
                lambda: sparse.observe_time_integral(foreign, quadrature=rule),
                lambda: sparse.observe_time_integral(observable),
                lambda: sparse.observe_time_integral(observable, quadrature='AcceptedStepSimpson'),
                lambda: sparse.observe_time_integral(observable, quadrature=None),
                lambda: sparse.observe_terminal('sample')):
    try:
        invalid()
    except (ValueError, TypeError):
        pass
    else:
        raise AssertionError('foreign reference or missing/untyped quadrature was admitted')
try:
    integral.value = 8
except AttributeError:
    pass
else:
    raise AssertionError('trajectory observation must be immutable')
"#), Some(&locals), Some(&locals))
    })
}

#[test]
fn python_event_functionals_use_reset_history_and_explicit_policy() -> PyResult<()> {
    let _fixture = MODULE_FIXTURE.lock().expect("Python package fixture lock");
    Python::initialize();
    Python::attach(|py| {
        let locals = PyDict::new(py);
        locals.set_item("eqiora", public_module(py)?)?;
        py.run(c_str!(r#"
import json
import math
source = "model Reset() { state x: 1; initial { x=0; } parameter threshold: 1=0.4; relation flow { derivative(x)=1[1/s]; } event hit=crossing(x-threshold,direction=rising); relation reset at hit { next(x)=0; } observable sample: 1=x; }"
model = eqiora.compile(source=source)
field, event = model.field('x'), model.activation('hit')
threshold = model.parameter('threshold')
observable = model.observable('sample')
assert isinstance(event, eqiora.ActivationRef)
assert model.activation(event.id) == event
assert eqiora.Model.from_bytes(model.to_bytes()).activation(event.id) == event
policy = eqiora.time.EventPolicy(max_events=1, guard_tolerances=(eqiora.time.GuardTolerance(event, 1e-10, eqiora.Dimension()),))
forward = eqiora.time.ForwardSensitivity(
    relative_tolerance=1e-10,
    absolute_tolerances=(eqiora.time.SensitivityTolerance(
        field, threshold, 1e-12, eqiora.Dimension(),
    ),),
)
def temporal(events):
    return eqiora.time.Tsitouras45(initial_step_s=1e-3, relative_tolerance=1e-11, absolute_tolerances={field:1e-13}, events=events, forward_sensitivities=forward)
plan = eqiora.resolve(model, temporal=temporal(policy))
rule = eqiora.time.TimeFunctionalQuadrature.AcceptedStepSimpson
sparse = eqiora.run(plan, state=eqiora.State.initial(plan), until_s=0.7, output_times_s=(0.2,))
dense = eqiora.run(plan, state=eqiora.State.initial(plan), until_s=0.7, output_times_s=(0.1,0.2,0.3,0.5,0.6))
terminal = sparse.observe_terminal(observable)
integral = sparse.observe_time_integral(observable, quadrature=rule)
# x(t)=t for t<0.4 and t-0.4 afterwards: x(0.7)=0.3 and J=(0.4^2+0.3^2)/2=0.125.
assert math.isclose(terminal.value, 0.3, abs_tol=1e-9, rel_tol=0)
assert math.isclose(integral.value, 0.125, abs_tol=1e-9, rel_tol=0)
assert terminal.value_type == eqiora.ValueType.real(eqiora.Dimension())
assert integral.value_type == eqiora.ValueType.real(eqiora.Dimension(time=1))
assert terminal.evaluation_kind == 'terminal' and terminal.quadrature is None
assert integral.evaluation_kind == 'time-integral' and integral.quadrature == rule
direction = {threshold: (eqiora.Dimension(), 1.0)}
terminal_jvp = sparse.observe_terminal_parameter_jvp(observable, direction)
integral_jvp = sparse.observe_time_integral_parameter_jvp(observable, direction, quadrature=rule)
# Raising the reset threshold delays the reset: dx(0.7)/dthreshold=-1 and dJ/dthreshold=0.1.
assert math.isclose(terminal_jvp.value, -1.0, abs_tol=1e-8, rel_tol=0)
assert math.isclose(integral_jvp.value, 0.1, abs_tol=1e-8, rel_tol=0)
assert terminal_jvp.parameter_jvp and integral_jvp.parameter_jvp
assert terminal_jvp.evaluation_kind == 'terminal-parameter-jvp'
assert integral_jvp.evaluation_kind == 'time-integral-parameter-jvp'
assert terminal.observable_id == integral.observable_id == observable.id
assert terminal.result_identity == integral.result_identity == json.loads(sparse.to_bytes())['identity']
assert dense.observe_terminal(observable).value == terminal.value
assert dense.observe_time_integral(observable, quadrature=rule).value == integral.value
assert dense.observe_time_integral(observable, quadrature=rule).result_identity != integral.result_identity
replayed_plan = eqiora.Plan.from_bytes(plan.to_bytes())
replayed = eqiora.Result.from_bytes(replayed_plan, sparse.to_bytes())
replayed_receipt = replayed.observe_time_integral(observable, quadrature=rule)
assert replayed_receipt.value == integral.value
assert replayed_receipt.result_identity == integral.result_identity
assert replayed_receipt.trajectory_identity == integral.trajectory_identity
assert replayed.observe_terminal(observable).value == terminal.value
assert replayed.observe_terminal_parameter_jvp(observable, direction).value == terminal_jvp.value
foreign = eqiora.compile(source=source.replace('1=0.4', '1=0.45'))
wrong_dimension = eqiora.time.EventPolicy(max_events=1, guard_tolerances=(eqiora.time.GuardTolerance(event, 1e-10, eqiora.Dimension(time=1)),))
foreign_policy = eqiora.time.EventPolicy(max_events=1, guard_tolerances=(eqiora.time.GuardTolerance(foreign.activation('hit'), 1e-10, eqiora.Dimension()),))
for invalid in (lambda: eqiora.resolve(model, temporal=temporal(None)),
                lambda: eqiora.resolve(model, temporal=temporal(wrong_dimension)),
                lambda: eqiora.resolve(model, temporal=temporal(foreign_policy)),
                lambda: eqiora.run(plan, until_s=0.9),
                lambda: sparse.observe_terminal(foreign.observable('sample')),
                lambda: sparse.observe_time_integral(observable)):
    try:
        invalid()
    except (eqiora.ValidationError, ValueError, TypeError):
        pass
    else:
        raise AssertionError('missing/wrong event policy, exhausted event bound, or foreign Observable was accepted')
for receipt in (terminal, integral):
    try:
        receipt.value = 8
    except AttributeError:
        pass
    else:
        raise AssertionError('event functional receipt must be frozen')
"#), Some(&locals), Some(&locals))
    })
}

fn public_module(py: Python<'_>) -> PyResult<Bound<'_, PyModule>> {
    let native = pyo3::wrap_pymodule!(_eqiora::_eqiora)(py);
    let package_directory = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
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
package = importlib.util.module_from_spec(spec)
sys.modules["eqiora"] = package
sys.modules["eqiora._eqiora"] = native
spec.loader.exec_module(package)
"#
        ),
        None,
        Some(&locals),
    )?;
    py.import("eqiora")
}
