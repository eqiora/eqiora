use std::path::Path;

use pyo3::ffi::c_str;
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyDictMethods, PyModule};

#[test]
fn python_common_ode_route_matches_independent_exponential_decay() -> PyResult<()> {
    Python::initialize();
    Python::attach(|py| {
        let module = public_module(py)?;
        let locals = PyDict::new(py);
        locals.set_item("eqiora", module)?;
        locals.set_item(
            "source",
            include_str!("../../../verify/interfaces/python-common-ode-lifecycle/models/decay.eqi"),
        )?;
        py.run(
            c_str!(
                r#"
import asyncio
import math

SOURCE = source

def resolve(model):
    field = model.field(model.field_ids[0])
    plan = eqiora.resolve(
        model,
        temporal=eqiora.time.Tsitouras45(
            initial_step_s=0.01,
            relative_tolerance=1.0e-9,
            absolute_tolerances={field: 1.0e-11},
        ),
    )
    return field, plan

model = eqiora.compile(source=SOURCE)
field, plan = resolve(model)
plan_bytes = plan.to_bytes()
portable_plan = eqiora.Plan.from_bytes(plan_bytes)
portable_field = portable_plan.fields[0]
assert portable_plan.identity == plan.identity
assert portable_plan.to_bytes() == plan_bytes
assert portable_plan.model.to_bytes() == model.to_bytes()
assert portable_plan.mesh is None
assert portable_plan.spatial is None
assert portable_plan.solve is None
assert portable_plan.temporal.initial_step_s == plan.temporal.initial_step_s
assert portable_plan.temporal.relative_tolerance == plan.temporal.relative_tolerance
assert portable_plan.temporal.absolute_tolerances == {portable_field: 1.0e-11}
try:
    eqiora.Plan.from_bytes(plan_bytes + b"\n")
except eqiora.ValidationError:
    pass
else:
    raise AssertionError("noncanonical Plan bytes must reject")
assert plan.mesh is None
assert plan.spatial is None
assert plan.solve is None
assert isinstance(plan.capability, eqiora.time.OdePlanView)
assert not hasattr(plan.capability, "scaling")
assert plan.solve is None
assert plan.temporal is not None
assert plan.temporal.absolute_tolerances == {field: 1.0e-11}
state = eqiora.State.initial(plan)
assert state.value(field) == 1.0
result = eqiora.run(
    plan,
    state=state,
    until_s=0.2,
    output_times_s=(0.1, 0.2),
)
series = result.series(field)
assert series.field == field
assert series.dimension == (0, 0, 0, 0, 0, 0, 0)
assert len(series) == 2
assert [time_s for time_s, _ in series] == [0.1, 0.2]
for time_s, value in series:
    assert math.isclose(value, math.exp(-time_s), rel_tol=2.0e-8, abs_tol=2.0e-10)
portable_result = eqiora.run(
    portable_plan,
    state=eqiora.State.initial(portable_plan),
    until_s=0.2,
    output_times_s=(0.2,),
)
assert math.isclose(
    portable_result.series(portable_field).values.numpy()[0],
    math.exp(-0.2),
    rel_tol=2.0e-8,
    abs_tol=2.0e-10,
)

replayed = eqiora.Model.from_bytes(model.to_bytes())
replayed_field, replayed_plan = resolve(replayed)
assert replayed_plan.identity == plan.identity
assert replayed_plan.model_digest == model.digest

recompiled = eqiora.compile(source=SOURCE)
_, recompiled_plan = resolve(recompiled)
assert recompiled_plan.identity == plan.identity
assert recompiled_plan.model_digest == plan.model_digest

other_source = SOURCE.replace("parameter rate: 1 / s = 1;", "parameter rate: 1 / s = 2;")
assert other_source != SOURCE
other = eqiora.compile(source=other_source)
other_field, other_plan = resolve(other)
assert other_plan.model_digest != plan.model_digest
foreign_temporal = eqiora.time.Tsitouras45(
    initial_step_s=0.01,
    relative_tolerance=1.0e-9,
    absolute_tolerances={other_field: 1.0e-11},
)
try:
    eqiora.resolve(
        model,
        temporal=foreign_temporal,
    )
except TypeError:
    pass
else:
    raise AssertionError("foreign absolute-tolerance FieldRef was admitted")
try:
    eqiora.run(
        plan,
        state=eqiora.State.initial(other_plan),
        until_s=0.2,
        output_times_s=(0.2,),
    )
except eqiora.ValidationError:
    pass
else:
    raise AssertionError("foreign ODE State was admitted")
for selector in (other_field, "x"):
    try:
        result.series(selector)
    except (TypeError, ValueError):
        pass
    else:
        raise AssertionError("non-exact ODE Series selector was admitted")

changed = eqiora.resolve(
    model,
    temporal=eqiora.time.Tsitouras45(
        initial_step_s=0.02,
        relative_tolerance=1.0e-8,
        absolute_tolerances={field: 2.0e-11},
    ),
)
assert changed.identity != plan.identity
assert changed.fields == plan.fields

for outputs in ((), (0.0,), (0.2, 0.1), (0.3,), (float("nan"),), (float("inf"),)):
    try:
        eqiora.run(
            plan,
            state=state,
            until_s=0.2,
            output_times_s=outputs,
        )
    except eqiora.ValidationError:
        pass
    else:
        raise AssertionError(f"invalid ODE output schedule was admitted: {outputs!r}")
for kwargs in (
    dict(initial_step_s=True, relative_tolerance=1.0e-9, absolute_tolerances={field: 1.0e-11}),
    dict(initial_step_s=0.0, relative_tolerance=1.0e-9, absolute_tolerances={field: 1.0e-11}),
    dict(initial_step_s=float("nan"), relative_tolerance=1.0e-9, absolute_tolerances={field: 1.0e-11}),
    dict(initial_step_s=0.01, relative_tolerance=0.0, absolute_tolerances={field: 1.0e-11}),
    dict(initial_step_s=0.01, relative_tolerance=1.0e-9, absolute_tolerances={}),
    dict(initial_step_s=0.01, relative_tolerance=1.0e-9, absolute_tolerances={field: -1.0e-11}),
):
    try:
        eqiora.time.Tsitouras45(**kwargs)
    except (TypeError, eqiora.ValidationError):
        pass
    else:
        raise AssertionError(f"invalid Tsitouras45 controls were admitted: {kwargs!r}")
try:
    eqiora.resolve(model, temporal=eqiora.time.BackwardEuler(0.01))
except TypeError:
    pass
else:
    raise AssertionError("BackwardEuler entered the no-Mesh ODE arm")
try:
    eqiora.resolve(model, spatial=object(), temporal=plan.temporal)
except TypeError:
    pass
else:
    raise AssertionError("spatial policy entered the no-Mesh ODE arm")

async def await_same_result():
    submitted = eqiora.submit(
        replayed_plan,
        state=eqiora.State.initial(replayed_plan),
        until_s=0.2,
        output_times_s=(0.2,),
    )
    assert submitted.adapter_version == "0.16.2"
    assert submitted.cancel() is False
    awaited = await submitted
    assert awaited is submitted.result()
    assert math.isclose(
        awaited.series(replayed_field).values[0],
        math.exp(-0.2),
        rel_tol=2.0e-8,
        abs_tol=2.0e-10,
    )

asyncio.run(await_same_result())
"#
            ),
            Some(&locals),
            Some(&locals),
        )
    })
}

#[test]
fn python_event_policy_binds_exact_activations_units_and_plan_bytes() -> PyResult<()> {
    Python::initialize();
    Python::attach(|py| {
        let locals = PyDict::new(py);
        locals.set_item("eqiora", public_module(py)?)?;
        py.run(c_str!(r#"
source = """
model Ball() {
 state height:m; state velocity:m/s;
 parameter ground:m=0; parameter restitution:1=0.8;
 initial {height=1[m];velocity=0[m/s];}
 relation flight {derivative(height)=velocity;derivative(velocity)=-9.81[m/s^2];}
 event impact_h=crossing(height-ground,direction=falling);
 event impact_v=crossing(height-ground,direction=falling);
 relation reset_h at impact_h {next(height)=ground;}
 relation reset_v at impact_v {next(velocity)=-restitution*pre(velocity);}
}
"""
model = eqiora.compile(source=source)
h, v = model.activation('impact_h'), model.activation('impact_v')
assert isinstance(h, eqiora.ActivationRef)
assert h.model_digest == model.digest
assert model.activation(h.id) == h
assert eqiora.Model.from_bytes(model.to_bytes()).activation(h.id) == h
length = eqiora.Dimension(length=1)
h_tol = eqiora.time.GuardTolerance(h, 1e-8, length)
v_tol = eqiora.time.GuardTolerance(v, 1e-8, length)
events = eqiora.time.EventPolicy(max_events=8, guard_tolerances=(h_tol, v_tol))
assert events.model_digest == model.digest
assert {entry.activation for entry in events.guard_tolerances} == {h, v}
assert all(entry.value == 1e-8 and entry.dimension == length for entry in events.guard_tolerances)
def temporal(events=None):
    return eqiora.time.Tsitouras45(initial_step_s=1e-3, relative_tolerance=1e-9, absolute_tolerances={model.field('height'):1e-11,model.field('velocity'):1e-11}, events=events)
assert temporal().events is None
try:
    eqiora.resolve(model, temporal=temporal())
except eqiora.ValidationError:
    pass
else:
    raise AssertionError('Event Model requires explicit policy')
plan = eqiora.resolve(model, temporal=temporal(events))
assert plan.temporal.events.max_events == 8
assert plan.temporal.events.model_digest == model.digest
reopened = eqiora.Plan.from_bytes(plan.to_bytes())
assert reopened.identity == plan.identity
assert reopened.to_bytes() == plan.to_bytes()
assert {entry.activation for entry in reopened.temporal.events.guard_tolerances} == {h, v}
changed = eqiora.resolve(model, temporal=temporal(eqiora.time.EventPolicy(max_events=9,guard_tolerances=(h_tol,v_tol))))
assert changed.identity != plan.identity
foreign = eqiora.compile(source=source.replace('restitution:1=0.8', 'restitution:1=0.7')).activation('impact_v')
for invalid in (
 lambda: model.activation('height'),
 lambda: eqiora.time.GuardTolerance('impact_h',1e-8,length),
 lambda: eqiora.time.GuardTolerance(h,0,length),
 lambda: eqiora.time.EventPolicy(max_events=0,guard_tolerances=(h_tol,v_tol)),
 lambda: eqiora.time.EventPolicy(max_events=8,guard_tolerances=(h_tol,h_tol)),
 lambda: eqiora.time.EventPolicy(max_events=8,guard_tolerances=(h_tol,eqiora.time.GuardTolerance(foreign,1e-8,length))),
 lambda: eqiora.resolve(model,temporal=temporal(eqiora.time.EventPolicy(max_events=8,guard_tolerances=(h_tol,)))),
 lambda: eqiora.resolve(model,temporal=temporal(eqiora.time.EventPolicy(max_events=8,guard_tolerances=(h_tol,eqiora.time.GuardTolerance(v,1e-8,eqiora.Dimension()))))),
 lambda: eqiora.resolve(model,temporal=temporal(eqiora.time.EventPolicy(max_events=8,guard_tolerances=(h_tol,eqiora.time.GuardTolerance(v,2e-8,length))))),
):
    try:
        invalid()
    except (TypeError, ValueError, eqiora.ValidationError):
        pass
    else:
        raise AssertionError('invalid Activation, unit, group tolerance or event budget was admitted')
"#),Some(&locals),Some(&locals))
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

#[test]
fn python_common_finite_route_owns_exact_plan_state_and_result() -> PyResult<()> {
    Python::initialize();
    Python::attach(|py| {
        let locals = PyDict::new(py);
        locals.set_item("eqiora", public_module(py)?)?;
        locals.set_item(
            "source",
            format!(
                "{}\n{}",
                include_str!(
                    "../../../crates/eqiora-api/packages/Eqiora.Electrical.Basic/src/basic.eqi"
                ),
                include_str!("../../../examples/voltage_divider.eqi")
            ),
        )?;
        py.run(c_str!(r#"
model = eqiora.compile(source=source)
plan = eqiora.resolve(model, solve=eqiora.solve.Linear(algorithm=eqiora.solve.LinearSolver.SparseLu, preconditioner=eqiora.solve.Preconditioner.Identity, reduction=eqiora.solve.Reduction.Fast, provider=eqiora.solve.SolverProvider.faer(), relative_tolerance=1e-12, absolute_tolerance=1e-14, maximum_iterations=100))
assert plan.mesh is None
assert plan.temporal is None
state = eqiora.State.initial(plan)
state = eqiora.State.from_bytes(plan, state.to_bytes())
result = eqiora.run(plan, state=state)
assert result.plan_key == plan.identity
assert result.to_bytes() == eqiora.Result.from_bytes(plan, result.to_bytes()).to_bytes()
"#), None, Some(&locals))
    })
}
