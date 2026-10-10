use pyo3::ffi::c_str;
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyDictMethods, PyModule};

#[test]
fn spatial_state_projects_every_stored_field_and_replays() -> PyResult<()> {
    Python::initialize();
    Python::attach(|py| {
        let locals = PyDict::new(py);
        locals.set_item("eqiora", public_module(py)?)?;
        py.run(
            c_str!(
                r#"
import numpy as np
graph = eqiora.geometry.GeometryGraph()
interval = graph.interval(bounds=(0., 1.))
geometry = graph.build(interval, named_topology={
    "body": interval.region, "left": interval.boundaries[0], "right": interval.boundaries[1]})
mesh = eqiora.meshing.generate(eqiora.meshing.resolve(
    geometry, eqiora.meshing.CartesianMesher(cells=(2,))))
model = eqiora.compile(source="""
model TwoStates() {
    domain body=box(0,1);
    domain left=boundary(body,axis=0,side=lower);
    domain right=boundary(body,axis=0,side=upper);
    state u:1 on body in h1;
    state v:1 on body in h1;
    initial {u=2;}
    law first on body {storage 1[s/m^2]*u; flux -grad(u); source 0[1/m^2];}
    law second on body {storage 1[s/m^2]*v; flux -grad(v); source 0[1/m^2];}
    relation fixed_left on left {trace(u)=2; trace(v)=7;}
    relation fixed_right on right {trace(u)=2; trace(v)=7;}
}
""")
linear = eqiora.solve.Linear(
    algorithm=eqiora.solve.LinearSolver.BiConjugateGradientStabilized,
    preconditioner=eqiora.solve.Preconditioner.Identity,
    reduction=eqiora.solve.Reduction.Reproducible,
    provider=eqiora.solve.SolverProvider.reference(),
    relative_tolerance=1e-12, absolute_tolerance=1e-14, maximum_iterations=100)
plan = eqiora.resolve(model, mesh=mesh, spatial=eqiora.fem.Q1(), solve=linear,
    temporal=eqiora.time.BackwardEuler(step_s=0.25))
plan = eqiora.Plan.from_bytes(plan.to_bytes())
provided = eqiora.InitialField(model.field("v"), vertex_values=[7.,7.,7.])
initial = eqiora.State.initial(plan, fields=(provided,))
try:
    eqiora.State.initial(plan, fields=(provided,
        eqiora.InitialField(model.field("u"), vertex_values=[2.,2.,2.])))
except eqiora.ValidationError:
    pass
else:
    raise AssertionError("duplicate source and supplied initial authority was accepted")
initial = eqiora.State.from_bytes(plan, initial.to_bytes())
try:
    eqiora.State.initial(plan, time_s=0.5, fields=(provided,))
except eqiora.ValidationError:
    pass
else:
    raise AssertionError("restart reused a source condition for an unsupplied Field")
restart = eqiora.State.initial(plan, time_s=0.5, fields=(provided,
    eqiora.InitialField(model.field("u"), vertex_values=[2.,2.,2.])))
restart = eqiora.State.from_bytes(plan, restart.to_bytes())
result = eqiora.run(plan, state=initial, steps=2, output_steps=(1,2))
result = eqiora.Result.from_bytes(plan, result.to_bytes())
# Constant Fields have zero diffusion and unchanged fixed traces. Each separate
# three-node Field must therefore retain its own constant through every step.
for state in (initial, restart, *result.trajectory.states):
    for name, expected in (("u",2.),("v",7.)):
        snapshot = state.field(model.field(name))
        values = np.asarray(snapshot.values("vertex"))
        assert values.shape == (3,)
        np.testing.assert_allclose(values, expected, rtol=0., atol=1e-12)
"#
            ),
            Some(&locals),
            Some(&locals),
        )
    })
}

#[test]
fn vector_spatial_state_preserves_components_and_replays() -> PyResult<()> {
    Python::initialize();
    Python::attach(|py| {
        let locals = PyDict::new(py);
        locals.set_item("eqiora", public_module(py)?)?;
        py.run(
            c_str!(
                r#"
import numpy as np
graph = eqiora.geometry.GeometryGraph()
rectangle = graph.rectangle(x_bounds=(0.,1.), y_bounds=(0.,1.))
geometry = graph.build(rectangle, named_topology={
    "region": rectangle.region, "left": rectangle.boundaries[0],
    "right": rectangle.boundaries[1], "bottom": rectangle.boundaries[2],
    "top": rectangle.boundaries[3]})
mesh = eqiora.meshing.generate(eqiora.meshing.resolve(
    geometry, eqiora.meshing.CartesianMesher(cells=(2,2))))
model = eqiora.compile(source="""
model VectorStorage() {
    domain region=box(0,1,0,1);
    domain left=boundary(region,axis=0,side=lower);
    domain right=boundary(region,axis=0,side=upper);
    domain bottom=boundary(region,axis=1,side=lower);
    domain top=boundary(region,axis=1,side=upper);
    variable g:m on region in smooth;
    relation potential on region {
        g=256[m]*(2*coordinate(0)/1[m]+3*coordinate(1)/1[m]-2.5)
            *(coordinate(0)/1[m])^2*(1-coordinate(0)/1[m])^2
            *(coordinate(1)/1[m])^2*(1-coordinate(1)/1[m])^2;
    }
    state u:vector<1,2> on region in smooth;
    relation balance on region {1[s/m^2]*derivative(u)=div(grad(u));}
    relation fixed_left on left {trace(u)=0;}
    relation fixed_right on right {trace(u)=0;}
    relation fixed_bottom on bottom {trace(u)=0;}
    relation fixed_top on top {trace(u)=0;}
}
""")
linear = eqiora.solve.Linear(
    algorithm=eqiora.solve.LinearSolver.BiConjugateGradientStabilized,
    preconditioner=eqiora.solve.Preconditioner.Identity,
    reduction=eqiora.solve.Reduction.Reproducible,
    provider=eqiora.solve.SolverProvider.reference(),
    relative_tolerance=1e-12, absolute_tolerance=1e-14, maximum_iterations=100)
plan = eqiora.resolve(model, mesh=mesh, spatial=eqiora.fem.Q1(), solve=linear,
    temporal=eqiora.time.BackwardEuler(step_s=1/24))
plan = eqiora.Plan.from_bytes(plan.to_bytes())
try:
    eqiora.State.initial(plan)
except eqiora.ValidationError:
    pass
else:
    raise AssertionError("missing exact initial assignment was accepted")
coefficients = np.zeros((9,2))
coefficients[4] = [2.,3.]
initial = eqiora.State.initial(plan, fields=(
    eqiora.InitialField(model.field("u"), vertex_values=coefficients),))
initial = eqiora.State.from_bytes(plan, initial.to_bytes())
result = eqiora.run(plan, state=initial, steps=3, output_steps=(1,2,3))
result = eqiora.Result.from_bytes(plan, result.to_bytes())
# Independently: center components start at (2,3), M=1/9, K=8/3,
# dt=1/24, hence each accepted step halves both components.
for step, state in enumerate((initial, *result.trajectory.states)):
    snapshot = state.field(model.field("u"))
    assert snapshot.value_shape == (2,)
    values = np.asarray(snapshot.values("vertex"))
    assert values.shape == (9,2)
    expected = np.zeros((9,2))
    expected[4] = np.array([2.,3.]) * 0.5**step
    np.testing.assert_allclose(values, expected, rtol=0., atol=1e-12)
"#
            ),
            Some(&locals),
            Some(&locals),
        )
    })
}

#[test]
fn kinematic_state_retains_displacement_rate_and_restart() -> PyResult<()> {
    Python::initialize();
    Python::attach(|py| {
        let locals = PyDict::new(py);
        locals.set_item("eqiora", public_module(py)?)?;
        py.run(c_str!(r#"
import numpy as np
graph = eqiora.geometry.GeometryGraph()
interval = graph.interval(bounds=(0.,1.))
geometry = graph.build(interval, named_topology={
    "body": interval.region, "left": interval.boundaries[0], "right": interval.boundaries[1]})
mesh = eqiora.meshing.generate(eqiora.meshing.resolve(geometry,
    eqiora.meshing.CartesianMesher(cells=(2,))))
model = eqiora.compile(source="""
model Wave() {
    domain body=box(0,1);
    domain left=boundary(body,axis=0,side=lower);
    domain right=boundary(body,axis=0,side=upper);
    variable g:m on body in smooth;
    relation potential on body {g=4*coordinate(0)*(1[m]-coordinate(0))/1[m];}
    state d:m on body in smooth;
    state v:m/s on body in smooth;
    relation kinematics on body {derivative(d)=v;}
    relation momentum on body {derivative(v)=1[m^2/s^2]*div(grad(d));}
    initial {d=g;v=0;}
    relation fixed_left on left {trace(v)=0;}
    relation fixed_right on right {trace(v)=0;}
}
""")
linear = eqiora.solve.Linear(
    algorithm=eqiora.solve.LinearSolver.BiConjugateGradientStabilized,
    preconditioner=eqiora.solve.Preconditioner.Identity,
    reduction=eqiora.solve.Reduction.Reproducible,
    provider=eqiora.solve.SolverProvider.reference(),
    relative_tolerance=1e-12, absolute_tolerance=1e-14, maximum_iterations=100)
plan = eqiora.resolve(model,mesh=mesh,spatial=eqiora.fem.Q1(),solve=linear,
    temporal=eqiora.time.BackwardEuler(step_s=0.25))
plan = eqiora.Plan.from_bytes(plan.to_bytes())
initial = eqiora.State.initial(plan)
initial = eqiora.State.from_bytes(plan,initial.to_bytes())
np.testing.assert_allclose(initial.field(model.field("d")).values("vertex"),[0.,1.,0.],rtol=0.,atol=1e-12)
np.testing.assert_allclose(initial.field(model.field("v")).values("vertex"),[0.,0.,0.],rtol=0.,atol=1e-12)
result = eqiora.run(plan,state=initial,steps=3,output_steps=(1,2,3))
result = eqiora.Result.from_bytes(plan,result.to_bytes())
d,v=1.,0.
assert len(result.trajectory.states)==3
for state in result.trajectory.states:
    # Independent two-element consistent-mass ratio K_ii/M_ii=4/(1/3)=12.
    d=(d+0.25*v)/1.75
    v=v-3*d
    for name,expected in (("d",d),("v",v)):
        values=np.asarray(state.field(model.field(name)).values("vertex"))
        assert values.shape==(3,)
        np.testing.assert_allclose(values,[0.,expected,0.],rtol=0.,atol=1e-12)
provided_d=eqiora.InitialField(model.field("d"),vertex_values=[0.,d,0.])
try:
    eqiora.State.initial(plan,time_s=0.75,fields=(provided_d,))
except eqiora.ValidationError:
    pass
else:
    raise AssertionError("restart omitted the exact rate Field")
restart=eqiora.State.initial(plan,time_s=0.75,fields=(provided_d,
    eqiora.InitialField(model.field("v"),vertex_values=[0.,v,0.])))
restart=eqiora.State.from_bytes(plan,restart.to_bytes())
resumed=eqiora.run(plan,state=restart,steps=1,output_steps=(1,))
d=(d+0.25*v)/1.75
v=v-3*d
for name,expected in (("d",d),("v",v)):
    np.testing.assert_allclose(resumed.trajectory.states[0].field(model.field(name)).values("vertex"),
        [0.,expected,0.],rtol=0.,atol=1e-12)
"#),Some(&locals),Some(&locals))
    })
}

#[test]
fn displacement_boundary_retains_exact_values_and_supplied_restart() -> PyResult<()> {
    Python::initialize();
    Python::attach(|py| {
        let locals = PyDict::new(py);
        locals.set_item("eqiora", public_module(py)?)?;
        py.run(c_str!(r#"
import numpy as np
graph = eqiora.geometry.GeometryGraph()
interval = graph.interval(bounds=(0.,1.))
geometry = graph.build(interval, named_topology={
    "body": interval.region, "left": interval.boundaries[0], "right": interval.boundaries[1]})
mesh = eqiora.meshing.generate(eqiora.meshing.resolve(geometry,
    eqiora.meshing.CartesianMesher(cells=(2,))))
model = eqiora.compile(source="""
model DrivenWave() {
    domain body=box(0,1);
    domain left=boundary(body,axis=0,side=lower);
    domain right=boundary(body,axis=0,side=upper);
    state d:m on body in smooth;
    state v:m/s on body in smooth;
    relation kinematics on body {derivative(d)=v;}
    relation momentum on body {derivative(v)=1[m^2/s^2]*div(grad(d));}
    initial {d=0;v=0;}
    relation fixed_left on left {trace(d)=1[m/s^2]*time()*time();}
    relation fixed_right on right {trace(d)=1[m/s^2]*time()*time();}
}
""")
linear = eqiora.solve.Linear(
    algorithm=eqiora.solve.LinearSolver.BiConjugateGradientStabilized,
    preconditioner=eqiora.solve.Preconditioner.Identity,
    reduction=eqiora.solve.Reduction.Reproducible,
    provider=eqiora.solve.SolverProvider.reference(),
    relative_tolerance=1e-12, absolute_tolerance=1e-14, maximum_iterations=100)
plan = eqiora.resolve(model,mesh=mesh,spatial=eqiora.fem.Q1(),solve=linear,
    temporal=eqiora.time.BackwardEuler(step_s=0.25))
plan = eqiora.Plan.from_bytes(plan.to_bytes())
initial = eqiora.State.initial(plan)
initial = eqiora.State.from_bytes(plan,initial.to_bytes())
result = eqiora.run(plan,state=initial,steps=3,output_steps=(1,2,3))
result = eqiora.Result.from_bytes(plan,result.to_bytes())
# Exact rational solution of the two-element consistent mass and stiffness system.
expected = ((1/16,1/4,1/112,1/28), (1/4,3/4,4/49,57/196),
            (9/16,5/4,1611/5488,1163/1372))
for state,(bd,bv,d,v) in zip(result.trajectory.states,expected,strict=True):
    state=eqiora.State.from_bytes(plan,state.to_bytes())
    for name,values in (("d",[bd,d,bd]),("v",[bv,v,bv])):
        np.testing.assert_allclose(state.field(model.field(name)).values("vertex"),values,rtol=0.,atol=1e-12)
mid=result.trajectory.states[1]
fields=tuple(eqiora.InitialField(model.field(name),vertex_values=mid.field(model.field(name)).values("vertex"))
    for name in ("d","v"))
try:
    eqiora.State.initial(plan,time_s=0.5,fields=(
        eqiora.InitialField(model.field("d"),vertex_values=[0.,4/49,1/4]),fields[1]))
except eqiora.ValidationError:
    pass
else:
    raise AssertionError("restart accepted displacement inconsistent with the boundary at its time")
restart=eqiora.State.initial(plan,time_s=0.5,fields=fields)
restart=eqiora.State.from_bytes(plan,restart.to_bytes())
resumed=eqiora.run(plan,state=restart,steps=1,output_steps=(1,))
for name in ("d","v"):
    np.testing.assert_allclose(resumed.trajectory.states[0].field(model.field(name)).values("vertex"),
        result.trajectory.states[2].field(model.field(name)).values("vertex"),rtol=0.,atol=1e-12)
"#),Some(&locals),Some(&locals))
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
    Ok(locals
        .get_item("package")?
        .expect("public package must load")
        .cast_into::<PyModule>()?)
}
