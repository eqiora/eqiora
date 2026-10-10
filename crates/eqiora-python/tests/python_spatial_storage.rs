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
