mod support;
use support::public_module;

use pyo3::ffi::c_str;
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyDictMethods};

const SOURCE: &str = include_str!("../../../examples/steady-flow-past-cylinder.eqi");

#[test]
fn python_compile_closes_geometry_and_reuses_the_root_resolver() -> PyResult<()> {
    Python::initialize();
    Python::attach(|py| {
        let module = public_module(py)?;
        let locals = PyDict::new(py);
        locals.set_item("eqiora", module)?;
        locals.set_item("source_text", SOURCE)?;
        py.run(
            c_str!(r#"
graph = eqiora.geometry.GeometryGraph()
rectangle = graph.rectangle(x_bounds=(0.0, 2.2), y_bounds=(0.0, 0.41))
circle = graph.circle(center=(0.2, 0.2), radius=0.05)
fluid = graph.subtract(rectangle, circle)
geometry = graph.build(fluid, named_topology={
    "fluid": fluid.region,
    "inlet": rectangle.boundaries[0],
    "outlet": rectangle.boundaries[1],
    "walls": rectangle.boundaries[2:],
    "cylinder": circle.boundaries[0],
})
parameters = {
    "dynamic_viscosity": 0.001,
    "zero_pressure": 0.0,
    "inlet_speed": 0.3,
    "channel_height": geometry.bounds[1][1] - geometry.bounds[1][0],
}
model = eqiora.compile(source=source_text, filename='cylinder.eqi', geometry=geometry, entry='SteadyFlowPastCylinder', bindings={'fluid': geometry.selection('fluid'), 'inlet': (geometry.selection('inlet'), geometry.selection('fluid')), 'outlet': (geometry.selection('outlet'), geometry.selection('fluid')), 'walls': (geometry.selection('walls'), geometry.selection('fluid')), 'cylinder': (geometry.selection('cylinder'), geometry.selection('fluid')), **parameters})
explicit = eqiora.compile(source=source_text, filename='renamed.eqi', geometry=geometry, entry='SteadyFlowPastCylinder', bindings={'fluid': geometry.selection('fluid'), 'inlet': (geometry.selection('inlet'), geometry.selection('fluid')), 'outlet': (geometry.selection('outlet'), geometry.selection('fluid')), 'walls': (geometry.selection('walls'), geometry.selection('fluid')), 'cylinder': (geometry.selection('cylinder'), geometry.selection('fluid')), **parameters})
assert model.digest == explicit.digest
replayed = eqiora.Model.from_bytes(model.to_bytes())
request = eqiora.meshing.GmshMesher(
    maximum_boundary_error=1e-4,
    minimum_mean_ratio=1e-5,
    maximum_boundary_facets=50,
)
mesh_plan = eqiora.meshing.resolve(geometry, request)
mesh = eqiora.meshing.generate(mesh_plan)
linear = eqiora.solve.Linear(
    algorithm=eqiora.solve.LinearSolver.SparseLu,
    preconditioner=eqiora.solve.Preconditioner.Identity,
    reduction=eqiora.solve.Reduction.Fast,
    provider=eqiora.solve.SolverProvider.faer(),
    relative_tolerance=1e-6,
    absolute_tolerance=1e-13,
    maximum_iterations=10000,
)
fresh_plan = eqiora.resolve(model, mesh=mesh, spatial=eqiora.fem.MiniP1(), solve=linear)
replayed_plan = eqiora.resolve(replayed, mesh=mesh, spatial=eqiora.fem.MiniP1(), solve=linear)
assert fresh_plan.identity == replayed_plan.identity
assert fresh_plan.model is model
assert fresh_plan.model_digest == model.digest == replayed.digest
assert fresh_plan.mesh is mesh
fresh_result = eqiora.run(fresh_plan)
replayed_result = eqiora.run(replayed_plan)
assert fresh_result.model_digest == model.digest == replayed_result.model_digest
assert fresh_result.plan_key == fresh_plan.identity == replayed_result.plan_key
assert fresh_result.output(fresh_plan.capability.pressure).coefficient_count("vertex") == mesh.vertex_count

for args, kwargs in (
    ((source_text,), {}),
    ((), {}),
    ((), {"source": source_text, "path": "ignored.eqi"}),
    ((), {"source": source_text, "filename": "x.eqi", "geometry": geometry, "parameters": {**parameters, "inlet_speed": True}}),
    ((), {"source": source_text, "filename": "x.eqi", "geometry": geometry, "parameters": {**parameters, "inlet_speed": float("nan")}}),
):
    try:
        eqiora.compile(*args, **kwargs)
    except (TypeError, ValueError, eqiora.ValidationError):
        pass
    else:
        raise AssertionError("invalid compile ingress was accepted")

for invalid in (
    {name: value for name, value in parameters.items() if name != "channel_height"},
    {**parameters, "extra": 1.0},
):
    try:
        eqiora.compile(source=source_text, geometry=geometry, entry='SteadyFlowPastCylinder', bindings={'fluid': geometry.selection('fluid'), 'inlet': (geometry.selection('inlet'), geometry.selection('fluid')), 'outlet': (geometry.selection('outlet'), geometry.selection('fluid')), 'walls': (geometry.selection('walls'), geometry.selection('fluid')), 'cylinder': (geometry.selection('cylinder'), geometry.selection('fluid')), **invalid})
    except eqiora.ValidationError:
        pass
    else:
        raise AssertionError("invalid Parameter inventory was accepted")
"#),
            None,
            Some(&locals),
        )
    })
}
