mod support;
use support::public_module;

use pyo3::ffi::c_str;
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyDictMethods};

const SOURCE: &str = include_str!("../../../examples/steady-flow-past-cylinder.eqi");

#[test]
fn python_exact_cylinder_stokes_uses_only_the_root_lifecycle() -> PyResult<()> {
    Python::initialize();
    Python::attach(|py| {
        let module = public_module(py)?;
        let locals = PyDict::new(py);
        locals.set_item("eqiora", module)?;
        locals.set_item("source_text", SOURCE)?;
        py.run(
            c_str!(
                r#"
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
model = eqiora.compile(source=source_text, filename='steady-flow-past-cylinder.eqi', geometry=geometry, entry='SteadyFlowPastCylinder', bindings={'fluid': geometry.selection('fluid'), 'inlet': (geometry.selection('inlet'), geometry.selection('fluid')), 'outlet': (geometry.selection('outlet'), geometry.selection('fluid')), 'walls': (geometry.selection('walls'), geometry.selection('fluid')), 'cylinder': (geometry.selection('cylinder'), geometry.selection('fluid')), **{'dynamic_viscosity': 0.001, 'zero_pressure': 0.0, 'inlet_speed': 0.3, 'channel_height': geometry.bounds[1][1] - geometry.bounds[1][0]}})
mesh_request = eqiora.meshing.GmshMesher(
    maximum_boundary_error=1e-4,
    minimum_mean_ratio=1e-5,
    maximum_boundary_facets=50,
)
mesh_plan = eqiora.meshing.resolve(geometry, mesh_request)
mesh = eqiora.meshing.generate(mesh_plan)
plan = eqiora.resolve(
    model,
    mesh=mesh,
    spatial=eqiora.fem.MiniP1(),
    solve=eqiora.solve.Linear(
        algorithm=eqiora.solve.LinearSolver.SparseLu,
        preconditioner=eqiora.solve.Preconditioner.Identity,
        reduction=eqiora.solve.Reduction.Fast,
        provider=eqiora.solve.SolverProvider.faer(),
        relative_tolerance=1e-6,
        absolute_tolerance=1e-13,
        maximum_iterations=10_000,
    ),
    scaling=None,
)
result = eqiora.run(plan)
evidence = eqiora.fluid.steady_stokes_evidence(result)
pressure = result.output(plan.capability.pressure)
result_bytes = result.to_bytes()
replayed_result = eqiora.Result.from_bytes(plan, result_bytes)
replayed_evidence = eqiora.fluid.steady_stokes_evidence(replayed_result)
assert replayed_result.to_bytes() == result_bytes
assert replayed_result.output(plan.capability.pressure).values("vertex").numpy().tolist() == pressure.values("vertex").numpy().tolist()
assert replayed_evidence.solve.true_residual_norm == evidence.solve.true_residual_norm
assert result.model_digest == model.digest
assert result.plan_key == plan.identity
assert pressure.coefficient_count("vertex") == mesh.vertex_count
assert evidence.plan_key == plan.identity
assert evidence.solve.true_residual_norm <= evidence.solve.residual_target
assert not hasattr(eqiora.fluid, "SteadyStokes")
assert not hasattr(eqiora.fluid, "resolve")
"#
            ),
            None,
            Some(&locals),
        )
    })
}
