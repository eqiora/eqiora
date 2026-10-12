use super::*;

#[test]
fn coupled_newton_projects_original_residuals_and_each_actual_correction() -> PyResult<()> {
    Python::initialize();
    Python::attach(|py| {
        let locals = PyDict::new(py);
        locals.set_item("eqiora", public_module(py)?)?;
        py.run(c_str!(r#"
import numpy as np

graph = eqiora.geometry.GeometryGraph()
rectangle = graph.rectangle(x_bounds=(0., 1.), y_bounds=(0., 1.))
geometry = graph.build(rectangle, named_topology={
    "body": rectangle.region, "left": rectangle.boundaries[0],
    "right": rectangle.boundaries[1], "bottom": rectangle.boundaries[2],
    "top": rectangle.boundaries[3]})
mesh = eqiora.meshing.generate(eqiora.meshing.resolve(
    geometry, eqiora.meshing.CartesianMesher(cells=(3, 3))))
linear = eqiora.solve.Linear(
    algorithm=eqiora.solve.LinearSolver.BiConjugateGradientStabilized,
    preconditioner=eqiora.solve.Preconditioner.Identity,
    reduction=eqiora.solve.Reduction.Reproducible,
    provider=eqiora.solve.SolverProvider.reference(),
    relative_tolerance=1e-12, absolute_tolerance=1e-14, maximum_iterations=1000)
for force in (3., 0.):
    source = r"""
public operator dyad(input left: spatial[1], input right: spatial[1]): spatial[2]
    = component(left, 0) * component(right, 1);
model Mixture() {
    domain body=box(0,1,0,1);
    domain left=boundary(body,axis=0,side=lower);
    domain right=boundary(body,axis=0,side=upper);
    domain bottom=boundary(body,axis=1,side=lower);
    domain top=boundary(body,axis=1,side=upper);
    parameter rho:kg/m^3=4;
    parameter mu:kg/(m*s)=0.1;
    parameter rate:kg/(m^3*s)=2;
"""
    source += f"parameter force:kg/(m^2*s^2)={force};\n"
    for name in ("a", "b", "c"):
        source += f"state {name}:vector<m/s,2> on body in smooth; initial {{ {name}=0; }}\n"
        source += f"relation balance_{name} on body {{ rho*derivative({name}) + div(rho*dyad(left={name},right={name})) - div(mu*grad({name}))"
        for other in ("a", "b", "c"):
            if other != name:
                source += f" + rate*({name}-{other})"
        if name == "a":
            source += " - force*grad(coordinate(0))"
        source += "=0; }\n"
        for side in ("left", "right", "bottom", "top"):
            source += f"relation {side}_{name} on {side} {{ trace({name})=0; }}\n"
    source += "}"
    model = eqiora.compile(source=source)
    plan = eqiora.resolve(model, mesh=mesh, spatial=eqiora.fem.Q1(),
        solve=eqiora.solve.Newton(linear=linear, relative_tolerance=1e-10,
            absolute_tolerance=1e-12, maximum_iterations=12),
        temporal=eqiora.time.BackwardEuler(0.125))
    assert plan.capability.kind == "spatial"
    plan = eqiora.Plan.from_bytes(plan.to_bytes())
    initial = eqiora.State.initial(plan)
    before = initial.to_bytes()
    result = eqiora.run(plan, state=initial, steps=2, output_steps=(2,))
    original = result.spatial_step(0).solve
    result = eqiora.Result.from_bytes(plan, result.to_bytes())
    observation = result.spatial_step(0)
    summary = observation.solve
    assert isinstance(summary, eqiora.NonlinearSolveSummary)
    assert observation.step_index == 2 and observation.output_index == 0
    assert initial.to_bytes() == before
    assert summary.true_residual_norm <= summary.residual_target
    assert summary.initial_residual_norm == original.initial_residual_norm
    assert summary.completed_iterations == original.completed_iterations
    assert len(summary.linear_solves) == summary.completed_iterations
    if force:
        assert summary.completed_iterations > 1
        for correction in summary.linear_solves:
            assert isinstance(correction, eqiora.LinearSolveSummary)
            assert correction.true_residual_norm <= correction.residual_target
        for name in ("a", "b", "c"):
            values = np.asarray(result.trajectory.states[-1].field(model.field(name)).values("vertex"))
            assert np.linalg.norm(values) > 1e-6
    else:
        assert summary.completed_iterations == 0 and summary.linear_solves == ()
    try:
        summary.completed_iterations = 99
    except AttributeError:
        pass
    else:
        raise AssertionError("Newton observations must be immutable")
"#), Some(&locals), Some(&locals))
    })
}
