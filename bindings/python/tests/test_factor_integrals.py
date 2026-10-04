"""Selected-factor observations use the installed Model/Plan/State/Result lifecycle."""
import pytest
import eqiora as q

SOURCE = """model Distribution(support position:interval(m), support velocity:interval(m/s)) {
  support phase:product(position,velocity);
  coordinate x:m on phase from position[0];
  coordinate v:m/s on phase from velocity[0];
  coordinate remaining_x:m on position from position[0];
  variable amplitude:s/m^2;
  relation amplitude_value { amplitude=3[s/m^2]; }
  let f:s/m^2 on phase=amplitude*(1+x/2[m])*(1+(v/4[m/s])^2);
  observable density:1/m on position=integral(f,measure(velocity));
  observable density_slope:1/m^2 on position=partial(density,wrt=remaining_x);
  observable current:1/s on position=integral(v*f,measure(velocity));
  observable mean:m/s on position=current/density;
  observable count:1=integral(density,measure(position));
}"""
LENGTH = q.Dimension(length=1)
SPEED = q.Dimension(length=1, time=-1)


def test_factor_integrals_installed_replay_and_dimensioned_output_points():
    model = q.compile(source=SOURCE, entry="Distribution", bindings={
        "position": q.CoordinateInterval(0, 2, dimension=LENGTH),
        "velocity": q.CoordinateInterval(-2, 4, dimension=SPEED),
    })
    outputs = {name: model.observable(name) for name in ("density", "current", "mean", "count", "density_slope")}
    model = q.Model.from_bytes(model.to_bytes())
    solve = q.solve.Linear(relative_tolerance=1e-12, absolute_tolerance=1e-14,
                           maximum_iterations=8, algorithm=q.solve.LinearSolver.SparseLu,
                           preconditioner=q.solve.Preconditioner.Identity,
                           reduction=q.solve.Reduction.Fast, provider=q.solve.SolverProvider.faer())
    plan = q.resolve(model, solve=solve)
    plan = q.Plan.from_bytes(plan.to_bytes())
    result = q.run(plan, state=q.State.initial(plan))
    result = q.Result.from_bytes(plan, result.to_bytes())
    for name, expected, dimension in [
        ("density", 135/4, q.Dimension(length=-1)),
        ("current", 351/8, q.Dimension(time=-1)),
        ("mean", 13/10, SPEED),
        ("density_slope", 45/4, q.Dimension(length=-2)),
    ]:
        observation = result.observe_at(outputs[name], [(1.0, LENGTH)], quadrature_points=3)
        assert observation.value == pytest.approx(expected, abs=1e-11)
        assert observation.value_type == q.ValueType.real(dimension)
        assert observation.point[0] in model.domain_ids
        assert observation.point[1] == [(1.0, LENGTH)]
        assert len(observation.quadratures) == 1
    total = result.observe(outputs["count"], quadrature_points=3)
    assert total.value == pytest.approx(135/2, abs=1e-11)
    assert total.point is None
    for point in [[], [(1.0, SPEED)], [(3.0, LENGTH)]]:
        with pytest.raises(q.ValidationError):
            result.observe_at(outputs["density"], point, quadrature_points=3)
    with pytest.raises(TypeError, match="booleans"):
        result.observe_at(outputs["density"], [(True, LENGTH)], quadrature_points=3)
    with pytest.raises(q.ValidationError, match="output point"):
        result.observe(outputs["density"], quadrature_points=3)


def test_factor_integrals_native_authoring_separates_output_and_integration_support():
    graph = q.geometry.GeometryGraph()
    square = graph.rectangle(x_bounds=(0.0, 1.0), y_bounds=(0.0, 1.0))
    geometry = graph.build(square, named_topology={"body": square.region, **{f"face_{i}": face for i, face in enumerate(square.boundaries)}})
    module = q.Module("main")
    component = module.component("Readings")
    body = component.volume("body", dimensions=2)
    field = component.field("u", on=body, role=q.FieldRole.Variable, value_type=q.ValueType.real())
    component.relation("hold", q.lang.equation(field, 0), on=body)
    component.observable("sample", 2*field, value_type=q.ValueType.real(), on=body)
    component.observable("total", field, value_type=q.ValueType.real(q.Dimension(length=2)), integrate_over=body)
    text = module.to_eqi()
    assert "observable sample: 1 on body" in text
    assert "integral(u, measure(body))" in text
    native = q.compile(source=module, entry="Readings", geometry=geometry, bindings={"body": geometry.selection("body")})
    emitted = q.compile(source=text, entry="Readings", geometry=geometry, bindings={"body": geometry.selection("body")})
    assert native.to_bytes() == emitted.to_bytes()
    assert native.structural_fingerprint == emitted.structural_fingerprint
