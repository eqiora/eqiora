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


@pytest.mark.parametrize("coupled", [False, True])
def test_factor_integrals_installed_replay_and_dimensioned_output_points(coupled):
    source = SOURCE.replace("amplitude=3[s/m^2]", "count=135/2") if coupled else SOURCE
    model = q.compile(source=source, entry="Distribution", bindings={
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


def test_spherical_factor_measure_installed_result_replay():
    import math
    source = """model Particle(support radius:interval(m)) {
      coordinate r:m on radius from radius[0];
      variable anchor:1; relation value {anchor=1;}
      observable total:1=integral(2[1/m^3]+3[1/m^5]*r^2,spherical_measure(radius));
      observable volume:m^3=integral(1,spherical_measure(radius));
      observable average:1/m^3=total/volume;
    }"""
    model = q.compile(source=source, entry="Particle", bindings={
        "radius": q.CoordinateInterval(0, 2, dimension=LENGTH),
    })
    outputs = {name: model.observable(name) for name in ("total", "volume", "average")}
    model = q.Model.from_bytes(model.to_bytes())
    solve = q.solve.Linear(relative_tolerance=1e-12, absolute_tolerance=1e-14,
                           maximum_iterations=8, algorithm=q.solve.LinearSolver.SparseLu,
                           preconditioner=q.solve.Preconditioner.Identity,
                           reduction=q.solve.Reduction.Fast, provider=q.solve.SolverProvider.faer())
    plan = q.Plan.from_bytes(q.resolve(model, solve=solve).to_bytes())
    result = q.run(plan, state=q.State.initial(plan))
    result = q.Result.from_bytes(plan, result.to_bytes())
    for name, expected, power in [("total", 1472*math.pi/15, 0),
                                  ("volume", 32*math.pi/3, 3),
                                  ("average", 46/5, -3)]:
        observation = result.observe(outputs[name], quadrature_points=3)
        assert observation.value == pytest.approx(expected, rel=0, abs=1e-11)
        assert observation.value_type == q.ValueType.real(q.Dimension(length=power))


@pytest.mark.parametrize("velocity_cells", [1, 3, 6])
def test_coordinate_field_grid_installed_solve_observe_and_replay(velocity_cells):
    source = """model Distribution(support position:interval(m), support velocity:interval(m/s)) {
      support phase:product(position,velocity);
      coordinate x:m on phase from position;
      coordinate v:m/s on phase from velocity;
      variable f:s/m^2 on phase;
      relation prescribed_density on phase {f=3[s/m^2]*(1+x/2[m])*(1+v*v/16[m^2/s^2]);}
      observable density:1/m on position=integral(f,measure(velocity));
      observable current:1/s on position=integral(v*f,measure(velocity));
      observable second:m/s^2 on position=integral(v*v*f,measure(velocity));
      observable count:1=integral(f,measure(phase));
    }"""
    model = q.compile(source=source, entry="Distribution", bindings={
        "position": q.CoordinateInterval(0, 2, dimension=LENGTH),
        "velocity": q.CoordinateInterval(-2, 4, dimension=SPEED),
    })
    phase = model.domain("phase")
    outputs = {name: model.observable(name) for name in ("density", "current", "second", "count")}
    mesh = q.meshing.Mesh.coordinate_factors(model, phase, [2, velocity_cells])
    assert mesh.realized_geometry_digest is None
    assert mesh.correspondence_digest is None
    assert mesh.cell_count == 2*velocity_cells
    assert mesh.selection_names == ()
    source_digest = mesh.source_digest
    mesh = q.meshing.Mesh.from_bytes(mesh.to_bytes())
    assert mesh.source_digest == source_digest
    refined = q.meshing.Mesh.coordinate_factors(model, phase, [2, velocity_cells*2])
    assert refined.source_digest == source_digest
    assert refined.digest != mesh.digest
    solve = q.solve.Linear(relative_tolerance=1e-12, absolute_tolerance=1e-14,
                           maximum_iterations=8, algorithm=q.solve.LinearSolver.SparseLu,
                           preconditioner=q.solve.Preconditioner.Identity,
                           reduction=q.solve.Reduction.Fast, provider=q.solve.SolverProvider.faer())
    model = q.Model.from_bytes(model.to_bytes())
    spatial = q.fvm.CellCentered()
    assert spatial.space == "cell-constant"
    plan = q.resolve(model, mesh=mesh, spatial=spatial, solve=solve)
    assert plan.geometry_digest is None
    assert plan.correspondence_digest is None
    assert plan.capability.coefficient_sampling == "quadrature-point"
    plan = q.Plan.from_bytes(plan.to_bytes())
    result = q.run(plan)
    result = q.Result.from_bytes(plan, result.to_bytes())
    n2 = velocity_cells**2
    factors = {"density": 15/2, "current": 39/4-9/(4*n2),
               "second": 186/5-18/n2+54/(5*n2*n2)}
    for name, factor in factors.items():
        for x, cell_midpoint in [(0.0, 0.5), (0.5, 0.5), (1.0, 1.5), (2.0, 1.5)]:
            observed = result.observe_at(outputs[name], [(x, LENGTH)], quadrature_points=2)
            assert observed.value == pytest.approx(3*(1+cell_midpoint/2)*factor, abs=1e-10)
    assert result.observe(outputs["count"], quadrature_points=2).value == pytest.approx(135/2, abs=1e-10)
    foreign = q.compile(source=source.replace("3[s/m^2]", "4[s/m^2]"), entry="Distribution", bindings={
        "position": q.CoordinateInterval(0, 2, dimension=LENGTH),
        "velocity": q.CoordinateInterval(-2, 4, dimension=SPEED),
    })
    with pytest.raises(q.ValidationError, match="foreign or stale"):
        q.meshing.Mesh.coordinate_factors(foreign, phase, [2, velocity_cells])
    for cells in [[2], [2, 0]]:
        with pytest.raises(q.ValidationError):
            q.meshing.Mesh.coordinate_factors(model, phase, cells)


@pytest.mark.parametrize("cells", [4, 8])
def test_radial_diffusion_installed_result_and_spherical_average(cells):
    from pathlib import Path
    source = Path(__file__).resolve().parents[3] / "verify/language/factor-integrals/models/radial-diffusion.eqi"
    model = q.compile(path=source, entry="Particle", bindings={
        "radius": q.CoordinateInterval(0, 1, dimension=LENGTH),
    })
    output = model.observable("average")
    radius = model.domain("radius")
    model = q.Model.from_bytes(model.to_bytes())
    mesh = q.meshing.Mesh.coordinate_factors(model, radius, [cells])
    mesh = q.meshing.Mesh.from_bytes(mesh.to_bytes())
    solve = q.solve.Linear(relative_tolerance=1e-12, absolute_tolerance=1e-14,
                           maximum_iterations=100, algorithm=q.solve.LinearSolver.SparseLu,
                           preconditioner=q.solve.Preconditioner.Identity,
                           reduction=q.solve.Reduction.Fast, provider=q.solve.SolverProvider.faer())
    plan = q.resolve(model, mesh=mesh, spatial=q.fvm.CellCentered(), solve=solve)
    plan = q.Plan.from_bytes(plan.to_bytes())
    result = q.Result.from_bytes(plan, q.run(plan).to_bytes())
    # Independent spherical cell volumes applied to c_i=1-r_i²+h²/4.
    h = 1/cells
    expected = 2/5+2*h*h/3-h**4/15
    observed = result.observe(output, quadrature_points=2)
    assert observed.value == pytest.approx(expected, abs=1e-10, rel=0)
    assert observed.value_type == q.ValueType.real(q.Dimension(length=-3))
    with pytest.raises(q.ValidationError, match="CellCentered"):
        q.resolve(model, mesh=mesh, spatial=q.fem.Q1(), solve=solve)


@pytest.mark.parametrize("a", [-1.0, 0.0, 1.0, 2.0])
@pytest.mark.parametrize("density,lower,upper,power,coefficient,derivative", [
    ("x*x", "0[m]", "a", 3, 1/3, 1),
    ("a*x", "a", "2*a", 3, 3/2, 9/2),
    ("1", "a", "2*a", 1, 1, 1),
])
def test_moving_endpoint_leibniz_installed_lifecycle(a, density, lower, upper, power, coefficient, derivative):
    source = f"""model Moving(support line:interval(m)) {{
      coordinate x:m on line from line;
      parameter a:m={a}[m];
      variable anchor:1; relation fixed {{anchor=1;}}
      observable total:m^{power}=integral({density},measure(line),lower={lower},upper={upper});
      observable slope:m^{power-1}=partial(total,wrt=a);
    }}"""
    model = q.compile(source=source, entry="Moving", bindings={
        "line": q.CoordinateInterval(-4, 4, dimension=LENGTH),
    })
    outputs = {name: model.observable(name) for name in ("total", "slope")}
    model = q.Model.from_bytes(model.to_bytes())
    solve = q.solve.Linear(relative_tolerance=1e-12, absolute_tolerance=1e-14,
                           maximum_iterations=8, algorithm=q.solve.LinearSolver.SparseLu,
                           preconditioner=q.solve.Preconditioner.Identity,
                           reduction=q.solve.Reduction.Fast, provider=q.solve.SolverProvider.faer())
    plan = q.Plan.from_bytes(q.resolve(model, solve=solve).to_bytes())
    result = q.Result.from_bytes(plan, q.run(plan, state=q.State.initial(plan)).to_bytes())
    for name, expected, dimension in [
        ("total", coefficient*a**power, q.Dimension(length=power)),
        ("slope", derivative*a**(power-1), q.Dimension(length=power-1)),
    ]:
        observed = result.observe(outputs[name], quadrature_points=2)
        assert observed.value == pytest.approx(expected, abs=1e-12, rel=0)
        assert observed.value_type == q.ValueType.real(dimension)
