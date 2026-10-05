"""Coordinate maps share the structured/source AST and the ordinary Result evaluator."""
import pytest
import eqiora as q


def specimen():
    module = q.Module("main")
    owner = module.model("Maps")
    body = owner.volume("body", dimensions=2)
    length = q.ValueType.real(q.Dimension(length=1))
    xi, eta, x, y = [owner.coordinate(name, value_type=length, on=body, factor=body, axis=axis)
                     for name, axis in (("xi", 0), ("eta", 1), ("x", 0), ("y", 1))]
    anchor = owner.field("anchor", value_type=q.ValueType.real(), role=q.FieldRole.Variable)
    owner.relation("hold", q.lang.equation(anchor, 0))
    bindings = dict(from_=(xi, eta), at=((x, 2*xi+eta), (y, 3*eta)))
    point = ((xi, q.lang.quantity(0.25, q.units.m)), (eta, q.lang.quantity(0.5, q.units.m)))
    for name, value, dimension in [
        ("value", q.lang.pullback(x*x+x*y, **bindings), q.Dimension(length=2)),
        ("signed", q.lang.jacobian_determinant(**bindings), q.Dimension()),
        ("volume", q.lang.volume_jacobian(**bindings), q.Dimension()),
        ("orientation", q.lang.map_orientation(**bindings), q.Dimension()),
    ]:
        owner.observable(name, q.lang.evaluate(value, at=point), value_type=q.ValueType.real(dimension))
    return module, (xi, eta, x, y)


def test_map_native_and_emitted_source_have_identical_model_bytes(tmp_path):
    module, _ = specimen()
    graph = q.geometry.GeometryGraph()
    region = graph.rectangle(x_bounds=(0, 3), y_bounds=(0, 3))
    geometry = graph.build(region, named_topology={
        "body": region.region, **{f"face_{i}": face for i, face in enumerate(region.boundaries)},
    })
    arguments = dict(entry="Maps", geometry=geometry, bindings={"body": geometry.selection("body")})
    direct = q.compile(source=module, **arguments)
    path = tmp_path / "maps.eqi"
    module.write_eqi(path)
    source = q.compile(path=path, **arguments)
    assert direct.to_bytes() == source.to_bytes()
    assert q.Model.from_bytes(direct.to_bytes()).to_bytes() == direct.to_bytes()
    text = path.read_text()
    for operator in ("pullback", "jacobian_determinant", "volume_jacobian", "map_orientation"):
        assert operator + "(" in text


def test_map_bindings_reject_duplicates_expressions_and_foreign_owners():
    _, (xi, eta, x, y) = specimen()
    _, (foreign, _, _, _) = specimen()
    for operation in (lambda **kwargs: q.lang.pullback(x*x, **kwargs),
                      q.lang.jacobian_determinant, q.lang.volume_jacobian, q.lang.map_orientation):
        for source, targets in [
            ((), ((x, xi),)), ((xi, xi), ((x, xi), (y, eta))),
            ((xi, eta), ()), ((xi, eta), ((x, xi), (x, eta))),
            ((xi+eta, eta), ((x, xi), (y, eta))),
        ]:
            with pytest.raises(q.lang.ModuleError):
                operation(from_=source, at=targets)
        with pytest.raises(q.lang.ModuleError, match="lexical owner"):
            operation(from_=(foreign, eta), at=((x, xi), (y, eta)))


def test_installed_affine_integral_uses_absolute_jacobian_after_result_replay():
    source = """model Maps(support a:interval(m),support b:interval(m),support c:interval(m),support d:interval(m)) {
        support reference:product(a,b); support target:product(c,d);
        coordinate xi:m on reference from a; coordinate eta:m on reference from b;
        coordinate x:m on target from c; coordinate y:m on target from d;
        variable anchor:1; relation hold {anchor=0;}
        observable total:m^4=integral(
            pullback(x*x+x*y,from=(xi,eta),at=(x=2*xi+eta,y=3*eta))
            *volume_jacobian(from=(xi,eta),at=(x=2*xi+eta,y=3*eta)),measure(reference));
    }"""
    model = q.compile(source=source, entry="Maps", bindings={
        name: q.CoordinateInterval(0, upper, dimension=q.Dimension(length=1))
        for name, upper in (("a", 1), ("b", 1), ("c", 3), ("d", 3))
    })
    output = model.observable("total")
    model = q.Model.from_bytes(model.to_bytes())
    solve = q.solve.Linear(relative_tolerance=1e-12, absolute_tolerance=1e-14,
                           maximum_iterations=8, algorithm=q.solve.LinearSolver.SparseLu,
                           preconditioner=q.solve.Preconditioner.Identity,
                           reduction=q.solve.Reduction.Fast, provider=q.solve.SolverProvider.faer())
    plan = q.Plan.from_bytes(q.resolve(model, solve=solve).to_bytes())
    result = q.run(plan, state=q.State.initial(plan))
    result = q.Result.from_bytes(plan, result.to_bytes())
    value = result.observe(output, quadrature_points=2)
    # Integral of (4xi²+10xi*eta+4eta²)*6 on the unit square is31.
    assert value.value == pytest.approx(31, rel=256*2**-52, abs=0)
    assert value.value_type == q.ValueType.real(q.Dimension(length=4))
