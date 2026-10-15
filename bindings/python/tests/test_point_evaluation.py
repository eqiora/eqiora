"""Exact point bindings share source/native authoring and ordinary Result reconstruction."""
import pytest
import eqiora as q

LENGTH = q.Dimension(length=1)
TEMPERATURE = q.Dimension(temperature=1)


def specimen():
    graph = q.geometry.GeometryGraph()
    region = graph.interval(bounds=(0.0, 1.0))
    geometry = graph.build(region, named_topology={
        "body": region.region, "left": region.boundaries[0], "right": region.boundaries[1],
    })
    module = q.Module("main")
    owner = module.component("Probe")
    body = owner.volume("body", dimensions=1)
    left = owner.boundary("left", parent=body)
    right = owner.boundary("right", parent=body)
    x = owner.coordinate("x", value_type=q.ValueType.real(LENGTH), on=body, factor=body, axis=0)
    temperature = owner.field("temperature", spatial_regularity=q.SpatialRegularity.H1, on=body, role=q.FieldRole.Variable,
                              value_type=q.ValueType.real(TEMPERATURE))
    owner.relation("balance", q.lang.equation(-q.lang.div(q.lang.grad(temperature)),
                   q.lang.quantity(0, q.units.K/q.units.m**2)), on=body)
    for name, boundary, value in (("left_value", left, 300), ("right_value", right, 302)):
        owner.relation(name, q.lang.equation(q.lang.trace(temperature),
                       q.lang.quantity(value, q.units.K)), on=boundary)
    profile = owner.observable("profile", temperature, on=body, value_type=q.ValueType.real(TEMPERATURE))
    point = ((x, q.lang.quantity(0.125, q.units.m)),)
    owner.observable("first", q.lang.evaluate(profile, at=point), value_type=q.ValueType.real(TEMPERATURE))
    owner.observable("second", q.lang.evaluate(profile, at=((x, q.lang.quantity(0.625, q.units.m)),)),
                     value_type=q.ValueType.real(TEMPERATURE))
    owner.observable("slope", q.lang.evaluate(q.lang.partial(temperature, wrt=x), at=point),
                     value_type=q.ValueType.real(q.Dimension(temperature=1, length=-1)))
    owner.observable("ramp", q.lang.evaluate(2*x + q.lang.quantity(1, q.units.m), at=point),
                     value_type=q.ValueType.real(LENGTH))
    owner.observable("sinusoid", q.lang.evaluate(q.lang.math.sin(x/q.lang.quantity(1, q.units.m)), at=point),
                     value_type=q.ValueType.real())
    owner.observable("outside", q.lang.evaluate(temperature, at=((x, q.lang.quantity(-1, q.units.m)),)),
                     value_type=q.ValueType.real(TEMPERATURE))
    parent = geometry.selection("body")
    bindings = {"body": parent, "left": (geometry.selection("left"), parent),
                "right": (geometry.selection("right"), parent)}
    return module, geometry, bindings, x


def test_point_evaluation_native_source_and_result_replay(tmp_path):
    module, geometry, bindings, _ = specimen()
    direct = q.compile(source=module, entry="Probe", geometry=geometry, bindings=bindings)
    path = tmp_path / "probe.eqi"
    module.write_eqi(path)
    emitted = q.compile(path=path, entry="Probe", geometry=geometry, bindings=bindings)
    assert direct.to_bytes() == emitted.to_bytes()
    outputs = {name: direct.observable(f"definition.{name}") for name in ("first", "second", "slope", "ramp", "sinusoid", "outside")}
    model = q.Model.from_bytes(direct.to_bytes())
    mesh = q.meshing.generate(q.meshing.resolve(geometry, q.meshing.CartesianMesher(cells=(4,))))
    plan = q.resolve(model, mesh=mesh, spatial=q.fem.Q1(), solve=q.solve.Linear(
        algorithm=q.solve.LinearSolver.SparseLu, preconditioner=q.solve.Preconditioner.Identity,
        reduction=q.solve.Reduction.Fast, provider=q.solve.SolverProvider.faer(),
        relative_tolerance=1e-12, absolute_tolerance=1e-14, maximum_iterations=100))
    plan = q.Plan.from_bytes(plan.to_bytes())
    result = q.run(plan)
    replay = q.Result.from_bytes(plan, result.to_bytes())
    # Affine T=300+2*x is in Q1 exactly. The sine oracle is the alternating
    # Taylor polynomial through x^9 at x=1/8; next term < 3e-18.
    z = 1/8
    sine = z-z**3/6+z**5/120-z**7/5040+z**9/362880
    profile = direct.observable("definition.profile")
    for candidate in (result, replay):
        before = candidate.to_bytes()
        for position, expected in ((0.125, 300.25), (0.625, 301.25)):
            sample = candidate.observe_at(profile, [(position, LENGTH)])
            assert sample.value == pytest.approx(expected, abs=1e-11, rel=0)
            assert sample.value_type == q.ValueType.real(TEMPERATURE)
        assert candidate.to_bytes() == before
        for name, expected, dimension in [
            ("first", 300.25, TEMPERATURE), ("second", 301.25, TEMPERATURE),
            ("slope", 2, q.Dimension(temperature=1, length=-1)),
            ("ramp", 1.25, LENGTH), ("sinusoid", sine, q.Dimension()),
        ]:
            observed = candidate.observe(outputs[name])
            assert observed.value == pytest.approx(expected, abs=1e-11, rel=0)
            assert observed.value_type == q.ValueType.real(dimension)
        with pytest.raises(q.ValidationError, match="outside"):
            candidate.observe(outputs["outside"])


def test_point_constructor_rejects_invalid_bindings_and_foreign_owners():
    _, _, _, x = specimen()
    _, _, _, foreign = specimen()
    point = q.lang.quantity(0.5, q.units.m)
    for bindings in ((), ((x, point), (x, point)), ((x+x, point),)):
        with pytest.raises(q.lang.ModuleError):
            q.lang.evaluate(x, at=bindings)
    with pytest.raises(q.lang.ModuleError, match="lexical owner"):
        q.lang.evaluate(x, at=((foreign, point),))
    with pytest.raises(q.lang.ModuleError, match="side"):
        q.lang.evaluate(x, at=((x, point),), side="nearest")


def test_planar_p1_points_keep_exact_region_ownership_and_result_replay():
    graph = q.geometry.GeometryGraph()
    left = graph.rectangle(x_bounds=(0.0, 1.0), y_bounds=(0.0, 1.0))
    right = graph.rectangle(x_bounds=(1.0, 2.0), y_bounds=(0.0, 1.0))
    partition = graph.partition(left, right, interface=(left.boundaries[1], right.boundaries[0]))
    geometry = graph.build(partition, named_topology={
        "left": left.region, "right": right.region,
        "left_outer": (left.boundaries[0], *left.boundaries[2:]),
        "right_outer": right.boundaries[1:],
        "left_face": left.boundaries[1], "right_face": right.boundaries[0],
    })
    mesh = q.meshing.generate(q.meshing.resolve(geometry, q.meshing.AffineTriangleMesher(cells=(2, 2))))
    source = """
public connector TemperatureBoundary {
    trace value:K;
    flux outward_flux:K/m;
    shape [];
    frame invariant;
    pairing euclidean_boundary_duality;
    orientation parent_outward;
}
public component TemperatureInterface(
    support body:volume(ambient_dimension=2),
    support face:boundary(parent=body),
    variable value:K on body in smooth,
    port edge:TemperatureBoundary over face
) {
    relation carrier on face {
        trace(value)=edge.value;
        normal(grad(value))=edge.outward_flux;
    }
}
public component HeatPair(
    support left:volume(ambient_dimension=2),
    support right:volume(ambient_dimension=2),
    support left_outer:boundary(parent=left),
    support right_outer:boundary(parent=right),
    support left_face:boundary(parent=left),
    support right_face:boundary(parent=right),
    support path:interval(m)
) {
    variable a:K on left in smooth;
    variable b:K on right in smooth;
    relation balance_left on left {-div(grad(a))=0[K/m^2];}
    relation balance_right on right {-div(grad(b))=0[K/m^2];}
    relation fixed_left on left_outer {trace(a)=300[K]+2[K/m]*coordinate(0)-3[K/m]*coordinate(1);}
    relation fixed_right on right_outer {trace(b)=300[K]+2[K/m]*coordinate(0)-3[K/m]*coordinate(1);}
    instance outgoing:TemperatureInterface(body=left,face=left_face,value=a);
    instance incoming:TemperatureInterface(body=right,face=right_face,value=b);
    connect outgoing.edge,incoming.edge;
    coordinate x:m on left from left[0];
    coordinate y:m on left from left[1];
    coordinate s:m on path from path;
    observable line:K on path=pullback(a,from=(s),at=(x=s,y=0.5[m]));
    observable left_profile:K on left=a;
    observable right_profile:K on right=b;
}
"""
    bindings = {name: geometry.selection(name) for name in ("left", "right")}
    bindings.update({name: (geometry.selection(name), bindings[parent]) for name, parent in (
        ("left_outer", "left"), ("right_outer", "right"),
        ("left_face", "left"), ("right_face", "right"),
    )})
    bindings["path"] = q.CoordinateInterval(0, 1, dimension=LENGTH)
    model = q.compile(source=source, entry="HeatPair", geometry=geometry, bindings=bindings)
    plan = q.resolve(model, mesh=mesh, spatial=(q.fem.P1().at(model.domain("left")),
                                             q.fem.P1().at(model.domain("right"))), solve=q.solve.Linear(
        algorithm=q.solve.LinearSolver.BiConjugateGradientStabilized,
        preconditioner=q.solve.Preconditioner.Identity,
        reduction=q.solve.Reduction.Reproducible, provider=q.solve.SolverProvider.reference(),
        relative_tolerance=1e-11, absolute_tolerance=1e-13, maximum_iterations=1000,
    ))
    result = q.run(plan)
    accepted = result.to_bytes()
    replay = q.Result.from_bytes(q.Plan.from_bytes(plan.to_bytes()), accepted)
    line = model.observable("definition.line")
    left_profile = model.observable("definition.left_profile")
    right_profile = model.observable("definition.right_profile")
    # The global affine harmonic T=300+2*x-3*y satisfies both interface laws.
    # Distinct regional coefficient inventories must reconstruct it independently.
    for candidate in (result, replay):
        for observable, position, expected in (
            (left_profile, (0.25, 0.125), 300.125),
            (right_profile, (1.5, 0.25), 302.25),
            (left_profile, (1.0, 0.25), 301.25),
            (right_profile, (1.0, 0.25), 301.25),
        ):
            sample = candidate.observe_at(observable, [(value, LENGTH) for value in position])
            assert sample.value == pytest.approx(expected, abs=1e-9, rel=0)
            assert sample.value_type == q.ValueType.real(TEMPERATURE)
        for position, expected in ((0, 298.5), (0.25, 299), (0.75, 300), (1, 300.5)):
            sample = candidate.observe_at(line, [(position, LENGTH)])
            assert sample.value == pytest.approx(expected, abs=1e-9, rel=0)
            assert sample.value_type == q.ValueType.real(TEMPERATURE)
        # The point belongs to the Mesh but is outside this Field's Region.
        with pytest.raises(q.ValidationError, match="outside"):
            candidate.observe_at(left_profile, [(1.5, LENGTH), (0.25, LENGTH)])
        assert candidate.to_bytes() == accepted
