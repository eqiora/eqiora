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
    temperature = owner.field("temperature", on=body, role=q.FieldRole.Variable,
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
    for candidate in (result, replay):
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
