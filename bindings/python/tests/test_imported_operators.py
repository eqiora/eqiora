"""Qualified calls retain exact providers and use shared polynomial admission."""

import pytest
import eqiora as q


POLYNOMIAL = """
public operator polynomial(input x: 1, input scale: 1): 1 = scale * (1 + x + x*x);
"""


def consumer(provider, *, path=None):
    main = q.Module("main", package="consumer.example")
    imported = (main.import_module("ops", path=path) if path else
                main.import_module("ops", provider))
    op = imported.operator("polynomial")
    model = main.model("Main")
    tick = model.clock("tick", period_s=1)
    x = model.input("x", value_type=q.ValueType.real(), at=tick)
    y = model.output("y", value_type=q.ValueType.real(), at=tick)
    model.relation("evaluate", q.lang.equation(y, op(scale=3, x=x)), at=tick)
    return main, op


def test_imported_operator_direct_emitted_local_and_artifact_replay(tmp_path):
    provider = q.Module("operators", package="provider.example")
    provider.operator("polynomial", inputs={"x": q.ValueType.real(), "scale": q.ValueType.real()},
                      result_type=q.ValueType.real(), body=lambda x, scale: scale * (1+x+x*x))
    direct, op = consumer(provider)
    text = direct.to_eqi()
    assert "ops.polynomial(x = x, scale = 3)" in text
    assert "public operator polynomial" not in text
    with pytest.raises(AttributeError):
        op.name = "replacement"
    parsed = q.Module.parse("main", text, package="consumer.example")
    parsed.import_module("ops", q.Module.parse("operators", provider.to_eqi(), package="provider.example"))
    path = tmp_path / "operators.eqi"
    path.write_text(POLYNOMIAL)
    local, _ = consumer(None, path=path)
    models = [q.compile(source=source, entry="Main") for source in (direct, parsed, local)]
    assert models[0].to_bytes() == models[1].to_bytes()
    for model in models:
        assert q.Model.from_bytes(model.to_bytes()).to_bytes() == model.to_bytes()
        for candidate in (model,):
            session = candidate.execution_session(end_time_s=2, max_step_s=0.1,
                                                  inputs={"x": ("tick", [-2.0, 0.0, 4.0])})
            assert session.advance_ticks(3) == 3
            # Independently: 3*(1+x+x²) gives 9, 3, 63 at -2, 0, 4.
            assert [session.output("y", i)[1] for i in range(3)] == [9.0, 3.0, 63.0]


@pytest.mark.parametrize("source", [POLYNOMIAL.replace("public", "private"),
                                      "public component polynomial() {}", "public component Other() {}"])
def test_descriptor_rejects_private_missing_and_wrong_kind(source):
    main = q.Module("main")
    imported = main.import_module("ops", q.Module.parse("ops", source))
    with pytest.raises((ValueError, q.EqioraError), match="public operator"):
        imported.operator("polynomial")


def test_imported_calls_reject_bad_names_and_foreign_owners():
    main = q.Module("main")
    op = main.import_module("ops", q.Module.parse("ops", POLYNOMIAL)).operator("polynomial")
    with pytest.raises(q.lang.ModuleError, match="named inputs"):
        op(x=1)
    with pytest.raises(q.lang.ModuleError, match="named inputs"):
        op(x=1, scale=2, typo=3)
    other = q.Module("other")
    body = other.model("Other")
    x = body.parameter("x", value_type=q.ValueType.real())
    local = main.model("Main")
    with pytest.raises(q.lang.ModuleError, match="Component"):
        local.let_alias("foreign", op(x=x, scale=2))
    with pytest.raises(q.lang.ModuleError, match="Module"):
        body.let_alias("bad", op(x=1, scale=2))


def test_wrong_units_reach_shared_operator_admission():
    main = q.Module("main")
    op = main.import_module("ops", q.Module.parse("ops", POLYNOMIAL)).operator("polynomial")
    body = main.model("Main")
    tick = body.clock("tick", period_s=1)
    out = body.output("y", value_type=q.ValueType.real(), at=tick)
    body.relation("bad", q.lang.equation(out, op(x=q.lang.quantity(1, q.units.m), scale=3)), at=tick)
    with pytest.raises(q.ValidationError) as error:
        q.compile(source=main, entry="Main")
    assert any(item.code == "EQ0603" for item in error.value.diagnostics)


def test_imported_operator_cycle_reaches_shared_definition_gate():
    source = POLYNOMIAL.replace("scale * (1 + x + x*x)", "polynomial(x=x, scale=scale)")
    main, _ = consumer(q.Module.parse("operators", source, package="provider.example"))
    with pytest.raises(q.ValidationError) as error:
        q.compile(source=main, entry="Main")
    assert any("cycl" in item.message or "recurs" in item.message for item in error.value.diagnostics)


def test_imported_dimensioned_operator_preserves_partial_bindings_and_argument_order():
    provider = q.Module.parse(
        "laws",
        "public dimension Stiffness = N/m; "
        "public operator energy(input x: m, input k: Stiffness): J = 0.5*k*x*x;",
        package="provider.example",
    )
    main = q.Module("main", package="consumer.example")
    energy = main.import_module("laws", provider).operator("energy")
    stiffness = q.ValueType.real(q.Dimension(mass=1, time=-2))
    length = q.ValueType.real(q.Dimension(length=1))
    force = q.ValueType.real(q.Dimension(mass=1, length=1, time=-2))
    model = main.model("Main")
    x = model.parameter("x", value_type=length)
    k = model.parameter("k", value_type=stiffness)
    model.set_default(x, q.lang.quantity(3, q.units.m))
    model.set_default(k, q.lang.quantity(4, q.units.N/q.units.m))
    output = model.field("force", role=q.FieldRole.Variable, value_type=force)
    model.relation("evaluate", q.lang.equation(
        output, q.lang.partial(energy(k=k, x=x), wrt=x, holding=(k,))))
    emitted = q.Module.parse("main", main.to_eqi(), package="consumer.example")
    emitted.import_module("laws", provider)
    models = [q.compile(source=source, entry="Main") for source in (main, emitted)]
    assert models[0].to_bytes() == models[1].to_bytes()
    for candidate in models:
        session = candidate.execution_session(end_time_s=0.1, max_step_s=0.1, inputs={})
        # E = k*x²/2, so the partial at fixed k is k*x = 12 N.
        assert session.field("force") == 12.0


@pytest.mark.parametrize("channels", (False, True))
@pytest.mark.parametrize("native", (False, True))
def test_imported_spatial_operator_retains_tensor_roles_and_rejects_channel_arrays(channels, native):
    if native:
        provider = q.Module("laws", package="provider.example")
        vector = q.ValueType.vector(q.ValueType.real(), 2)
        provider.operator("dyadic", inputs={"left": vector, "right": vector},
                          result_type=q.ValueType.tensor(q.ValueType.real(), 2, 2),
                          body=lambda left, right: q.lang.component(left, 0) * q.lang.component(right, 1))
    else:
        provider = q.Module.parse(
            "laws",
            "public operator dyadic(input left: spatial[1], input right: spatial[1]): spatial[2] "
            "= component(left, 0) * component(right, 1);",
            package="provider.example",
        )
    main = q.Module("main")
    dyadic = main.import_module("laws", provider).operator("dyadic")
    model = main.model("Main")
    support = model.volume("body", dimensions=2)
    scalar = q.ValueType.real()
    value_type = q.ValueType.array(scalar, 2) if channels else q.ValueType.vector(scalar, 2)
    left = model.field("left", role=q.FieldRole.Variable, value_type=value_type, on=support)
    right = model.field("right", role=q.FieldRole.Variable, value_type=value_type, on=support)
    model.relation("balance", q.lang.equation(q.lang.div(q.lang.div(dyadic(right=right, left=left))), 0), on=support)
    graph = q.geometry.GeometryGraph()
    rectangle = graph.rectangle(x_bounds=(0, 1), y_bounds=(0, 1))
    geometry = graph.build(rectangle, named_topology={
        "body": rectangle.region,
        **dict(zip(("left", "right", "bottom", "top"), rectangle.boundaries)),
    })
    bindings = {"body": geometry.selection("body")}
    if channels:
        with pytest.raises(q.ValidationError):
            q.compile(source=main, entry="Main", geometry=geometry, bindings=bindings)
        return
    source = main.to_eqi()
    assert "laws.dyadic(left = left, right = right)" in source
    emitted = q.Module.parse("main", source)
    emitted.import_module("laws", provider)
    direct = q.compile(source=main, entry="Main", geometry=geometry, bindings=bindings)
    replayed = q.compile(source=emitted, entry="Main", geometry=geometry, bindings=bindings)
    assert direct.to_bytes() == replayed.to_bytes()
