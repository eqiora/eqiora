"""Exact coordinate binders use native structured authoring and current Model replay."""
import json
import pytest
import eqiora

q = eqiora.lang
u = eqiora.units


def geometry():
    graph = eqiora.geometry.GeometryGraph()
    rectangle = graph.rectangle(x_bounds=(0, 2), y_bounds=(0, 3))
    return graph.build(rectangle, named_topology={"body": rectangle.region, **dict(zip(("left", "right", "bottom", "top"), rectangle.boundaries))})


def test_structured_coordinate_partials_emit_and_replay(tmp_path):
    module = eqiora.Module("main")
    owner = module.model("Derivatives")
    body = owner.volume("body", dimensions=2)
    length = eqiora.ValueType.real(dimension=eqiora.Dimension(length=1))
    x = owner.coordinate("x", value_type=length, on=body, factor=body, axis=0)
    y = owner.coordinate("y", value_type=length, on=body, factor=body, axis=1)
    field = owner.field("f", on=body, value_type=eqiora.ValueType.real(), role=eqiora.FieldRole.Variable)
    owner.relation("analytic", q.equation(q.partial(x*y, wrt=x), y), on=body)
    owner.relation("unknown", q.equation(q.partial(field, wrt=y), q.quantity(0, u.one/u.m)), on=body)
    domain = geometry()
    arguments = dict(entry="Derivatives", geometry=domain, bindings={"body": domain.selection("body")})
    direct = eqiora.compile(source=module, **arguments)
    path = tmp_path / "coordinates.eqi"
    module.write_eqi(path)
    assert "coordinate y: m on body from body[1];" in path.read_text()
    emitted = eqiora.compile(path=path, **arguments)
    replay = eqiora.Model.from_bytes(direct.to_bytes())
    assert direct.to_bytes() == emitted.to_bytes() == replay.to_bytes()
    assert "coordinate-partial" in direct.to_bytes().decode()
    assert json.loads(direct.to_bytes())["schema"] == "eqiora.model-envelope/v45"


def test_coordinate_builder_rejects_foreign_owner_and_bad_axis_before_claiming_name():
    module = eqiora.Module("main")
    owner = module.model("Owner")
    body = owner.volume("body", dimensions=2)
    foreign = module.component("Other").volume("body", dimensions=2)
    length = eqiora.ValueType.real(dimension=eqiora.Dimension(length=1))
    with pytest.raises(q.ModuleError, match="belong"):
        owner.coordinate("x", value_type=length, on=body, factor=foreign, axis=0)
    for axis in (-1, True, 0.5):
        with pytest.raises(TypeError, match="nonnegative integer"):
            owner.coordinate("x", value_type=length, on=body, factor=body, axis=axis)
    owner.coordinate("x", value_type=length, on=body, factor=body, axis=0)


def test_installed_position_velocity_partials_keep_separate_coordinate_units():
    source = """model Phase(support position:interval(m),support velocity:interval(m/s)) {
        support phase:product(position,velocity);
        coordinate x:m on phase from position;
        coordinate v:m/s on phase from velocity;
        relation derivatives on phase { partial(x*v,wrt=x)=v; partial(x*v,wrt=v)=x; }
    }"""
    arguments = dict(entry="Phase", bindings={
        "position": eqiora.CoordinateInterval(0, 2, dimension=eqiora.Dimension(length=1)),
        "velocity": eqiora.CoordinateInterval(0, 4, dimension=eqiora.Dimension(length=1, time=-1)),
    })
    model = eqiora.compile(source=source, **arguments)
    assert eqiora.Model.from_bytes(model.to_bytes()).to_bytes() == model.to_bytes()
    with pytest.raises(eqiora.ValidationError):
        eqiora.compile(source=source.replace("partial(x*v,wrt=v)=x", "partial(x*v,wrt=v)=v"), **arguments)
