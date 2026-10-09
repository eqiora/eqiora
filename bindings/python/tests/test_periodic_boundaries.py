"""The admitted scalar transport specimen authors periodic topology explicitly."""

import pytest

import eqiora

from test_boundary_families import bindings, rectangle

q = eqiora.lang
CONCENTRATION = eqiora.ValueType.real(eqiora.Dimension(temperature=1))
FLUX = eqiora.ValueType.real(eqiora.Dimension(temperature=1, length=1, time=-1))
POTENTIAL = eqiora.ValueType.real(eqiora.Dimension(length=2, time=-1))
VELOCITY = eqiora.ValueType.real(eqiora.Dimension(length=1, time=-1))


def periodic_transport(*, wrong_axis=False):
    module = eqiora.Module("main")
    connector = module.field_connector("ScalarTransportBoundary",
                                        trace=("concentration", CONCENTRATION),
                                        flux=("transport_flux", FLUX))
    side = module.component("PeriodicTransportSide2d")
    body = side.volume("body", dimensions=2)
    face = side.boundary("face", parent=body)
    concentration = side.field_requirement("concentration", spatial_regularity=eqiora.SpatialRegularity.Smooth, value_type=CONCENTRATION,
                                            role=eqiora.FieldRole.Variable, on=body)
    potential = side.field_requirement("flow_potential", spatial_regularity=eqiora.SpatialRegularity.Smooth, value_type=POTENTIAL,
                                        role=eqiora.FieldRole.Variable, on=body)
    diffusivity = side.parameter("diffusivity", value_type=POTENTIAL)
    port = side.port("transport", connector=connector, on=face)
    side.relation("periodic_binding",
                  q.equation(q.trace(concentration) - port.concentration, 0),
                  q.equation(q.normal(concentration * q.grad(potential)
                                       - diffusivity * q.grad(concentration)) - port.transport_flux, 0),
                  on=face)

    root = module.model("Main")
    body = root.volume("body", dimensions=2)
    faces = {name: root.boundary(name, parent=body) for name in ("left", "right", "bottom", "top")}
    concentration = root.field("concentration", spatial_regularity=eqiora.SpatialRegularity.Smooth, value_type=CONCENTRATION,
                                role=eqiora.FieldRole.State, on=body)
    root.initial(left=concentration, right=q.quantity(1, eqiora.units.K))
    potential = root.field("flow_potential", spatial_regularity=eqiora.SpatialRegularity.Smooth, value_type=POTENTIAL,
                           role=eqiora.FieldRole.Variable, on=body)
    speed = root.parameter("speed", value_type=VELOCITY)
    diffusivity = root.parameter("diffusivity", value_type=POTENTIAL)
    root.set_default(speed, q.quantity(1, eqiora.units.m / eqiora.units.s))
    root.set_default(diffusivity, q.quantity(0.2, eqiora.units.m**2 / eqiora.units.s))
    root.relation("flow_definition", q.equation(potential - speed * q.coordinate(0), 0), on=body)
    root.relation("transport", q.equation(q.derivative(concentration)
                                           + q.div(concentration * q.grad(potential))
                                           - q.div(diffusivity * q.grad(concentration)), 0), on=body)
    endpoints = []
    for name in ("left", "bottom" if wrong_axis else "right"):
        instance = root.instance(name + "_periodic", component=side, bindings={
            "body": body, "face": faces[name], "concentration": concentration,
            "flow_potential": potential, "diffusivity": diffusivity,
        })
        endpoints.append(instance["transport"])
    for name in ("bottom", "top"):
        root.relation(name + "_flux", q.equation(q.normal(diffusivity * q.grad(concentration)), 0), on=faces[name])
    root.connect_periodic(*endpoints)
    return module


def test_periodic_transport_retains_abstract_supports_and_exact_geometry_on_replay(tmp_path):
    module = periodic_transport()
    geometry = rectangle()
    text = module.to_eqi()
    assert "connect periodic left_periodic.transport, right_periodic.transport;" in text
    assert "domain " not in text
    direct = eqiora.compile(source=module, geometry=geometry, entry="Main", bindings=bindings(geometry))
    path = tmp_path / "transport.eqi"
    module.write_eqi(path)
    emitted = eqiora.compile(path=path, geometry=geometry, entry="Main", bindings=bindings(geometry))
    assert direct.to_bytes() == emitted.to_bytes()
    assert eqiora.Model.from_bytes(direct.to_bytes()).digest == direct.digest


def test_periodic_pairing_does_not_follow_equal_field_types_on_another_axis():
    geometry = rectangle()
    with pytest.raises(eqiora.ValidationError) as error:
        eqiora.compile(source=periodic_transport(wrong_axis=True), geometry=geometry,
                       entry="Main", bindings=bindings(geometry))
    assert any("periodic" in diagnostic.message.lower() for diagnostic in error.value.diagnostics)
    assert all("unsupported" not in diagnostic.message.lower() for diagnostic in error.value.diagnostics)
