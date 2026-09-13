"""Fixed-domain physical Laws share source/native authoring and Relation identity."""

import re

import pytest

import eqiora
from test_language_source import rectangle_geometry

q = eqiora.lang


def balance_module(field_dimension, coefficient_dimension, production_dimension):
    module = eqiora.Module("conservation")
    component = module.component("Balance")
    body = component.volume("body", dimensions=2)
    value = component.field("value", on=body, role=eqiora.FieldRole.Variable,
                            value_type=eqiora.ValueType.real(field_dimension))
    coefficient = component.parameter("coefficient", value_type=eqiora.ValueType.real(coefficient_dimension))
    production = component.parameter("production", value_type=eqiora.ValueType.real(production_dimension))
    handle = component.law("balance", on=body, flux=-coefficient * q.grad(value),
                           source=production, doc="Physical outward flux.")
    return module, component, body, value, handle


@pytest.mark.parametrize("dimensions", [
    (eqiora.Dimension(temperature=1),
     eqiora.Dimension(mass=1, length=1, time=-3, temperature=-1),
     eqiora.Dimension(mass=1, length=-1, time=-3)),
    (eqiora.Dimension(mass=1, length=-3), eqiora.Dimension(length=2, time=-1),
     eqiora.Dimension(mass=1, length=-3, time=-1)),
])
def test_heat_and_mass_laws_compile_from_native_python_and_emitted_source(dimensions, tmp_path):
    module, _, _, _, handle = balance_module(*dimensions)
    assert isinstance(handle, q.Relation)
    text = module.to_eqi()
    assert "law balance on body" in text
    assert "flux -coefficient * grad(value);" in text
    assert "source production;" in text
    assert "storage" not in text
    geometry = rectangle_geometry()
    bindings = {"body": geometry.selection("region"),
                "coefficient": 2.0, "production": 4.0}
    direct = eqiora.compile(source=module, geometry=geometry, entry="Balance", bindings=bindings)
    path = tmp_path / "conservation.eqi"
    path.write_text(text)
    replay = eqiora.compile(path=path, geometry=geometry, entry="Balance", bindings=bindings)
    assert direct.structurally_equivalent(replay)
    # The equation has identical ordered operands but does not own physical
    # flux/source meaning. Structural identity must retain that distinction.
    equations = re.sub(
        r"law balance on body \{\s*flux ([^;]+);\s*source ([^;]+);\s*\}",
        r"relation balance on body { div(\1) = \2; }", text,
    )
    assert equations != text
    ordinary = eqiora.compile(source=equations, geometry=geometry,
                               entry="Balance", bindings=bindings)
    assert not direct.structurally_equivalent(ordinary)


def test_law_rejects_foreign_terms_and_boundary_support_before_mutating_draft():
    module = eqiora.Module("conservation")
    left = module.component("Left")
    right = module.component("Right")
    body = left.volume("body", dimensions=2)
    boundary = left.boundary("surface", parent=body)
    other = right.volume("other", dimensions=2)
    value = left.field("value", on=body, role=eqiora.FieldRole.Variable,
                       value_type=eqiora.ValueType.real())
    foreign = right.field("foreign", on=other, role=eqiora.FieldRole.Variable,
                          value_type=eqiora.ValueType.real())
    with pytest.raises(q.ModuleError, match="belong to this Component"):
        left.law("balance", on=body, flux=q.grad(foreign), source=0)
    with pytest.raises(q.ModuleError, match="volume support"):
        left.law("balance", on=boundary, flux=q.grad(value), source=0)
    with pytest.raises(q.ModuleError):
        left.law("balance", on=other, flux=q.grad(value), source=0)
    with pytest.raises(TypeError, match="source"):
        left.law("balance", on=body, flux=q.grad(value))
    left.law("balance", on=body, flux=q.grad(value), source=0)
    assert module.to_eqi().count("law balance") == 1


def test_component_relation_before_law_preserves_both_authoring_paths():
    module = eqiora.Module("mixed_relation_law")
    component = module.component("Balance")
    body = component.volume("body", dimensions=1)
    boundary = component.boundary("left", parent=body)
    value = component.field("value", on=body, role=eqiora.FieldRole.Variable,
                            value_type=eqiora.ValueType.real())
    component.relation("trace", q.equation(q.trace(value), 0), on=boundary)
    component.law("balance", on=body, flux=-q.grad(value), source=0)
    text = module.to_eqi()
    assert "relation trace on left" in text
    assert "law balance on body" in text


def test_steady_law_model_replay_executes_with_exact_geometry_admission():
    import numpy as np
    from test_external_geometry_round_trip import bindings, execute, interval

    geometry = interval()
    module = eqiora.Module("balance")
    component = module.component("Diffusion")
    body = component.volume("body", dimensions=1)
    left = component.boundary("left", parent=body)
    right = component.boundary("right", parent=body)
    value = component.field("u", on=body, role=eqiora.FieldRole.Variable,
                            value_type=eqiora.ValueType.real())
    component.law("balance", on=body, flux=-2 * q.grad(value),
                  source=q.quantity(4, eqiora.units.one / eqiora.units.m**2))
    for name, boundary in (("left_value", left), ("right_value", right)):
        component.relation(name, q.equation(q.trace(value), 0), on=boundary)
    direct = eqiora.compile(source=module, geometry=geometry,
                            entry="Diffusion", bindings=bindings(geometry))
    field_id = direct.field("definition.u").id
    replay = eqiora.Model.from_bytes(direct.to_bytes())
    mesh = eqiora.meshing.generate(eqiora.meshing.resolve(
        geometry, eqiora.meshing.CartesianMesher(cells=(4,)),
    ))
    # -2*u''=4 m^-2 and u(0)=u(1 m)=0 imply u=x*(1-x).
    # Exact integration of this constant load reproduces the quadratic at Q1
    # nodes; tolerance covers binary64 linear-solve roundoff only.
    x = mesh.coordinates[:, 0]
    expected = x * (1.0 - x)
    for model in (direct, replay):
        np.testing.assert_allclose(execute(model, mesh, field_id), expected,
                                   rtol=0, atol=1e-12)
    # Identical coordinates with foreign Geometry identity cannot admit replay.
    foreign = interval(label="foreign_body")
    foreign_mesh = eqiora.meshing.generate(eqiora.meshing.resolve(
        foreign, eqiora.meshing.CartesianMesher(cells=(4,)),
    ))
    np.testing.assert_array_equal(foreign_mesh.coordinates, mesh.coordinates)
    with pytest.raises(eqiora.EqioraError):
        execute(replay, foreign_mesh, field_id)

    reversed_source = module.to_eqi().replace("flux -2 * grad(u);", "flux 2 * grad(u);")
    assert reversed_source != module.to_eqi()
    reversed_model = eqiora.compile(source=reversed_source, geometry=geometry,
                                    entry="Diffusion", bindings=bindings(geometry))
    with pytest.raises(eqiora.ExecutionError, match="positive finite diffusion"):
        execute(reversed_model, mesh, reversed_model.field("definition.u").id)
