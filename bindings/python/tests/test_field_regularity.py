"""Both authoring surfaces retain the same explicit continuum assertion."""
import json

import eqiora
import pytest


@pytest.mark.parametrize("regularity,admitted", [
    (eqiora.SpatialRegularity.Unspecified, False),
    (eqiora.SpatialRegularity.L2, False),
    (eqiora.SpatialRegularity.H1, True),
    (eqiora.SpatialRegularity.Smooth, True),
])
def test_native_trace_requires_field_regularity(regularity, admitted):
    body = eqiora.Domain.box("body", (0.0, 1.0), (0.0, 1.0))
    wall = body.boundary("wall", axis=0, side=eqiora.BoundarySide.Lower)
    u = eqiora.Field("u", role=eqiora.FieldRole.Variable, domain=body,
                     spatial_regularity=regularity)
    assert u.spatial_regularity == regularity
    module = eqiora.Module("M", body, wall, u,
        eqiora.Relation("wall_value", domain=wall,
                        equations=[(eqiora.trace(u, on=wall), 0)]))
    if not admitted:
        with pytest.raises(eqiora.ValidationError, match="regularity"):
            eqiora.compile(source=module)
        return
    model = eqiora.compile(source=module)
    assert eqiora.Model.from_bytes(model.to_bytes()).to_bytes() == model.to_bytes()


def test_module_field_assertion_reaches_source_and_model():
    module = eqiora.Module("main")
    owner = module.component("C")
    body = owner.volume("body", dimensions=2)
    u = owner.field("u", role=eqiora.FieldRole.Variable,
                    value_type=eqiora.ValueType.real(), on=body,
                    spatial_regularity=eqiora.SpatialRegularity.H1)
    owner.relation("law", eqiora.lang.equation(u, 0), on=body)
    source = module.to_eqi()
    assert "variable u: 1 on body in h1;" in source
    source += "\nmodel M() { domain body=box(0,1,0,1); instance c:C(body=body); }"
    model = eqiora.compile(source=source, entry="M")
    fields = [node["definition"] for node in json.loads(model.to_bytes())["nodes"]
              if node["definition"]["kind"] == "field"]
    assert len(fields) == 1
    assert fields[0]["spatial_regularity"] == "h1"


@pytest.mark.parametrize("profile,admitted", [("h1", True), ("smooth", True), ("l2", False)])
def test_borrowed_field_keeps_the_callers_regularity(profile, admitted):
    module = eqiora.Module("main")
    owner = module.component("BoundaryValue")
    body = owner.volume("body", dimensions=2)
    wall = owner.boundary("wall", parent=body)
    u = owner.field_requirement("u", role=eqiora.FieldRole.Variable,
                                value_type=eqiora.ValueType.real(), on=body,
                                spatial_regularity=eqiora.SpatialRegularity.H1)
    owner.relation("value", eqiora.lang.equation(eqiora.lang.trace(u), 0), on=wall)
    source = module.to_eqi()
    assert "variable u: 1 on body in h1" in source
    source += (
        "\nmodel M() { domain body=box(0,1,0,1); "
        "domain wall=boundary(body,axis=0,side=lower); "
        f"variable u: 1 on body in {profile}; "
        "instance c:BoundaryValue(body=body,wall=wall,u=u); }"
    )
    if not admitted:
        with pytest.raises(eqiora.ValidationError, match="regularity"):
            eqiora.compile(source=source, entry="M")
        return
    model = eqiora.compile(source=source, entry="M")
    fields = [node["definition"] for node in json.loads(model.to_bytes())["nodes"]
              if node["definition"]["kind"] == "field"]
    assert len(fields) == 1
    assert fields[0]["spatial_regularity"] == profile
