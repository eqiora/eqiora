"""Oriented authoring retains spatial graph meaning through source and exact replay."""
import json

import eqiora
import pytest


@pytest.mark.parametrize("complex_values", [False, True])
def test_hcurl_test_declaration_survives_python_source_compilation(complex_values):
    q = eqiora.lang
    module = eqiora.Module("main")
    owner = module.component("CurlForm")
    body = owner.volume("body", dimensions=2)
    scalar = eqiora.ValueType.complex() if complex_values else eqiora.ValueType.real()
    u = owner.field("u", role=eqiora.FieldRole.Variable,
                    value_type=eqiora.ValueType.vector(scalar, 2), on=body)
    law = owner.relation("law", q.equation(q.curl(q.curl(u)), 0), on=body)
    v = owner.test("v", for_=u, regularity="hcurl")
    test_curl = q.math.conj(q.curl(v)) if complex_values else q.curl(v)
    owner.weak_form("weak", [law], equations=[(q.integrate(body, test_curl * q.curl(u)), 0)])
    graph = eqiora.geometry.GeometryGraph()
    rectangle = graph.rectangle(x_bounds=(0.0, 1.0), y_bounds=(0.0, 1.0))
    geometry = graph.build(rectangle, named_topology={
        "body": rectangle.region,
        "left": rectangle.boundaries[0], "right": rectangle.boundaries[1],
        "bottom": rectangle.boundaries[2], "top": rectangle.boundaries[3],
    })
    bindings = {"body": geometry.selection("body")}
    source = module.to_eqi()
    model = eqiora.compile(source=source, entry="CurlForm", geometry=geometry, bindings=bindings)
    assert model.authored_formulations[0].test_restrictions[0][4] == "hcurl"
    with pytest.raises(eqiora.EqioraError, match="declared regularity"):
        eqiora.compile(source=source.replace("in hcurl", "in l2"), entry="CurlForm",
                       geometry=geometry, bindings=bindings)


@pytest.mark.parametrize("dimensions", [2, 3])
@pytest.mark.parametrize("complex_values", [False, True])
def test_curl_and_tangential_trace_retain_exact_support(dimensions, complex_values, tmp_path):
    q = eqiora.lang
    module = eqiora.Module("main")
    owner = module.component("Oriented")
    body = owner.volume("body", dimensions=dimensions)
    face = owner.boundary("face", parent=body)
    scalar = eqiora.ValueType.complex() if complex_values else eqiora.ValueType.real()
    vector = eqiora.ValueType.vector(scalar, dimensions)
    u = owner.field("u", role=eqiora.FieldRole.Variable, value_type=vector, on=body)
    owner.relation("interior", q.equation(q.curl(q.curl(u)), -q.div(q.grad(u))), on=body)
    # The trace has the same oriented boundary scope but different planar output shape.
    f = owner.field("f", role=eqiora.FieldRole.Variable, value_type=scalar, on=body)
    boundary_value = q.trace(u) if dimensions == 3 else q.trace(f)
    owner.relation("boundary_value", q.equation(q.tangential_trace(u), boundary_value), on=face)
    emitted = module.to_eqi()
    assert "curl(curl(u))" in emitted
    assert "tangential_trace(u)" in emitted
    bounds = ",".join(["0,1"] * dimensions)
    emitted += (
        f"\nmodel M() {{ domain body=box({bounds}); "
        "domain face=boundary(body,axis=0,side=upper); "
        "instance oriented:Oriented(body=body,face=face); }"
    )
    model = eqiora.compile(source=emitted, entry="M")
    path = tmp_path / "oriented.eqi"
    path.write_text(emitted)
    assert eqiora.compile(path=path, entry="M").to_bytes() == model.to_bytes()
    assert eqiora.Model.from_bytes(model.to_bytes()).to_bytes() == model.to_bytes()
    relations = [node["definition"] for node in json.loads(model.to_bytes())["nodes"]
                 if node["definition"]["kind"] == "relation"]
    ops = {node["op"] for relation in relations for node in relation["expression"]["nodes"]}
    assert {"gradient", "normal-component", "pure-operator-application"} <= ops


def test_oriented_unary_operators_preserve_lexical_ownership():
    q = eqiora.lang
    module = eqiora.Module("main")
    first, second = (module.component(name) for name in ("First", "Second"))
    body = first.volume("body", dimensions=3)
    u = first.field("u", role=eqiora.FieldRole.Variable,
                    value_type=eqiora.ValueType.vector(eqiora.ValueType.real(), 3), on=body)
    for value in (q.curl(u), q.tangential_trace(u)):
        with pytest.raises(q.ModuleError):
            second.relation("foreign", q.equation(value, 0))
