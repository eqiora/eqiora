"""Cross authoring retains the shared typed operator through source and replay."""
import json

import eqiora
import pytest


@pytest.mark.parametrize("complex_values", [False, True])
def test_cross_module_source_and_exact_replay(complex_values):
    q = eqiora.lang
    module = eqiora.Module("main")
    owner = module.model("Cross")
    scalar = eqiora.ValueType.complex() if complex_values else eqiora.ValueType.real()
    kind = eqiora.ValueType.vector(scalar, 3)
    a, b, c = [owner.field(name, role=eqiora.FieldRole.Variable, value_type=kind)
               for name in ("a", "b", "c")]
    owner.relation("law", q.equation(q.cross(a, b), c))
    emitted = module.to_eqi()
    assert "cross(a, b)" in emitted
    model = eqiora.compile(source=module, entry="Cross")
    assert eqiora.compile(source=emitted, entry="Cross").to_bytes() == model.to_bytes()
    assert eqiora.Model.from_bytes(model.to_bytes()).to_bytes() == model.to_bytes()
    relations = [node["definition"] for node in json.loads(model.to_bytes())["nodes"]
                 if node["definition"]["kind"] == "relation"]
    assert len(relations) == 1
    assert any(node["op"] == "pure-operator-application"
               for node in relations[0]["expression"]["nodes"])


def test_cross_rejects_foreign_lexical_owners():
    module = eqiora.Module("main")
    kind = eqiora.ValueType.vector(eqiora.ValueType.real(), 3)
    a, b = [module.component(name).field("v", role=eqiora.FieldRole.Variable, value_type=kind)
            for name in ("First", "Second")]
    with pytest.raises(eqiora.lang.ModuleError):
        eqiora.lang.cross(a, b)
