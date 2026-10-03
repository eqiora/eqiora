"""Formal differentiation retains independent binding identity in installed authoring."""
import json

import pytest
import eqiora

q = eqiora.lang


def test_formal_partial_direct_emitted_and_model_replay_execute(tmp_path):
    source = eqiora.Module("main")
    scalar = eqiora.ValueType.real()
    f = source.operator("f", inputs={"x": scalar, "y": scalar}, result_type=scalar,
                        body=lambda x, y: x*x*y)
    slope = source.operator("slope", inputs={"x": scalar, "y": scalar}, result_type=scalar,
                            body=lambda x, y: q.partial(f(x=x, y=y), wrt=x, holding=(y,)))
    model = source.model("Partial")
    tick = model.clock("tick", period_s=1)
    memory = model.field("memory", role=eqiora.FieldRole.State, value_type=scalar, at=tick)
    model.initial((memory, 0))
    model.relation("evaluate", q.equation(q.next(memory), slope(x=3, y=5)), at=tick)
    direct = eqiora.compile(source=source, entry="Partial")
    path = tmp_path / "partial.eqi"
    source.write_eqi(path)
    emitted = eqiora.compile(path=path, entry="Partial")
    replay = eqiora.Model.from_bytes(direct.to_bytes())
    assert direct.to_bytes() == emitted.to_bytes() == replay.to_bytes()
    fields = [node["id"]["ulid"] for node in json.loads(direct.to_bytes())["nodes"]
              if node["definition"]["kind"] == "field"]
    assert len(fields) == 1
    for candidate in (direct, emitted, replay):
        session = candidate.execution_session(end_time_s=0.1, max_step_s=0.1, inputs={})
        assert session.advance_ticks(1) == 1
        assert session.field(fields[0]) == 30.0


def test_partial_constructor_rejects_binding_expressions_duplicates_and_foreign_owner():
    source = eqiora.Module("main")
    scalar = eqiora.ValueType.real()
    for body in (lambda x, y: q.partial(x*x, wrt=x*x),
                 lambda x, y: q.partial(x*x, wrt=x, holding=(x,)),
                 lambda x, y: q.partial(x*x, wrt=x, holding=(y, y))):
        with pytest.raises(q.ModuleError):
            source.operator("bad", inputs={"x": scalar, "y": scalar}, result_type=scalar, body=body)
    saved = []
    source.operator("capture", inputs={"x": scalar}, result_type=scalar,
                    body=lambda x: saved.append(x) or x)
    with pytest.raises(q.ModuleError, match="lexical owner"):
        source.operator("foreign", inputs={"x": scalar}, result_type=scalar,
                        body=lambda x: q.partial(x*x, wrt=saved[0]))


@pytest.mark.parametrize("component, direction, expected", [
    (0, (1, 0), 10), (1, (1, 0), 6),
    (0, (0, 1), 6), (1, (0, 1), 30),
    (0, (2, -1), 14), (1, (2, -1), -18),
    (2, (0, 0), 210), (3, (0, 0), 588),
])
def test_ordered_jacobian_actions_direct_emitted_and_replay(tmp_path, component, direction, expected):
    # f=x²y+y³ has H=[[2y,2x],[2x,6y]]. These exact integer
    # values at (3,5) are derived independently of the executable graph.
    source = eqiora.Module("main")
    scalar = eqiora.ValueType.real()

    def body(x, y, dx, dy):
        f = x*x*y + y*y*y
        if component >= 2:
            return q.vjp(f, wrt=(x, y)[component - 2], cotangent=7)
        first = q.partial(f, wrt=(x, y)[component])
        return q.jvp(first, wrt=(x, y), tangent=(dx, dy))

    action = source.operator("action", inputs={name: scalar for name in ("x", "y", "dx", "dy")},
                             result_type=scalar, body=body)
    model = source.model("HessianAction")
    tick = model.clock("tick", period_s=1)
    memory = model.field("memory", role=eqiora.FieldRole.State, value_type=scalar, at=tick)
    model.initial((memory, 0))
    model.relation("evaluate", q.equation(q.next(memory),
                   action(x=3, y=5, dx=direction[0], dy=direction[1])), at=tick)
    direct = eqiora.compile(source=source, entry="HessianAction")
    path = tmp_path / "hessian.eqi"
    source.write_eqi(path)
    emitted = eqiora.compile(path=path, entry="HessianAction")
    replay = eqiora.Model.from_bytes(direct.to_bytes())
    assert direct.to_bytes() == emitted.to_bytes() == replay.to_bytes()
    fields = [node["id"]["ulid"] for node in json.loads(direct.to_bytes())["nodes"]
              if node["definition"]["kind"] == "field"]
    assert len(fields) == 1
    for candidate in (direct, emitted, replay):
        session = candidate.execution_session(end_time_s=0.1, max_step_s=0.1, inputs={})
        assert session.advance_ticks(1) == 1
        assert session.field(fields[0]) == expected
