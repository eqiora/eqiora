"""Continuous expression derivatives use the canonical structured authoring path."""

import eqiora

q = eqiora.lang


def test_time_and_alias_chain_derivative_compile_emit_and_replay(tmp_path):
    source = eqiora.Module("main")
    owner = source.model("Storage")
    state = owner.field("amount", role=eqiora.FieldRole.State,
                        value_type=eqiora.ValueType.real())
    stored = owner.let_alias("stored", state * state)
    second = q.quantity(1, eqiora.units.s)
    owner.initial((state, 1))
    owner.relation("flow", q.equation(q.derivative(stored), q.time() / second / second))
    direct = eqiora.compile(source=source, entry="Storage")
    path = tmp_path / "storage.eqi"
    source.write_eqi(path)
    assert "time()" in path.read_text()
    emitted = eqiora.compile(path=path, entry="Storage")
    replay = eqiora.Model.from_bytes(direct.to_bytes())
    assert direct.to_bytes() == emitted.to_bytes() == replay.to_bytes()
