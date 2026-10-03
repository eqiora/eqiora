"""Crossing event handles preserve nominal activation and lexical guard ownership."""

import pytest

import eqiora

q = eqiora.lang


def event_source(*, wrong_activation=False):
    source = eqiora.Module("main")
    owner = source.model("Impact")
    velocity = owner.field("velocity", role=eqiora.FieldRole.State,
                           value_type=eqiora.ValueType.real(eqiora.Dimension(length=1, time=-1)))
    height = owner.field("height", role=eqiora.FieldRole.State,
                         value_type=eqiora.ValueType.real(eqiora.Dimension(length=1)))
    impact = owner.event("impact", height, direction="falling", doc="Downward ground crossing.")
    old = owner.let_alias("old_velocity", q.pre(velocity), at=impact)
    activation = owner.event("other", height, direction="falling") if wrong_activation else impact
    owner.initial((height, q.quantity(1, eqiora.units.m)),
                  (velocity, q.quantity(-1, eqiora.units.m / eqiora.units.s)))
    owner.relation("reset", q.equation(q.next(velocity), -old), at=activation)
    return source


def test_event_and_alias_authoring_emits_exact_distinct_activation():
    text = event_source().to_eqi()
    assert "event impact = crossing(height, direction = falling);" in text
    assert "Downward ground crossing." in text
    assert "let old_velocity at impact = pre(velocity);" in text
    assert "relation reset at impact" in text
    assert "next(velocity) = -old_velocity;" in text


@pytest.mark.parametrize("direction", ("any", "rising", "falling"))
def test_event_requires_explicit_closed_direction_and_is_immutable(direction):
    source = eqiora.Module("main")
    owner = source.model("Events")
    first = owner.event("first", 1, direction=direction)
    second = owner.event("second", 1, direction=direction)
    assert isinstance(first, q.Event)
    assert not isinstance(first, q.Clock)
    assert len({first, second}) == 2
    with pytest.raises(AttributeError):
        first.name = "changed"
    with pytest.raises(TypeError):
        owner.event("missing", 1)
    assert f"crossing(1, direction = {direction})" in source.to_eqi()
    with pytest.raises(q.ModuleError, match="frozen"):
        owner.event("late", 1, direction=direction)


@pytest.mark.parametrize("direction", (None, True, "up", "", 1))
def test_invalid_event_direction_does_not_reserve_a_name(direction):
    owner = eqiora.Module("main").model("Events")
    with pytest.raises(q.ModuleError, match="direction"):
        owner.event("bad", 1, direction=direction)
    owner.let_alias("bad", 1)


@pytest.mark.parametrize("same_source", (True, False))
def test_event_rejects_foreign_guard_and_activation_before_name_reservation(same_source):
    source = eqiora.Module("main")
    owner = source.model("Local")
    foreign = (source if same_source else eqiora.Module("main")).component("Foreign")
    guard = foreign.parameter("guard", value_type=eqiora.ValueType.real())
    event = foreign.event("impact", guard, direction="any")
    with pytest.raises(q.ModuleError, match="Component"):
        owner.event("bad", guard, direction="falling")
    with pytest.raises(q.ModuleError, match="event.*Component"):
        owner.relation("bad", q.equation(1, 1), at=event)
    with pytest.raises(q.ModuleError, match="event.*Component"):
        owner.let_alias("bad", 1, at=event)
    owner.let_alias("bad", 1)


def test_event_guard_rejects_foreign_constant_call_and_escaped_reduction_binder():
    source = eqiora.Module("main")
    owner = source.model("Local")
    foreign = eqiora.Module("main")
    constant = foreign.operator("constant", inputs={}, result_type=eqiora.ValueType.real(), body=lambda: 1)
    with pytest.raises(q.ModuleError, match="Module"):
        owner.event("bad", constant(), direction="any")
    rows = owner.index_set("Rows", extent=2)
    escaped = []
    owner.sum(lambda i: escaped.append(i) or 1, over=rows)
    with pytest.raises(q.ModuleError, match="binder"):
        owner.event("bad", q.ordinal(escaped[0]), direction="any")
    owner.let_alias("bad", 1)


@pytest.mark.parametrize("method", ("field", "input", "output"))
def test_event_is_not_a_periodic_field_or_port_clock(method):
    owner = eqiora.Module("main").model("Events")
    event = owner.event("impact", 1, direction="any")
    arguments = {"value_type": eqiora.ValueType.real(), "at": event}
    if method == "field":
        arguments["role"] = eqiora.FieldRole.State
    with pytest.raises(q.ModuleError, match="clock"):
        getattr(owner, method)("bad", **arguments)
    owner.let_alias("bad", 1)


def test_event_cannot_satisfy_a_borrowed_periodic_clock_requirement():
    source = eqiora.Module("main")
    child = source.component("Child")
    tick = child.clock_requirement("tick")
    owner = source.model("Events")
    event = owner.event("impact", 1, direction="any")
    with pytest.raises(q.ModuleError, match="Clock"):
        owner.instance("bad", component=child, bindings={'tick': event})
    owner.let_alias("bad", 1)


def test_event_guard_budget_is_checked_before_reserving_the_event():
    owner = eqiora.Module("main").model("Events")
    guard = q.math.pi
    for _ in range(11):
        guard = guard + guard
    owner.event("first", guard, direction="any")  # 4095 authored nodes
    owner.event("second", 1, direction="any")
    with pytest.raises(q.ModuleError, match="node limit"):
        owner.event("bad", 1, direction="any")
    owner.let_alias("bad", 1)


def test_event_guard_reset_and_alias_compile_from_source_and_file(tmp_path):
    source = event_source()
    model = eqiora.compile(source=source, entry="Impact")
    path = tmp_path / "impact.eqi"
    source.write_eqi(path)
    assert eqiora.compile(path=path, entry="Impact").to_bytes() == model.to_bytes()


def test_event_alias_cannot_be_consumed_at_an_equal_guard_foreign_event():
    with pytest.raises(eqiora.ValidationError) as error:
        eqiora.compile(source=event_source(wrong_activation=True), entry="Impact")
    assert any("event" in diagnostic.message.lower() or "activation" in diagnostic.message.lower()
               for diagnostic in error.value.diagnostics)


@pytest.mark.parametrize("priority", (-(1 << 63), -2, 0, 2, (1 << 63) - 1))
def test_event_priority_has_exact_canonical_source(priority):
    source = eqiora.Module("main")
    owner = source.model("Events")
    owner.event("selected", 1, direction="rising", priority=priority)
    suffix = f", priority = {priority}" if priority else ""
    assert f"crossing(1, direction = rising{suffix})" in source.to_eqi()


@pytest.mark.parametrize("priority", (True, False, 1.0, "1", None, -(1 << 63) - 1, 1 << 63))
def test_invalid_event_priority_does_not_reserve_a_name(priority):
    owner = eqiora.Module("main").model("Events")
    with pytest.raises(q.ModuleError, match="priority"):
        owner.event("available", 1, direction="rising", priority=priority)
    owner.let_alias("available", 1)
