"""Mathematical condition kinds survive the installed Python source/Model route."""

import pytest
import eqiora

q = eqiora.lang
LENGTH = eqiora.ValueType.real(eqiora.Dimension(length=1))
FORCE = eqiora.ValueType.real(eqiora.Dimension(mass=1, length=1, time=-2))


def contact(*, condition=None, load=6, mixed=False):
    source = eqiora.Module("main")
    model = source.model("Contact")
    gap = model.field("gap", value_type=LENGTH, role=eqiora.FieldRole.Variable)
    force = model.field("force", value_type=FORCE, role=eqiora.FieldRole.Variable)
    zero_gap = q.quantity(0, eqiora.units.m)
    zero_force = q.quantity(0, eqiora.units.N)
    law = q.complementarity(q.greater_equal(gap, zero_gap),
                            q.less_equal(zero_force, force)) if condition is None else condition(gap, force)
    model.relation("contact",
                   q.equation(q.quantity(2, eqiora.units.N / eqiora.units.m) * gap - force,
                              q.quantity(load, eqiora.units.N)),
                   law,
                   q.inequality(q.quantity(4, eqiora.units.m), gap))
    if mixed:
        connector = source.connector("Pin", across=("position", LENGTH), through=("force", FORCE))
        terminal = source.component("Terminal")
        port = terminal.port("pin", connector=connector)
        terminal.relation("law", q.equation(port.force, q.quantity(0, eqiora.units.N)))
        first = model.instance("first", component=terminal, bindings={})
        second = model.instance("second", component=terminal, bindings={})
        model.connect(first["pin"], second["pin"])
    model.observable("opening", gap, value_type=LENGTH)
    return source


def test_condition_model_round_trip_preserves_types_and_order(tmp_path):
    source = contact()
    text = source.to_eqi()
    assert "complementarity(" in text and "inequality(" in text
    direct = eqiora.compile(source=source, entry="Contact")
    path = tmp_path / "contact.eqi"
    source.write_eqi(path)
    emitted = eqiora.compile(path=path, entry="Contact")
    assert direct.to_bytes() == emitted.to_bytes()
    assert direct.structural_fingerprint == emitted.structural_fingerprint
    assert eqiora.Model.from_bytes(direct.to_bytes()).digest == direct.digest


@pytest.mark.parametrize("condition", [
    lambda gap, force: q.complementarity(gap, force),
    lambda gap, force: q.complementarity(q.less_equal(gap, q.quantity(0, eqiora.units.m)),
                                          q.greater_equal(force, q.quantity(0, eqiora.units.N))),
    lambda gap, force: q.complementarity(q.greater_equal(gap, q.quantity(0, eqiora.units.N)),
                                          q.greater_equal(force, q.quantity(0, eqiora.units.N))),
    lambda gap, force: q.inequality(gap, force),
    lambda gap, force: q.inequality(q.greater_equal(gap, q.quantity(0, eqiora.units.m)), True),
])
def test_invalid_mathematical_conditions_fail_at_shared_compiler(condition):
    with pytest.raises(eqiora.ValidationError) as error:
        eqiora.compile(source=contact(condition=condition), entry="Contact")
    assert error.value.diagnostics


def test_conditions_are_immutable_distinct_from_boolean_and_exactly_owned():
    first = eqiora.Module("first").model("A")
    second = eqiora.Module("second").model("B")
    a = first.field("a", value_type=LENGTH, role=eqiora.FieldRole.Variable)
    b = second.field("b", value_type=LENGTH, role=eqiora.FieldRole.Variable)
    condition = q.inequality(a, q.quantity(0, eqiora.units.m))
    assert isinstance(condition, q.Inequality)
    assert not isinstance(condition, (q.Expression, q.Equation))
    with pytest.raises(TypeError, match="truth"):
        bool(condition)
    with pytest.raises(AttributeError, match="immutable"):
        condition._lhs = b
    with pytest.raises(q.ModuleError, match="owner"):
        q.inequality(a, b)
    with pytest.raises(q.ModuleError, match="Component"):
        second.relation("foreign", condition)
    with pytest.raises(TypeError, match="requires"):
        first.relation("boolean", q.greater_equal(a, q.quantity(0, eqiora.units.m)))
    with pytest.raises(TypeError):
        first.initial(condition)


@pytest.mark.parametrize("load,gap,force,activity", [(6, 3, 0, "inactive"), (-6, 0, 6, "active")])
def test_ordinary_finite_lifecycle_replays_original_conditions(load, gap, force, activity):
    model = eqiora.compile(source=contact(load=load), entry="Contact")
    length = eqiora.Dimension(length=1)
    force_unit = eqiora.Dimension(mass=1, length=1, time=-2)
    reference = model.constraint("contact", 1)
    assert reference.kind == "complementarity"
    for ordinal in (0, 3):
        with pytest.raises(TypeError):
            model.constraint("contact", ordinal)
    linear = eqiora.solve.Linear(relative_tolerance=1e-12, absolute_tolerance=1e-14,
                                  maximum_iterations=8, algorithm=eqiora.solve.LinearSolver.SparseLu,
                                  preconditioner=eqiora.solve.Preconditioner.Identity,
                                  reduction=eqiora.solve.Reduction.Fast, provider=eqiora.solve.SolverProvider.faer())
    with pytest.raises(eqiora.ValidationError):
        eqiora.resolve(model, solve=linear)
    tolerance = eqiora.solve.ConstraintTolerance.complementarity(reference, 1e-10, length, 1e-10, force_unit)
    bound = eqiora.solve.ConstraintTolerance.inequality(model.constraint("contact", 2), 1e-10, length)
    enforcement = eqiora.solve.ActiveSet(tolerances=(tolerance, bound), max_active_sets=2)
    plan = eqiora.resolve(model, solve=linear, enforcement=enforcement)
    assert len(plan.fields) == 2
    restored_plan = eqiora.Plan.from_bytes(plan.to_bytes())
    state = eqiora.State.initial(restored_plan)
    state = eqiora.State.from_bytes(restored_plan, state.to_bytes())
    result = eqiora.run(restored_plan, state=state)
    restored_result = eqiora.Result.from_bytes(restored_plan, result.to_bytes())
    assert restored_result.to_bytes() == result.to_bytes()
    assert restored_plan.enforcement.tolerances[0].reference == reference
    assert len(restored_result.constraints) == 2
    receipt = restored_result.constraints[0]
    assert receipt.reference == reference and receipt.activity == activity
    assert receipt.left_value == pytest.approx(gap, abs=1e-10)
    assert receipt.right_value == pytest.approx(force, abs=1e-10)
    assert (receipt.left_dimension, receipt.right_dimension) == (length, force_unit)
    assert (receipt.left_tolerance, receipt.right_tolerance) == (1e-10, 1e-10)
    assert restored_result.constraints[1].activity == "inequality"
    assert restored_result.observe(model.observable("opening")).value == pytest.approx(gap, abs=1e-10)
    other = eqiora.compile(source=contact(load=12), entry="Contact")
    with pytest.raises(TypeError, match="another exact Model"):
        eqiora.resolve(other, solve=linear, enforcement=enforcement)
    wrong = eqiora.solve.ConstraintTolerance.complementarity(reference, 1e-10, force_unit, 1e-10, length)
    with pytest.raises(eqiora.ValidationError):
        eqiora.resolve(model, solve=linear, enforcement=eqiora.solve.ActiveSet(tolerances=(wrong, bound), max_active_sets=2))
    with pytest.raises(TypeError, match="spatial or temporal"):
        eqiora.resolve(model, solve=linear, enforcement=enforcement, temporal=object())


def test_explicit_finite_constraints_reject_mixed_port_and_field_owners():
    model = eqiora.compile(source=contact(mixed=True), entry="Contact")
    length = eqiora.Dimension(length=1)
    force = eqiora.Dimension(mass=1, length=1, time=-2)
    enforcement = eqiora.solve.ActiveSet(tolerances=(
        eqiora.solve.ConstraintTolerance.complementarity(model.constraint("contact", 1), 1e-10, length, 1e-10, force),
        eqiora.solve.ConstraintTolerance.inequality(model.constraint("contact", 2), 1e-10, length),
    ), max_active_sets=2)
    linear = eqiora.solve.Linear(relative_tolerance=1e-12, absolute_tolerance=1e-14,
                                maximum_iterations=8, algorithm=eqiora.solve.LinearSolver.SparseLu,
                                preconditioner=eqiora.solve.Preconditioner.Identity,
                                reduction=eqiora.solve.Reduction.Fast, provider=eqiora.solve.SolverProvider.faer())
    with pytest.raises(eqiora.ValidationError, match="Ports"):
        eqiora.resolve(model, solve=linear, enforcement=enforcement)


def test_infeasible_inequality_is_not_dropped_or_relaxed():
    model = eqiora.compile(source=contact(load=12), entry="Contact")
    length = eqiora.Dimension(length=1)
    force = eqiora.Dimension(mass=1, length=1, time=-2)
    enforcement = eqiora.solve.ActiveSet(tolerances=(
        eqiora.solve.ConstraintTolerance.complementarity(
            model.constraint("contact", 1), 1e-10, length, 1e-10, force),
        eqiora.solve.ConstraintTolerance.inequality(
            model.constraint("contact", 2), 1e-10, length),
    ), max_active_sets=2)
    linear = eqiora.solve.Linear(
        relative_tolerance=1e-12,
        absolute_tolerance=1e-14,
        maximum_iterations=8,
        algorithm=eqiora.solve.LinearSolver.SparseLu,
        preconditioner=eqiora.solve.Preconditioner.Identity,
        reduction=eqiora.solve.Reduction.Fast,
        provider=eqiora.solve.SolverProvider.faer(),
    )
    plan = eqiora.resolve(model, solve=linear, enforcement=enforcement)
    with pytest.raises(eqiora.ExecutionError, match="inequality"):
        eqiora.run(plan, state=eqiora.State.initial(plan))
