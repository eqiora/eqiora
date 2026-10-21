"""Typed material inputs use ordinary authoring, finite execution and replay."""

import numpy as np
import pytest
import eqiora as q


@pytest.mark.parametrize(
    "shaped,complex_domain",
    [(False, False), (False, True), (True, True), (True, False)],
)
def test_value_only_inputs_keep_complete_type_and_ordinary_execution(
    shaped, complex_domain
):
    lang = q.lang
    source = q.Module("main")
    scalar = q.ValueType.complex() if complex_domain else q.ValueType.real()
    kind = q.ValueType.array(scalar, 2) if shaped else scalar
    owner = source.property_contract(
        "Response",
        value_type=kind,
        inputs={"x": kind},
        derivatives="value_only",
        branch="single",
    )
    x = owner.input("x")
    component = lang.component(x, 0) if shaped else x
    value = (
        component * lang.math.conj(component) + lang.math.complex(0, 1) * component
        if complex_domain
        else component * component + 2 * component
    )
    measured = source.property_release(
        "Measured",
        implements=owner,
        value=value,
        source_unit=q.units.one,
        source_scale=1,
        citation="org.example.manufactured",
        license="spdx.CC0_1_0",
        validity=lang.greater_equal(lang.math.real(component), 0),
        outside="reject",
        branch="single",
    )
    consumer = source.component("Consumer")
    response = consumer.property("response", contract=owner)
    x = consumer.parameter("x", value_type=kind)
    y = consumer.field("y", role=q.FieldRole.Variable, value_type=kind)
    consumer.relation("law", lang.equation(y, response(x=x)))
    consumer.observable("output", y, value_type=kind)
    root = source.model("Main")
    p = root.parameter("p", value_type=kind)
    values = [1 + 2j, 3 - 4j] if complex_domain else [1.0, 3.0]
    root.set_default(p, lang.array(values) if shaped else values[0])
    root.instance("sample", component=consumer, bindings={"x": p, "response": measured})
    model = q.compile(source=source, entry="Main")
    parsed = q.compile(source=source.to_eqi(), entry="Main")
    assert model.to_bytes() == parsed.to_bytes()
    # z*conj(z)+i*z = a²+b²-b+i*a; the real law is x²+2x.
    expected = [3 + 1j, 29 + 3j] if complex_domain else [3.0, 15.0]
    expected = expected if shaped else expected[0]
    solve = q.solve.Linear(
        relative_tolerance=1e-13,
        absolute_tolerance=1e-15,
        maximum_iterations=8,
        algorithm=q.solve.LinearSolver.BiConjugateGradientStabilized,
        preconditioner=q.solve.Preconditioner.Identity,
        reduction=q.solve.Reduction.Reproducible,
        provider=q.solve.SolverProvider.reference(),
    )
    output = model.observable("sample.output")
    parameter = model.parameter("p")
    for candidate in (model, parsed, q.Model.from_bytes(model.to_bytes())):
        plan = q.resolve(candidate, solve=solve)
        plan = q.Plan.from_bytes(plan.to_bytes())
        initial = q.State.initial(plan)
        before = initial.to_bytes()
        result = q.run(plan, state=initial)
        reopened = q.Result.from_bytes(plan, result.to_bytes())
        assert reopened.to_bytes() == result.to_bytes()
        np.testing.assert_allclose(
            result.observe(output).value,
            expected,
            rtol=0,
            atol=1e-10,
        )
        assert initial.to_bytes() == before
        # Admission of a complete value type grants no active-input derivative.
        with pytest.raises(q.ValidationError, match="value-only contract"):
            q.diff.compile(plan, inputs=(parameter,), outputs=(plan.fields[0],))
