"""Exact nominal input ownership and deterministic analytic property authoring."""
import pytest
import eqiora

q = eqiora.lang
u = eqiora.units


def contract(source, name="Response", **kwargs):
    return source.property_contract(name, value_type=eqiora.ValueType.real(),
                                    inputs={"x": eqiora.ValueType.real(), "y": eqiora.ValueType.real()},
                                    derivatives="first_partials", **kwargs)


def release(source, owner, value, **kwargs):
    return source.property_release("Measured", implements=owner, value=value,
                                   source_unit=u.one, source_scale=1,
                                   citation="org.example.measurement", license="spdx.CC0_1_0", **kwargs)


def test_exact_contract_inputs_and_release_validity_cannot_capture_foreign_formals():
    source = eqiora.Module("main")
    first = contract(source)
    second = contract(source, "Other")
    with pytest.raises(q.ModuleError, match="exact contract"):
        release(source, first, second.input("x"))
    with pytest.raises(q.ModuleError, match="exact contract"):
        release(source, first, first.input("x"), validity=q.greater(second.input("x"), 0))
    with pytest.raises(q.ModuleError, match="declared contract formal"):
        first.input("absent")
    with pytest.raises(AttributeError):
        first.input("x")._owner = second
    with pytest.raises(AttributeError):
        first._profile.branch = "changed"
    # Failed attempts did not consume the release name.
    release(source, first, first.input("x") * first.input("y"), validity=q.greater_equal(first.input("x"), 0))


def test_branch_and_outside_policy_are_explicit_and_validation_is_atomic():
    source = eqiora.Module("main")
    owner = contract(source, branch="liquid")
    for kwargs in [{}, {"branch": "gas"}, {"branch": "liquid", "outside": "extrapolate"}]:
        with pytest.raises(q.ModuleError):
            release(source, owner, owner.input("x"), **kwargs)
    with pytest.raises(TypeError):
        release(source, owner, lambda: 1, branch="liquid")
    release(source, owner, owner.input("x"), branch="liquid", outside="reject")


def test_named_application_preserves_order_and_rejects_foreign_components():
    source = eqiora.Module("main")
    owner = contract(source)
    release(source, owner, owner.input("x") * owner.input("y"))
    component = source.component("Consumer")
    requirement = component.property("response", contract=owner)
    assert isinstance(requirement, q.PropertyRequirement)
    x = component.parameter("x", value_type=eqiora.ValueType.real())
    y = component.parameter("y", value_type=eqiora.ValueType.real())
    component.let_alias("first", requirement(y=y, x=x))
    component.let_alias("second", requirement(x=x, y=y))
    with pytest.raises(q.ModuleError, match="exactly its named inputs"):
        requirement(x=x)
    foreign = source.component("Other")
    foreign_x = foreign.parameter("x", value_type=eqiora.ValueType.real())
    with pytest.raises(q.ModuleError, match="same Component"):
        requirement(x=foreign_x, y=y)
    text = source.to_eqi()
    assert text.count("response(x = x, y = y)") == 2
    assert "first_partials" in text


def test_ordered_input_mapping_is_snapshotted_and_invalid_profiles_do_not_reserve_names():
    source = eqiora.Module("main")
    with pytest.raises(q.ModuleError, match="derivatives"):
        source.property_contract("Response", value_type=eqiora.ValueType.real(), derivatives="automatic")
    inputs = {"y": eqiora.ValueType.real(), "x": eqiora.ValueType.real()}
    owner = source.property_contract("Response", value_type=eqiora.ValueType.real(), inputs=inputs)
    inputs.clear()
    assert tuple(name for name, _ in owner._profile.inputs) == ("y", "x")
    assert owner.input("x") is not None


def test_analytic_inputs_execute_through_direct_source_and_model_replay(tmp_path):
    source = eqiora.Module("main")
    owner = contract(source)
    measured = release(source, owner, owner.input("x") ** 2 + 3 * owner.input("y"),
                       validity=q.greater_equal(owner.input("x"), 0))
    component = source.component("Consumer")
    tick = component.clock_requirement("tick")
    response = component.property("response", contract=owner)
    x = component.parameter("x", value_type=eqiora.ValueType.real())
    y = component.parameter("y", value_type=eqiora.ValueType.real())
    output = component.output("result", value_type=eqiora.ValueType.real(), at=tick)
    component.relation("apply", q.equation(output, response(x=x, y=y)), at=tick)
    root = source.model("Main")
    clock = root.clock("tick", period_s=1)
    for name, first, second in [("first", 2, 5), ("second", 3, 7)]:
        inner = root.instance(name, component=component,
                              bindings={"tick": clock, "response": measured, "x": first, "y": second})
        observed = root.output(f"result_{name}", value_type=eqiora.ValueType.real(), at=clock)
        root.relation(f"observe_{name}", q.equation(observed, inner["result"]), at=clock)
    direct = eqiora.compile(source=source, entry="Main")
    path = tmp_path / "analytic.eqi"
    source.write_eqi(path)
    parsed = eqiora.compile(path=path, entry="Main")
    assert direct.to_bytes() == parsed.to_bytes()
    reopened = eqiora.Model.from_bytes(direct.to_bytes())
    for model in [direct, parsed, reopened]:
        assert len(model.property_bindings) == 2
        assert all(binding.inputs == ["x", "y"] and binding.normalized_value is None
                   for binding in model.property_bindings)
    for model in [direct, parsed]:
        session = model.execution_session(end_time_s=1, max_step_s=0.1, inputs={})
        assert session.advance_ticks(1) == 1
        assert session.output("result_first", 0)[1] == 19.0
        assert session.output("result_second", 0)[1] == 30.0


def test_complex_response_uses_common_plan_parameter_products_and_model_replay(tmp_path):
    import numpy as np

    source = eqiora.Module("main")
    real, complex_type = eqiora.ValueType.real(), eqiora.ValueType.complex()
    owner = source.property_contract(
        "Response", value_type=complex_type, inputs={"x": real, "y": real},
        derivatives="first_partials", branch="principal",
    )
    x, y = owner.input("x"), owner.input("y")
    measured = release(
        source, owner, x*x/q.math.complex(1, x) + q.math.complex(0, y),
        validity=q.logical_and(q.greater_equal(x, 0), q.less_equal(x, 4)),
        branch="principal",
    )
    consumer = source.component("Consumer")
    response = consumer.property("response", contract=owner)
    x = consumer.parameter("x", value_type=real)
    y = consumer.parameter("y", value_type=real)
    z = consumer.field("z", role=eqiora.FieldRole.Variable, value_type=complex_type)
    consumer.relation("law", q.equation(z, response(x=x, y=y)))
    consumer.observable("value", z, value_type=complex_type)
    root = source.model("Main")
    for name, x_value, y_value in [("first", 2, 3), ("second", 1, 2)]:
        x = root.parameter(f"{name}_x", value_type=real)
        y = root.parameter(f"{name}_y", value_type=real)
        root.set_default(x, x_value)
        root.set_default(y, y_value)
        root.instance(name, component=consumer,
                      bindings={"response": measured, "x": x, "y": y})
    direct = eqiora.compile(source=source, entry="Main")
    path = tmp_path / "complex-response.eqi"
    source.write_eqi(path)
    parsed = eqiora.compile(path=path, entry="Main")
    assert parsed.to_bytes() == direct.to_bytes()
    outputs = [direct.observable(f"{name}.value") for name in ("first", "second")]
    inputs = [direct.parameter(f"first_{name}") for name in ("x", "y")]
    for model in [direct, parsed, eqiora.Model.from_bytes(direct.to_bytes())]:
        plan = eqiora.resolve(model, solve=eqiora.solve.Linear(
            relative_tolerance=1e-13, absolute_tolerance=1e-15, maximum_iterations=8,
            algorithm=eqiora.solve.LinearSolver.BiConjugateGradientStabilized,
            preconditioner=eqiora.solve.Preconditioner.Identity,
            reduction=eqiora.solve.Reduction.Reproducible,
            provider=eqiora.solve.SolverProvider.reference(),
        ))
        result = eqiora.run(plan, state=eqiora.State.initial(plan))
        result = eqiora.Result.from_bytes(plan, result.to_bytes())
        for output, expected in zip(outputs, [0.8+1.4j, 0.5+1.5j], strict=True):
            assert result.observe(output).value == pytest.approx(expected, rel=0, abs=1e-10)
        program = eqiora.diff.compile(plan, inputs=tuple(inputs), outputs=(outputs[0],))
        for x, y, expected, dx in [(2., 3., 0.8+1.4j, 0.16-1.12j),
                                   (1., 2., 0.5+1.5j, 0.5-1j)]:
            point = program.evaluate((x, y))
            np.testing.assert_allclose(point.primal().output[0], [expected], rtol=0, atol=1e-10)
            np.testing.assert_allclose(point.jvp((1., 0.)).tangent[0], [dx], rtol=0, atol=1e-10)
            # Real pairing with i selects Im(g): d/dx=Im(g_x), d/dy=1.
            np.testing.assert_allclose(point.vjp((1j,)).input_cotangent,
                                       [dx.imag, 1.], rtol=0, atol=1e-10)
        with pytest.raises(eqiora.ValidationError):
            program.evaluate((5., 3.))


@pytest.mark.parametrize("composed", (False, True))
def test_tensor_response_preserves_native_types_units_products_and_replay(tmp_path, composed):
    import numpy as np

    source = eqiora.Module("main")
    unit = u.W / u.m / u.K
    dimension = eqiora.Dimension(mass=1, length=1, time=-3, temperature=-1)
    scalar = eqiora.ValueType.complex(dimension)
    tensor = eqiora.ValueType.tensor(scalar, 2, 2)
    inputs = {"T": eqiora.ValueType.real(eqiora.Dimension(temperature=1)),
              "p": eqiora.ValueType.real()}
    def law(T, p):
        x = T/q.quantity(1, u.K)
        return (q.math.complex(x, x*x)*q.delta(0, 1)+p)*q.quantity(1, unit)
    operation = source.operator("conductivity", inputs=inputs, result_type=tensor, body=law) if composed else law
    owner = source.property_contract("Conductivity", value_type=tensor, inputs=inputs,
                                     derivatives="first_partials", branch="principal")
    T, p = owner.input("T"), owner.input("p")
    measured = source.property_release("Measured", implements=owner, value=operation(T=T, p=p),
        source_unit=unit, source_scale=1, citation="org.example.tensor", license="spdx.CC0_1_0",
        validity=q.logical_and(q.logical_and(q.greater_equal(T, q.quantity(0, u.K)),
                                            q.less_equal(T, q.quantity(4, u.K))),
                               q.logical_and(q.greater_equal(p, 0), q.less_equal(p, 5))),
        branch="principal", outside="reject")
    consumer = source.component("Consumer")
    requirement = consumer.property("response", contract=owner)
    T = consumer.parameter("T", value_type=inputs["T"])
    p = consumer.parameter("p", value_type=inputs["p"])
    z = consumer.field("z", role=eqiora.FieldRole.Variable, value_type=scalar)
    consumer.relation("law", q.equation(z, q.matrix_trace(requirement(T=T, p=p))))
    consumer.observable("value", z, value_type=scalar)
    root = source.model("Main")
    for name, temperature, factor in [("first", 2, 3), ("second", 1, 2)]:
        T = root.parameter(f"{name}_T", value_type=inputs["T"])
        p = root.parameter(f"{name}_p", value_type=inputs["p"])
        root.set_default(T, q.quantity(temperature, u.K))
        root.set_default(p, factor)
        root.instance(name, component=consumer, bindings={"T": T, "p": p, "response": measured})
    direct = eqiora.compile(source=source, entry="Main")
    path = tmp_path / "tensor-response.eqi"
    source.write_eqi(path)
    parsed = eqiora.compile(path=path, entry="Main")
    assert direct.to_bytes() == parsed.to_bytes()
    outputs = [direct.observable(f"{name}.value") for name in ("first", "second")]
    selected = [direct.parameter(f"first_{name}") for name in ("T", "p")]
    for model in [direct, parsed, eqiora.Model.from_bytes(direct.to_bytes())]:
        plan = eqiora.resolve(model, solve=eqiora.solve.Linear(
            relative_tolerance=1e-13, absolute_tolerance=1e-15, maximum_iterations=8,
            algorithm=eqiora.solve.LinearSolver.BiConjugateGradientStabilized,
            preconditioner=eqiora.solve.Preconditioner.Identity,
            reduction=eqiora.solve.Reduction.Reproducible, provider=eqiora.solve.SolverProvider.reference()))
        result = eqiora.run(plan, state=eqiora.State.initial(plan))
        result = eqiora.Result.from_bytes(plan, result.to_bytes())
        for output, expected in zip(outputs, [10+8j, 6+2j], strict=True):
            observed = result.observe(output)
            assert observed.value_type == scalar
            assert observed.value == pytest.approx(expected, rel=0, abs=1e-10)
        program = eqiora.diff.compile(plan, inputs=tuple(selected), outputs=(outputs[0],))
        # tr(K)=2T+2p+2iT² in conductivity units, with T in kelvin.
        # Holding p fixed: d/dT=2+4iT. Holding T fixed: d/dp=2.
        for T, p in [(2., 3.), (1., 2.)]:
            point = program.evaluate((T, p))
            np.testing.assert_allclose(point.primal().output[0], [2*T+2*p+2j*T*T], rtol=0, atol=1e-10)
            np.testing.assert_allclose(point.jvp((1., 0.)).tangent[0], [2+4j*T], rtol=0, atol=1e-10)
            np.testing.assert_allclose(point.jvp((0., 1.)).tangent[0], [2.], rtol=0, atol=1e-10)
            np.testing.assert_allclose(point.vjp((1j,)).input_cotangent, [4*T, 0.], rtol=0, atol=1e-10)
        for point in [(5., 3.), (2., 6.)]:
            with pytest.raises(eqiora.ValidationError):
                program.evaluate(point)
