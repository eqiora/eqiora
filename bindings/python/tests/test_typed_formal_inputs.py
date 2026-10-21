"""Formal scopes retain typed held inputs and compose with ordinary total products."""

import numpy as np
import pytest
import eqiora as e


@pytest.mark.parametrize(
    "shaped,complex_domain",
    [(False, False), (False, True), (True, False), (True, True)],
)
@pytest.mark.parametrize("order", [1, 2])
def test_typed_held_inputs_keep_values_replay_and_total_products(
    shaped, complex_domain, order
):
    q = e.lang
    source = e.Module("main")
    real = e.ValueType.real()
    scalar = e.ValueType.complex() if complex_domain else real
    kind = e.ValueType.array(scalar, 2) if shaped else scalar

    def body(x, z):
        component = q.component(z, 0) if shaped else z
        value = x * x * q.math.conj(component)
        for _ in range(order):
            value = q.partial(value, wrt=x, holding=(z,))
        return value

    derivative = source.operator(
        "partial_value", inputs={"x": real, "z": kind}, result_type=kind, body=body
    )
    root = source.model("Main")
    x = root.parameter("x", value_type=real)
    z = root.parameter("z", value_type=kind)
    points = [2 + 5j, 7 - 4j] if complex_domain else [2.0, 7.0]
    root.set_default(x, 3.0)
    root.set_default(z, q.array(points) if shaped else points[0])
    field = root.field("value", role=e.FieldRole.Variable, value_type=kind)
    root.relation("law", q.equation(field, derivative(x=x, z=z)))
    root.observable("output", field, value_type=kind)
    model = e.compile(source=source, entry="Main")
    parsed = e.compile(source=source.to_eqi(), entry="Main")
    assert parsed.to_bytes() == model.to_bytes()
    output = model.observable("output")
    inputs = (model.parameter("x"), model.parameter("z"))
    point_z = np.asarray(points) if shaped else points[0]
    # f=x²*conj(z): f_x=2x*conj(z), f_xx=2*conj(z).
    scale = 6 if order == 1 else 2
    expected = scale * np.conjugate(point_z)
    dx = 2 * np.conjugate(point_z) if order == 1 else np.zeros_like(point_z)
    seed_value = 1 + 2j if complex_domain else 1.0
    seed = np.ones_like(point_z) * seed_value if shaped else seed_value
    zero_z = np.zeros_like(point_z) if shaped else (0j if complex_domain else 0.0)
    expected_x_dual = np.real(np.vdot(seed, dx))
    expected_z_dual = scale * np.conjugate(seed)
    solve = e.solve.Linear(
        relative_tolerance=1e-13,
        absolute_tolerance=1e-15,
        maximum_iterations=8,
        algorithm=e.solve.LinearSolver.SparseLu,
        preconditioner=e.solve.Preconditioner.Identity,
        reduction=e.solve.Reduction.Fast,
        provider=e.solve.SolverProvider.faer(),
    )
    for candidate in (model, parsed, e.Model.from_bytes(model.to_bytes())):
        plan = e.resolve(candidate, solve=solve)
        plan = e.Plan.from_bytes(plan.to_bytes())
        state = e.State.initial(plan)
        before = state.to_bytes()
        result = e.run(plan, state=state)
        replay = e.Result.from_bytes(plan, result.to_bytes())
        np.testing.assert_allclose(
            replay.observe(output).value, expected, rtol=0, atol=1e-10
        )
        assert state.to_bytes() == before
        program = e.diff.compile(plan, inputs=inputs, outputs=(output,))
        point = program.evaluate((3.0, point_z))
        np.testing.assert_allclose(
            point.primal().output[0], expected, rtol=0, atol=1e-10
        )
        np.testing.assert_allclose(
            point.jvp((1.0, zero_z)).tangent[0], dx, rtol=0, atol=1e-10
        )
        pullback = point.vjp((seed,)).input_cotangent
        np.testing.assert_allclose(pullback[0], expected_x_dual, rtol=0, atol=1e-10)
        np.testing.assert_allclose(pullback[1], expected_z_dual, rtol=0, atol=1e-10)
