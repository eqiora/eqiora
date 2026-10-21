"""Real-linear actions share typed authoring, ordinary Plans and immutable replay."""

import numpy as np
import pytest
import eqiora as e


@pytest.mark.parametrize("shaped", [False, True])
@pytest.mark.parametrize("local_operator", [False, True])
@pytest.mark.parametrize("mode", ["jvp", "vjp", "nested"])
def test_complex_actions_keep_real_duals_and_whole_directions(shaped, local_operator, mode):
    q = e.lang
    source = e.Module("main")
    scalar = e.ValueType.complex()
    kind = e.ValueType.array(scalar, 2) if shaped else scalar

    def value_body(z):
        z = q.component(z, 0) if shaped else z
        return z * q.math.conj(z) + 1j * z

    value = source.operator("value", inputs={"z": kind}, result_type=kind, body=value_body)

    def action_body(z, dz, seed):
        f = value(z=z)
        if mode == "jvp":
            return q.jvp(f, wrt=(z,), tangent=(dz,))
        reverse = q.vjp(f, wrt=z, cotangent=seed)
        if mode == "vjp":
            return reverse
        return q.jvp(reverse, wrt=(z,), tangent=(dz,))

    action = source.operator(
        "action", inputs={name: kind for name in ("z", "dz", "seed")},
        result_type=kind, body=action_body,
    )
    root = source.model("Main")
    z, dz, seed = (root.parameter(name, value_type=kind) for name in ("z", "dz", "seed"))
    for parameter, values in (
        (z, [2 + 5j, 7 - 4j]),
        (dz, [3 - 4j, -2 + 3j]),
        (seed, [7 + 11j, -3 + 2j]),
    ):
        root.set_default(parameter, q.array(values) if shaped else values[0])
    field = root.field("value", role=e.FieldRole.Variable, value_type=kind)
    result = action(z=z, dz=dz, seed=seed) if local_operator else action_body(z, dz, seed)
    root.relation("law", q.equation(field, result))
    root.observable("output", field, value_type=kind)
    # f(a+ib)=(a²+b²-b)+ia, hence its real Jacobian is [[2a,2b-1],[1,0]].
    # The nested action differentiates Df^T(seed): 2*Re(seed)*dz.
    expected = {
        "jvp": [-24 + 3j, -55 - 2j],
        "vjp": [39 + 63j, -40 + 27j],
        "nested": [42 - 56j, 12 - 18j],
    }[mode]
    expected = np.asarray(expected) if shaped else expected[0]
    assert_replay(source, expected)


@pytest.mark.parametrize("local_operator", [False, True])
@pytest.mark.parametrize("mode", ["jvp", "vjp", "nested"])
def test_indexed_action_keeps_whole_value_axes(local_operator, mode):
    q = e.lang
    source = e.Module("main")
    scalar = e.ValueType.complex()
    array = e.ValueType.array(scalar, 2)

    def action_body(z, dz, seed):
        if mode == "jvp":
            return q.jvp(z, wrt=(z,), tangent=(dz,))[1]
        reverse = q.vjp(z[0] * q.math.conj(z[1]), wrt=z, cotangent=seed)
        if mode == "vjp":
            return reverse[1]
        return q.jvp(reverse, wrt=(z,), tangent=(dz,))[1]

    action = source.operator(
        "action", inputs={"z": array, "dz": array, "seed": scalar},
        result_type=scalar, body=action_body,
    )
    root = source.model("Main")
    z = root.parameter("z", value_type=array)
    dz = root.parameter("dz", value_type=array)
    seed = root.parameter("seed", value_type=scalar)
    root.set_default(z, q.array([2 + 5j, 7 - 4j]))
    root.set_default(dz, q.array([3 - 4j, -2 + 3j]))
    root.set_default(seed, 7 + 11j)
    field = root.field("value", role=e.FieldRole.Variable, value_type=scalar)
    value = action(z=z, dz=dz, seed=seed) if local_operator else action_body(z, dz, seed)
    root.relation("law", q.equation(field, value))
    root.observable("output", field, value_type=scalar)
    # For f=z0*conj(z1), the second real-adjoint block is conj(seed)*z0.
    expected = {"jvp": -2 + 3j, "vjp": 69 + 13j, "nested": -23 - 61j}[mode]
    assert_replay(source, expected)


@pytest.mark.parametrize("local_operator", [False, True])
@pytest.mark.parametrize("sign", [-1, 1])
@pytest.mark.parametrize("mode", ["partial", "second", "jvp", "vjp"])
def test_held_branch_actions_preserve_lazy_domains_through_replay(local_operator, sign, mode):
    q = e.lang
    source = e.Module("main")
    scalar = e.ValueType.real()

    def action_body(x, k):
        value = q.if_else(q.greater(k, 0), q.math.log(x), q.math.log(-x))
        if mode == "partial":
            return q.partial(value, wrt=x)
        if mode == "second":
            return q.partial(q.partial(value, wrt=x), wrt=x)
        if mode == "jvp":
            return q.jvp(value, wrt=(x,), tangent=(3,))
        return q.vjp(value, wrt=x, cotangent=4)

    action = source.operator(
        "action", inputs={"x": scalar, "k": scalar}, result_type=scalar, body=action_body,
    )
    root = source.model("Main")
    x = root.parameter("x", value_type=scalar)
    k = root.parameter("k", value_type=scalar)
    root.set_default(x, sign * 2)
    root.set_default(k, sign)
    field = root.field("value", role=e.FieldRole.Variable, value_type=scalar)
    value = action(x=x, k=k) if local_operator else action_body(x, k)
    root.relation("law", q.equation(field, value))
    root.observable("output", field, value_type=scalar)
    # d log(±x)/dx = 1/x and d² log(±x)/dx² = -1/x².
    expected = {"partial": sign * 0.5, "second": -0.25, "jvp": sign * 1.5, "vjp": sign * 2}
    assert_replay(source, expected[mode])


@pytest.mark.parametrize("local_operator", [False, True])
@pytest.mark.parametrize("other", [1, -1])
@pytest.mark.parametrize("mode", ["value", "partial", "second", "jvp", "vjp"])
def test_indexed_actions_retain_complete_output_domains(local_operator, other, mode):
    q = e.lang
    source = e.Module("main")
    scalar = e.ValueType.real()
    array = e.ValueType.array(scalar, 2)
    f = source.operator(
        "f", inputs={"x": scalar, "z": array}, result_type=array,
        body=lambda x, z: x * q.math.log(q.component(z, 0)),
    )

    def action_body(x, z):
        value = f(x=x, z=z)[0]
        if mode == "value":
            return value
        if mode == "partial":
            return q.partial(value, wrt=x)
        if mode == "second":
            return q.partial(q.partial(value, wrt=x), wrt=x)
        if mode == "jvp":
            return q.jvp(value, wrt=(x,), tangent=(1,))
        return q.vjp(value, wrt=x, cotangent=1)

    action = source.operator(
        "action", inputs={"x": scalar, "z": array},
        result_type=scalar, body=action_body,
    )
    root = source.model("Main")
    x = root.parameter("x", value_type=scalar)
    z = root.parameter("z", value_type=array)
    root.set_default(x, 1)
    root.set_default(z, q.array([1, other]))
    field = root.field("value", role=e.FieldRole.Variable, value_type=scalar)
    value = action(x=x, z=z) if local_operator else action_body(x, z)
    root.relation("law", q.equation(field, value))
    root.observable("output", field, value_type=scalar)
    # The selected log(1) and every selected derivative are zero, but the
    # complete output still requires the other logarithm to be defined.
    if other > 0:
        assert_replay(source, 0)
    else:
        with pytest.raises(e.ValidationError, match="EQ0507|EQ0807"):
            assert_replay(source, 0)


def assert_replay(source, expected):
    model = e.compile(source=source, entry="Main")
    parsed = e.compile(source=source.to_eqi(), entry="Main")
    assert parsed.to_bytes() == model.to_bytes()
    output = model.observable("output")
    solve = e.solve.Linear(
        relative_tolerance=1e-13, absolute_tolerance=1e-15, maximum_iterations=8,
        algorithm=e.solve.LinearSolver.SparseLu,
        preconditioner=e.solve.Preconditioner.Identity,
        reduction=e.solve.Reduction.Fast, provider=e.solve.SolverProvider.faer(),
    )
    for candidate in (model, parsed, e.Model.from_bytes(model.to_bytes())):
        plan = e.Plan.from_bytes(e.resolve(candidate, solve=solve).to_bytes())
        state = e.State.initial(plan)
        before = state.to_bytes()
        result = e.run(plan, state=state)
        replay = e.Result.from_bytes(plan, result.to_bytes())
        np.testing.assert_array_equal(replay.observe(output).value, expected)
        assert state.to_bytes() == before
