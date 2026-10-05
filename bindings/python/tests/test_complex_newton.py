"""Independent cubic roots through the common finite Newton lifecycle."""
import math
import unittest
import eqiora as q


def test_complex_newton_retains_shaped_seeds_and_exact_replay():
    check = unittest.TestCase()
    equations = "".join(
        f"z[{i}]+0.25*math.abs2(z[{i}])*z[{i}]=math.complex(2.25,4.5);"
        for i in range(6)
    )
    norm = "+".join(f"math.abs2(z[{i}])" for i in range(6))
    model = q.compile(source=f"""model M(){{
        variable z:array<complex<1>,6>; variable w:1;
        relation r{{{equations}w+0.25*w*w*w=4;}}
        observable output:1={norm}+w*w;
    }}""")
    linear = q.solve.Linear(relative_tolerance=1e-13, absolute_tolerance=1e-15,
        maximum_iterations=32, algorithm=q.solve.LinearSolver.SparseLu,
        preconditioner=q.solve.Preconditioner.Identity,
        reduction=q.solve.Reduction.Fast, provider=q.solve.SolverProvider.faer())
    newton = q.solve.Newton(linear=linear,
        relative_tolerance=0.0, absolute_tolerance=1e-12,
        maximum_iterations=32, maximum_line_search_steps=16)
    scales = {model.constraint("r", i): (4., q.Dimension()) for i in range(6)}
    scales[model.constraint("r", 6)] = (2., q.Dimension())
    plan = q.resolve(model, solve=newton, scaling=scales)
    assert q.resolve(model, solve=newton, scaling=dict(reversed(tuple(scales.items())))).identity == plan.identity
    with check.assertRaises(q.ValidationError):
        q.resolve(model, solve=newton, scaling={model.constraint("r", 0): (4., q.Dimension(length=1))})
    with check.assertRaises(q.ValidationError):
        q.resolve(model, solve=newton, scaling={model.constraint("r", 0): (0., q.Dimension())})
    plan = q.Plan.from_bytes(plan.to_bytes())
    assert plan.capability.unknown_count == 13
    fields = (q.InitialField(model.field("z"), value=[0j] * 6),
              q.InitialField(model.field("w"), value=0.0))
    state = q.State.initial(plan, fields=fields)
    state = q.State.from_bytes(plan, state.to_bytes())
    result = q.run(plan, state=state)
    # Six roots 1+2i and one root 2 give sum |z|²+w²=34.
    assert abs(result.observe(model.observable("output")).value - 34.) < 1e-10
    expected_initial_norm = math.sqrt(6*((2.25/4)**2+(4.5/4)**2)+(4/2)**2)
    assert abs(result.solve.initial_residual_norm - expected_initial_norm) < 1e-13
    assert result.solve.true_residual_norm <= 1e-12
    assert q.Result.from_bytes(plan, result.to_bytes()).to_bytes() == result.to_bytes()
    exact = q.State.initial(plan, fields=(
        q.InitialField(model.field("z"), value=[1+2j] * 6),
        q.InitialField(model.field("w"), value=2.0)))
    assert q.run(plan, state=exact).solve.completed_iterations == 0
    with check.assertRaisesRegex(q.ValidationError, "shape"):
        q.State.initial(plan, fields=(q.InitialField(model.field("z"), value=[[0j]*3]*2), fields[1]))
    with check.assertRaises(q.ValidationError):
        q.State.initial(plan, fields=(fields[0], q.InitialField(model.field("w"), value=1j)))
    for value in ([[0], [0, 1]], [], [0, [1]]):
        with check.assertRaisesRegex(ValueError, "shape"):
            q.InitialField(model.field("z"), value=value)
    cyclic = []
    cyclic.append(cyclic)
    with check.assertRaisesRegex(ValueError, "cycle"):
        q.InitialField(model.field("z"), value=cyclic)

