"""Finite midpoint stages consume the ordinary explicit solver hierarchy."""

import math

import eqiora as e
import pytest


@pytest.mark.parametrize("nonlinear", [False, True])
def test_midpoint_retains_solve_policy_and_replays_an_actual_update(nonlinear):
    rhs = "-u*u*1[1/s]" if nonlinear else "-u*1[1/s]"
    model = e.compile(
        source=f"""model Decay() {{
            state u:1;
            initial {{ u=1; }}
            relation evolution {{ derivative(u)={rhs}; }}
        }}"""
    )
    u = model.field("u")
    linear = e.solve.Linear(
        algorithm=e.solve.LinearSolver.BiConjugateGradientStabilized,
        preconditioner=e.solve.Preconditioner.Identity,
        reduction=e.solve.Reduction.Reproducible,
        provider=e.solve.SolverProvider.reference(),
        relative_tolerance=0.0,
        absolute_tolerance=1e-14,
        maximum_iterations=100,
    )
    solve = (
        e.solve.Newton(
            linear=linear,
            relative_tolerance=0.0,
            absolute_tolerance=1e-12,
            maximum_iterations=16,
            maximum_line_search_steps=16,
        )
        if nonlinear
        else linear
    )
    plan = e.resolve(
        model,
        temporal=e.time.ImplicitMidpoint(
            step_s=1.0,
            relative_tolerance=1e-12,
            absolute_tolerances={(u, 0, 0, False): 1e-12},
        ),
        solve=solve,
    )
    plan = e.Plan.from_bytes(plan.to_bytes())
    assert isinstance(plan.solve, e.solve.ResolvedNewton if nonlinear else e.solve.ResolvedLinear)
    state = e.State.initial(plan)
    before = state.to_bytes()
    result = e.run(plan, state=state, until_s=1.0, output_times_s=(0.5, 1.0))
    # y1-1 = -(1+y1)/2 gives y1=1/3. For the quadratic rate,
    # y1-1 = -((1+y1)/2)^2 gives y1^2+6*y1-3=0; select
    # the branch continuously connected to the initial value as h -> 0.
    expected = 2 * math.sqrt(3) - 3 if nonlinear else 1 / 3
    series = list(result.series(u))
    assert series[-1][1] == pytest.approx(expected, rel=0, abs=1e-11)
    assert series[0][1] == pytest.approx((1 + expected) / 2, rel=0, abs=1e-11)
    assert state.to_bytes() == before
    assert e.Result.from_bytes(plan, result.to_bytes()).to_bytes() == result.to_bytes()
    restored = e.State.from_result(plan, result, time_s=1.0)
    assert e.State.from_bytes(plan, restored.to_bytes()).value(u) == restored.value(u)


def test_midpoint_rejects_missing_policy_and_false_linearity_without_changing_state():
    # This polynomial vanishes at 0 and 1. Sampling those states cannot prove
    # affinity; the structural check must reject it even at a stationary seed.
    model = e.compile(
        source="""model Nonlinear() {
        state u:1; initial {u=1;}
        relation evolution {derivative(u)=(u*u*u-u)*1[1/s];}
    }"""
    )
    u = model.field("u")
    temporal = e.time.ImplicitMidpoint(
        step_s=1.0,
        relative_tolerance=1e-12,
        absolute_tolerances={(u, 0, 0, False): 1e-12},
    )
    with pytest.raises(e.ValidationError, match="explicit Linear or Newton"):
        e.resolve(model, temporal=temporal)
    linear = e.solve.Linear(
        objective=e.solve.Robust,
        relative_tolerance=0.0,
        absolute_tolerance=1e-14,
        maximum_iterations=100,
    )
    plan = e.Plan.from_bytes(
        e.resolve(model, temporal=temporal, solve=linear).to_bytes()
    )
    state = e.State.initial(plan)
    before = state.to_bytes()
    with pytest.raises(e.ExecutionError, match="affine"):
        e.run(plan, state=state, until_s=1.0, output_times_s=(1.0,))
    assert state.to_bytes() == before


def test_midpoint_honors_newton_iteration_budget_and_rejects_erased_solve_intent():
    import json

    model = e.compile(
        source="""model Nonlinear() {
        state u:1; initial {u=1;}
        relation evolution {derivative(u)=-u*u*1[1/s];}
    }"""
    )
    u = model.field("u")
    linear = e.solve.Linear(
        objective=e.solve.Robust,
        relative_tolerance=0.0,
        absolute_tolerance=1e-14,
        maximum_iterations=100,
    )
    solve = e.solve.Newton(
        linear=linear,
        relative_tolerance=0.0,
        absolute_tolerance=1e-12,
        maximum_iterations=1,
        maximum_line_search_steps=0,
    )
    plan = e.resolve(
        model,
        solve=solve,
        temporal=e.time.ImplicitMidpoint(
            step_s=1.0,
            relative_tolerance=1e-12,
            absolute_tolerances={(u, 0, 0, False): 1e-12},
        ),
    )
    wire = json.loads(plan.to_bytes())
    wire["solve"] = None
    with pytest.raises(e.ValidationError, match="explicit Linear or Newton"):
        e.Plan.from_bytes(json.dumps(wire).encode())
    plan = e.Plan.from_bytes(plan.to_bytes())
    assert plan.solve.maximum_iterations == 1
    state = e.State.initial(plan)
    before = state.to_bytes()
    with pytest.raises(e.ExecutionError, match="1 iterations"):
        e.run(plan, state=state, until_s=1.0, output_times_s=(1.0,))
    assert state.to_bytes() == before
