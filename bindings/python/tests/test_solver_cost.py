"""Declared cost proxies and payload admission share ordinary finite execution."""
import struct

import pytest
import eqiora as q


def _model():
    return q.compile(source="""model Costed() {
      parameter load:1 = 6;
      variable x:1; variable y:1;
      relation first {4*x+y=load;}
      relation second {2*x+3*y=8;}
      observable loss:1 = x+10*y;
    }""", entry="Costed")


def _manual(*, cost=None, faer=False):
    return q.solve.Linear(
        algorithm=(q.solve.LinearSolver.SparseLu if faer else
                   q.solve.LinearSolver.BiConjugateGradientStabilized),
        preconditioner=q.solve.Preconditioner.Identity,
        reduction=q.solve.Reduction.Fast if faer else q.solve.Reduction.Reproducible,
        provider=q.solve.SolverProvider.faer() if faer else q.solve.SolverProvider.reference(),
        relative_tolerance=1e-12, absolute_tolerance=1e-14,
        maximum_iterations=100, cost=cost)


def _run_loss(model, plan):
    result = q.run(plan, state=q.State.initial(plan))
    result = q.Result.from_bytes(plan, result.to_bytes())
    assert result.observe(model.observable("loss")).value == pytest.approx(21., abs=1e-10)
    return result


def _assumptions(cost):
    return (cost.solves, cost.structures, cost.numeric_preparations,
            cost.identity_iterations, cost.jacobi_iterations,
            cost.numeric_payload_limit, cost.declared_horizon)


def _costs(plan):
    return [(c.candidate, c.setup_work, c.solve_work, c.acceptance_work,
             c.total_work, c.numeric_payload_bytes) for c in plan.solve.planning_costs]


def test_solver_cost_constructor_and_request_inspection():
    assert _assumptions(q.solve.SolverCost()) == (1, 1, 1, None, None, None, False)
    cost = q.solve.SolverCost(solves=2, structures=2, numeric_preparations=2,
                            identity_iterations=3, jacobi_iterations=1,
                            numeric_payload_limit=4096)
    assert _assumptions(_manual(cost=cost).cost) == (2, 2, 2, 3, 1, 4096, True)
    with pytest.raises(ValueError, match="together"):
        q.solve.SolverCost(solves=2)
    with pytest.raises(ValueError, match="structures"):
        q.solve.SolverCost(solves=1, structures=2, numeric_preparations=2)
    for kwargs in ({"identity_iterations": 0}, {"numeric_payload_limit": 0},
                   {"solves": 0, "structures": 1, "numeric_preparations": 1}):
        with pytest.raises(ValueError, match="positive"):
            q.solve.SolverCost(**kwargs)


def test_finite_cost_replay_and_selected_field_observable_derivatives():
    model = _model()
    cost = q.solve.SolverCost(solves=2, structures=2, numeric_preparations=2,
                            identity_iterations=3, jacobi_iterations=3)
    request = q.solve.Linear(objective=q.solve.SolverPlanningObjective.LowMemory,
                             relative_tolerance=1e-12, absolute_tolerance=1e-14,
                             maximum_iterations=100, cost=cost)
    plan = q.resolve(model, solve=request)
    assert (plan.solve.planning_rows, plan.solve.planning_stored_slots,
            plan.solve.planning_slots_exact) == (2, 4, True)
    assert plan.solve.backend == "eqiora.reference"
    identity = next(c for c in plan.solve.planning_costs
                    if c.candidate == "eqiora.reference.bicgstab-general-identity-reproducible-f64")
    # n=2,z=4,k=3,H=S=F=2: structure=12, recurrence=2*3*(16+40),
    # acceptance=2*(8+10). Payload: solver200 + seven host indices + values/RHS48.
    assert (identity.setup_work, identity.solve_work, identity.acceptance_work,
            identity.total_work, identity.numeric_payload_bytes) == (
                12, 336, 36, 384, 248+7*struct.calcsize("P"))
    replay = q.Plan.from_bytes(plan.to_bytes())
    assert _assumptions(replay.solve.cost_assumptions) == _assumptions(cost)
    assert _costs(replay) == _costs(plan)
    _run_loss(model, replay)
    program = q.diff.compile(replay, inputs=(model.parameter("load"),),
                             outputs=(model.field("x"), model.observable("loss")))
    point = program.evaluate((6.,))
    assert point.primal().output == pytest.approx((1., 21.), abs=1e-10)
    # Inverse [[4,1],[2,3]] has first column (3/10,-1/5).
    assert point.jvp((1.,)).tangent == pytest.approx((.3, -1.7), abs=1e-10)
    assert point.vjp((0., 2.)).input_cotangent == pytest.approx((-3.4,), abs=1e-10)


def test_exact_manual_controls_survive_cost_forecasts_and_replay():
    model = _model()
    cost = q.solve.SolverCost(identity_iterations=1, jacobi_iterations=1)
    plan = q.resolve(model, solve=_manual(cost=cost))
    replay = q.Plan.from_bytes(plan.to_bytes())
    assert replay.solve.algorithm == "bicgstab"
    assert replay.solve.preconditioner == "identity"
    assert replay.solve.reduction == "reproducible"
    assert replay.solve.backend == "eqiora.reference"
    assert replay.solve.maximum_iterations == 100
    assert replay.solve.relative_tolerance == 1e-12
    assert replay.solve.absolute_tolerance == 1e-14
    assert _assumptions(replay.solve.cost_assumptions) == _assumptions(cost)
    _run_loss(model, replay)


def test_numeric_payload_limits_reject_insufficient_and_unknown_bounds():
    model = _model()
    # Nonvacuous controls: both exact providers solve the same analytical system.
    _run_loss(model, q.resolve(model, solve=_manual()))
    _run_loss(model, q.resolve(model, solve=_manual(faer=True)))
    # Exact logical bound: reference BiCG workspace200, values/RHS48,
    # and seven host-sized CSR indices. Equality is admitted.
    bound = 248+7*struct.calcsize("P")
    _run_loss(model, q.resolve(model, solve=_manual(
        cost=q.solve.SolverCost(numeric_payload_limit=bound))))
    with pytest.raises(q.ValidationError, match="numeric-payload-limit-exceeded"):
        q.resolve(model, solve=_manual(cost=q.solve.SolverCost(numeric_payload_limit=1)))
    with pytest.raises(q.ValidationError, match="numeric-payload-bound-unproved"):
        q.resolve(model, solve=_manual(faer=True,
                  cost=q.solve.SolverCost(numeric_payload_limit=1_000_000)))


def test_common_plan_rejects_unimplemented_retained_preparation_horizon():
    # The lower-level planner may compare this hypothetical retained horizon;
    # ordinary finite Run has no admitted retained preparation owner.
    cost = q.solve.SolverCost(solves=100, structures=1, numeric_preparations=1,
                            identity_iterations=3, jacobi_iterations=3)
    with pytest.raises(q.ValidationError, match="retained preparation horizon"):
        q.resolve(_model(), solve=_manual(cost=cost))
