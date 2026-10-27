"""Explicit Run planning keeps source proofs, scheduling and physical acceptance together."""
import json
import math
import os
import platform
import sys
import struct

import numpy as np
import pytest

import eqiora as q


def _heat(*, cells=2, manual=False, cost=None, capacity="1", diffusion="1", boundary="300[K]", initial="300[K]", extra=""):
    graph = q.geometry.GeometryGraph()
    rectangle = graph.rectangle(x_bounds=(0., 1.), y_bounds=(0., 1.))
    sides = ("left", "right", "bottom", "top")
    geometry = graph.build(rectangle, named_topology={
        "body": rectangle.region, **dict(zip(sides, rectangle.boundaries, strict=True)),
    })
    body = geometry.selection("body")
    source = """public component Heat(
        support body:volume(ambient_dimension=2),
        support left:boundary(parent=body),support right:boundary(parent=body),
        support bottom:boundary(parent=body),support top:boundary(parent=body)) {
        state u:K on body in h1; initial {u=INITIAL;}
        EXTRA
        relation balance on body {CAPACITY*1[s/m^2]*derivative(u)
            -div(DIFFUSION*grad(u))=1[K/m^2];}
        relation left_value on left {trace(u)=BOUNDARY;}
        relation right_value on right {trace(u)=BOUNDARY;}
        relation bottom_value on bottom {trace(u)=BOUNDARY;}
        relation top_value on top {trace(u)=BOUNDARY;}
        observable clock:s=time();
    }""".replace("EXTRA", extra).replace("CAPACITY", capacity).replace("DIFFUSION", diffusion).replace("BOUNDARY", boundary).replace("INITIAL", initial)
    model = q.compile(source=source, entry="Heat", geometry=geometry,
        bindings={"body": body, **{side: (geometry.selection(side), body) for side in sides}})
    mesh = q.meshing.generate(q.meshing.resolve(geometry,
        q.meshing.CartesianMesher(cells=(cells, cells))))
    common = dict(relative_tolerance=1e-12, absolute_tolerance=1e-14,
                  maximum_iterations=100, cost=cost)
    solve = (q.solve.Linear(algorithm=q.solve.LinearSolver.SparseLu,
        preconditioner=q.solve.Preconditioner.Identity, reduction=q.solve.Reduction.Fast,
        provider=q.solve.SolverProvider.faer(), **common) if manual else
        q.solve.Linear(objective=q.solve.SolverPlanningObjective.Fast, **common))
    return model, q.resolve(model, mesh=mesh, spatial=q.fem.Q1(), solve=solve,
        temporal=q.time.BackwardEuler(1/24))


def _work(plan):
    return [(c.candidate, c.setup_work, c.solve_work, c.acceptance_work, c.run_work,
             c.total_work, c.numeric_payload_bytes, c.retained_numeric_bytes)
            for c in plan.solve.planning_costs]


def _factor_calls(profile, phase):
    return sum(item.calls for item in profile.phases
               if item.path[-1] == phase and item.fields.get("backend") == "faer")


def _measure(plan, steps, outputs):
    # Reuse the installed ordinary Run/Profile owner. Resolve, State construction
    # and subsequent result inspection stay outside its measured execution span.
    state = q.State.initial(plan)
    before = state.to_bytes()
    result = q.run(plan, state=state, steps=steps, output_steps=outputs, profile=True)
    assert state.to_bytes() == before
    profile = result.profile
    assert profile is not None and math.isfinite(profile.total_seconds) and profile.total_seconds > 0
    finalizations = [phase for phase in profile.phases
                     if phase.fields.get("role") == "scalar_region_finalization"]
    assert sum(phase.calls for phase in finalizations) == 2*steps+len(outputs)
    assert all(0 <= phase.self_seconds <= phase.inclusive_seconds for phase in finalizations)
    return result, profile


def _check_four_cell_oracle(model, result, steps):
    assert len(model.field_ids) == 1
    # Existing heated-body Q1 authority: M=1/9,K=8/3,F=1/4,
    # dt=1/24. The center's steady excess3/32 has a half-step deficit.
    for state, step in zip(result.trajectory.states, steps, strict=True):
        values = state.field(model.field(model.field_ids[0])).values("vertex").reshape(-1)
        assert values.shape == (9,)
        np.testing.assert_allclose(np.delete(values, 4), 300., atol=1e-10, rtol=0)
        assert values[4] == pytest.approx(300+3/32*(1-0.5**step), abs=1e-10, rel=0)


def test_common_run_cost_one_shot_and_repeated_contexts_are_explicit_and_deterministic():
    cost = q.solve.SolverCost(solves=2, structures=2, numeric_preparations=2,
                            identity_iterations=1, jacobi_iterations=1)
    model, original = _heat(cells=4, cost=cost, boundary="0[K]", initial="0[K]")
    before = original.to_bytes()
    single = original.for_run(steps=1, output_steps=(1,))
    repeated = original.for_run(steps=5, output_steps=(1, 5))
    assert len({original.identity, single.identity, repeated.identity}) == 3
    assert original.to_bytes() == before
    assert single.solve.algorithm != "sparse-lu"
    assert repeated.solve.algorithm == "sparse-lu" and repeated.solve.backend == "eqiora.faer"
    assert repeated.solve.cost_assumptions.solves == 5
    assert repeated.solve.cost_assumptions.identity_iterations == 1
    assert repeated.solve.maximum_iterations == original.solve.maximum_iterations == 100
    assert ("run", "run.schedule-replaces-caller-horizon") in repeated.solve.planning_reasons
    for plan in (single, repeated):
        replay = q.Plan.from_bytes(plan.to_bytes())
        assert replay.to_bytes() == plan.to_bytes() and _work(replay) == _work(plan)
    canonical = repeated.to_bytes()
    assert canonical.count(b'"identity_iterations":1') == 1
    with pytest.raises(q.ValidationError):
        q.Plan.from_bytes(canonical.replace(b'"identity_iterations":1', b'"identity_iterations":2'))
    one, _ = _measure(single, 1, (1,))
    many, many_profile = _measure(repeated, 5, (1, 5))
    assert _factor_calls(many_profile, "symbolic_factorization") == 1
    assert _factor_calls(many_profile, "numeric_factorization") == 1
    # The same first accepted step is retained across the two explicit workloads.
    np.testing.assert_allclose(one.trajectory.states[0].field(model.field(model.field_ids[0])).values("vertex"),
        many.trajectory.states[0].field(model.field(model.field_ids[0])).values("vertex"), atol=1e-10, rtol=0)
    for candidate in repeated.solve.planning_costs:
        assert candidate.total_work == (candidate.setup_work + candidate.solve_work
            + candidate.acceptance_work + candidate.run_work)
    selected = next(c for c in repeated.solve.planning_costs
                    if c.candidate == repeated.solve.selected_candidate_id)
    # n=9,z<=81: one symbolic envelope90 + one numeric envelope729.
    assert selected.setup_work == 819
    assert selected.numeric_payload_bytes is None
    assert selected.retained_numeric_bytes == (2+1)*4*25*8


def test_common_run_cost_manual_controls_complete_run_counts_and_existing_oracle(tmp_path):
    model, original = _heat(manual=True)
    planned = original.for_run(steps=3, output_steps=(1, 3))
    for name in ("algorithm", "backend", "provider", "preconditioner", "reduction",
                 "relative_tolerance", "absolute_tolerance", "maximum_iterations"):
        assert getattr(planned.solve, name) == getattr(original.solve, name)
    ordinary, ordinary_profile = _measure(original, 3, (1, 3))
    forecast, forecast_profile = _measure(planned, 3, (1, 3))
    for result in (ordinary, forecast):
        _check_four_cell_oracle(model, result, (1, 3))
    for profile in (ordinary_profile, forecast_profile):
        assert _factor_calls(profile, "symbolic_factorization") == 1
        assert _factor_calls(profile, "numeric_factorization") == 1
    replay = q.Plan.from_bytes(planned.to_bytes())
    restored = q.Result.from_bytes(replay, forecast.to_bytes())
    _check_four_cell_oracle(model, restored, (1, 3))
    # Preserve complete-Run measurements for this invocation without a speedup
    # assertion, microbenchmark ranker, or portable timing/memory claim.
    (tmp_path/"complete-runs.json").write_text(json.dumps({
        "measurement": "ordinary installed Run/Profile; initialization excluded",
        "platform": platform.platform(), "python": sys.version,
        "affinity": sorted(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else None,
        "plan_identity": planned.identity, "steps": 3, "output_steps": [1, 3],
        "ordinary_seconds": ordinary_profile.total_seconds,
        "planned_seconds": forecast_profile.total_seconds,
        "expected_finalizations": 8,
    }, indent=2))


def test_common_run_cost_schedule_and_persisted_proof_tampering_reject():
    _, original = _heat(manual=True)
    planned = original.for_run(steps=3, output_steps=(1, 3))
    state = q.State.initial(planned)
    before = state.to_bytes()
    for steps, outputs in [(4, (1, 3)), (3, (3,))]:
        with pytest.raises(q.ValidationError, match="workload differs"):
            q.run(planned, state=state, steps=steps, output_steps=outputs)
    assert state.to_bytes() == before
    wire = json.loads(planned.to_bytes())
    for key, value in (("steps", 4), ("proof", "claimed"), ("assembly_passes", 7),
                       ("common_work", 0), ("retained_numeric_bytes", 1)):
        changed = {**wire, "run_planning": {**wire["run_planning"], key: value}}
        with pytest.raises(q.ValidationError):
            q.Plan.from_bytes(json.dumps(changed, separators=(",", ":")).encode())
    for steps, outputs in [(0, (1,)), (3, ()), (3, (3, 1)), (3, (1, 4))]:
        with pytest.raises(q.ValidationError):
            original.for_run(steps=steps, output_steps=outputs)


def test_common_run_cost_payload_cap_selects_reference_without_retained_factor_credit():
    # n=1,z<=1: 12 recurrence scalars+one reduction partial, CSR/RHS2scalars
    # and3host CSR indices. Retained physical State payload is a separate term.
    bound = 120+3*struct.calcsize("P")
    cost = q.solve.SolverCost(identity_iterations=1, jacobi_iterations=1,
                            numeric_payload_limit=bound)
    model, original = _heat(cost=cost)
    planned = original.for_run(steps=3, output_steps=(1, 3))
    assert planned.solve.backend == "eqiora.reference"
    assert planned.solve.cost_assumptions.numeric_payload_limit == bound
    selected = next(item for item in planned.solve.planning_costs
                    if item.candidate == planned.solve.selected_candidate_id)
    assert selected.setup_work == 3*(1+1)
    assert selected.numeric_payload_bytes == bound
    assert selected.retained_numeric_bytes > bound
    assert any(reason == "resource.numeric-payload-bound-unproved"
               for _, reason in planned.solve.planning_reasons)
    result, profile = _measure(planned, 3, (1, 3))
    _check_four_cell_oracle(model, result, (1, 3))
    assert _factor_calls(profile, "symbolic_factorization") == 0
    assert _factor_calls(profile, "numeric_factorization") == 0
    with pytest.raises(q.ValidationError, match="numeric-payload-limit-exceeded"):
        _heat(cost=q.solve.SolverCost(numeric_payload_limit=bound-1))


@pytest.mark.parametrize("data", [
    {"capacity": "(1+time()/1[s])"},
    {"boundary": "300[K]+time()*1[K/s]"},
    {"extra": "variable log_k:1 on body in smooth; relation coefficient on body {log_k=time()/1[s];}",
     "diffusion": "math.exp(log_k)"},
])
def test_common_run_cost_source_time_dependencies_cannot_receive_reuse_credit(data):
    _, original = _heat(manual=True, **data)
    with pytest.raises(q.ValidationError, match="autonomous"):
        original.for_run(steps=3, output_steps=(1, 3))


@pytest.mark.parametrize("coefficient", ["coordinate(0)/1[m]", "(if coordinate(0)<0.5[m] then 0 else coordinate(0)/1[m])"])
def test_common_run_cost_fixed_coordinate_coefficients_and_changed_model_rederive(coefficient):
    data = dict(extra="variable log_k:1 on body in smooth; relation coefficient on body {log_k=COEFFICIENT;}".replace("COEFFICIENT", coefficient),
                diffusion="math.exp(log_k)")
    _, original = _heat(manual=True, **data)
    planned = original.for_run(steps=2, output_steps=(2,))
    _, changed = _heat(manual=True, capacity="2", **data)
    changed = changed.for_run(steps=2, output_steps=(2,))
    assert changed.identity != planned.identity
    with pytest.raises(q.ValidationError):
        q.run(changed, state=q.State.initial(planned), steps=2, output_steps=(2,))
    for candidate in (planned, changed):
        _, profile = _measure(candidate, 2, (2,))
        assert _factor_calls(profile, "symbolic_factorization") == 1
        assert _factor_calls(profile, "numeric_factorization") == 1
