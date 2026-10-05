from __future__ import annotations

import math
import asyncio
import gc

import numpy as np
import pytest

import eqiora


DECAY = """
model decay() {
  state x: 1;
  initial { x = 1; }
  parameter rate: 1 / s = 1;
  relation flow {
    derivative(x) + rate * x = 0;
  }
}
"""


def resolve_decay(model: eqiora.Model) -> tuple[eqiora.Plan, eqiora.FieldRef]:
    field = model.field(model.field_ids[0])
    plan = eqiora.resolve(
        model,
        temporal=eqiora.time.Tsitouras45(
            initial_step_s=0.01,
            relative_tolerance=1.0e-9,
            absolute_tolerances={(field, 0): 1.0e-11},
        ),
    )
    return plan, field


def test_higher_order_initial_time_and_formulation_survive_installed_replay() -> None:
    model = eqiora.compile(source="""
model Timed() {
  state q:m;
  initial {
    q=time()*time()*time()*1[m/s^3]/6;
    derivative(q)=time()*time()*1[m/s^3]/2;
  }
  relation flow { derivative(derivative(q))=time()*1[m/s^3]; }
}
""")
    q = model.field("q")
    plan = eqiora.resolve(model, temporal=eqiora.time.Tsitouras45(
        initial_step_s=0.001, relative_tolerance=1e-10,
        absolute_tolerances={(q, order): 1e-12 for order in (0, 1)},
    ))
    plan = eqiora.Plan.from_bytes(plan.to_bytes())
    assert plan.formulation.effective == eqiora.FormulationKind.FirstOrderEvolution
    assert plan.formulation.state_coordinates == ((q, 0), (q, 1))
    state = eqiora.State.initial(plan, time_s=2.)
    state = eqiora.State.from_bytes(plan, state.to_bytes())
    assert state.time_s == 2.
    assert math.isclose(state.value(q), 8/6, rel_tol=0, abs_tol=1e-9)
    assert math.isclose(state.value(q, derivative_order=1), 2., rel_tol=0, abs_tol=1e-9)
    result = eqiora.run(plan, state=state, until_s=3., output_times_s=(3.,))
    result = eqiora.Result.from_bytes(plan, result.to_bytes())
    for order, expected in ((0, 27/6), (1, 9/2)):
        assert math.isclose(list(result.series(q, derivative_order=order))[-1][1], expected,
                            rel_tol=0, abs_tol=1e-9)
    for time_s in (-1., -0., math.inf, math.nan):
        with pytest.raises(eqiora.ValidationError):
            eqiora.State.initial(plan, time_s=time_s)


def test_model_first_no_mesh_decay_owns_exact_lineage_and_adaptive_series() -> None:
    model = eqiora.compile(source=DECAY, filename="decay.eqi")
    plan, field = resolve_decay(model)

    assert plan.model is model
    assert plan.model_digest == model.digest
    assert plan.fields == (field,)
    assert plan.mesh is None
    assert plan.mesh_digest is None
    assert plan.spatial is None
    assert isinstance(plan.capability, eqiora.time.OdePlanView)
    assert not hasattr(plan.capability, "scaling")
    assert plan.solve is None
    assert plan.execution.placement == "host-serial"
    assert plan.capability.backend == "eqiora.time.diffsol"
    assert plan.capability.backend_version == "0.16.2"

    initial = eqiora.State.initial(plan)
    assert initial.model is model
    assert initial.mesh is None
    assert initial.time_s == 0.0
    assert initial.state_coordinates == ((field, 0),)
    assert initial.value(field) == 1.0
    assert initial.source_kind == "initial"
    initial_bytes = initial.to_bytes()
    replayed_initial = eqiora.State.from_bytes(plan, initial_bytes)
    assert replayed_initial == initial
    assert replayed_initial.to_bytes() == initial_bytes
    assert replayed_initial.source_kind == "artifact"
    with pytest.raises(eqiora.ValidationError):
        eqiora.State.from_bytes(plan, initial_bytes + b"\n")

    result = eqiora.run(
        plan,
        state=initial,
        until_s=0.2,
        output_times_s=(0.1, 0.2),
    )
    series = result.series(field)
    assert series.field == field
    np.testing.assert_array_equal(series.time.numpy(), np.array([0.1, 0.2]))
    np.testing.assert_allclose(
        series.values.numpy(),
        np.array([math.exp(-0.1), math.exp(-0.2)]),
        rtol=2.0e-8,
        atol=2.0e-10,
    )
    with pytest.raises(TypeError):
        result.series("x")  # type: ignore[arg-type]


def test_replay_and_fresh_compile_share_canonical_identity_and_track_source_changes() -> None:
    first = eqiora.compile(source=DECAY)
    replayed = eqiora.Model.from_bytes(first.to_bytes())
    second = eqiora.compile(source=DECAY)
    first_plan, _ = resolve_decay(first)
    replayed_plan, _ = resolve_decay(replayed)
    second_plan, _ = resolve_decay(second)

    assert replayed_plan.identity == first_plan.identity
    assert replayed_plan.model_digest == replayed.digest == first.digest
    assert second_plan.model_digest == second.digest
    assert second_plan.model_digest == first_plan.model_digest
    assert second_plan.identity == first_plan.identity

    changed_source = DECAY.replace("parameter rate: 1 / s = 1;", "parameter rate: 1 / s = 2;")
    assert changed_source != DECAY
    changed = eqiora.compile(source=changed_source)
    changed_plan, _ = resolve_decay(changed)
    assert changed_plan.model_digest == changed.digest
    assert changed_plan.model_digest != first_plan.model_digest
    assert changed_plan.identity != first_plan.identity


def test_output_cadence_changes_results_without_changing_model_identity() -> None:
    model = eqiora.compile(source=DECAY)
    plan, field = resolve_decay(model)
    fingerprint = model.structural_fingerprint
    model_bytes = model.to_bytes()
    results = []
    for output_times in ((0.2,), (0.1, 0.2)):
        result = eqiora.run(
            plan,
            state=eqiora.State.initial(plan),
            until_s=0.2,
            output_times_s=output_times,
        )
        assert tuple(result.series(field).time.numpy()) == output_times
        results.append(result.to_bytes())
        assert model.structural_fingerprint == fingerprint
        assert model.to_bytes() == model_bytes
    assert results[0] != results[1]


def test_restart_is_a_new_adaptive_run_and_step_controls_are_rejected() -> None:
    model = eqiora.compile(source=DECAY)
    plan, field = resolve_decay(model)
    first = eqiora.run(
        plan,
        state=eqiora.State.initial(plan),
        until_s=0.1,
        output_times_s=(0.1,),
    )
    restart = eqiora.State.from_result(plan, first, time_s=0.1)
    second = eqiora.run(
        plan,
        state=restart,
        until_s=0.2,
        output_times_s=(0.2,),
    )
    assert second.series(field).values.numpy()[0] == pytest.approx(
        math.exp(-0.2), rel=2.0e-8, abs=2.0e-10
    )

    with pytest.raises(TypeError, match="steps/output_steps"):
        eqiora.submit(
            plan,
            state=restart,
            steps=1,
            output_steps=(1,),
        )


def test_async_run_and_array_ownership_use_the_same_common_result() -> None:
    async def exercise() -> tuple[eqiora.Result, np.ndarray]:
        model = eqiora.compile(source=DECAY)
        plan, field = resolve_decay(model)
        submitted = eqiora.submit(
            plan,
            state=eqiora.State.initial(plan),
            until_s=0.2,
            output_times_s=(0.1, 0.2),
        )
        assert submitted.cancel() is False
        result = await submitted
        assert result is submitted.result()
        values = result.series(field).values.numpy(copy=False)
        assert not values.flags.writeable
        return result, values

    result, retained = asyncio.run(exercise())
    del result
    gc.collect()
    np.testing.assert_allclose(
        retained,
        np.array([math.exp(-0.1), math.exp(-0.2)]),
        rtol=2.0e-8,
        atol=2.0e-10,
    )
