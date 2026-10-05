from __future__ import annotations

import asyncio
import gc
import math

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


def admitted() -> tuple[eqiora.Model, eqiora.FieldRef, eqiora.Plan, eqiora.State]:
    model = eqiora.compile(source=DECAY)
    field = model.field(model.field_ids[0])
    plan = eqiora.resolve(
        model,
        temporal=eqiora.time.Tsitouras45(
            initial_step_s=0.01,
            relative_tolerance=1.0e-9,
            absolute_tolerances={(field, 0): 1.0e-11},
        ),
    )
    return model, field, plan, eqiora.State.initial(plan)


def submit_decay(plan: eqiora.Plan, state: eqiora.State) -> eqiora.Run:
    return eqiora.submit(
        plan,
        state=state,
        until_s=0.2,
        output_times_s=(0.1, 0.2),
    )


def test_sync_and_submitted_runs_share_one_result_contract() -> None:
    model, field, plan, state = admitted()
    synchronous = eqiora.run(
        plan,
        state=state,
        until_s=0.2,
        output_times_s=(0.1, 0.2),
    )
    submitted = submit_decay(plan, state)
    first = submitted.result()
    second = submitted.result()

    assert first is second
    assert submitted.status == eqiora.RunStatus.Completed
    assert submitted.done
    assert submitted.history == (
        eqiora.RunStatus.Created,
        eqiora.RunStatus.Validating,
        eqiora.RunStatus.Queued,
        eqiora.RunStatus.Running,
        eqiora.RunStatus.Completed,
    )
    assert submitted.cancellation is None
    assert submitted.cancel() is False
    assert submitted.model_id == model.model_id == first.model_id
    assert submitted.model_digest == model.digest == first.model_digest
    assert submitted.model_revision == model.revision.number == first.model_revision
    assert plan.package_compilation_digest is None
    assert submitted.package_compilation_digest is None
    assert submitted.plan_key == first.plan_key
    assert submitted.adapter == first.adapter == "eqiora.time.diffsol"
    assert submitted.adapter_version == first.adapter_version == "0.16.2"
    assert first.elapsed_seconds >= 0.0
    np.testing.assert_array_equal(
        first.series(field).time.numpy(), synchronous.series(field).time.numpy()
    )
    np.testing.assert_allclose(
        first.series(field).values.numpy(),
        np.exp(-np.array([0.1, 0.2])),
        rtol=2.0e-8,
        atol=2.0e-10,
    )

    retained = first.series(field).values.numpy()
    del first, second, submitted, synchronous, model, plan, state
    gc.collect()
    np.testing.assert_allclose(
        retained,
        np.exp(-np.array([0.1, 0.2])),
        rtol=2.0e-8,
        atol=2.0e-10,
    )


def test_common_ode_run_forms_are_exact_and_fail_closed() -> None:
    _, _, plan, state = admitted()
    with pytest.raises(TypeError, match="steps/output_steps"):
        eqiora.submit(plan, state=state, steps=2, output_steps=(1, 2))
    with pytest.raises(eqiora.ValidationError):
        eqiora.submit(
            plan,
            state=state,
            until_s=0.2,
            output_times_s=(0.2, 0.1),
        )
    with pytest.raises(TypeError, match="requires state"):
        eqiora.submit(plan, until_s=0.2, output_times_s=(0.2,))


def test_await_and_task_cancellation_do_not_redefine_native_execution() -> None:
    async def exercise() -> None:
        _, field, plan, state = admitted()
        completed = submit_decay(plan, state)
        result = await completed
        assert result is completed.result()
        assert result.series(field).values.numpy()[-1] == pytest.approx(
            math.exp(-0.2), rel=2.0e-8, abs=2.0e-10
        )

        _, _, live_plan, live_state = admitted()
        live = submit_decay(live_plan, live_state)
        task = asyncio.ensure_future(live)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        assert live.result() is live.result()
        assert live.status == eqiora.RunStatus.Completed

    asyncio.run(exercise())


def test_cancellation_claim_stops_at_the_available_adapter_boundary() -> None:
    _, _, plan, state = admitted()
    submitted = submit_decay(plan, state)
    # Diffsol exposes no accepted-step callback through this seam. The Run must
    # not claim a cancellation request that its worker cannot observe.
    assert submitted.cancel() is False
    result = submitted.result()
    assert submitted.status == eqiora.RunStatus.Completed
    assert result is submitted.result()


def test_profile_is_opt_in_and_does_not_change_ode_values_or_plan_identity() -> None:
    _, field, plan, state = admitted()
    plan_identity = plan.identity
    ordinary = eqiora.run(
        plan, state=state, until_s=0.2, output_times_s=(0.1, 0.2)
    )
    profiled = eqiora.run(
        plan,
        state=state,
        until_s=0.2,
        output_times_s=(0.1, 0.2),
        profile=True,
    )
    assert ordinary.profile is None
    assert profiled.profile is not None
    assert {tuple(phase.path) for phase in profiled.profile.phases} == {
        ("run",),
        ("run", "setup"),
        ("run", "solve"),
        ("run", "postprocess"),
    }
    assert profiled.profile.events
    assert profiled.plan_key == ordinary.plan_key
    assert plan.identity == plan_identity
    np.testing.assert_array_equal(
        profiled.series(field).values.numpy(), ordinary.series(field).values.numpy()
    )
