"""Exact native-history continuation through installed Python and a fresh process."""
from __future__ import annotations

import base64
import json
import math
import subprocess
import sys
import time

import numpy as np
import pytest
import eqiora as q


def problem(events: bool):
    source = "model Flow(){state x:1;initial{x=1;}"
    source += (
        "relation flow{derivative(x)=-1[1/s]*x;}"
        "event hit=crossing(x-0.5,direction=falling);relation reset at hit{next(x)=1;}"
        if events else
        "relation flow{derivative(x)=-25[1/s]*x+math.sin(100[1/s]*time())*1[1/s];}"
    )
    model = q.compile(source=source + "}")
    field = model.field("x")
    policy = q.time.EventPolicy(max_events=2000, guard_tolerances=(
        q.time.GuardTolerance(model.activation("hit"), 1e-8, q.Dimension()),
    )) if events else None
    plan = q.resolve(model, temporal=q.time.Tsitouras45(
        initial_step_s=1., relative_tolerance=1e-9,
        absolute_tolerances={(field, 0, 0, False): 1e-11}, events=policy,
    ))
    horizon = 1000. if events else 100.
    return plan, field, horizon, (0.2, 1.5, horizon)


def trajectory_bytes(result: q.Result) -> bytes:
    # Execution elapsed time is incidental; the exact claim is the accepted
    # numerical history, reset receipts and output schedule in the Trajectory.
    payload = json.loads(result.to_bytes())["content"]["payload"]
    return base64.b64decode(payload["trajectory_base64"])


@pytest.mark.parametrize("events", [False, True])
def test_progress_cancellation_and_process_reopening_preserve_exact_history(tmp_path, events):
    plan, field, horizon, outputs = problem(events)
    initial = q.State.initial(plan)
    run = q.submit(plan, state=initial, until_s=horizon, output_times_s=outputs)
    deadline = time.monotonic() + 30.
    previous = 0
    while True:
        progress = run.progress
        if progress is not None:
            assert isinstance(progress, q.RunProgress)
            assert progress.maximum_steps is None
            assert progress.accepted_steps >= previous
            previous = progress.accepted_steps
            if progress.model_time_s > 0.75:
                break
        assert not run.done, "the bounded long fixture must reach an observable accepted prefix"
        assert time.monotonic() < deadline
        time.sleep(0.0005)
    assert run.cancel()
    with pytest.raises(q.CancellationError):
        run.result()
    cancelled = run.cancellation
    assert isinstance(cancelled, q.RunCancellation)
    assert cancelled.progress == run.progress
    assert cancelled.progress.accepted_steps >= previous
    assert cancelled.state.time_s == cancelled.progress.model_time_s
    assert cancelled.state.source_kind == "checkpoint"
    assert cancelled.request_identity == run.plan_key
    checkpoint = cancelled.checkpoint
    assert isinstance(checkpoint, q.ExecutionCheckpoint)
    saved = checkpoint.to_bytes()
    recorded = checkpoint.record()
    assert isinstance(recorded, q.RecordedTimeCheckpoint)
    assert recorded.to_bytes() == saved
    assert recorded.request_identity == cancelled.request_identity
    recorded = q.RecordedTimeCheckpoint.from_bytes(plan, saved)
    assert recorded.restore().to_bytes() == saved
    assert not hasattr(recorded, "state")
    assert q.ExecutionCheckpoint.from_bytes(plan, saved).to_bytes() == saved
    checkpoint.write(tmp_path / "saved.eqcheckpoint")
    plan.write(tmp_path / "plan.eqplan")

    for controls in ({"state": initial}, {"until_s": horizon}, {"output_times_s": outputs}, {"steps": 1}):
        with pytest.raises(TypeError, match="original State, horizon and output schedule"):
            q.submit(plan, checkpoint=checkpoint, **controls)
    foreign, _, _, _ = problem(not events)
    with pytest.raises((ValueError, q.ValidationError), match="exact Plan"):
        q.submit(foreign, checkpoint=checkpoint)
    with pytest.raises(q.ValidationError):
        q.ExecutionCheckpoint.from_bytes(plan, saved[:-1])
    # A separate interpreter must reconstruct both owners from files. It cannot
    # retain this Run, observer, provider object or solver heap.
    subprocess.run([sys.executable, "-I", "-c", """
from pathlib import Path
import sys
import eqiora as q
root = Path(sys.argv[1])
plan = q.Plan.read(root / 'plan.eqplan')
checkpoint = q.ExecutionCheckpoint.read(plan, root / 'saved.eqcheckpoint')
run = q.submit(plan, checkpoint=checkpoint)
result = run.result()
assert run.progress.model_time_s == float(sys.argv[2])
assert run.progress.maximum_steps is None
(root / 'resumed.json').write_bytes(result.to_bytes())
(root / 'accepted_steps.txt').write_text(str(run.progress.accepted_steps))
""", str(tmp_path), str(horizon)], check=True, timeout=120)
    resumed = q.Result.from_bytes(plan, (tmp_path / "resumed.json").read_bytes())
    baseline = q.run(plan, state=initial, until_s=horizon, output_times_s=outputs)
    assert trajectory_bytes(resumed) == trajectory_bytes(baseline)
    mapped = q.EvaluationMapPlan.from_time(
        plan, [initial], until_s=horizon, output_times_s=outputs,
        checkpoints=[cancelled.checkpoint],
        # Budget for this already serialized history, including dense samples.
        # Only incidental elapsed-time metadata may change its encoded size;
        # reserve 1 KiB for that, without changing the public storage defaults.
        member_bytes_limit=len(baseline.to_bytes()) + 1024,
        storage_bytes_limit=128 * 1024**2,
    ).execute()
    assert isinstance(mapped, q.CompleteEvaluationMap), mapped.diagnostics
    assert trajectory_bytes(mapped[0]) == trajectory_bytes(baseline)
    history = json.loads(trajectory_bytes(resumed))["payload"]["history"]
    assert int((tmp_path / "accepted_steps.txt").read_text()) == len(history)
    values = np.asarray(resumed.series(field).values)
    times = np.asarray(outputs)
    if events:
        exact = np.exp(-(times - np.floor(times / math.log(2)) * math.log(2)))
        # Finite guard tolerance may accumulate over at most 2000 resets;
        # this comparison is a regression bound, not a certified global bound.
        np.testing.assert_allclose(values, exact, atol=8e-5, rtol=0)
    else:
        exact = (1 + 100/10625) * np.exp(-25*times) + (25*np.sin(100*times) - 100*np.cos(100*times))/10625
        np.testing.assert_allclose(values, exact, atol=2e-8, rtol=0)


def test_restart_and_resume_are_distinct_operations():
    # A physical State can start a fresh request; it cannot substitute for the
    # accepted controller/history required by checkpoint continuation.
    plan, _, _, _ = problem(False)
    with pytest.raises(TypeError):
        q.submit(plan, checkpoint=q.State.initial(plan))
    assert not hasattr(q, "TransientRunProgress")
    assert not hasattr(q, "TransientRunCancellation")
