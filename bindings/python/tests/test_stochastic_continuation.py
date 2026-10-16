"""Serial stochastic execution through the installed Plan/Run/persistence owners."""
from __future__ import annotations

import base64
import json
import subprocess
import sys
import time

import numpy as np
import pytest
import eqiora as q


def problem(jumps: bool):
    laws = (
        "noise N:poisson(intensity=512[1/s]);"
        "law first for x {jump N=2;} law second for y {jump N=1;}"
        if jumps else
        "noise W:wiener(covariance=1);"
        "law first for x {calculus ito;drift 2[1/s];diffusion W=1[1/s^(1/2)];}"
        "law second for y {calculus ito;drift 1[1/s];diffusion W=0.5[1/s^(1/2)];}"
    )
    model = q.compile(source=(
        "model Process(){state x:1;state y:1;initial{x=0;y=0;}" + laws + "}"
    ))
    temporal = q.time.PoissonEvents(max_events=4096) if jumps else q.time.EulerMaruyama(depth=9)
    return q.resolve(model, temporal=temporal)


def trajectory_bytes(result):
    return base64.b64decode(json.loads(result.to_bytes())["content"]["payload"]["trajectory_base64"])


@pytest.mark.parametrize("jumps", [False, True])
def test_process_checkpoint_reopens_in_a_new_interpreter(tmp_path, jumps):
    plan = problem(jumps)
    assert isinstance(plan.capability, q.time.TimePlanView)
    assert plan.capability.kind == "time"
    assert plan.mesh is None
    assert plan.solve is None
    reopened = q.Plan.from_bytes(plan.to_bytes())
    assert reopened.to_bytes() == plan.to_bytes()
    assert reopened.temporal == plan.temporal
    initial = q.State.initial(plan, time_s=2.0)
    assert q.State.from_bytes(plan, initial.to_bytes()).to_bytes() == initial.to_bytes()
    identity = q.SamplingIdentity(b"member", b"path", lineage=(4, 9))
    assert identity.sample_id == b"member"
    assert identity.channel == b"path"
    assert identity.lineage == [4, 9]
    assert identity.coupling_id is None
    same_identity = q.SamplingIdentity(b"member", b"path", lineage=(4, 9))
    assert same_identity == identity
    assert hash(same_identity) == hash(identity)
    controls = dict(state=initial, until_s=3., output_times_s=(2.25, 2.5, 2.75),
                    master_seed=bytes(range(32)), sampling=identity)
    run = q.submit(plan, **controls)
    deadline = time.monotonic() + 30.
    while run.progress is None or run.progress.accepted_steps < 1:
        assert not run.done, "fixture must expose an accepted prefix before completion"
        assert time.monotonic() < deadline
        time.sleep(0.0005)
    assert run.cancel()
    with pytest.raises(q.CancellationError):
        run.result()
    cancelled = run.cancellation
    assert 0 < cancelled.progress.accepted_steps
    assert 2. < cancelled.state.time_s < 3.
    assert cancelled.state.source_kind == "checkpoint"
    assert cancelled.progress == run.progress
    saved = cancelled.checkpoint.to_bytes()
    assert q.ExecutionCheckpoint.from_bytes(plan, saved).to_bytes() == saved
    cancelled.checkpoint.write(tmp_path / "saved.eqcheckpoint")
    plan.write(tmp_path / "plan.eqplan")

    # Reopen only durable artifacts in a separate interpreter, with no live RNG.
    subprocess.run([sys.executable, "-I", "-c", """
from pathlib import Path
import sys
import eqiora as q
root = Path(sys.argv[1])
plan = q.Plan.read(root / 'plan.eqplan')
checkpoint = q.ExecutionCheckpoint.read(plan, root / 'saved.eqcheckpoint')
run = q.submit(plan, checkpoint=checkpoint)
result = run.result()
assert run.progress.model_time_s == 3.
result.write(root / 'resumed.eqresult')
""", str(tmp_path)], check=True, timeout=120)
    resumed = q.Result.read(plan, tmp_path / "resumed.eqresult")
    baseline = q.run(plan, **controls)
    assert trajectory_bytes(resumed) == trajectory_bytes(baseline)
    x, y = plan.model.field("x"), plan.model.field("y")
    xs, ys = np.asarray(resumed.series(x).values), np.asarray(resumed.series(y).values)
    # Every authored increment has dx=2*dy; binary arithmetic cancels the shared
    # Wiener noise here. Poisson counts give exact integral x=2*y as well.
    np.testing.assert_array_equal(xs, 2 * ys)
    if jumps:
        assert np.all(ys == np.floor(ys))
        assert np.all(np.diff(ys) >= 0)
    state = q.State.from_result(plan, resumed, time_s=2.5)
    assert state.value(x) == xs[1]
    assert q.Result.from_bytes(plan, resumed.to_bytes()).to_bytes() == resumed.to_bytes()
    with pytest.raises(TypeError):
        q.submit(plan, checkpoint=cancelled.checkpoint, master_seed=controls["master_seed"], sampling=identity)
    with pytest.raises(q.ValidationError):
        q.ExecutionCheckpoint.from_bytes(plan, saved + b"\n")


def test_sampling_controls_are_explicit_and_do_not_cross_execution_families():
    plan = problem(False)
    controls = dict(state=q.State.initial(plan), until_s=1., output_times_s=(1.,))
    identity = q.SamplingIdentity(b"member", b"path")
    with pytest.raises(q.ValidationError):
        q.submit(plan, **controls)
    with pytest.raises(TypeError):
        q.submit(plan, **controls, sampling=identity)
    with pytest.raises(ValueError, match="32 bytes"):
        q.submit(plan, **controls, sampling=identity, master_seed=b"short")
    with pytest.raises(q.ValidationError):
        q.SamplingIdentity(b"", b"path")
    with pytest.raises(q.ValidationError):
        q.resolve(plan.model, temporal=q.time.EulerMaruyama(depth=53))
    with pytest.raises(q.ValidationError):
        q.resolve(plan.model, temporal=q.time.PoissonEvents(max_events=4096))
    model = q.compile(source="model Deterministic(){state x:1;initial{x=1;}relation flow{derivative(x)=0[1/s];}}")
    field = model.field("x")
    deterministic = q.resolve(model, temporal=q.time.Tsitouras45(
        initial_step_s=.01, relative_tolerance=1e-6,
        absolute_tolerances={(field, 0, 0, False): 1e-8},
    ))
    with pytest.raises(q.ValidationError):
        q.submit(deterministic, state=q.State.initial(deterministic), until_s=1.,
                 output_times_s=(1.,), sampling=identity, master_seed=bytes(32))
