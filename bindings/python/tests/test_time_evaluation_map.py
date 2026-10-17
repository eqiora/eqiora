"""The common map projects ordinary independent time Runs and accepted Results."""
from pathlib import Path

import eqiora as q
import numpy as np
import pytest

from test_adaptive_continuation import problem as decay_problem, trajectory_bytes


def stochastic_problem():
    root = Path(__file__).resolve().parents[3]
    components = (root / "crates/eqiora-api/packages/Eqiora.Stochastic.Processes/src/processes.eqi").read_text()
    source = (root / "examples/stochastic-processes/src/main.eqi").read_text()
    source = source.replace("import Eqiora.Stochastic.Processes.processes as processes;", "").replace("processes.", "")
    model = q.compile(source=components + source)
    return q.resolve(model, temporal=q.time.EulerMaruyama(depth=4))


@pytest.mark.parametrize("chunk,workers", [(1, 1), (4, 3)])
def test_mapped_ou_gbm_preserve_ordinary_results_samples_and_reductions(chunk, workers):
    plan = stochastic_problem()
    states = [q.State.initial(plan) for _ in range(4)]
    samples = [q.SamplingIdentity(bytes([i]), b"map") for i in (2, 1, 2, 3)]
    controls = dict(until_s=1., output_times_s=(.25, .5, 1.), master_seed=bytes(32))
    expected = [q.run(plan, state=state, sampling=sample, **controls) for state, sample in zip(states, samples)]
    mapped = q.EvaluationMapPlan.from_time(plan, states, sampling=samples, chunk_size=chunk, workers=workers, **controls)
    assert mapped.time_plan is plan
    assert mapped.point_shape == (4,)
    assert len(mapped) == 4
    assert mapped[-1].to_bytes() == states[-1].to_bytes()
    assert mapped.occurrence_coordinates(2) == (2,)
    with pytest.raises(TypeError, match="time maps"):
        _ = mapped.program
    complete = mapped.execute()
    assert isinstance(complete, q.CompleteEvaluationMap)
    assert complete.statuses == ["accepted"] * 4
    results = [complete[i] for i in range(len(complete))]
    for index, result in enumerate(results):
        assert isinstance(result, q.Result)
        assert trajectory_bytes(result) == trajectory_bytes(expected[index])
        reopened = q.Result.from_bytes(plan, result.to_bytes())
        assert trajectory_bytes(reopened) == trajectory_bytes(result)
    assert trajectory_bytes(results[0]) == trajectory_bytes(results[2])
    assert trajectory_bytes(results[0]) != trajectory_bytes(results[1])
    output = plan.model.observable("gbm.sample")
    actual = q.MomentEstimator.from_time_results(results, [output], time_s=1.)
    reference = q.MomentEstimator.from_time_results(expected, [output], time_s=1.)
    assert actual.value == reference.value
    with pytest.raises(q.ValidationError):
        complete.jvp([])
    with pytest.raises(IndexError):
        _ = complete[4]


def test_mapped_decay_uses_distinct_initial_states_event_histories_and_clocks():
    plan, field, _, _ = decay_problem(True)
    prepared = q.run(plan, state=q.State.initial(plan), until_s=.5, output_times_s=(.5,))
    # A physical State from an accepted Run and an authored State at the same time.
    states = [q.State.from_result(plan, prepared, time_s=.5), q.State.initial(plan, time_s=.5)]
    controls = dict(until_s=2.2, output_times_s=(.6, .8, 1., 1.5, 2.1))
    expected = [q.run(plan, state=state, **controls) for state in states]
    assert trajectory_bytes(expected[0]) != trajectory_bytes(expected[1])
    mapped = q.EvaluationMapPlan.from_time(plan, states, chunk_size=2, workers=2, **controls)
    complete = mapped.execute()
    for index, result in enumerate(expected):
        assert trajectory_bytes(complete[index]) == trajectory_bytes(result)
        np.testing.assert_array_equal(complete[index].series(field).values, result.series(field).values)


def test_time_map_cancellation_and_admission_do_not_publish_partial_results():
    plan = stochastic_problem()
    initial = q.State.initial(plan)
    controls = dict(until_s=1., output_times_s=(1.,), master_seed=bytes(32), sampling=[q.SamplingIdentity(b"a", b"map")])
    mapped = q.EvaluationMapPlan.from_time(plan, [initial], **controls)
    token = q.EvaluationMapCancellation()
    token.cancel()
    report = mapped.execute(cancellation=token)
    assert isinstance(report, q.EvaluationMapTerminalReport)
    assert report.cancelled
    assert report.stopped_index == 0
    assert report.statuses == ["cancelled"]
    assert report[0] is None
    assert report.checkpoint(0) is None
    with pytest.raises(TypeError):
        q.MomentEstimator.from_time_results([report[0]], [plan.model.observable("ou.sample")], time_s=1.)
    with pytest.raises(ValueError, match="sampling identity"):
        q.EvaluationMapPlan.from_time(plan, [initial, initial], **controls)
    with pytest.raises(q.ValidationError):
        q.EvaluationMapPlan.from_time(plan, [initial], workers=2, chunk_size=1, **controls)
    failed = q.EvaluationMapPlan.from_time(plan, [initial], member_bytes_limit=1, **controls).execute()
    assert isinstance(failed, q.EvaluationMapTerminalReport)
    assert failed.statuses == ["failed"]
    assert failed[0] is None
