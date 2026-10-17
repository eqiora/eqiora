"""Installed collections retain exact attribution without recreating live executions."""
import copy
import hashlib
import json
import subprocess
import sys

import eqiora as q
import numpy as np
import pytest

from test_adaptive_continuation import trajectory_bytes
from test_evaluation_map import program_for, scalar_leaves
from test_time_evaluation_map import stochastic_problem


def assert_same_primal(record, original):
    for actual, expected in zip(record.output, original.output, strict=True):
        np.testing.assert_array_equal(actual, expected)
        if isinstance(actual, np.ndarray):
            assert not actual.flags.writeable
    assert record.evidence.plan_identity == original.evidence.plan_identity


def test_exact_offline_grouped_reopening_and_distinct_retry_occurrences(tmp_path):
    _, program, inputs = program_for()
    mapped = program.map(scalar_leaves(np.array([[3., .2], [1., -.1], [3., .2]])),
                         point_shape=(3,), shared_inputs=[inputs[1]], shared=(2.,))
    complete = mapped.execute()
    collection = complete.record()
    assert collection.complete and len(collection) == 1
    group = collection[0]
    assert group.complete and len(group) == 3
    assert group.shared_input_ids == mapped.shared_input_ids
    assert group.inputs == program.inputs and group.outputs == program.outputs
    assert group.program_bytes is not None
    assert len({member.occurrence_id for member in group}) == 3
    assert group[0].reference == group[2].reference
    for i, member in enumerate(group):
        assert_same_primal(member.result, complete[i].primal())
        np.testing.assert_array_equal(member.point, complete[i].point)
        assert member.sampling is None and member.checkpoint is None
        assert member.diagnostics == []
        assert not hasattr(member.result, "jvp")
    retry = mapped.execute_recorded()
    assert retry[0].request_identity == group.request_identity
    assert retry[0].execution_id != group.execution_id
    assert retry[0][0].occurrence_id != group[0].occurrence_id
    with pytest.raises(q.ValidationError, match="duplicate execution"):
        q.EvaluationCollection.combine([collection, collection])
    _, other, _ = program_for(q.fvm.CellCenteredTpfa())
    other_collection = other.map((np.array([1.]), np.array([2.]), np.array([0.])),
                                 point_shape=(1,)).execute_recorded()
    grouped = q.EvaluationCollection.combine([collection, other_collection])
    assert grouped[0].plan.identity != grouped[1].plan.identity
    path = tmp_path / "saved.eqcollection"
    grouped.write(path)
    expected = grouped.to_bytes()
    assert q.EvaluationCollection.read(path).to_bytes() == expected
    # A new interpreter sees only saved bytes, not the originating Program/Plan/RNG.
    script = """
import pathlib, sys
import eqiora as q
path = pathlib.Path(sys.argv[1])
def forbidden(*args, **kwargs):
    raise AssertionError('offline opening invoked execution')
q.run = q.submit = q.resolve = q.compile = q.differentiate = forbidden
value = q.EvaluationCollection.read(path)
assert value.to_bytes() == path.read_bytes()
assert value.complete and len(value) == 2
assert value[0][0].reference == value[0][2].reference
assert value[0][0].occurrence_id != value[0][2].occurrence_id
assert value[0][0].result.output[0].shape
assert value[0].inputs
assert value[0].plan.identity
"""
    subprocess.run([sys.executable, "-I", "-c", script, str(path)], check=True, cwd=tmp_path, timeout=120)
    with pytest.raises(IndexError):
        _ = grouped[2]
    with pytest.raises(IndexError):
        _ = group[-4]


def test_partial_failure_and_cancellation_retain_individual_authority():
    _, program, _ = program_for()
    mapped = program.map((np.array([1., 1., 3.]), np.array([2., -1., 2.]), np.zeros(3)),
                         point_shape=(3,))
    report = mapped.execute()
    collection = q.EvaluationCollection.from_bytes(report.record().to_bytes())
    assert not collection.complete and not collection[0].complete
    assert [v.status for v in collection[0]] == ["accepted", "failed", "not_started"]
    assert_same_primal(collection[0][0].result, report[0].primal())
    assert collection[0][1].result is collection[0][2].result is None
    assert collection[0][1].diagnostics[0].code == report.diagnostics[0].code
    assert collection[0][1].diagnostics[0].message == report.diagnostics[0].message
    assert collection[0][1].diagnostics[0].severity == report.diagnostics[0].severity
    wire = json.loads(collection.to_bytes())
    wire["complete"] = True
    with pytest.raises(q.ValidationError, match="completion"):
        q.EvaluationCollection.from_bytes(json.dumps(wire, separators=(",", ":")).encode())
    token = q.EvaluationMapCancellation()
    token.cancel()
    cancelled = mapped.execute_recorded(cancellation=token)
    assert [v.status for v in cancelled[0]] == ["cancelled", "not_started", "not_started"]
    assert q.EvaluationCollection.from_bytes(cancelled.to_bytes()).to_bytes() == cancelled.to_bytes()


def test_stochastic_records_preserve_repeated_sample_identity_and_exact_results(tmp_path):
    plan = stochastic_problem()
    samples = [q.SamplingIdentity(bytes([i]), b"record") for i in (2, 1, 2)]
    mapped = q.EvaluationMapPlan.from_time(plan, [q.State.initial(plan)] * 3,
            until_s=1., output_times_s=(.25, .5, 1.), master_seed=bytes(32), sampling=samples)
    complete = mapped.execute()
    collection = complete.record()
    statistic = q.MomentEstimator.from_time_results([complete[i] for i in range(3)],
            [plan.model.observable("gbm.sample")], time_s=1., weights=[1., 2., 0.])
    collection = collection.with_estimator(0, statistic)
    reopened = q.EvaluationCollection.from_bytes(collection.to_bytes())
    saved_statistic = reopened[0].estimators[0]
    assert saved_statistic.value == statistic.value
    assert saved_statistic.value_type == statistic.value_type
    assert saved_statistic.weights == statistic.weights
    assert saved_statistic.reduction == statistic.reduction
    assert saved_statistic.time_s == statistic.time_s
    assert saved_statistic.model_digest == statistic.model_digest
    assert saved_statistic.result_identities == statistic.result_identities
    assert saved_statistic.occurrence_ids == [member.occurrence_id for member in reopened[0]]
    assert saved_statistic.gradient_estimator == "fixed-samples"
    assert saved_statistic.gradient_entities == {}
    assert saved_statistic.output_indices is None
    assert not hasattr(saved_statistic, "standard_error")
    wrong_order = q.MomentEstimator.from_time_results([complete[i] for i in (1, 0, 2)],
            [plan.model.observable("gbm.sample")], time_s=1.)
    with pytest.raises(q.ValidationError, match="ordered membership"):
        collection.with_estimator(0, wrong_order)
    assert reopened.to_bytes() == collection.to_bytes()
    assert reopened[0].program_bytes is None and reopened[0].inputs is None
    for i, member in enumerate(reopened[0]):
        assert member.sampling == samples[i]
        assert member.result.to_bytes() == complete[i].to_bytes()
        assert member.time_request_bytes is not None and member.point is None
    # Wall-clock execution provenance may differ; the retained physical path must match.
    assert trajectory_bytes(reopened[0][0].result) == trajectory_bytes(reopened[0][2].result)
    assert reopened[0][0].occurrence_id != reopened[0][2].occurrence_id
    token = q.EvaluationMapCancellation()
    token.cancel()
    partial = mapped.execute_recorded(cancellation=token)
    partial = q.EvaluationCollection.from_bytes(partial.to_bytes())
    assert not partial.complete
    assert [v.sampling for v in partial[0]] == samples
    with pytest.raises(q.ValidationError, match="complete execution group"):
        partial.with_estimator(0, statistic)
    path = tmp_path / "stochastic.eqcollection"
    collection.write(path)
    script = """
import pathlib, sys
import eqiora as q
path = pathlib.Path(sys.argv[1])
value = q.EvaluationCollection.read(path)
assert value.to_bytes() == path.read_bytes()
import hashlib
assert hashlib.sha256(value[0][0].result.to_bytes()).hexdigest() == sys.argv[3]
assert value[0][0].sampling == value[0][2].sampling
assert value[0][0].sampling != value[0][1].sampling
record = value[0].estimators[0]
assert record.value == float(sys.argv[2])
assert record.occurrence_ids == [member.occurrence_id for member in value[0]]
assert record.result_identities == [member.reference for member in value[0]]
assert record.weights == [1., 2., 0.]
assert record.reduction == 'empirical-mean'
assert record.time_s == 1.
assert not hasattr(record, 'standard_error')
"""
    subprocess.run([sys.executable, "-I", "-c", script, str(path), str(statistic.value), hashlib.sha256(complete[0].to_bytes()).hexdigest()], check=True, cwd=tmp_path, timeout=120)


def test_collection_falsifiers_reach_membership_and_content_boundaries():
    _, program, _ = program_for()
    collection = program.map((np.array([1., 2.]), np.ones(2), np.zeros(2)),
                             point_shape=(2,)).execute_recorded()
    raw = collection.to_bytes()
    original = json.loads(raw)
    mutants = []
    missing = copy.deepcopy(original)
    missing["groups"][0]["members"].pop()
    mutants.append((missing, "missing planned"))
    duplicate = copy.deepcopy(original)
    duplicate["groups"][0]["members"][1]["occurrence"] = duplicate["groups"][0]["members"][0]["occurrence"]
    mutants.append((duplicate, "attribution"))
    reordered = copy.deepcopy(original)
    reordered["groups"][0]["members"].reverse()
    mutants.append((reordered, "attribution"))
    foreign = copy.deepcopy(original)
    foreign["groups"][0]["members"][0]["outcome"]["reference"] = "absent"
    mutants.append((foreign, "reference is missing"))
    for wire, message in mutants:
        with pytest.raises(q.ValidationError, match=message):
            q.EvaluationCollection.from_bytes(json.dumps(wire, separators=(",", ":")).encode())
    for corrupt in (b"", raw[:len(raw)//2], raw[:-1], raw + b"x"):
        with pytest.raises(q.ValidationError):
            q.EvaluationCollection.from_bytes(corrupt)
