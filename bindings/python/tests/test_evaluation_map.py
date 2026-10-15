"""Installed native batches: independent composition checks, not new solver claims."""

from __future__ import annotations

import subprocess
import sys
import threading

import numpy as np
import pytest

import eqiora
from test_differentiation import DLPackProducer, model_and_plan


def program_for(method=None):
    model, plan = model_and_plan(method or eqiora.fem.Q1())
    inputs = tuple(model.parameter(name) for name in
                   ("source_scale", "diffusion", "boundary_offset"))
    return model, eqiora.diff.compile(plan, inputs=inputs, outputs=(plan.capability.fields[0],)), inputs


def close(actual, expected):
    # Same existing accepted-point comparison precision; never fitted to outputs.
    np.testing.assert_allclose(actual, expected, rtol=1e-10, atol=1e-12)


@pytest.mark.parametrize("method", [eqiora.fem.Q1(), eqiora.fvm.CellCenteredTpfa()])
def test_ordered_duplicates_shared_native_products_and_analytic_composition(method):
    model, program, inputs = program_for(method)
    original = model.to_bytes()
    mapped = np.array([[3.0, 0.2], [1.0, -0.1], [3.0, 0.2]])
    plan = program.map(mapped, shared_inputs=[inputs[1]], shared=np.array([2.0]))
    frozen = plan.points.copy()
    mapped[:] = 99.0
    assert plan.point_shape == (3,)
    assert plan.input_shape == (3, 3)
    assert plan.output_shape == (3, *program.output_shape)
    assert plan.shared_input_ids == [program.input_ids[1]]
    assert plan.mapped_input_ids == [program.input_ids[0], program.input_ids[2]]
    assert plan.occurrence_coordinates(2) == (2,)
    assert plan.program.model_digest == program.model_digest
    complete = plan.execute()
    assert isinstance(complete, eqiora.CompleteEvaluationMap)
    assert complete.statuses == ["accepted"] * 3
    assert len(complete) == 3
    np.testing.assert_array_equal(plan[-1], frozen[-1])
    np.testing.assert_array_equal(complete[-1].point, frozen[-1])
    assert len(list(complete)) == 3
    with pytest.raises(IndexError):
        plan[-4]
    points = [program.evaluate(tuple(point)) for point in frozen]
    close(complete.primal(), [p.primal().output[0] for p in points])
    np.testing.assert_array_equal(complete.primal()[0], complete.primal()[2])
    np.testing.assert_array_equal(complete.member(2).point, frozen[2])
    assert not complete.primal().flags.writeable
    assert not plan.points.flags.writeable
    with pytest.raises(ValueError):
        plan.points.setflags(write=True)

    # For the already admitted discrete solve, u_h(a,k,b)=(a/k)r_h+b.
    # This also falsifies accidentally swapping source and diffusion coordinates.
    response = program.evaluate(tuple(np.array([1.0, 1.0, 0.0]))).primal().output[0]
    close(complete.primal(), [a / k * response + b for a, k, b in frozen])
    direction = np.array([[0.4, 0.3], [-0.2, 0.7], [0.1, -0.5]])
    dk = -0.25
    jvp = complete.jvp(direction, shared=np.array([dk]))
    expected_jvp = [(da / k - a * dk / k**2) * response + db
                    for (a, k, _), (da, db) in zip(frozen, direction, strict=True)]
    close(jvp.tangent, expected_jvp)
    close(jvp.tangent, [p.jvp((d[0], dk, d[1])).tangent[0]
                        for p, d in zip(points, direction, strict=True)])
    close(jvp.output, complete.primal())
    assert jvp.member(2).evidence.plan_identity == program.plan_identity
    cotangents = np.arange(complete.primal().size, dtype=np.float64).reshape(plan.output_shape) / 10.0
    vjp = complete.vjp(cotangents)
    expected = np.array([p.vjp((c,)).input_cotangent
                         for p, c in zip(points, cotangents, strict=True)])
    close(vjp.shared_cotangents, [expected[:, 1].sum()])
    close(vjp.mapped_cotangents, expected[:, [0, 2]])
    close(np.sum(jvp.tangent * cotangents),
          np.sum(vjp.mapped_cotangents * direction) + vjp.shared_cotangents[0] * dk)
    assert model.to_bytes() == original


def test_nested_interleaved_grids_and_product_lineage():
    _, program, inputs = program_for()
    mapped = np.array([[[1.0, 0.0], [2.0, 0.2]], [[3.0, -0.1], [1.0, 0.0]]])
    plan = program.map(mapped, shared_inputs=[inputs[1]], shared=np.array([2.0]))
    complete = plan.execute()
    assert plan.occurrence_coordinates(2) == (1, 0)
    # Combined axes are [point0, seed0, point1], not flattened/outer-product guessed.
    directions = np.arange(2 * 3 * 2 * 2, dtype=np.float64).reshape(2, 3, 2, 2) / 20.0
    shared = np.array([[0.1], [-0.3], [0.2]])
    jvp = complete.jvp(directions, shared=shared, seed_shape=(3,), point_axes=(0, 2))
    assert jvp.point_axes == (0, 2)
    assert jvp.seed_shape == (3,)
    assert jvp.plan.point_shape == (2, 2)
    assert jvp.shape == (2, 3, 2, *program.output_shape)
    covectors = np.ones(jvp.shape)
    vjp = complete.vjp(covectors, seed_shape=(3,), point_axes=(0, 2))
    expected_shared = np.zeros((3, 1))
    for i, s, j in np.ndindex(2, 3, 2):
        point = complete.member(i * 2 + j)
        direction = np.array([directions[i, s, j, 0], shared[s, 0], directions[i, s, j, 1]])
        close(jvp.tangent[i, s, j], point.jvp(tuple(direction)).tangent[0])
        expected = np.asarray(point.vjp((covectors[i, s, j],)).input_cotangent)
        close(vjp.mapped_cotangents[i, s, j], expected[[0, 2]])
        expected_shared[s, 0] += expected[1]
    close(vjp.shared_cotangents, expected_shared)
    assert vjp.point_axes == (0, 2)
    with pytest.raises(eqiora.EqioraError):
        complete.jvp(directions, shared=shared, seed_shape=(3,), point_axes=(0, 0))
    with pytest.raises(BufferError, match="shape"):
        complete.jvp(directions, shared=np.ones((2, 3, 1)), seed_shape=(3,), point_axes=(0, 2))
    bad_output = covectors.copy()
    bad_output[1, 2, 0, 1] = np.inf
    with pytest.raises(BufferError, match=r"output cotangents.*grid coordinates \[1, 2, 0\], output coordinate 1"):
        complete.vjp(bad_output, seed_shape=(3,), point_axes=(0, 2))
    bad_input = mapped.copy()
    bad_input[1, 0, 1] = np.nan
    with pytest.raises(BufferError, match=r"mapped parameters.*grid coordinates \[1, 0\], input coordinate 1"):
        program.map(bad_input, shared_inputs=[inputs[1]], shared=np.array([2.0]))
    bad_shared = shared.copy()
    bad_shared[2, 0] = np.nan
    with pytest.raises(BufferError, match=r"shared tangents.*grid coordinates \[2\], input coordinate 0"):
        complete.jvp(directions, shared=bad_shared, seed_shape=(3,), point_axes=(0, 2))


def test_zero_singleton_all_shared_and_zero_seed_shapes():
    _, program, inputs = program_for()
    token = eqiora.EvaluationMapCancellation()
    token.cancel()
    empty = program.map(np.empty((2, 0, 3))).execute(cancellation=token)
    assert isinstance(empty, eqiora.CompleteEvaluationMap)
    assert empty.primal().shape == (2, 0, *program.output_shape)
    assert empty.statuses == []
    product = empty.jvp(np.empty((2, 0, 3)))
    assert product.tangent.shape == (2, 0, *program.output_shape)
    singleton = program.map(np.array([1.0, 2.0, 0.0])).execute()
    assert singleton.plan.point_shape == ()
    assert singleton.primal().shape == program.output_shape
    assert singleton.plan.occurrence_coordinates(0) == ()
    zero_seeds = singleton.jvp(np.empty((0, 3)), seed_shape=(0,))
    assert zero_seeds.tangent.shape == (0, *program.output_shape)
    shared_only = program.map(np.empty((2, 0)), shared_inputs=inputs,
                              shared=np.array([1.0, 2.0, 0.0])).execute()
    np.testing.assert_array_equal(shared_only.primal()[0], shared_only.primal()[1])
    product = shared_only.jvp(np.empty((2, 0)), shared=np.array([1.0, 0.0, 0.0]))
    assert product.tangent.shape == (2, *program.output_shape)
    vjp = shared_only.vjp(np.ones((2, *program.output_shape)))
    assert vjp.mapped_cotangents.shape == (2, 0)
    assert vjp.shared_cotangents.shape == (3,)


def test_metadata_only_physical_failure_and_native_cancellation():
    _, program, _ = program_for()
    # Negative diffusion is finite and has a valid input schema: only execute solves/admit physics.
    plan = program.map(np.array([[1.0, 2.0, 0.0], [1.0, -1.0, 0.0], [3.0, 2.0, 0.0]]))
    assert plan.output_shape == (3, *program.output_shape)
    assert plan.estimated_storage_bytes > 0
    report = plan.execute()
    assert isinstance(report, eqiora.EvaluationMapTerminalReport)
    assert report.stopped_index == 1
    assert report.statuses == ["accepted", "failed", "not_started"]
    assert report.member(0) is not None
    assert report.member(1) is report.member(2) is None
    assert report[-1] is None
    assert report.diagnostics
    assert not report.cancelled
    for name in ("primal", "jvp", "vjp"):
        assert not hasattr(report, name)
    token = eqiora.EvaluationMapCancellation()
    assert not token.requested
    token.cancel()
    cancelled = plan.execute(cancellation=token)
    assert cancelled.cancelled and cancelled.stopped_index == 0
    assert cancelled.statuses == ["cancelled", "not_started", "not_started"]
    assert cancelled.member(0) is None
    with pytest.raises(IndexError):
        cancelled.member(3)


def test_exact_input_admission_dlpack_once_and_frozen_aliases():
    _, program, inputs = program_for()
    values = np.array([[1.0, 2.0, 0.0], [3.0, 2.0, 0.1]])
    producer = DLPackProducer(values)
    plan = program.map(producer)
    assert len(producer.exports) == 1
    values[:] = 55.0
    np.testing.assert_array_equal(plan.points, [[1.0, 2.0, 0.0], [3.0, 2.0, 0.1]])
    # Existing Eqiora Array is rank one: one point, with the same staging contract.
    point = program.evaluate(tuple(np.array([1.0, 2.0, 0.0]))).point
    assert program.map(np.array(point)).point_shape == ()
    for bad in (np.ones((2, 3), dtype=np.float32), np.ones((2, 3), dtype=np.int64),
                np.ones((3, 2)).T, np.ones((2, 3), dtype=">f8"),
                np.ones((2, 6))[:, ::2], np.ones((2, 2)), np.array(2.0),
                np.array([[1.0, np.nan, 0.0]])):
        with pytest.raises((BufferError, TypeError)):
            program.map(bad)
    wrong_device = DLPackProducer(np.ones((2, 3)), device=(2, 0))
    with pytest.raises(BufferError, match="CPU"):
        program.map(wrong_device)
    assert not wrong_device.exports
    with pytest.raises(BufferError, match="shared"):
        program.map(np.empty((0, 2)), shared_inputs=[inputs[0]], shared=np.array([np.nan]))
    with pytest.raises(BufferError, match="distinct"):
        program.map(np.ones((1, 1)), shared_inputs=[inputs[0], inputs[0]], shared=np.ones(2))
    other_model, _ = model_and_plan(eqiora.fem.Q1(), diffusion=3.0)
    with pytest.raises(BufferError, match="exact Program/Model"):
        program.map(np.ones((1, 2)), shared_inputs=[other_model.parameter("diffusion")], shared=np.ones(1))
    with pytest.raises((BufferError, eqiora.EqioraError), match="byte"):
        program.map(np.ones((1, 3)), retained_bytes_limit=1)


def test_cancellation_token_can_be_requested_from_another_python_thread():
    _, program, _ = program_for()
    plan = program.map(np.tile(np.array([1.0, 2.0, 0.0]), (1024, 1)))
    token = eqiora.EvaluationMapCancellation()
    started = threading.Event()
    outcomes = []

    def execute():
        started.set()
        outcomes.append(plan.execute(cancellation=token))

    worker = threading.Thread(target=execute)
    worker.start()
    assert started.wait(timeout=10)
    token.cancel()
    worker.join(timeout=10)
    assert not worker.is_alive()
    assert len(outcomes) == 1
    assert isinstance(outcomes[0], eqiora.EvaluationMapTerminalReport)
    assert outcomes[0].cancelled
    assert outcomes[0].statuses[outcomes[0].stopped_index] == "cancelled"


def test_basic_import_has_no_optional_framework():
    subprocess.run([sys.executable, "-c", "import sys, eqiora; assert 'jax' not in sys.modules; assert 'torch' not in sys.modules; assert eqiora.diff.EvaluationMapPlan is eqiora.EvaluationMapPlan"], check=True)
