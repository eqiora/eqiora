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


def scalar_leaves(values):
    """Project this fixture's scalar columns into ordered contiguous leaves."""
    return tuple(np.ascontiguousarray(values[..., i]) for i in range(values.shape[-1]))


def close(actual, expected):
    # Same existing accepted-point comparison precision; never fitted to outputs.
    np.testing.assert_allclose(actual, expected, rtol=1e-10, atol=1e-12)


@pytest.mark.parametrize("method", [eqiora.fem.Q1(), eqiora.fvm.CellCenteredTpfa()])
def test_ordered_duplicates_shared_native_products_and_analytic_composition(method):
    model, program, inputs = program_for(method)
    original = model.to_bytes()
    mapped = np.array([[3.0, 0.2], [1.0, -0.1], [3.0, 0.2]])
    plan = program.map(scalar_leaves(mapped), point_shape=mapped.shape[:-1],
                       shared_inputs=[inputs[1]], shared=(2.0,))
    frozen = np.stack(plan.points, axis=-1)
    mapped[:] = 99.0
    assert plan.point_shape == (3,)
    assert all(leaf.shape == (3,) for leaf in plan.points)
    assert plan.shared_input_ids == [program.inputs[1].id]
    assert plan.mapped_input_ids == [program.inputs[0].id, program.inputs[2].id]
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
    close(complete.primal()[0], [p.primal().output[0] for p in points])
    np.testing.assert_array_equal(complete.primal()[0][0], complete.primal()[0][2])
    np.testing.assert_array_equal(complete.member(2).point, frozen[2])
    assert not complete.primal()[0].flags.writeable
    assert not plan.points[0].flags.writeable
    with pytest.raises(ValueError):
        plan.points[0].setflags(write=True)

    # For the already admitted discrete solve, u_h(a,k,b)=(a/k)r_h+b.
    # This also falsifies accidentally swapping source and diffusion coordinates.
    response = program.evaluate(tuple(np.array([1.0, 1.0, 0.0]))).primal().output[0]
    close(complete.primal()[0], [a / k * response + b for a, k, b in frozen])
    direction = np.array([[0.4, 0.3], [-0.2, 0.7], [0.1, -0.5]])
    dk = -0.25
    jvp = complete.jvp(scalar_leaves(direction), shared=(dk,))
    expected_jvp = [(da / k - a * dk / k**2) * response + db
                    for (a, k, _), (da, db) in zip(frozen, direction, strict=True)]
    close(jvp.tangent[0], expected_jvp)
    close(jvp.tangent[0], [p.jvp((d[0], dk, d[1])).tangent[0]
                        for p, d in zip(points, direction, strict=True)])
    close(jvp.output[0], complete.primal()[0])
    assert jvp.member(2).evidence.plan_identity == program.plan_identity
    cotangents = np.arange(complete.primal()[0].size, dtype=np.float64).reshape(complete.primal()[0].shape) / 10.0
    vjp = complete.vjp((cotangents,))
    expected = np.array([p.vjp((c,)).input_cotangent
                         for p, c in zip(points, cotangents, strict=True)])
    assert vjp.shared_shape == ()
    assert vjp.mapped_shape == (3,)
    close(vjp.shared_cotangents[0], expected[:, 1].sum())
    for leaf, column in zip(vjp.mapped_cotangents, [0, 2], strict=True):
        close(leaf, expected[:, column])
        assert not leaf.flags.writeable
    assert not vjp.shared_cotangents[0].flags.writeable
    close(np.sum(jvp.tangent[0] * cotangents),
          sum(np.sum(leaf * direction[:, index]) for index, leaf in enumerate(vjp.mapped_cotangents)) + vjp.shared_cotangents[0] * dk)
    assert model.to_bytes() == original


def test_nested_interleaved_grids_and_product_lineage():
    _, program, inputs = program_for()
    mapped = np.array([[[1.0, 0.0], [2.0, 0.2]], [[3.0, -0.1], [1.0, 0.0]]])
    plan = program.map(scalar_leaves(mapped), point_shape=mapped.shape[:-1],
                       shared_inputs=[inputs[1]], shared=(2.0,))
    complete = plan.execute()
    assert plan.occurrence_coordinates(2) == (1, 0)
    # Combined axes are [point0, seed0, point1], not flattened/outer-product guessed.
    directions = np.arange(2 * 3 * 2 * 2, dtype=np.float64).reshape(2, 3, 2, 2) / 20.0
    shared = np.array([[0.1], [-0.3], [0.2]])
    jvp = complete.jvp(scalar_leaves(directions), shared=scalar_leaves(shared), seed_shape=(3,), point_axes=(0, 2))
    assert jvp.point_axes == (0, 2)
    assert jvp.seed_shape == (3,)
    assert jvp.plan.point_shape == (2, 2)
    assert jvp.shape == (2, 3, 2)
    covectors = np.ones((*jvp.shape, *program.outputs[0].shape))
    vjp = complete.vjp((covectors,), seed_shape=(3,), point_axes=(0, 2))
    expected_shared = np.zeros((3, 1))
    for i, s, j in np.ndindex(2, 3, 2):
        point = complete.member(i * 2 + j)
        direction = np.array([directions[i, s, j, 0], shared[s, 0], directions[i, s, j, 1]])
        close(jvp.tangent[0][i, s, j], point.jvp(tuple(direction)).tangent[0])
        expected = np.asarray(point.vjp((covectors[i, s, j],)).input_cotangent)
        close([leaf[i, s, j] for leaf in vjp.mapped_cotangents], expected[[0, 2]])
        expected_shared[s, 0] += expected[1]
    close(vjp.shared_cotangents[0], expected_shared[:, 0])
    assert vjp.shared_shape == (3,)
    assert vjp.mapped_shape == (2, 3, 2)
    assert vjp.point_axes == (0, 2)
    with pytest.raises(eqiora.EqioraError):
        complete.jvp(scalar_leaves(directions), shared=scalar_leaves(shared), seed_shape=(3,), point_axes=(0, 0))
    with pytest.raises(BufferError, match="shape"):
        complete.jvp(scalar_leaves(directions), shared=(np.ones((2, 3)),), seed_shape=(3,), point_axes=(0, 2))
    bad_output = covectors.copy()
    bad_output[1, 2, 0, 1] = np.inf
    with pytest.raises(BufferError, match=r"output cotangents leaf 0.*finite"):
        complete.vjp((bad_output,), seed_shape=(3,), point_axes=(0, 2))
    bad_input = mapped.copy()
    bad_input[1, 0, 1] = np.nan
    with pytest.raises(BufferError, match=r"mapped parameters leaf 1.*finite"):
        program.map(scalar_leaves(bad_input), point_shape=(2, 2),
                        shared_inputs=[inputs[1]], shared=(2.0,))
    bad_shared = shared.copy()
    bad_shared[2, 0] = np.nan
    with pytest.raises(BufferError, match=r"shared tangents leaf 0.*finite"):
        complete.jvp(scalar_leaves(directions), shared=scalar_leaves(bad_shared), seed_shape=(3,), point_axes=(0, 2))


def test_zero_singleton_all_shared_and_zero_seed_shapes():
    _, program, inputs = program_for()
    token = eqiora.EvaluationMapCancellation()
    token.cancel()
    empty = program.map(tuple(np.empty((2, 0)) for _ in inputs), point_shape=(2, 0)).execute(cancellation=token)
    assert isinstance(empty, eqiora.CompleteEvaluationMap)
    assert empty.primal()[0].shape == (2, 0, *program.outputs[0].shape)
    assert empty.statuses == []
    product = empty.jvp(tuple(np.empty((2, 0)) for _ in inputs))
    assert product.tangent[0].shape == (2, 0, *program.outputs[0].shape)
    singleton = program.map((1.0, 2.0, 0.0), point_shape=()).execute()
    assert singleton.plan.point_shape == ()
    assert singleton.primal()[0].shape == program.outputs[0].shape
    assert singleton.plan.occurrence_coordinates(0) == ()
    zero_seeds = singleton.jvp(tuple(np.empty((0,)) for _ in inputs), seed_shape=(0,))
    assert zero_seeds.tangent[0].shape == (0, *program.outputs[0].shape)
    shared_only = program.map((), point_shape=(2,), shared_inputs=inputs,
                              shared=(1.0, 2.0, 0.0)).execute()
    np.testing.assert_array_equal(shared_only.primal()[0][0], shared_only.primal()[0][1])
    product = shared_only.jvp((), shared=(1.0, 0.0, 0.0))
    assert product.tangent[0].shape == (2, *program.outputs[0].shape)
    vjp = shared_only.vjp((np.ones((2, *program.outputs[0].shape)),))
    assert vjp.mapped_cotangents == ()
    assert vjp.mapped_shape == (2,)
    assert len(vjp.shared_cotangents) == 3
    assert all(leaf.shape == () for leaf in vjp.shared_cotangents)


def test_metadata_only_physical_failure_and_native_cancellation():
    _, program, _ = program_for()
    # Negative diffusion is finite and has a valid input schema: only execute solves/admit physics.
    plan = program.map((np.array([1.0, 1.0, 3.0]), np.array([2.0, -1.0, 2.0]),
                        np.zeros(3)), point_shape=(3,))
    assert plan.point_shape == (3,)
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
    values = np.array([1.0, 3.0])
    producer = DLPackProducer(values)
    plan = program.map((producer, np.full(2, 2.0), np.array([0.0, 0.1])), point_shape=(2,))
    assert len(producer.exports) == 1
    values[:] = 55.0
    np.testing.assert_array_equal(np.stack(plan.points, axis=-1),
                                  [[1.0, 2.0, 0.0], [3.0, 2.0, 0.1]])
    point = program.evaluate((1.0, 2.0, 0.0)).point
    assert program.map(point, point_shape=()).point_shape == ()
    for bad in (np.ones(2, dtype=np.float32), np.ones(2, dtype=np.int64),
                np.ones(2, dtype=">f8"), np.ones(4)[::2], np.ones(3),
                np.array(2.0), np.array([1.0, np.nan])):
        with pytest.raises((BufferError, TypeError)):
            program.map((bad, np.ones(2), np.zeros(2)), point_shape=(2,))
    wrong_device = DLPackProducer(np.ones(2), device=(2, 0))
    with pytest.raises(BufferError, match="CPU"):
        program.map((wrong_device, np.ones(2), np.zeros(2)), point_shape=(2,))
    assert not wrong_device.exports
    with pytest.raises(BufferError, match="shared"):
        program.map((np.empty(0), np.empty(0)), point_shape=(0,),
                    shared_inputs=[inputs[0]], shared=(np.array(np.nan),))
    with pytest.raises(BufferError, match="distinct"):
        program.map((np.ones(1),), point_shape=(1,),
                    shared_inputs=[inputs[0], inputs[0]], shared=(1.0, 1.0))
    other_model, _ = model_and_plan(eqiora.fem.Q1(), diffusion=3.0)
    with pytest.raises(BufferError, match="exact Program/Model"):
        program.map((np.ones(1), np.ones(1)), point_shape=(1,),
                    shared_inputs=[other_model.parameter("diffusion")], shared=(1.0,))
    with pytest.raises((BufferError, eqiora.EqioraError), match="byte"):
        program.map((np.ones(1), np.ones(1), np.zeros(1)), point_shape=(1,),
                    retained_bytes_limit=1)


def test_cancellation_token_can_be_requested_from_another_python_thread():
    _, program, _ = program_for()
    plan = program.map((np.ones(1024), np.full(1024, 2.0), np.zeros(1024)),
                       point_shape=(1024,))
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
