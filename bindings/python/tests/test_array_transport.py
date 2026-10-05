from __future__ import annotations

import gc
import subprocess
import sys

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


def result_array() -> tuple[eqiora.Result, eqiora.Array]:
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
    result = eqiora.run(
        plan,
        state=eqiora.State.initial(plan),
        until_s=0.2,
        output_times_s=(0.05, 0.1, 0.2),
    )
    return result, result.series(field).values


def test_result_owns_a_lazy_exact_cpu_descriptor() -> None:
    program = f"""
import sys
import eqiora
assert "numpy" not in sys.modules
model = eqiora.compile(source={DECAY!r})
field = model.field(model.field_ids[0])
plan = eqiora.resolve(
    model,
    temporal=eqiora.time.Tsitouras45(
        initial_step_s=0.01,
        relative_tolerance=1.0e-9,
        absolute_tolerances={{(field, 0): 1.0e-11}},
    ),
)
result = eqiora.run(
    plan,
    state=eqiora.State.initial(plan),
    until_s=0.2,
    output_times_s=(0.05, 0.1, 0.2),
)
array = result.series(field).values
assert "numpy" not in sys.modules
assert array.shape == (3,)
assert array.strides == (8,)
assert "numpy" not in sys.modules

class ReentrantNumpyFinder:
    attempted = False

    def find_spec(self, fullname, _path, _target=None):
        if not self.attempted and fullname.startswith("numpy"):
            self.attempted = True
            try:
                array.numpy(copy=False)
            except RuntimeError as error:
                assert "already in progress" in str(error)
            else:
                raise AssertionError("reentrant materialization was admitted")
        return None

finder = ReentrantNumpyFinder()
sys.meta_path.insert(0, finder)
array.numpy(copy=False)
sys.meta_path.remove(finder)
assert finder.attempted
assert "numpy" in sys.modules
"""
    completed = subprocess.run(
        [sys.executable, "-c", program],
        check=False,
        capture_output=True,
        text=True,
        timeout=10.0,
    )
    assert completed.returncode == 0, completed.stderr

    _, array = result_array()
    assert array.device == "cpu"
    assert array.device_id == 0
    assert array.dtype == "float64"
    assert array.byte_order == sys.byteorder
    assert array.shape == (3,)
    assert array.strides == (np.dtype(np.float64).itemsize,)
    assert array.c_contiguous
    assert array.aligned
    assert array.readonly
    assert array.ownership == "owned"
    assert array.origin_copy_occurred is False
    assert len(array) == 3


def test_numpy_projection_is_zero_copy_read_only_and_lifetime_safe() -> None:
    result, array = result_array()
    selected = array.numpy(copy=False)
    assert selected is array.numpy(copy=None)
    assert selected.flags.c_contiguous
    assert selected.flags.aligned
    assert not selected.flags.owndata
    assert not selected.flags.writeable

    with pytest.raises(ValueError):
        selected[0] = 9.0
    with pytest.raises(ValueError):
        selected.setflags(write=True)

    copied = array.numpy(copy=True)
    assert copied.flags.writeable
    assert not np.shares_memory(copied, selected)
    copied[0] = 9.0
    assert selected[0] == pytest.approx(np.exp(-0.05), rel=2.0e-8, abs=2.0e-10)

    del result, array
    gc.collect()
    np.testing.assert_allclose(
        selected,
        np.exp(-np.array([0.05, 0.1, 0.2])),
        rtol=2.0e-8,
        atol=2.0e-10,
    )


def test_dlpack_is_a_versioned_cpu_snapshot_not_a_mutable_alias() -> None:
    result, array = result_array()
    original = array.numpy(copy=False)
    assert array.__dlpack_device__() == (1, 0)

    snapshot = np.from_dlpack(array)
    assert snapshot.ctypes.data != original.ctypes.data
    assert not np.shares_memory(snapshot, original)
    np.testing.assert_array_equal(snapshot, original)
    if snapshot.flags.writeable:
        snapshot[0] = 11.0
    else:
        with pytest.raises(ValueError):
            snapshot.setflags(write=True)
    assert original[0] == pytest.approx(np.exp(-0.05), rel=2.0e-8, abs=2.0e-10)

    explicit = np.from_dlpack(array, copy=True)
    assert not np.shares_memory(explicit, original)
    on_cpu = np.from_dlpack(array, device="cpu")
    assert not np.shares_memory(on_cpu, original)
    assert not np.shares_memory(on_cpu, snapshot)
    with pytest.raises(BufferError):
        np.from_dlpack(array, copy=False)

    with pytest.raises(BufferError):
        array.__dlpack__()
    with pytest.raises(BufferError):
        array.__dlpack__(stream=1, max_version=(1, 0))
    with pytest.raises(BufferError):
        array.__dlpack__(max_version=(1, 0), dl_device=(2, 0))
    with pytest.raises(BufferError):
        array.__dlpack__(max_version=(2, 0))

    capsule = array.__dlpack__(max_version=(1, 0), dl_device=(1, 0), copy=True)
    assert '"dltensor_versioned"' in repr(capsule)

    class SingleUseProducer:
        def __dlpack_device__(self):
            return (1, 0)

        def __dlpack__(self, **_kwargs):
            return capsule

    producer = SingleUseProducer()
    consumed = np.from_dlpack(producer)
    np.testing.assert_array_equal(consumed, original)
    with pytest.raises(ValueError):
        np.from_dlpack(producer)

    expected_tail = original[1:].copy()
    del result, array, original
    gc.collect()
    np.testing.assert_array_equal(snapshot[1:], expected_tail)
