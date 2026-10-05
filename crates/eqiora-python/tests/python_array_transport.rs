use std::path::Path;

use pyo3::ffi::c_str;
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyDictMethods, PyModule};

const DECAY: &str = r#"
model decay() {
  state x: 1;
  initial { x = 1; }
  parameter rate: 1 / s = 1;
  relation flow {
    derivative(x) + rate * x = 0;
  }
}
"#;

#[test]
fn python_array_transport_is_owned_explicit_and_fail_closed() -> PyResult<()> {
    Python::initialize();
    Python::attach(|py| {
        let module = public_module(py)?;
        let locals = PyDict::new(py);
        locals.set_item("eqiora", &module)?;
        locals.set_item("decay_source", DECAY)?;
        py.run(
            c_str!(
                r#"
import gc
import sys

assert "numpy" not in sys.modules
model = eqiora.compile(source=decay_source)
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
array = result.series(field).values
assert "numpy" not in sys.modules
assert array.device == "cpu"
assert array.device_id == 0
assert array.dtype == "float64"
assert array.byte_order == sys.byteorder
assert array.shape == (3,)
assert array.strides == (8,)
assert array.c_contiguous and array.aligned and array.readonly
assert array.ownership == "owned"
assert not array.origin_copy_occurred

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
view = array.numpy(copy=False)
sys.meta_path.remove(finder)
assert finder.attempted

import numpy as np

assert view is array.numpy(copy=None)
assert not view.flags.owndata
assert not view.flags.writeable
try:
    view.setflags(write=True)
except ValueError:
    pass
else:
    raise AssertionError("immutable Result storage became writeable")

copied = array.numpy(copy=True)
assert copied.flags.writeable
assert not np.shares_memory(copied, view)
original = view.copy()
copied[0] = 9.0
assert view[0] == original[0]

snapshot = np.from_dlpack(array)
assert snapshot.ctypes.data != view.ctypes.data, (
    snapshot.ctypes.data,
    view.ctypes.data,
    snapshot.base,
)
assert not np.shares_memory(snapshot, view)
if snapshot.flags.writeable:
    snapshot[0] = 11.0
else:
    try:
        snapshot.setflags(write=True)
    except ValueError:
        pass
    else:
        raise AssertionError("a read-only DLPack snapshot became writeable")
assert view[0] == original[0]

second_snapshot = np.from_dlpack(array, device="cpu")
assert second_snapshot.ctypes.data != view.ctypes.data
assert second_snapshot.ctypes.data != snapshot.ctypes.data
assert not np.shares_memory(second_snapshot, view)
assert not np.shares_memory(second_snapshot, snapshot)
try:
    np.from_dlpack(array, copy=False)
except BufferError:
    pass
else:
    raise AssertionError("DLPack copy=False exposed immutable Result storage")

try:
    array.__dlpack__()
except BufferError:
    pass
else:
    raise AssertionError("legacy DLPack export was accepted")

for kwargs in (
    {"stream": 1, "max_version": (1, 0)},
    {"max_version": (1, 0), "dl_device": (2, 0)},
    {"max_version": (2, 0)},
):
    try:
        array.__dlpack__(**kwargs)
    except BufferError:
        pass
    else:
        raise AssertionError(f"unsupported DLPack request was accepted: {kwargs}")

capsule = array.__dlpack__(
    max_version=(1, 0), dl_device=(1, 0), copy=True
)

class SingleUseProducer:
    def __dlpack_device__(self):
        return (1, 0)

    def __dlpack__(self, **_kwargs):
        return capsule

producer = SingleUseProducer()
consumed = np.from_dlpack(producer)
np.testing.assert_array_equal(consumed, view)
try:
    np.from_dlpack(producer)
except ValueError:
    pass
else:
    raise AssertionError("one DLPack capsule was consumed twice")

del result, array
gc.collect()
np.testing.assert_allclose(
    view,
    np.exp(-np.array([0.05, 0.1, 0.2])),
    rtol=1.0e-8,
    atol=1.0e-10,
)
np.testing.assert_allclose(snapshot[1:], view[1:])
"#
            ),
            Some(&locals),
            Some(&locals),
        )?;
        Ok(())
    })
}

fn public_module(py: Python<'_>) -> PyResult<Bound<'_, PyModule>> {
    let native = pyo3::wrap_pymodule!(_eqiora::_eqiora)(py);
    let package_directory = Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../../bindings/python/python/eqiora")
        .canonicalize()?;
    let locals = PyDict::new(py);
    locals.set_item("native", native.bind(py))?;
    locals.set_item("package_directory", package_directory.to_string_lossy())?;
    py.run(
        c_str!(
            r#"
import importlib.util
import pathlib
import sys

package_path = pathlib.Path(package_directory)
spec = importlib.util.spec_from_file_location(
    "eqiora",
    package_path / "__init__.py",
    submodule_search_locations=[str(package_path)],
)
assert spec is not None and spec.loader is not None
package = importlib.util.module_from_spec(spec)
sys.modules["eqiora"] = package
sys.modules["eqiora._eqiora"] = native
spec.loader.exec_module(package)
"#
        ),
        None,
        Some(&locals),
    )?;
    Ok(locals
        .get_item("package")?
        .expect("the package loader must bind eqiora")
        .cast_into::<PyModule>()?)
}
