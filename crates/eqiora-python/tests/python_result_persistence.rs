use std::path::Path;

use pyo3::ffi::c_str;
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyDictMethods, PyModule};

#[test]
fn python_ode_result_round_trips_complete_canonical_content() -> PyResult<()> {
    Python::initialize();
    Python::attach(|py| {
        let module = public_module(py)?;
        let locals = PyDict::new(py);
        locals.set_item("eqiora", module)?;
        py.run(
            c_str!(
                r#"
import os
import pathlib
import tempfile

source = """
model result_decay() {
  state x: 1;
  initial { x = 1; }
  parameter rate: 1 / s = 1;
  relation flow {
    derivative(x) + rate * x = 0;
  }
}
"""

model = eqiora.compile(source=source)
field = model.field(model.field_ids[0])
plan = eqiora.resolve(
    model,
    temporal=eqiora.time.Tsitouras45(
        initial_step_s=0.01,
        relative_tolerance=1.0e-9,
        absolute_tolerances={field: 1.0e-11},
    ),
)
result = eqiora.run(
    plan,
    state=eqiora.State.initial(plan),
    until_s=0.2,
    output_times_s=(0.1, 0.2),
)
result_bytes = result.to_bytes()
assert b"eqiora.common-result/v5" in result_bytes
replayed = eqiora.Result.from_bytes(plan, result_bytes)
assert replayed.to_bytes() == result_bytes
assert replayed.plan_key == result.plan_key
assert replayed.elapsed_seconds == result.elapsed_seconds
assert replayed.series(field).values.numpy().tolist() == result.series(field).values.numpy().tolist()

result_directory_owner = tempfile.TemporaryDirectory()
result_directory = pathlib.Path(result_directory_owner.name)
result_path = result_directory / "run.eqresult"
result_path.write_bytes(b"incomplete previous output")
result.write(result_path)
assert result_path.read_bytes() == result_bytes
assert not list(result_directory.glob(".eqiora-result-*.tmp"))
reopened = eqiora.Result.read(plan, result_path)
assert reopened.to_bytes() == result_bytes
assert reopened.plan_key == result.plan_key
assert reopened.series(field).values.numpy().tolist() == result.series(field).values.numpy().tolist()

for name, rejected in (
    ("truncated.eqresult", result_bytes[:-1]),
    ("trailing.eqresult", result_bytes + b"\n"),
    ("unknown-version.eqresult", result_bytes.replace(b"common-result/v5", b"common-result/v9")),
    ("retired-version.eqresult", result_bytes.replace(b"common-result/v5", b"common-result/v2")),
):
    assert rejected != result_bytes
    path = result_directory / name
    path.write_bytes(rejected)
    try:
        eqiora.Result.read(plan, path)
    except eqiora.CompatibilityError:
        pass
    else:
        raise AssertionError(f"hostile Result file must reject: {name}")

wrong_suffix = result_directory / "run.json"
for operation in (
    lambda: result.write(wrong_suffix),
    lambda: eqiora.Result.read(plan, wrong_suffix),
):
    try:
        operation()
    except eqiora.CompatibilityError:
        pass
    else:
        raise AssertionError("Result file paths require the exact .eqresult suffix")

directory_path = result_directory / "directory.eqresult"
directory_path.mkdir()
for operation in (
    lambda: result.write(directory_path),
    lambda: eqiora.Result.read(plan, directory_path),
):
    try:
        operation()
    except eqiora.CompatibilityError:
        pass
    else:
        raise AssertionError("Result file I/O must reject non-regular paths")

if os.name != "nt":
    target = result_directory / "target.eqresult"
    target.write_bytes(result_bytes)
    symlink = result_directory / "symlink.eqresult"
    symlink.symlink_to(target)
    for operation in (
        lambda: result.write(symlink),
        lambda: eqiora.Result.read(plan, symlink),
    ):
        try:
            operation()
        except eqiora.CompatibilityError:
            pass
        else:
            raise AssertionError("Result file I/O must reject symlinks")

oversized = result_directory / "oversized.eqresult"
with oversized.open("wb") as stream:
    stream.truncate(512 * 1024 * 1024 + 1)
try:
    eqiora.Result.read(plan, oversized)
except eqiora.CompatibilityError:
    pass
else:
    raise AssertionError("Result pre-read must enforce the decoder bound")

try:
    eqiora.Result.from_bytes(plan, result_bytes + b"\n")
except eqiora.ValidationError:
    pass
else:
    raise AssertionError("noncanonical Result bytes were admitted")

changed = eqiora.resolve(
    model,
    temporal=eqiora.time.Tsitouras45(
        initial_step_s=0.02,
        relative_tolerance=1.0e-8,
        absolute_tolerances={field: 2.0e-11},
    ),
)
try:
    eqiora.Result.from_bytes(changed, result_bytes)
except eqiora.ValidationError:
    pass
else:
    raise AssertionError("Result bytes were crossed with a different Plan")
try:
    eqiora.Result.read(changed, result_path)
except eqiora.CompatibilityError:
    pass
else:
    raise AssertionError("Result file was crossed with a different Plan")
result_directory_owner.cleanup()
"#
            ),
            Some(&locals),
            Some(&locals),
        )
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
