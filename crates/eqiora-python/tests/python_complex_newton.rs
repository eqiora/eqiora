use pyo3::ffi::c_str;
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyDictMethods};

#[test]
fn python_complex_newton_uses_the_installed_public_test() -> PyResult<()> {
    Python::initialize();
    Python::attach(|py| {
        let native = pyo3::wrap_pymodule!(_eqiora::_eqiora)(py);
        let package = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("../../bindings/python/python/eqiora")
            .canonicalize()?;
        let locals = PyDict::new(py);
        locals.set_item("native", native.bind(py))?;
        locals.set_item("package_directory", package.to_string_lossy())?;
        locals.set_item(
            "tests",
            include_str!("../../../bindings/python/tests/test_complex_newton.py"),
        )?;
        py.run(c_str!(r#"
import importlib.util, pathlib, sys
path = pathlib.Path(package_directory)
spec = importlib.util.spec_from_file_location("eqiora", path / "__init__.py", submodule_search_locations=[str(path)])
eqiora = importlib.util.module_from_spec(spec)
sys.modules["eqiora"] = eqiora
sys.modules["eqiora._eqiora"] = native
spec.loader.exec_module(eqiora)
exec(tests, globals())
test_complex_newton_retains_shaped_seeds_and_exact_replay()
"#), Some(&locals), None)
    })
}
