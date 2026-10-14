mod support;

use pyo3::ffi::c_str;
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyDictMethods};

#[test]
fn python_complex_newton_uses_the_installed_public_test() -> PyResult<()> {
    Python::initialize();
    Python::attach(|py| {
        let locals = PyDict::new(py);
        locals.set_item("eqiora", support::public_module(py)?)?;
        locals.set_item(
            "tests",
            include_str!("../../../bindings/python/tests/test_complex_newton.py"),
        )?;
        py.run(
            c_str!(
                r#"
exec(tests, globals())
test_complex_newton_retains_shaped_seeds_and_exact_replay()
"#
            ),
            Some(&locals),
            None,
        )
    })
}
