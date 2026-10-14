mod support;
use support::public_module;

use pyo3::ffi::c_str;
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyDictMethods};

#[test]
fn python_hermitian_plan_run_result_and_replay() -> PyResult<()> {
    Python::initialize();
    Python::attach(|py| {
        let module = public_module(py)?;
        let locals = PyDict::new(py);
        locals.set_item("eqiora", &module)?;
        let source = std::ffi::CString::new(include_str!(
            "../../../bindings/python/tests/test_hermitian_eigen.py"
        ))
        .unwrap();
        py.run(&source, Some(&locals), Some(&locals))?;
        py.run(
            c_str!("test_python_hermitian_plan_run_result_and_replay(); test_python_source_coordinate_embedding_and_original_residual(); test_python_authored_finite_weak_form_execution_and_replay()"),
            Some(&locals),
            Some(&locals),
        )
    })
}
