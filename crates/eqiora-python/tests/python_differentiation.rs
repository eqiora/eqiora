mod support;

use pyo3::ffi::c_str;
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyDictMethods};

#[test]
fn python_differentiable_program_is_exact_paired_and_fail_closed() -> PyResult<()> {
    Python::initialize();
    Python::attach(|py| {
        let locals = PyDict::new(py);
        locals.set_item("eqiora", support::public_module(py)?)?;
        locals.set_item(
            "tests",
            include_str!("../../../bindings/python/tests/_differentiation_contract.py"),
        )?;
        locals.set_item(
            "complex_tests",
            include_str!("../../../bindings/python/tests/test_complex_differentiation.py"),
        )?;
        py.run(
            c_str!(
                r#"
exec(tests, globals())
for method in (eqiora.fem.Q1(), eqiora.fvm.CellCenteredTpfa()):
    check_complete_field_actions(method)
exec(complex_tests, globals())
test_common_complex_sensitivity_uses_real_pairing_at_each_parameter_point()
test_common_typed_parameter_coordinates_preserve_selected_order_and_complex_parts()
test_structured_outputs_keep_complex_fields_loss_and_margin_associated()
test_structured_outputs_preserve_intrinsic_shapes_and_distinct_units()
"#
            ),
            Some(&locals),
            Some(&locals),
        )
    })
}
