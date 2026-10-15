//! Numeric input projection through the source compiler's exact unit owner.
use super::PyAstExpression;
use super::expression::syntax_error;
use crate::modeling::PyDimension;
use eqiora::compiler::InputUnitCatalog;
use eqiora::language::DecimalLiteral;
use pyo3::prelude::*;

#[pyfunction]
pub(super) fn _normalize_input_quantity(
    text: &str,
    unit: &PyAstExpression,
) -> PyResult<(f64, PyDimension)> {
    let value = DecimalLiteral::parse(text).map_err(syntax_error)?;
    let quantity = InputUnitCatalog::quantity(&value, &unit.value).map_err(syntax_error)?;
    Ok((quantity.value(), PyDimension::from_native(quantity.dim())))
}
