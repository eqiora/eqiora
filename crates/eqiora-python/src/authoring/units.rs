//! Numeric input projection through the source compiler's exact unit owner.
use super::PyAstExpression;
use super::expression::syntax_error;
use crate::modeling::PyDimension;
use eqiora::compiler::UnitCatalog;
use eqiora::language::DecimalLiteral;
use pyo3::prelude::*;

#[pyfunction]
pub(super) fn _normalize_input_quantity(
    text: &str,
    unit: &PyAstExpression,
) -> PyResult<(f64, PyDimension)> {
    let value = DecimalLiteral::parse(text).map_err(syntax_error)?;
    let quantity = UnitCatalog::quantity(&value, &unit.value).map_err(syntax_error)?;
    Ok((quantity.value(), PyDimension::from_native(quantity.dim())))
}

/// Project an existing coherent value without a second numeric coercion in Python.
#[pyfunction]
pub(super) fn _present_quantity(
    value: &Bound<'_, PyAny>,
    dimension: &PyDimension,
    unit: &PyAstExpression,
    decimal_places: u8,
    locale: &str,
) -> PyResult<(f64, f64, String, &'static str)> {
    let stored = crate::modeling::value_literal::real(value)?;
    let quantity = eqiora::DynQuantity::new(stored, dimension.value);
    let (displayed, text) =
        UnitCatalog::present_quantity(quantity, &unit.value, decimal_places, locale)
            .map_err(pyo3::exceptions::PyValueError::new_err)?;
    Ok((stored, displayed, text, UnitCatalog::PRESENTATION_ROUNDING))
}
