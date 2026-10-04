//! Native Observable declarations and spatial reduction expressions.
use super::*;
use eqiora::language::DraftObservable;

/// Typed derived declaration that adds no solve unknown.
#[pyclass(
    name = "Observable",
    module = "eqiora._eqiora",
    frozen,
    skip_from_py_object
)]
#[derive(Debug, Clone)]
pub(super) struct PyObservable {
    pub(super) value: DraftObservable,
}

#[pymethods]
impl PyObservable {
    fn __repr__(&self) -> String {
        format!("Observable(name={:?})", self.value.name())
    }

    #[new]
    #[pyo3(signature = (name, *, value_type, expression))]
    fn new(
        name: String,
        value_type: &PyValueType,
        expression: &Bound<'_, PyAny>,
    ) -> PyResult<Self> {
        Ok(Self {
            value: DraftObservable::new(
                name,
                value_type.value.clone(),
                expression_from_python(expression)?,
            ),
        })
    }

    #[getter]
    fn name(&self) -> &str {
        self.value.name()
    }

    #[getter]
    fn value_type(&self) -> PyValueType {
        PyValueType {
            value: self.value.value_type().clone(),
        }
    }

    #[getter]
    fn expression(&self) -> PyExpression {
        PyExpression::new(self.value.expression().clone())
    }
    fn __neg__(&self) -> PyExpression {
        PyExpression::new(-DraftExpression::observable(&self.value))
    }

    fn __bool__(&self) -> PyResult<bool> {
        Err(PyTypeError::new_err(
            "symbolic Eqiora expressions have no truth value",
        ))
    }

    fn __add__(&self, other: &Bound<'_, PyAny>) -> PyResult<PyExpression> {
        binary(
            DraftExpression::observable(&self.value),
            other,
            Binary::Add,
            false,
        )
    }

    fn __radd__(&self, other: &Bound<'_, PyAny>) -> PyResult<PyExpression> {
        binary(
            DraftExpression::observable(&self.value),
            other,
            Binary::Add,
            true,
        )
    }

    fn __sub__(&self, other: &Bound<'_, PyAny>) -> PyResult<PyExpression> {
        binary(
            DraftExpression::observable(&self.value),
            other,
            Binary::Subtract,
            false,
        )
    }

    fn __rsub__(&self, other: &Bound<'_, PyAny>) -> PyResult<PyExpression> {
        binary(
            DraftExpression::observable(&self.value),
            other,
            Binary::Subtract,
            true,
        )
    }

    fn __mul__(&self, other: &Bound<'_, PyAny>) -> PyResult<PyExpression> {
        binary(
            DraftExpression::observable(&self.value),
            other,
            Binary::Multiply,
            false,
        )
    }

    fn __rmul__(&self, other: &Bound<'_, PyAny>) -> PyResult<PyExpression> {
        binary(
            DraftExpression::observable(&self.value),
            other,
            Binary::Multiply,
            true,
        )
    }

    fn __truediv__(&self, other: &Bound<'_, PyAny>) -> PyResult<PyExpression> {
        binary(
            DraftExpression::observable(&self.value),
            other,
            Binary::Divide,
            false,
        )
    }

    fn __rtruediv__(&self, other: &Bound<'_, PyAny>) -> PyResult<PyExpression> {
        binary(
            DraftExpression::observable(&self.value),
            other,
            Binary::Divide,
            true,
        )
    }
}

#[pyfunction]
fn integral(value: &Bound<'_, PyAny>, measure: &Bound<'_, PyAny>) -> PyResult<PyExpression> {
    Ok(PyExpression::new(DraftExpression::integral(
        expression_from_python(value)?,
        expression_from_python(measure)?,
    )))
}

#[pyfunction]
fn measure(domain: &PyDomain) -> PyExpression {
    PyExpression::new(DraftExpression::measure(&domain.value))
}

pub(super) fn register(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_class::<PyObservable>()?;
    module.add_function(wrap_pyfunction!(integral, module)?)?;
    module.add_function(wrap_pyfunction!(measure, module)?)?;
    Ok(())
}
