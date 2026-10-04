use super::*;

pub(super) fn declaration_from_python(value: &Bound<'_, PyAny>) -> PyResult<DraftDeclaration> {
    if let Ok(definition) = value.extract::<PyRef<'_, enumeration::PyEnum>>() {
        let name = definition.name.clone().ok_or_else(|| {
            PyValueError::new_err("native enum declarations require an authored lexical name")
        })?;
        return Ok(DraftDeclaration::Enum {
            name,
            definition: definition.value.clone(),
        });
    }
    if let Ok(space) = value.extract::<PyRef<'_, nominal::PyFiniteSpace>>() {
        return Ok(DraftDeclaration::FiniteSpace {
            name: space.name.clone(),
            definition: space.value.clone(),
        });
    }
    if let Ok(set) = value.extract::<PyRef<'_, nominal::PyIndexSet>>() {
        return Ok(DraftDeclaration::IndexSet {
            name: set.name.clone(),
            definition: set.value.clone(),
        });
    }
    if let Ok(domain) = value.extract::<PyRef<'_, PyDomain>>() {
        return Ok(domain.value.clone().into());
    }
    if let Ok(initial) = value.extract::<PyRef<'_, PyInitial>>() {
        return Ok(DraftDeclaration::Initial(initial.equations.clone()));
    }
    if let Ok(field) = value.extract::<PyRef<'_, PyField>>() {
        return Ok(field.value.clone().into());
    }
    if let Ok(parameter) = value.extract::<PyRef<'_, PyParameter>>() {
        return Ok(parameter.value.clone().into());
    }
    if let Ok(domain) = value.extract::<PyRef<'_, PyPhysicalDomain>>() {
        return Ok(domain.value.clone().into());
    }
    if let Ok(port) = value.extract::<PyRef<'_, PyConservingPort>>() {
        return Ok(port.value.clone().into());
    }
    if let Ok(observable) = value.extract::<PyRef<'_, observable::PyObservable>>() {
        return Ok(observable.value.clone().into());
    }
    if let Ok(relation) = value.extract::<PyRef<'_, PyRelation>>() {
        return Ok(relation.value.clone().into());
    }
    if let Ok(connection) = value.extract::<PyRef<'_, PyConnection>>() {
        return Ok(connection.value.clone().into());
    }
    Err(PyTypeError::new_err(
        "Module arguments must be model declaration objects",
    ))
}

pub(super) fn expression_from_python(value: &Bound<'_, PyAny>) -> PyResult<DraftExpression> {
    if let Ok(value) = value.extract::<PyRef<'_, enumeration::PyEnumValue>>() {
        return DraftExpression::enum_value(value.value.clone())
            .map_err(|error| PyValueError::new_err(error.to_string()));
    }
    if let Ok(expression) = value.extract::<PyRef<'_, PyExpression>>() {
        return Ok(expression.value.clone());
    }
    if let Ok(observable) = value.extract::<PyRef<'_, observable::PyObservable>>() {
        return Ok(DraftExpression::observable(&observable.value));
    }
    if let Ok(field) = value.extract::<PyRef<'_, PyField>>() {
        return Ok(field.value.expression());
    }
    if let Ok(parameter) = value.extract::<PyRef<'_, PyParameter>>() {
        return Ok(parameter.value.expression());
    }
    if let Ok(value) = value.cast::<PyComplex>() {
        return Ok(DraftExpression::complex(value.real(), value.imag()));
    }
    if value.is_instance_of::<PyBool>() {
        return Ok(DraftExpression::boolean(value.extract()?));
    }
    if value.is_instance_of::<PyInt>() {
        return value_literal::expression(value);
    }
    value
        .extract::<f64>()
        .map_err(|_| expression_type_error())
        .and_then(|value| {
            eqiora::language::DecimalLiteral::from_f64(value)
                .map_err(|error| PyValueError::new_err(error.to_string()))
        })
        .map(DraftExpression::constant)
}

#[derive(Debug, Clone, Copy)]
pub(super) enum Binary {
    Add,
    Subtract,
    Multiply,
    Divide,
}

pub(super) fn binary(
    own: DraftExpression,
    other: &Bound<'_, PyAny>,
    operator: Binary,
    reverse: bool,
) -> PyResult<PyExpression> {
    let other = expression_from_python(other)?;
    let (left, right) = if reverse { (other, own) } else { (own, other) };
    let value = match operator {
        Binary::Add => left + right,
        Binary::Subtract => left - right,
        Binary::Multiply => left * right,
        Binary::Divide => left / right,
    };
    Ok(PyExpression::new(value))
}

fn expression_type_error() -> PyErr {
    PyTypeError::new_err(
        "expected an Expression, Field, Parameter, Observable, or real/complex number",
    )
}

pub(super) fn equation_pairs(
    values: &Bound<'_, PyAny>,
) -> PyResult<Vec<(DraftExpression, DraftExpression)>> {
    if !(values.is_instance_of::<PyTuple>() || values.is_instance_of::<PyList>()) {
        return Err(PyTypeError::new_err(
            "equations must be an ordered tuple or list of pairs",
        ));
    }
    values
        .try_iter()?
        .map(|pair| {
            let pair = pair?;
            if !(pair.is_instance_of::<PyTuple>() || pair.is_instance_of::<PyList>())
                || pair.len()? != 2
            {
                return Err(PyTypeError::new_err(
                    "each equation requires exactly two explicit sides",
                ));
            }
            Ok((
                expression_from_python(&pair.get_item(0)?)?,
                expression_from_python(&pair.get_item(1)?)?,
            ))
        })
        .collect()
}
