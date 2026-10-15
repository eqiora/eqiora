//! Python projections of the shared complete mathematical value owner.

use eqiora::{ScalarDomain, ValueLiteral, ValueType};
use pyo3::IntoPyObjectExt;
use pyo3::exceptions::{PyTypeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::{PyBool, PyComplex, PyFloat, PyInt, PyList, PyTuple};

pub(crate) fn scalar(value: &Bound<'_, PyAny>) -> PyResult<(f64, f64)> {
    if let Ok(value) = value.cast::<PyComplex>() {
        return Ok((value.real(), value.imag()));
    }
    if value.is_instance_of::<PyBool>()
        || !(value.is_instance_of::<PyFloat>() || value.is_instance_of::<PyInt>())
    {
        return Err(PyTypeError::new_err(
            "value components must be real or complex numbers, not bool",
        ));
    }
    Ok((value.extract()?, 0.0))
}

pub(crate) fn integer<'py>(value: &Bound<'py, PyAny>) -> PyResult<Bound<'py, PyInt>> {
    if !value.is_instance_of::<PyBool>()
        && let Ok(value) = value.cast::<PyInt>()
    {
        return Ok(value.clone());
    }
    Err(PyTypeError::new_err(
        "integer values require exact int components, not bool or float",
    ))
}

fn collect<T>(
    value: &Bound<'_, PyAny>,
    extents: &[std::num::NonZeroU32],
    output: &mut Vec<T>,
    component: &impl Fn(&Bound<'_, PyAny>) -> PyResult<T>,
) -> PyResult<()> {
    let Some((extent, tail)) = extents.split_first() else {
        output.push(component(value)?);
        return Ok(());
    };
    if !(value.is_instance_of::<PyList>() || value.is_instance_of::<PyTuple>()) {
        return Err(PyValueError::new_err(
            "value must match every declared shape axis; scalar broadcasting is not admitted",
        ));
    }
    if value.len()? != extent.get() as usize {
        return Err(PyValueError::new_err(
            "value shape differs from the complete declared type",
        ));
    }
    for item in value.try_iter()? {
        collect(&item?, tail, output, component)?;
    }
    Ok(())
}

pub(crate) fn from_python(
    value: &Bound<'_, PyAny>,
    value_type: ValueType,
) -> PyResult<ValueLiteral> {
    eqiora::language::ValueTypeSyntax::validate_checked(&value_type)
        .map_err(|error| PyValueError::new_err(error.to_string()))?;
    if value_type.scalar_domain() == ScalarDomain::Enum {
        let value = value
            .extract::<PyRef<'_, super::enumeration::PyEnumValue>>()
            .map_err(|_| {
                PyTypeError::new_err(
                    "enum values require an exact EnumValue, not strings or numbers",
                )
            })?;
        if value.value.value_type() != &value_type {
            return Err(PyValueError::new_err(
                "enum value belongs to a different nominal type",
            ));
        }
        return Ok(value.value.clone());
    }
    if value_type.scalar_domain() == ScalarDomain::Boolean {
        if !value.is_instance_of::<PyBool>() {
            return Err(PyTypeError::new_err(
                "Boolean values require bool, not numeric values",
            ));
        }
        return Ok(ValueLiteral::boolean(value.extract()?));
    }
    if value_type.scalar_domain() == ScalarDomain::Integer {
        let integer = |value: &Bound<'_, PyAny>| -> PyResult<i64> { integer(value)?.extract() };
        let result = if value.is_instance_of::<PyList>() || value.is_instance_of::<PyTuple>() {
            let mut components = Vec::new();
            collect(
                value,
                value_type.shape().extents(),
                &mut components,
                &integer,
            )?;
            ValueLiteral::integer(value_type, components)
        } else {
            ValueLiteral::from_integer(value_type, integer(value)?)
        };
        return result.map_err(|error| PyValueError::new_err(error.to_string()));
    }
    if !value.is_instance_of::<PyList>() && !value.is_instance_of::<PyTuple>() {
        let (real, imaginary) = scalar(value)?;
        let result = if imaginary == 0.0 {
            ValueLiteral::from_real(value_type, real)
        } else {
            ValueLiteral::new(value_type, [(real, imaginary)])
        };
        return result.map_err(|error| PyValueError::new_err(error.to_string()));
    }

    let mut components = Vec::new();
    collect(
        value,
        value_type.shape().extents(),
        &mut components,
        &scalar,
    )?;
    ValueLiteral::new(value_type, components)
        .map_err(|error| PyValueError::new_err(error.to_string()))
}

pub(crate) fn to_python(py: Python<'_>, value: &ValueLiteral) -> PyResult<Py<PyAny>> {
    eqiora::language::ValueTypeSyntax::validate_checked(value.value_type())
        .map_err(|error| PyValueError::new_err(error.to_string()))?;
    if value.enum_tag().is_some() {
        return Py::new(
            py,
            super::enumeration::PyEnumValue {
                value: value.clone(),
            },
        )
        .map(Py::into_any);
    }
    if let Some(value) = value.as_bool() {
        return value.into_py_any(py);
    }
    fn nested(
        py: Python<'_>,
        value: &ValueLiteral,
        axis: usize,
        offset: &mut usize,
    ) -> PyResult<Py<PyAny>> {
        if let Some(extent) = value.value_type().shape().extents().get(axis) {
            let items = (0..extent.get())
                .map(|_| nested(py, value, axis + 1, offset))
                .collect::<PyResult<Vec<_>>>()?;
            return Ok(PyTuple::new(py, items)?.into_any().unbind());
        }
        if value.value_type().scalar_domain() == ScalarDomain::Integer {
            let integer = value
                .integer_component(*offset)
                .expect("bounded integer component");
            *offset += 1;
            return integer.into_py_any(py);
        }
        let (real, imaginary) = value.component(*offset).expect("bounded component");
        *offset += 1;
        if value.value_type().scalar_domain() == ScalarDomain::Complex {
            Ok(PyComplex::from_doubles(py, real, imaginary)
                .into_any()
                .unbind())
        } else {
            real.into_py_any(py)
        }
    }
    nested(py, value, 0, &mut 0)
}

pub(crate) fn expression(value: &Bound<'_, PyAny>) -> PyResult<eqiora::language::DraftExpression> {
    fn build(
        value: &Bound<'_, PyAny>,
        depth: usize,
        nodes: &mut usize,
    ) -> PyResult<eqiora::language::DraftExpression> {
        use eqiora::language::DraftExpression;
        *nodes += 1;
        if depth > 64 || *nodes > 4096 {
            return Err(PyValueError::new_err(
                "typed input exceeds expression depth or node bounds",
            ));
        }
        if value.is_instance_of::<PyList>() || value.is_instance_of::<PyTuple>() {
            if value.len()? == 0 {
                return Err(PyValueError::new_err("channel arrays must be nonempty"));
            }
            let items = value
                .try_iter()?
                .map(|item| build(&item?, depth + 1, nodes))
                .collect::<PyResult<Vec<_>>>()?;
            return Ok(DraftExpression::array(items));
        }
        if value.is_instance_of::<PyBool>() {
            return Ok(DraftExpression::boolean(value.extract()?));
        }
        if value.is_instance_of::<PyInt>() {
            let bits: usize = value.call_method0("bit_length")?.extract()?;
            if bits > 851 {
                return Err(PyValueError::new_err(
                    "numeric literal exceeds the 256-byte decimal limit",
                ));
            }
            let literal = eqiora::language::DecimalLiteral::parse(value.str()?.to_str()?)
                .map_err(|error| PyValueError::new_err(error.to_string()))?;
            return Ok(DraftExpression::constant(literal));
        }
        let (real, imaginary) = scalar(value)?;
        if !real.is_finite() || !imaginary.is_finite() {
            return Err(PyValueError::new_err(
                "typed input components must be finite",
            ));
        }
        if value.is_instance_of::<PyComplex>() {
            Ok(DraftExpression::complex(real, imaginary))
        } else {
            Ok(DraftExpression::constant(
                eqiora::language::DecimalLiteral::from_f64(real)
                    .map_err(|error| PyValueError::new_err(error.to_string()))?,
            ))
        }
    }
    build(value, 1, &mut 0)
}
