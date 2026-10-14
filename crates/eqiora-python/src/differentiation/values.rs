//! Python leaf transport projects the native value signatures once.

use super::*;
use eqiora::api::DifferentiableValues;
use pyo3::types::{PySequence, PyTuple};

pub(super) fn stage_parameters(
    py: Python<'_>,
    program: &DifferentiableProgram,
    value: &Bound<'_, PyAny>,
) -> PyResult<DifferentiableValues> {
    let inputs = program.identity().inputs();
    let sequence = value.cast::<PySequence>()?;
    if sequence.len()? != inputs.len() {
        return Err(pyo3::exceptions::PyBufferError::new_err(format!(
            "parameters require {} ordered value leaves",
            inputs.len()
        )));
    }
    let components = inputs
        .iter()
        .enumerate()
        .map(|(index, signature)| {
            crate::array::stage_value_components(
                py,
                &sequence.get_item(index)?,
                signature.mathematical_type().value_type.scalar_domain(),
                signature.shape(),
                &format!("parameter leaf {index}"),
            )
        })
        .collect::<PyResult<Vec<_>>>()?;
    DifferentiableValues::from_components(
        inputs,
        &components.iter().map(Vec::as_slice).collect::<Vec<_>>(),
    )
    .map_err(|diagnostic| diagnostic_error(py, &[diagnostic]))
}

/// Publish one immutable numerical leaf per exact native signature.
pub(super) fn project(py: Python<'_>, values: &DifferentiableValues) -> PyResult<Py<PyTuple>> {
    let leaves = (0..values.signatures().len())
        .map(|index| {
            let leaf = values.leaf(index).expect("selected leaf");
            let buffer = match leaf
                .signature()
                .mathematical_type()
                .value_type
                .scalar_domain()
            {
                eqiora::ScalarDomain::Real => PyArrayBuffer::from_owned_result(
                    py,
                    leaf.components().map(|(real, _)| real).collect(),
                )?,
                eqiora::ScalarDomain::Complex => PyArrayBuffer::from_owned_complex_result(
                    py,
                    leaf.components()
                        .map(|(real, imaginary)| numpy::Complex64::new(real, imaginary))
                        .collect(),
                )?,
                _ => unreachable!("admitted differentiable scalar domain"),
            };
            buffer
                .bind(py)
                .borrow()
                .numpy_shaped(py, leaf.signature().shape())
        })
        .collect::<PyResult<Vec<_>>>()?;
    Ok(PyTuple::new(py, leaves)?.unbind())
}
