//! Python leaf transport projects the native value signatures once.

use super::*;
use eqiora::api::DifferentiableValues;
use pyo3::types::PySequence;

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
