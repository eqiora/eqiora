//! Python leaf transport projects the native value signatures once.

use super::*;
use eqiora::api::{DifferentiableValueSignature, DifferentiableValues};
use pyo3::types::{PySequence, PyTuple};

/// Bound the minimum typed inventory charge before staging dense occurrences.
/// Native map admission still owns the full retained-state storage estimate.
pub(crate) fn points_from_coordinates(
    signatures: &[DifferentiableValueSignature],
    coordinates: &[f64],
    count: usize,
    bytes_limit: usize,
) -> Result<Vec<DifferentiableValues>, eqiora::Diagnostic> {
    let invalid = || {
        eqiora::Diagnostic::error(
            eqiora::diagnostic::codes::INVALID_LINEARIZATION,
            "dense map point inventory exceeds its byte limit or coordinate shape",
        )
    };
    let bytes = count
        .checked_mul(size_of::<DifferentiableValues>())
        .and_then(|bytes| {
            coordinates
                .len()
                .checked_mul(size_of::<f64>())?
                .checked_add(bytes)
        })
        .ok_or_else(invalid)?;
    if bytes > bytes_limit
        || (count == 0 && !coordinates.is_empty())
        || (count != 0 && !coordinates.len().is_multiple_of(count))
    {
        return Err(invalid());
    }
    let width = coordinates.len().checked_div(count).unwrap_or(0);
    (0..count)
        .map(|index| from_coordinates(signatures, &coordinates[index * width..(index + 1) * width]))
        .collect()
}

/// Admit the existing dense map/FFI transport at its private adapter boundary.
/// Logical components are bound through the same native value constructor as
/// ordinary Python leaves; no flat coordinates cross the public map API.
pub(crate) fn from_coordinates(
    signatures: &[DifferentiableValueSignature],
    coordinates: &[f64],
) -> Result<DifferentiableValues, eqiora::Diagnostic> {
    let invalid = || {
        eqiora::Diagnostic::error(
            eqiora::diagnostic::codes::INVALID_LINEARIZATION,
            "dense transport differs from the exact value signatures",
        )
    };
    let mut remaining = coordinates.iter().copied();
    let mut leaves = Vec::with_capacity(signatures.len());
    for signature in signatures {
        let count = signature
            .shape()
            .iter()
            .try_fold(1usize, |count, &extent| {
                count.checked_mul(extent).ok_or_else(invalid)
            })?;
        if count > remaining.len() {
            return Err(invalid());
        }
        let complex = signature.mathematical_type().value_type.scalar_domain()
            == eqiora::ScalarDomain::Complex;
        let mut components = Vec::with_capacity(count);
        for _ in 0..count {
            let real = remaining.next().ok_or_else(invalid)?;
            let imaginary = if complex {
                remaining.next().ok_or_else(invalid)?
            } else {
                0.0
            };
            components.push((real, imaginary));
        }
        leaves.push(components);
    }
    if remaining.next().is_some() {
        return Err(invalid());
    }
    DifferentiableValues::from_components(
        signatures,
        &leaves.iter().map(Vec::as_slice).collect::<Vec<_>>(),
    )
}

/// Stage each leaf against its base signature and construct its mathematical role.
pub(super) fn stage<T>(
    py: Python<'_>,
    signatures: &[DifferentiableValueSignature],
    value: &Bound<'_, PyAny>,
    role: &str,
    bind: impl FnOnce(
        &[DifferentiableValueSignature],
        &[&[(f64, f64)]],
    ) -> Result<T, eqiora::Diagnostic>,
) -> PyResult<T> {
    let sequence = value.cast::<PySequence>()?;
    if sequence.len()? != signatures.len() {
        return Err(pyo3::exceptions::PyBufferError::new_err(format!(
            "{role} requires {} ordered value leaves",
            signatures.len()
        )));
    }
    let components = signatures
        .iter()
        .enumerate()
        .map(|(index, signature)| {
            crate::array::stage_value_components(
                py,
                &sequence.get_item(index)?,
                signature.mathematical_type().value_type.scalar_domain(),
                signature.shape(),
                &format!("{role} leaf {index}"),
            )
        })
        .collect::<PyResult<Vec<_>>>()?;
    bind(
        signatures,
        &components.iter().map(Vec::as_slice).collect::<Vec<_>>(),
    )
    .map_err(|diagnostic| diagnostic_error(py, &[diagnostic]))
}

/// Publish primal leaves through their existing semantic value view.
pub(super) fn project(py: Python<'_>, values: &DifferentiableValues) -> PyResult<Py<PyTuple>> {
    let leaves = (0..values.signatures().len())
        .map(|index| {
            let leaf = values.leaf(index).expect("admitted leaf");
            project_leaf(
                py,
                leaf.signature(),
                leaf.signature().shape(),
                leaf.components(),
            )
        })
        .collect::<PyResult<Vec<_>>>()?;
    Ok(PyTuple::new(py, leaves)?.unbind())
}

/// Derivative roles expose logical components without pretending to be primal values.
pub(super) fn project_components<I: Iterator<Item = (f64, f64)>>(
    py: Python<'_>,
    signatures: &[DifferentiableValueSignature],
    components: impl Fn(usize) -> I,
) -> PyResult<Py<PyTuple>> {
    let leaves = signatures
        .iter()
        .enumerate()
        .map(|(index, signature)| project_leaf(py, signature, signature.shape(), components(index)))
        .collect::<PyResult<Vec<_>>>()?;
    Ok(PyTuple::new(py, leaves)?.unbind())
}

/// A collection adds only declared grid axes outside each covector's intrinsic leaves.
pub(super) fn project_covectors(
    py: Python<'_>,
    signatures: &[DifferentiableValueSignature],
    grid_shape: &[usize],
    values: &[eqiora::api::DifferentiableCotangent],
) -> PyResult<Py<PyTuple>> {
    let leaves = signatures
        .iter()
        .enumerate()
        .map(|(index, signature)| {
            let shape = grid_shape
                .iter()
                .chain(signature.shape())
                .copied()
                .collect::<Vec<_>>();
            project_leaf(
                py,
                signature,
                &shape,
                values
                    .iter()
                    .flat_map(|value| value.components(index).expect("admitted leaf")),
            )
        })
        .collect::<PyResult<Vec<_>>>()?;
    Ok(PyTuple::new(py, leaves)?.unbind())
}

/// One owner for scalar domain, immutable storage and intrinsic shape projection.
fn project_leaf(
    py: Python<'_>,
    signature: &DifferentiableValueSignature,
    shape: &[usize],
    components: impl Iterator<Item = (f64, f64)>,
) -> PyResult<Py<PyAny>> {
    let buffer = match signature.mathematical_type().value_type.scalar_domain() {
        eqiora::ScalarDomain::Real => {
            PyArrayBuffer::from_owned_result(py, components.map(|(real, _)| real).collect())?
        }
        eqiora::ScalarDomain::Complex => PyArrayBuffer::from_owned_complex_result(
            py,
            components
                .map(|(real, imaginary)| numpy::Complex64::new(real, imaginary))
                .collect(),
        )?,
        _ => unreachable!("admitted differentiable scalar domain"),
    };
    buffer.bind(py).borrow().numpy_shaped(py, shape)
}

/// Project the current dense frontend transport only after native typed admission.
/// The closure supplies logical components; complex scalar parts never become axes.
pub(crate) fn coordinate_projection<I: Iterator<Item = (f64, f64)>>(
    signatures: &[DifferentiableValueSignature],
    components: impl Fn(usize) -> I,
) -> Vec<f64> {
    signatures
        .iter()
        .enumerate()
        .flat_map(|(index, signature)| {
            let complex = signature.mathematical_type().value_type.scalar_domain()
                == eqiora::ScalarDomain::Complex;
            components(index)
                .flat_map(move |(re, im)| [Some(re), complex.then_some(im)].into_iter().flatten())
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn empty_coordinate_buffers_cannot_hide_an_unbounded_occurrence_inventory() {
        assert!(points_from_coordinates(&[], &[], usize::MAX, 1024).is_err());
        assert!(points_from_coordinates(&[], &[], 1_000_000, 1024).is_err());
        assert!(points_from_coordinates(&[], &[1.0], 0, 1024).is_err());
        assert!(points_from_coordinates(&[], &[], 0, 0).unwrap().is_empty());
        let all_shared = points_from_coordinates(&[], &[], 3, 1024).unwrap();
        assert_eq!(all_shared.len(), 3);
        assert!(all_shared.iter().all(|point| point.signatures().is_empty()));
    }
}
