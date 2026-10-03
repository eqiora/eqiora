use eqiora_core::{Diagnostic, Scalar};
use num_complex::ComplexFloat;

use super::invalid_realization;
use crate::LinearOperatorProperties as Properties;

// Verify the declared exact symmetry of the newly admitted captured profiles.
// This is not property inference, a positivity proof, or a matrix-free claim.
pub(super) fn validate_declared_symmetry<S: Scalar + ComplexFloat<Real = f64>>(
    offsets: &[usize],
    columns: &[usize],
    values: &[S],
    properties: Properties,
) -> Result<(), Diagnostic> {
    let conjugate = match properties {
        Properties::Symmetric | Properties::ComplexSymmetric => false,
        Properties::Hermitian | Properties::HermitianPositiveDefinite => true,
        _ => return Ok(()),
    };
    for row in 0..offsets.len() - 1 {
        for entry in offsets[row]..offsets[row + 1] {
            let column = columns[entry];
            let mirror_range = offsets[column]..offsets[column + 1];
            let mirror = columns[mirror_range.clone()]
                .binary_search(&row)
                .map_or_else(|_| S::zero(), |index| values[mirror_range.start + index]);
            let expected = if conjugate { mirror.conj() } else { mirror };
            if values[entry] != expected {
                return Err(invalid_realization(format!(
                    "canonical CSR coefficients violate declared {properties:?} symmetry at ({row}, {column})"
                )));
            }
        }
    }
    Ok(())
}
