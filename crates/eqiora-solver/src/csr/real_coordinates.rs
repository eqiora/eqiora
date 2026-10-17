//! The real-coordinate representation of an exact complex-linear CSR action.
use super::{CanonicalCsrSystemView, CsrStorage, invalid_realization};
use crate::{LinearOperatorOrientation, LinearOperatorProperties};
use eqiora_core::Diagnostic;
use num_complex::Complex64;

impl CanonicalCsrSystemView<Complex64> {
    /// Represent this oriented complex-linear system in adjacent real/imaginary coordinates.
    /// Each `a+ib` becomes `[[a,-b],[b,a]]`. The supplied RHS belongs to the oriented
    /// problem and may differ from the RHS captured with this source operator.
    /// Real transposition of the result represents the complex Euclidean adjoint,
    /// not the unconjugated complex transpose. Its fingerprint identifies this
    /// real representation, independently of the original complex system.
    ///
    /// # Errors
    /// Rejects dimension overflow, incompatible or non-finite RHS values, and invalid CSR.
    pub fn real_coordinate_system(
        &self,
        orientation: LinearOperatorOrientation,
        right_hand_side: &[Complex64],
    ) -> Result<CanonicalCsrSystemView<f64>, Diagnostic> {
        if right_hand_side.len() != self.rows()
            || right_hand_side.iter().any(|value| !value.is_finite())
        {
            return Err(invalid_realization(
                "complex real-coordinate system requires an exact finite RHS",
            ));
        }
        let storage = RealBlockStorage::new(self, orientation, right_hand_side)?;
        let properties = match self.properties() {
            LinearOperatorProperties::HermitianPositiveDefinite => {
                LinearOperatorProperties::SymmetricPositiveDefinite
            }
            LinearOperatorProperties::Hermitian => LinearOperatorProperties::Symmetric,
            _ => LinearOperatorProperties::General,
        };
        CanonicalCsrSystemView::new(&storage, storage.rhs(), properties)
    }
}

// For each complex coordinate j, [Re(x_j), Im(x_j)] are adjacent.
// Each a+ib becomes [[a,-b],[b,a]]. No imaginary component is discarded.
struct RealBlockStorage {
    offsets: Vec<usize>,
    columns: Vec<usize>,
    values: Vec<f64>,
    rhs: Vec<f64>,
}

impl RealBlockStorage {
    fn new(
        source: &CanonicalCsrSystemView<Complex64>,
        orientation: LinearOperatorOrientation,
        rhs: &[Complex64],
    ) -> Result<Self, Diagnostic> {
        let dimension = source
            .rows()
            .checked_mul(2)
            .ok_or_else(|| invalid_realization("complex real-block dimension overflowed"))?;
        let mut rows = vec![Vec::new(); dimension];
        for row in 0..source.rows() {
            for entry in source.row_offsets()[row]..source.row_offsets()[row + 1] {
                let column = source.column_indices()[entry];
                let value = source.values()[entry];
                let (row, column, value) = match orientation {
                    LinearOperatorOrientation::Normal => (row, column, value),
                    LinearOperatorOrientation::Transposed => (column, row, value),
                    LinearOperatorOrientation::ConjugateTransposed => (column, row, value.conj()),
                };
                rows[2 * row].extend([(2 * column, value.re), (2 * column + 1, -value.im)]);
                rows[2 * row + 1].extend([(2 * column, value.im), (2 * column + 1, value.re)]);
            }
        }
        let mut storage = Self {
            offsets: vec![0],
            columns: Vec::new(),
            values: Vec::new(),
            rhs: rhs.iter().flat_map(|value| [value.re, value.im]).collect(),
        };
        for row in &mut rows {
            row.sort_unstable_by_key(|(column, _)| *column);
            for &(column, value) in row.iter() {
                storage.columns.push(column);
                storage.values.push(value);
            }
            storage.offsets.push(storage.values.len());
        }
        Ok(storage)
    }
}

impl CsrStorage for RealBlockStorage {
    fn rows(&self) -> usize {
        self.rhs.len()
    }
    fn columns(&self) -> usize {
        self.rhs.len()
    }
    fn row_offsets(&self) -> &[usize] {
        &self.offsets
    }
    fn column_indices(&self) -> &[usize] {
        &self.columns
    }
    fn values(&self) -> &[f64] {
        &self.values
    }
}

impl RealBlockStorage {
    pub(crate) fn rhs(&self) -> &[f64] {
        &self.rhs
    }
}
