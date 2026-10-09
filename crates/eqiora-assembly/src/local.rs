use eqiora_core::diagnostic::codes;
use eqiora_core::{Diagnostic, Scalar};
use num_complex::ComplexFloat;

/// Dense local matrix and right-hand-side contribution in local ordering.
///
/// Local rows and columns are intentionally anonymous. A separate
/// [`crate::AssemblyMap`] supplies equations, free unknowns, and fixed values
/// when the contribution is scattered.
#[derive(Debug, Clone, PartialEq)]
pub struct LocalContribution<S> {
    rows: usize,
    columns: usize,
    matrix: Vec<S>,
    rhs: Vec<S>,
}

impl<S: Scalar + ComplexFloat> LocalContribution<S> {
    /// Construct a finite row-major local contribution.
    ///
    /// # Errors
    /// Returns `EQ0805` for zero rows, shape overflow/mismatch, or non-finite
    /// matrix/right-hand-side entries.
    pub fn new(
        rows: usize,
        columns: usize,
        matrix: Vec<S>,
        rhs: Vec<S>,
    ) -> Result<Self, Diagnostic> {
        if rows == 0 {
            return Err(invalid_local(
                "local contribution requires at least one row",
            ));
        }
        let entry_count = rows
            .checked_mul(columns)
            .ok_or_else(|| invalid_local("local contribution matrix dimensions overflow usize"))?;
        if matrix.len() != entry_count || rhs.len() != rows {
            return Err(invalid_local(format!(
                "local contribution shape is {rows}x{columns} with {} matrix and {} rhs entries",
                matrix.len(),
                rhs.len()
            )));
        }
        if matrix.iter().chain(&rhs).any(|value| !value.is_finite()) {
            return Err(invalid_local(
                "local contribution entries must all be finite",
            ));
        }
        Ok(Self {
            rows,
            columns,
            matrix,
            rhs,
        })
    }

    /// Local row count.
    #[must_use]
    pub const fn rows(&self) -> usize {
        self.rows
    }

    /// Local column count.
    #[must_use]
    pub const fn columns(&self) -> usize {
        self.columns
    }

    /// Row-major local matrix.
    #[must_use]
    pub fn matrix(&self) -> &[S] {
        &self.matrix
    }

    /// Local right-hand side.
    #[must_use]
    pub fn rhs(&self) -> &[S] {
        &self.rhs
    }

    /// One local matrix entry.
    #[must_use]
    pub fn entry(&self, row: usize, column: usize) -> Option<S> {
        (row < self.rows && column < self.columns).then(|| self.matrix[row * self.columns + column])
    }

    /// Express local test and trial coordinates in signed entity orientations.
    /// Applies `diag(row_signs) A diag(column_signs)` and `diag(row_signs) b`.
    /// Signs are real, so this also preserves complex test duality.
    ///
    /// # Errors
    /// Returns `EQ0805` unless both sign arrays have the exact local widths and
    /// every entry is `-1` or `+1`.
    pub fn reoriented(mut self, row_signs: &[i8], column_signs: &[i8]) -> Result<Self, Diagnostic> {
        if row_signs.len() != self.rows
            || column_signs.len() != self.columns
            || row_signs
                .iter()
                .chain(column_signs)
                .any(|sign| !matches!(sign, -1 | 1))
        {
            return Err(invalid_local(
                "local orientation requires exact row/column widths and signs",
            ));
        }
        for (row, row_sign) in row_signs.iter().enumerate() {
            if *row_sign < 0 {
                self.rhs[row] = -self.rhs[row];
            }
            for (column, column_sign) in column_signs.iter().enumerate() {
                if row_sign != column_sign {
                    let entry = &mut self.matrix[row * self.columns + column];
                    *entry = -*entry;
                }
            }
        }
        Ok(self)
    }
}

fn invalid_local(message: impl Into<String>) -> Diagnostic {
    Diagnostic::error(codes::INVALID_LOCAL_CONTRIBUTION, message)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn complex_contributions_retain_both_components_and_reject_nonfinite_parts() {
        use num_complex::Complex64 as C;
        let value = LocalContribution::new(
            1,
            2,
            vec![C::new(2., 3.), C::new(5., -7.)],
            vec![C::new(11., 13.)],
        )
        .unwrap();
        assert_eq!(value.entry(0, 1), Some(C::new(5., -7.)));
        assert_eq!(value.rhs(), &[C::new(11., 13.)]);
        for invalid in [C::new(f64::NAN, 0.), C::new(0., f64::INFINITY)] {
            assert!(LocalContribution::new(1, 1, vec![invalid], vec![C::new(0., 0.)]).is_err());
            assert!(LocalContribution::new(1, 1, vec![C::new(1., 0.)], vec![invalid]).is_err());
        }
    }

    #[test]
    fn signed_local_coordinates_transform_rows_columns_and_complex_loads() {
        use num_complex::Complex64 as C;
        let original = LocalContribution::new(
            2,
            2,
            vec![
                C::new(1.0, 2.0),
                C::new(3.0, 4.0),
                C::new(5.0, 6.0),
                C::new(7.0, 8.0),
            ],
            vec![C::new(9.0, 10.0), C::new(11.0, 12.0)],
        )
        .unwrap();
        let transformed = original.clone().reoriented(&[-1, 1], &[1, -1]).unwrap();
        assert_eq!(
            transformed.matrix(),
            &[
                C::new(-1.0, -2.0),
                C::new(3.0, 4.0),
                C::new(5.0, 6.0),
                C::new(-7.0, -8.0)
            ]
        );
        assert_eq!(
            transformed.rhs(),
            &[C::new(-9.0, -10.0), C::new(11.0, 12.0)]
        );
        assert_eq!(
            transformed.reoriented(&[-1, 1], &[1, -1]).unwrap(),
            original
        );
        for (rows, columns) in [
            (vec![1], vec![1, 1]),
            (vec![1, 0], vec![1, 1]),
            (vec![1, 1], vec![2, 1]),
        ] {
            assert!(original.clone().reoriented(&rows, &columns).is_err());
        }
    }

    #[test]
    fn local_contribution_checks_dense_shape_and_values() {
        assert_eq!(
            LocalContribution::new(2, 2, vec![1.0; 3], vec![0.0; 2])
                .unwrap_err()
                .code(),
            codes::INVALID_LOCAL_CONTRIBUTION
        );
        assert_eq!(
            LocalContribution::new(1, 1, vec![f64::NAN], vec![0.0])
                .unwrap_err()
                .code(),
            codes::INVALID_LOCAL_CONTRIBUTION
        );
    }
}
