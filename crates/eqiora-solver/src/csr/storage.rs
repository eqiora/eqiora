//! Borrowed sparse actions shared by admitted assembly and captured solver storage.
use std::ops::Range;

use eqiora_core::{Diagnostic, Scalar};
use num_complex::ComplexFloat;

use super::solve_failed;
use crate::LinearOperatorOrientation;

/// Immutable matrix storage, independent of a linear system's right-hand side.
///
/// Implementors expose CSR arrays only, never an overridable mathematical action
/// or identity. The inherent actions on `dyn CsrStorage` use Eqiora's arithmetic;
/// canonical capture copies and validates these arrays before assigning identity.
pub trait CsrStorage<S = f64> {
    /// Matrix row count.
    fn rows(&self) -> usize;
    /// Matrix column count.
    fn columns(&self) -> usize;
    /// CSR row offsets.
    fn row_offsets(&self) -> &[usize];
    /// CSR column indices.
    fn column_indices(&self) -> &[usize];
    /// CSR nonzero values, in stored reduction order.
    fn values(&self) -> &[S];
}

impl<S: Scalar + ComplexFloat + Sync> dyn CsrStorage<S> + '_ {
    /// Apply a contiguous range of rows to a complete finite input vector.
    ///
    /// No storage is copied or allocated. Each row uses compensated addition in
    /// stored order, independently for each real/imaginary component.
    /// All input coordinates must be finite, including structurally unused ones.
    /// The caller owns disjoint input and output buffers. Output after an
    /// arithmetic/storage failure is unspecified; invalid input is rejected
    /// before writing output. This action does not certify canonical storage.
    ///
    /// # Errors
    /// Returns `EQ0802` for invalid dimensions/ranges, nonfinite input or results,
    /// inconsistent CSR array lengths, or invalid offsets/columns in visited rows.
    pub fn apply_rows(
        &self,
        rows: Range<usize>,
        input: &[S],
        output: &mut [S],
    ) -> Result<(), Diagnostic> {
        let storage = BorrowedCsr::new(self)?;
        let count = rows
            .end
            .checked_sub(rows.start)
            .ok_or_else(|| solve_failed("CSR row action requires a nondecreasing row range"))?;
        if rows.end > storage.rows || input.len() != storage.columns || output.len() != count {
            return Err(solve_failed(
                "CSR row action dimensions or range do not agree",
            ));
        }
        require_finite_input(input)?;
        for (row, result) in rows.zip(output) {
            let mut sum = S::zero();
            let mut correction = S::zero();
            for entry in storage.row(row)? {
                let column = storage.column_indices[entry];
                let value = input
                    .get(column)
                    .ok_or_else(|| solve_failed("CSR column is outside the input vector"))?;
                super::accumulation::add(&mut sum, &mut correction, storage.values[entry] * *value);
            }
            sum = sum + correction;
            if !sum.is_finite() {
                return Err(solve_failed(format!(
                    "CSR row {row} produced a non-finite value"
                )));
            }
            *result = sum;
        }
        Ok(())
    }

    /// Apply the normal, transpose, or conjugate-transpose matrix action.
    ///
    /// Transpose and adjoint visit source rows and their entries in stored order.
    /// One temporary compensation value per output coordinate retains low-order
    /// contributions, using the same addition as the normal row action.
    /// This shares the finite-input and output-on-failure contract of
    /// `apply_rows`, without allocating or copying matrix storage.
    ///
    /// # Errors
    /// Returns `EQ0802` for invalid dimensions, nonfinite input or results,
    /// inconsistent CSR array lengths, or invalid offsets/columns.
    pub fn apply_oriented(
        &self,
        orientation: LinearOperatorOrientation,
        input: &[S],
        output: &mut [S],
    ) -> Result<(), Diagnostic> {
        if orientation == LinearOperatorOrientation::Normal {
            return self.apply_rows(0..self.rows(), input, output);
        }
        let storage = BorrowedCsr::new(self)?;
        if input.len() != storage.rows || output.len() != storage.columns {
            return Err(solve_failed(
                "transposed CSR action dimensions do not agree",
            ));
        }
        require_finite_input(input)?;
        output.fill(S::zero());
        let mut corrections = vec![S::zero(); output.len()];
        for (row, input_value) in input.iter().enumerate() {
            for entry in storage.row(row)? {
                let coefficient = if orientation == LinearOperatorOrientation::ConjugateTransposed {
                    storage.values[entry].conj()
                } else {
                    storage.values[entry]
                };
                let column = storage.column_indices[entry];
                let result = output
                    .get_mut(column)
                    .ok_or_else(|| solve_failed("CSR column is outside the output vector"))?;
                super::accumulation::add(
                    result,
                    &mut corrections[column],
                    coefficient * *input_value,
                );
            }
        }
        for (result, correction) in output.iter_mut().zip(corrections) {
            *result = *result + correction;
        }
        if output.iter().any(|value| !value.is_finite()) {
            return Err(solve_failed(
                "transposed CSR action produced a non-finite value",
            ));
        }
        Ok(())
    }
}

fn require_finite_input<S: ComplexFloat>(input: &[S]) -> Result<(), Diagnostic> {
    if input.iter().any(|value| !value.is_finite()) {
        return Err(solve_failed("CSR action input must be finite"));
    }
    Ok(())
}

struct BorrowedCsr<'a, S> {
    rows: usize,
    columns: usize,
    row_offsets: &'a [usize],
    column_indices: &'a [usize],
    values: &'a [S],
}

impl<'a, S> BorrowedCsr<'a, S> {
    fn new(storage: &'a dyn CsrStorage<S>) -> Result<Self, Diagnostic> {
        let result = Self {
            rows: storage.rows(),
            columns: storage.columns(),
            row_offsets: storage.row_offsets(),
            column_indices: storage.column_indices(),
            values: storage.values(),
        };
        if result.rows == 0
            || result.columns == 0
            || result.rows.checked_add(1) != Some(result.row_offsets.len())
            || result.column_indices.len() != result.values.len()
            || result.row_offsets.first() != Some(&0)
            || result.row_offsets.last() != Some(&result.values.len())
        {
            return Err(solve_failed("CSR action storage shape is inconsistent"));
        }
        Ok(result)
    }

    fn row(&self, row: usize) -> Result<Range<usize>, Diagnostic> {
        let range = self.row_offsets[row]..self.row_offsets[row + 1];
        if range.start > range.end || range.end > self.values.len() {
            return Err(solve_failed(
                "CSR row offsets are outside the stored entries",
            ));
        }
        Ok(range)
    }
}
