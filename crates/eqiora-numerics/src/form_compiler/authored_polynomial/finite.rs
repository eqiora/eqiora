//! Finite actions use the same exact real/imaginary coefficient ring as spatial forms.
use super::*;

impl Context<'_> {
    pub(super) fn closed_component(
        &mut self,
        shape: &[usize],
        values: &[(f64, f64)],
        indices: &[usize],
    ) -> Option<Polynomial> {
        if shape.len() != indices.len() || shape.contains(&0) {
            return None;
        }
        let count = shape
            .iter()
            .try_fold(1usize, |n, extent| n.checked_mul(*extent))?;
        if count != values.len() {
            return None;
        }
        let mut flat = 0usize;
        for (index, extent) in indices.iter().zip(shape) {
            if *index >= *extent {
                return None;
            }
            flat = flat.checked_mul(*extent)?.checked_add(*index)?;
        }
        let (real, imag) = *values.get(flat)?;
        Polynomial::complex(
            Polynomial::constant(number(real)?),
            Polynomial::constant(number(imag)?),
        )
    }

    pub(super) fn apply(
        &mut self,
        matrix: &E,
        vector: &E,
        row: usize,
        depth: usize,
    ) -> Option<Polynomial> {
        let shape = self.shape(matrix, depth + 1)?;
        let [rows, columns] = shape.as_slice() else {
            return None;
        };
        if row >= *rows || self.shape(vector, depth + 1)? != [*columns] {
            return None;
        }
        self.remaining = self.remaining.checked_sub(*columns)?;
        let mut sum = Polynomial::constant(ExactRational::integer(0));
        for column in 0..*columns {
            let term = self
                .tensor(matrix, row, column, depth + 1)?
                .checked_mul(&self.vector(vector, column, depth + 1)?)
                .ok()?;
            sum = sum.checked_add(&term).ok()?;
        }
        Some(sum)
    }
}
