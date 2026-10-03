//! Recover the exact complex linear action from paired semantic coordinates.
//! This is an equivalence check, never an inferred symmetry or definiteness claim.
use super::*;
use eqiora_ir::ComponentScalarization;
use eqiora_solver::{CanonicalCsrSystemView, CompleteCsrStorage, LinearOperatorProperties};
use num_complex::Complex64;
use std::collections::BTreeMap;

impl FiniteConstraintProblem {
    pub(crate) fn complex_linear_system(
        &self,
    ) -> Result<Option<CanonicalCsrSystemView<Complex64>>, Diagnostic> {
        if self.enforcement.is_some() || !self.coordinates.len().is_multiple_of(2) {
            return Ok(None);
        }
        for pair in self.coordinates.as_chunks::<2>().0 {
            if pair[0].is_imaginary()
                || !pair[1].is_imaginary()
                || pair[0].symbol() != pair[1].symbol()
                || pair[0].component_index() != pair[1].component_index()
            {
                return Ok(None);
            }
        }
        for relation in &self.relations {
            let mut count = 0;
            let Some(expression) = preparation::branch_expression(relation, 0, &mut count)? else {
                return Ok(None);
            };
            let typed = coordinates::typed_expression(&self.kernel, &expression)?;
            let components = ComponentScalarization::lower(&typed)?;
            let (pairs, remainder) = components.rows().as_chunks::<2>();
            if !remainder.is_empty() {
                return Ok(None);
            }
            for pair in pairs {
                if pair[0].is_imaginary()
                    || !pair[1].is_imaginary()
                    || pair[0].root_index() != pair[1].root_index()
                    || pair[0].component_index() != pair[1].component_index()
                {
                    return Ok(None);
                }
            }
        }
        let real = solve::branch_system(self, 0)?;
        let mut storage = ComplexStorage {
            offsets: vec![0],
            columns: Vec::new(),
            values: Vec::new(),
            rhs: real
                .right_hand_side()
                .as_chunks::<2>()
                .0
                .iter()
                .map(|pair| Complex64::new(pair[0], pair[1]))
                .collect(),
        };
        for row in (0..real.rows()).step_by(2) {
            let mut blocks = BTreeMap::<usize, [f64; 4]>::new();
            for part in 0..2 {
                for entry in real.row_offsets()[row + part]..real.row_offsets()[row + part + 1] {
                    let column = real.column_indices()[entry];
                    blocks.entry(column / 2).or_default()[2 * part + column % 2] =
                        real.values()[entry];
                }
            }
            for (column, [rr, ri, ir, ii]) in blocks {
                if rr != ii || ri != -ir {
                    return Ok(None);
                }
                storage.columns.push(column);
                storage.values.push(Complex64::new(rr, ir));
            }
            storage.offsets.push(storage.values.len());
        }
        CanonicalCsrSystemView::new(&storage, LinearOperatorProperties::General).map(Some)
    }
}

struct ComplexStorage {
    offsets: Vec<usize>,
    columns: Vec<usize>,
    values: Vec<Complex64>,
    rhs: Vec<Complex64>,
}
impl CompleteCsrStorage<Complex64> for ComplexStorage {
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
    fn values(&self) -> &[Complex64] {
        &self.values
    }
    fn right_hand_side(&self) -> &[Complex64] {
        &self.rhs
    }
}
