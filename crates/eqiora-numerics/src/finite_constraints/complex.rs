//! Recover the exact complex linear action from paired semantic coordinates.
//! This is an equivalence check, never an inferred symmetry or definiteness claim.
use super::*;
use eqiora_ir::ComponentScalarization;
use eqiora_solver::CanonicalCsrSystemView;
use num_complex::Complex64;

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
        solve::branch_system(self, 0)?.complex_coordinate_system()
    }
}
