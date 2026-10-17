//! Accumulate a residual correction separately from the current approximation.
use eqiora_core::Scalar;
use num_complex::ComplexFloat;

pub(super) struct SolutionCorrection<S> {
    base: Vec<S>,
    candidate: Vec<S>,
    original_target: f64,
    initial_norm: f64,
}

impl<S: Scalar + ComplexFloat> SolutionCorrection<S> {
    pub(super) fn new(updates: &mut [S], target: f64, initial_norm: f64) -> Self {
        let base = updates.to_vec();
        updates.fill(S::zero());
        Self {
            base,
            candidate: vec![S::zero(); updates.len()],
            original_target: target,
            initial_norm,
        }
    }

    pub(super) fn candidate(&mut self, updates: &[S]) -> &[S] {
        for ((candidate, base), update) in self.candidate.iter_mut().zip(&self.base).zip(updates) {
            *candidate = *base + *update;
        }
        &self.candidate
    }

    pub(super) fn restart(&mut self, updates: &mut [S], residual_norm: f64) -> f64 {
        self.base.copy_from_slice(&self.candidate);
        updates.fill(S::zero());
        // Preserve the original relative reduction for the correction equation.
        // Reusing the absolute outer target would repeatedly truncate tiny
        // corrections before they can change the rounded approximation.
        // The caller still accepts only against the original true residual and
        // counts every correction iteration against the original solve budget.
        self.original_target
            .min(self.original_target * (residual_norm / self.initial_norm))
    }

    pub(super) fn into_values(self) -> Vec<S> {
        self.candidate
    }
}
