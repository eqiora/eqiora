//! Partial residual/output actions, distinct from implicit solution products.
use super::*;
use eqiora_ir::{RelationCotangent, RelationTangent};

impl DifferentiableEvaluation {
    /// All accepted free unknowns in step order; stationary evaluations have one step.
    #[must_use]
    pub fn accepted_unknowns(&self) -> &[f64] {
        &self.unknowns
    }

    /// Apply R_w dw + R_p dp with each step's unknowns independent.
    pub fn residual_jvp(&self, unknown: &[f64], parameter: &[f64]) -> Result<Vec<f64>, Diagnostic> {
        self.native.partial(|relation, _| {
            let mut output = vec![0.; relation.residual_dimension()];
            relation.jvp(RelationTangent::Both { unknown, parameter }, &mut output)?;
            Ok(output)
        })
    }

    /// Apply the complete residual transpose without implicit solves.
    pub fn residual_vjp(&self, cotangent: &[f64]) -> Result<(Vec<f64>, Vec<f64>), Diagnostic> {
        self.native.partial(|relation, _| {
            let mut unknown = vec![0.; relation.unknown_dimension()];
            let mut parameter = vec![0.; relation.parameter_dimension()];
            relation.vjp(
                cotangent,
                RelationCotangent::Both {
                    unknown: &mut unknown,
                    parameter: &mut parameter,
                },
            )?;
            Ok((unknown, parameter))
        })
    }

    /// Apply O_w dw + O_p dp at the selected physical times, without solving.
    pub fn output_partial_jvp(
        &self,
        unknown: &[f64],
        parameter: &[f64],
    ) -> Result<Vec<f64>, Diagnostic> {
        self.native.partial(|_, output| {
            let mut values = vec![0.; output.output_dimension()];
            output.jvp(unknown, parameter, &mut values)?;
            self.projection.gather_flat(&values)
        })
    }

    /// Transpose selected outputs into all independent unknowns and common Parameters.
    pub fn output_partial_vjp(
        &self,
        cotangent: &[f64],
    ) -> Result<(Vec<f64>, Vec<f64>), Diagnostic> {
        let all: Vec<_> = self
            .projection
            .scatter(cotangent)?
            .into_iter()
            .flatten()
            .collect();
        self.native.partial(|_, output| {
            let mut unknown = vec![0.; output.unknown_dimension()];
            let mut parameter = vec![0.; output.parameter_dimension()];
            output.vjp(&all, &mut unknown, &mut parameter)?;
            Ok((unknown, parameter))
        })
    }
}
