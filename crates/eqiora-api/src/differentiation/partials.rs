//! Partial residual/output actions, distinct from implicit reduced solution actions.
use super::*;
use eqiora_ir::{RelationCotangent, RelationTangent};

impl DifferentiableEvaluation {
    /// Accepted unknown coordinates paired with this immutable linearization.
    #[must_use]
    pub fn accepted_unknowns(&self) -> &[f64] {
        self.native.relation().accepted_unknowns()
    }

    /// Apply R_w dw + R_p dp without solving an implicit sensitivity system.
    pub fn residual_jvp(&self, unknown: &[f64], parameter: &[f64]) -> Result<Vec<f64>, Diagnostic> {
        let relation = self.native.relation();
        let mut output = vec![0.0; relation.residual_dimension()];
        relation.jvp(RelationTangent::Both { unknown, parameter }, &mut output)?;
        Ok(output)
    }

    /// Apply the residual transpose to both independent coordinate roles.
    pub fn residual_vjp(&self, cotangent: &[f64]) -> Result<(Vec<f64>, Vec<f64>), Diagnostic> {
        let relation = self.native.relation();
        let mut unknown = vec![0.0; relation.unknown_dimension()];
        let mut parameter = vec![0.0; relation.parameter_dimension()];
        relation.vjp(
            cotangent,
            RelationCotangent::Both {
                unknown: &mut unknown,
                parameter: &mut parameter,
            },
        )?;
        Ok((unknown, parameter))
    }

    /// Apply O_w dw + O_p dp with unknowns and Parameters independent.
    pub fn output_partial_jvp(
        &self,
        unknown: &[f64],
        parameter: &[f64],
    ) -> Result<Vec<f64>, Diagnostic> {
        let mut output = vec![0.0; self.native.output_dimension()];
        self.native.jvp(unknown, parameter, &mut output)?;
        Ok(output)
    }

    /// Apply the output transpose to independent unknown and Parameter roles.
    pub fn output_partial_vjp(
        &self,
        cotangent: &[f64],
    ) -> Result<(Vec<f64>, Vec<f64>), Diagnostic> {
        let mut unknown = vec![0.0; self.native.unknown_dimension()];
        let mut parameter = vec![0.0; self.native.parameter_dimension()];
        self.native.vjp(cotangent, &mut unknown, &mut parameter)?;
        Ok((unknown, parameter))
    }
}
