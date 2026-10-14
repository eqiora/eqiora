//! Owned real-coordinate output values and their paired dense partial actions.

use eqiora_core::{Diagnostic, diagnostic::codes};

use crate::LinearizedOutput;

/// Accepted-point values with output-row-major state and Parameter Jacobians.
///
/// Real coordinates also represent shaped and complex outputs. Their semantic
/// identities, types and coordinate ordering belong to the selecting Program;
/// this numerical representation assumes neither scalar outputs nor holomorphy.
/// A scalar objective is the one-row case of the same output contract.
#[derive(Debug, Clone, PartialEq)]
pub struct DenseOutputLinearization {
    values: Vec<f64>,
    unknown_jacobian: Vec<f64>,
    parameter_jacobian: Vec<f64>,
}

impl DenseOutputLinearization {
    /// Bind nonempty finite output values to their two row-major partial actions.
    ///
    /// Column counts follow from the matrix lengths. Compatibility with the
    /// accepted relation is checked when composing the output and relation.
    ///
    /// # Errors
    /// Rejects empty outputs, incomplete matrix rows and nonfinite entries.
    pub fn new(
        values: Vec<f64>,
        unknown_jacobian: Vec<f64>,
        parameter_jacobian: Vec<f64>,
    ) -> Result<Self, Diagnostic> {
        if values.is_empty()
            || unknown_jacobian.len() % values.len() != 0
            || parameter_jacobian.len() % values.len() != 0
            || values
                .iter()
                .chain(&unknown_jacobian)
                .chain(&parameter_jacobian)
                .any(|value| !value.is_finite())
        {
            return Err(invalid(
                "dense output requires finite values and complete Jacobian rows",
            ));
        }
        Ok(Self {
            values,
            unknown_jacobian,
            parameter_jacobian,
        })
    }

    /// Values in the selected output coordinate order.
    #[must_use]
    pub fn values(&self) -> &[f64] {
        &self.values
    }

    /// State partials in output-row-major order.
    #[must_use]
    pub fn unknown_jacobian(&self) -> &[f64] {
        &self.unknown_jacobian
    }

    /// Direct Parameter partials in output-row-major order.
    #[must_use]
    pub fn parameter_jacobian(&self) -> &[f64] {
        &self.parameter_jacobian
    }
}

impl LinearizedOutput<f64> for DenseOutputLinearization {
    fn unknown_dimension(&self) -> usize {
        self.unknown_jacobian.len() / self.values.len()
    }
    fn parameter_dimension(&self) -> usize {
        self.parameter_jacobian.len() / self.values.len()
    }
    fn output_dimension(&self) -> usize {
        self.values.len()
    }

    fn primal(&self, output: &mut [f64]) -> Result<(), Diagnostic> {
        if output.len() != self.values.len() {
            return Err(invalid("dense output primal shape mismatch"));
        }
        output.copy_from_slice(&self.values);
        Ok(())
    }

    fn jvp(
        &self,
        unknown: &[f64],
        parameter: &[f64],
        output: &mut [f64],
    ) -> Result<(), Diagnostic> {
        if unknown.len() != self.unknown_dimension()
            || parameter.len() != self.parameter_dimension()
            || output.len() != self.values.len()
            || unknown
                .iter()
                .chain(parameter)
                .any(|value| !value.is_finite())
        {
            return Err(invalid("dense output JVP shape or value mismatch"));
        }
        for (row, value) in output.iter_mut().enumerate() {
            *value = self.unknown_jacobian[row * unknown.len()..(row + 1) * unknown.len()]
                .iter()
                .zip(unknown)
                .chain(
                    self.parameter_jacobian[row * parameter.len()..(row + 1) * parameter.len()]
                        .iter()
                        .zip(parameter),
                )
                .map(|(entry, tangent)| entry * tangent)
                .sum();
        }
        finite(output)
    }

    fn vjp(
        &self,
        output: &[f64],
        unknown: &mut [f64],
        parameter: &mut [f64],
    ) -> Result<(), Diagnostic> {
        if output.len() != self.values.len()
            || unknown.len() != self.unknown_dimension()
            || parameter.len() != self.parameter_dimension()
            || output.iter().any(|value| !value.is_finite())
        {
            return Err(invalid("dense output VJP shape or value mismatch"));
        }
        for (jacobian, result) in [
            (&self.unknown_jacobian, unknown),
            (&self.parameter_jacobian, parameter),
        ] {
            let columns = result.len();
            for (column, value) in result.iter_mut().enumerate() {
                *value = output
                    .iter()
                    .enumerate()
                    .map(|(row, cotangent)| jacobian[row * columns + column] * cotangent)
                    .sum();
            }
            finite(result)?;
        }
        Ok(())
    }
}

fn finite(values: &[f64]) -> Result<(), Diagnostic> {
    if values.iter().any(|value| !value.is_finite()) {
        Err(invalid("dense output action produced a non-finite value"))
    } else {
        Ok(())
    }
}

fn invalid(message: &str) -> Diagnostic {
    Diagnostic::error(codes::INVALID_LINEARIZATION, message)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn output_rows_preserve_both_partial_actions_and_their_transpose() {
        // O=(u0+2*u1+p, 3*u0-u1+4*p), at u=(2,3), p=5.
        let output =
            DenseOutputLinearization::new(vec![13., 23.], vec![1., 2., 3., -1.], vec![1., 4.])
                .unwrap();
        let mut primal = [0.; 2];
        output.primal(&mut primal).unwrap();
        assert_eq!(primal, [13., 23.]);
        let mut tangent = [0.; 2];
        output.jvp(&[7., 11.], &[13.], &mut tangent).unwrap();
        assert_eq!(tangent, [42., 62.]);
        let mut unknown = [0.; 2];
        let mut parameter = [0.; 1];
        output
            .vjp(&[17., 19.], &mut unknown, &mut parameter)
            .unwrap();
        assert_eq!(unknown, [74., 15.]);
        assert_eq!(parameter, [93.]);
        assert_eq!(
            tangent[0] * 17. + tangent[1] * 19.,
            unknown[0] * 7. + unknown[1] * 11. + parameter[0] * 13.
        );
        assert!(output.primal(&mut [0.]).is_err());
        assert!(output.jvp(&[7.], &[13.], &mut tangent).is_err());
        assert!(output.vjp(&[17.], &mut unknown, &mut parameter).is_err());
    }

    #[test]
    fn output_admission_rejects_incomplete_or_nonfinite_actions() {
        assert!(DenseOutputLinearization::new(vec![], vec![], vec![]).is_err());
        assert!(DenseOutputLinearization::new(vec![1., 2.], vec![1.], vec![]).is_err());
        assert!(DenseOutputLinearization::new(vec![1., 2.], vec![], vec![1.]).is_err());
        assert!(DenseOutputLinearization::new(vec![f64::NAN], vec![], vec![]).is_err());
        assert!(DenseOutputLinearization::new(vec![1.], vec![f64::INFINITY], vec![]).is_err());
        assert!(DenseOutputLinearization::new(vec![1.], vec![], vec![f64::NEG_INFINITY]).is_err());
        let constant = DenseOutputLinearization::new(vec![1., 2.], vec![], vec![]).unwrap();
        let mut tangent = [1.; 2];
        constant.jvp(&[], &[], &mut tangent).unwrap();
        assert_eq!(tangent, [0., 0.]);
        constant.vjp(&[3., 4.], &mut [], &mut []).unwrap();
    }
    #[test]
    fn scalar_objective_is_one_paired_output_projection() {
        let objective = DenseOutputLinearization::new(vec![11.0], vec![2.0, 3.0], vec![5.0])
            .expect("finite objective");
        let mut primal = [0.0];
        objective.primal(&mut primal).unwrap();
        assert_eq!(primal, [11.0]);

        let mut tangent = [0.0];
        objective.jvp(&[7.0, 11.0], &[13.0], &mut tangent).unwrap();
        assert_eq!(tangent, [112.0]);

        let mut state = [0.0; 2];
        let mut parameter = [0.0];
        objective.vjp(&[17.0], &mut state, &mut parameter).unwrap();
        assert_eq!(state, [34.0, 51.0]);
        assert_eq!(parameter, [85.0]);
        assert_eq!(
            tangent[0] * 17.0,
            7.0 * state[0] + 11.0 * state[1] + 13.0 * parameter[0]
        );

        let overflowing =
            DenseOutputLinearization::new(vec![0.0], vec![f64::MAX], Vec::new()).unwrap();
        assert!(overflowing.jvp(&[2.0], &[], &mut [0.0]).is_err());
        assert!(overflowing.vjp(&[2.0], &mut [0.0], &mut []).is_err());
    }
}
