//! Complete Field coefficient projections over method-native algebraic states.

use eqiora_core::Diagnostic;
use eqiora_core::diagnostic::codes;
use eqiora_ir::LinearizedOutput;

/// Complete Field coefficients and their paired indexed JVP/VJP projection.
///
/// Essential elimination and shared trace quotients map physical coefficients
/// to free coordinates. The direct Parameter action retains prescribed values.
/// Component, entity and functional ordering is owned by the bound Field map.
#[derive(Debug, Clone, PartialEq)]
pub struct FieldCoefficientLinearization {
    primal: Vec<f64>,
    unknown_dimension: usize,
    parameter_dimension: usize,
    output_unknowns: Vec<Option<usize>>,
    unknown_weights: Vec<f64>,
    direct_parameter_jacobian: Vec<f64>,
}

impl FieldCoefficientLinearization {
    /// Construct one finite, shape-consistent Field projection.
    ///
    /// `direct_parameter_jacobian` is output-row-major. `None` in
    /// `output_unknowns` denotes an eliminated or directly derived output.
    ///
    /// # Errors
    /// Returns `EQ0704` for invalid dimensions, indices, or non-finite data.
    pub(crate) fn new(
        primal: Vec<f64>,
        unknown_dimension: usize,
        parameter_dimension: usize,
        output_unknowns: Vec<Option<usize>>,
        direct_parameter_jacobian: Vec<f64>,
    ) -> Result<Self, Diagnostic> {
        let expected_direct = primal
            .len()
            .checked_mul(parameter_dimension)
            .ok_or_else(|| invalid("Field output Jacobian shape overflows usize"))?;
        if primal.is_empty()
            || output_unknowns.len() != primal.len()
            || direct_parameter_jacobian.len() != expected_direct
            || output_unknowns
                .iter()
                .flatten()
                .any(|&index| index >= unknown_dimension)
        {
            return Err(invalid(
                "Field output linearization has incompatible primal, state, or Parameter shape",
            ));
        }
        if primal
            .iter()
            .chain(&direct_parameter_jacobian)
            .any(|value| !value.is_finite())
        {
            return Err(invalid(
                "Field output linearization requires finite primal values and actions",
            ));
        }
        Ok(Self {
            unknown_weights: vec![1.; primal.len()],
            primal,
            unknown_dimension,
            parameter_dimension,
            output_unknowns,
            direct_parameter_jacobian,
        })
    }

    /// Rebind physical free coordinates to the joint implicit unknown layout.
    pub(crate) fn project_unknowns(
        mut self,
        dimension: usize,
        coordinates: &[(usize, f64)],
    ) -> Result<Self, Diagnostic> {
        if coordinates.len() != self.unknown_dimension
            || coordinates
                .iter()
                .any(|&(index, weight)| index >= dimension || !weight.is_finite())
        {
            return Err(invalid(
                "Field output projection has a foreign unknown layout",
            ));
        }
        for (index, weight) in self
            .output_unknowns
            .iter_mut()
            .zip(&mut self.unknown_weights)
        {
            if let Some(index) = index {
                let (joint, factor) = coordinates[*index];
                *index = joint;
                *weight *= factor;
                if !weight.is_finite() {
                    return Err(invalid("Field projection weight is non-finite"));
                }
            }
        }
        self.unknown_dimension = dimension;
        Ok(self)
    }

    pub(crate) fn state_entries(&self) -> impl Iterator<Item = Option<(usize, f64)>> + '_ {
        self.output_unknowns
            .iter()
            .zip(&self.unknown_weights)
            .map(|(index, weight)| index.map(|index| (index, *weight)))
    }

    /// Complete accepted Field values in the Realization's canonical order.
    #[must_use]
    pub fn values(&self) -> &[f64] {
        &self.primal
    }
}

impl LinearizedOutput<f64> for FieldCoefficientLinearization {
    fn unknown_dimension(&self) -> usize {
        self.unknown_dimension
    }

    fn parameter_dimension(&self) -> usize {
        self.parameter_dimension
    }

    fn output_dimension(&self) -> usize {
        self.primal.len()
    }

    fn primal(&self, output: &mut [f64]) -> Result<(), Diagnostic> {
        if output.len() != self.primal.len() {
            return Err(invalid("Field output primal shape mismatch"));
        }
        output.copy_from_slice(&self.primal);
        Ok(())
    }

    fn jvp(
        &self,
        unknown_tangent: &[f64],
        parameter_tangent: &[f64],
        output_tangent: &mut [f64],
    ) -> Result<(), Diagnostic> {
        if unknown_tangent.len() != self.unknown_dimension
            || parameter_tangent.len() != self.parameter_dimension
            || output_tangent.len() != self.primal.len()
            || unknown_tangent
                .iter()
                .chain(parameter_tangent)
                .any(|value| !value.is_finite())
        {
            return Err(invalid("Field output JVP shape or value mismatch"));
        }
        for (row, output) in output_tangent.iter_mut().enumerate() {
            let state = self.output_unknowns[row].map_or(0.0, |index| {
                self.unknown_weights[row] * unknown_tangent[index]
            });
            let direct = self.direct_parameter_jacobian
                [row * self.parameter_dimension..(row + 1) * self.parameter_dimension]
                .iter()
                .zip(parameter_tangent)
                .map(|(entry, tangent)| entry * tangent)
                .sum::<f64>();
            *output = state + direct;
        }
        finite(output_tangent, "Field output JVP")
    }

    fn vjp(
        &self,
        output_cotangent: &[f64],
        unknown_cotangent: &mut [f64],
        parameter_cotangent: &mut [f64],
    ) -> Result<(), Diagnostic> {
        if output_cotangent.len() != self.primal.len()
            || unknown_cotangent.len() != self.unknown_dimension
            || parameter_cotangent.len() != self.parameter_dimension
            || output_cotangent.iter().any(|value| !value.is_finite())
        {
            return Err(invalid("Field output VJP shape or value mismatch"));
        }
        unknown_cotangent.fill(0.0);
        parameter_cotangent.fill(0.0);
        for (row, &cotangent) in output_cotangent.iter().enumerate() {
            if let Some(index) = self.output_unknowns[row] {
                unknown_cotangent[index] += self.unknown_weights[row] * cotangent;
            }
            for (column, parameter) in parameter_cotangent.iter_mut().enumerate() {
                *parameter += self.direct_parameter_jacobian
                    [row * self.parameter_dimension + column]
                    * cotangent;
            }
        }
        finite(unknown_cotangent, "Field output state VJP")?;
        finite(parameter_cotangent, "Field output Parameter VJP")
    }
}

fn finite(values: &[f64], operation: &str) -> Result<(), Diagnostic> {
    if values.iter().any(|value| !value.is_finite()) {
        Err(invalid(format!("{operation} produced a non-finite value")))
    } else {
        Ok(())
    }
}

fn invalid(message: impl Into<String>) -> Diagnostic {
    Diagnostic::error(codes::INVALID_LINEARIZATION, message)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn indexed_projection_jvp_and_vjp_are_paired() {
        let output = FieldCoefficientLinearization::new(
            vec![2.0, 3.0, 5.0],
            2,
            1,
            vec![None, Some(0), Some(1)],
            vec![4.0, 0.0, 0.0],
        )
        .unwrap();
        let mut tangent = vec![0.0; 3];
        output.jvp(&[7.0, 11.0], &[13.0], &mut tangent).unwrap();
        assert_eq!(tangent, vec![52.0, 7.0, 11.0]);

        let cotangent = [17.0, 19.0, 23.0];
        let mut state = vec![0.0; 2];
        let mut parameter = vec![0.0; 1];
        output.vjp(&cotangent, &mut state, &mut parameter).unwrap();
        assert_eq!(state, vec![19.0, 23.0]);
        assert_eq!(parameter, vec![68.0]);
        let left = tangent
            .iter()
            .zip(cotangent)
            .map(|(left, right)| left * right)
            .sum::<f64>();
        let right = state
            .iter()
            .zip([7.0, 11.0])
            .map(|(left, right)| left * right)
            .sum::<f64>()
            + parameter[0] * 13.0;
        assert_eq!(left, right);
    }
}
