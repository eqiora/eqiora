//! One accepted spatial relation with separate predecessor and Model actions.
use super::*;
use crate::region_assembly::mapping::FieldDof;
use crate::spatial_design::SpatialDesignCoordinate;
use eqiora_ir::{DiscreteStepLinearization, RelationCotangent, RelationTangent};
use eqiora_solver::{LinearOperator, LinearOperatorOrientation, OrientedLinearOperator};
use std::sync::Arc;

/// One full real coordinate in an exact spatial Plan and constrained Field map.
/// Model components, mesh entities, functional slots and real/imaginary parts
/// retain their separate meanings. Only the owning Plan constructs this identity.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SpatialStepCoordinate {
    pub(in crate::numerical_admission) plan: Arc<str>,
    pub(in crate::numerical_admission) key: FieldDof,
    pub(in crate::numerical_admission) imaginary: bool,
}

#[derive(Debug, Clone, PartialEq)]
pub(super) struct StepHistory {
    pub(super) jacobian: eqiora_assembly::CsrMatrix<f64>,
    pub(super) coordinates: Vec<SpatialStepCoordinate>,
    pub(super) previous: Vec<f64>,
    pub(super) interval: [f64; 2],
    pub(super) output: crate::linearized_output::FieldCoefficientLinearization,
    pub(super) selected_output_history: Option<eqiora_assembly::CsrMatrix<f64>>,
}

/// Borrow an accepted spatial step as `[previous State, Model Parameters]`.
/// The original point's Parameter products still hold physical history fixed.
/// This view exposes the additional sparse history action for shared discrete
/// trajectory differentiation; it performs no solve or history interpolation.
#[derive(Debug, Clone, Copy)]
pub struct CommonDiscreteStepLinearization<'a> {
    pub(super) point: &'a CommonDifferentiationLinearization,
    pub(super) history: &'a StepHistory,
}

/// Borrow a step output with independent `[previous State, Model Parameters]` inputs.
/// Its original direct Parameter action and any direct predecessor interpolation
/// action are retained separately from the implicit residual solve.
#[derive(Debug, Clone, Copy)]
struct CommonDiscreteStepOutput<'a> {
    source: &'a dyn LinearizedOutput<f64>,
    previous_dimension: usize,
    previous_jacobian: Option<&'a eqiora_assembly::CsrMatrix<f64>>,
}

impl<'point> CommonDiscreteStepLinearization<'point> {
    /// Selected complete outputs paired with this same accepted step.
    #[must_use]
    pub fn selected_output(&self) -> impl LinearizedOutput<f64> + 'point + use<'point> {
        CommonDiscreteStepOutput {
            source: self.point,
            previous_dimension: self.history.previous.len(),
            previous_jacobian: self.history.selected_output_history.as_ref(),
        }
    }

    fn full_output(&self) -> CommonDiscreteStepOutput<'point> {
        CommonDiscreteStepOutput {
            source: &self.history.output,
            previous_dimension: self.history.previous.len(),
            previous_jacobian: None,
        }
    }

    fn split<'a>(&self, values: &'a [f64]) -> Result<(&'a [f64], &'a [f64]), Diagnostic> {
        if values.len() != LinearizedRelation::parameter_dimension(self)
            || values.iter().any(|v| !v.is_finite())
        {
            return Err(invalid(
                "step tangent requires exact finite predecessor and Parameter coordinates",
            ));
        }
        Ok(values.split_at(self.history.previous.len()))
    }

    fn history_jvp(&self, previous: &[f64], output: &mut [f64]) -> Result<(), Diagnostic> {
        let mut action = vec![0.; self.residual_dimension()];
        self.history.jacobian.apply(previous, &mut action)?;
        for (value, action) in output.iter_mut().zip(action) {
            *value += action;
        }
        if output.iter().any(|value| !value.is_finite()) {
            return Err(invalid("step tangent accumulation is non-finite"));
        }
        Ok(())
    }

    fn parameter_vjp(&self, input: &[f64], output: &mut [f64]) -> Result<(), Diagnostic> {
        if output.len() != LinearizedRelation::parameter_dimension(self) {
            return Err(invalid(
                "step cotangent has an incompatible coordinate layout",
            ));
        }
        let (previous, parameters) = output.split_at_mut(self.history.previous.len());
        self.history.jacobian.apply_oriented(
            LinearOperatorOrientation::ConjugateTransposed,
            input,
            previous,
        )?;
        self.point
            .relation
            .vjp(input, RelationCotangent::Parameter(parameters))
    }
}

impl LinearizedRelation<f64> for CommonDiscreteStepLinearization<'_> {
    fn unknown_dimension(&self) -> usize {
        self.point.relation.unknown_dimension()
    }
    fn residual_dimension(&self) -> usize {
        self.point.relation.residual_dimension()
    }
    fn parameter_dimension(&self) -> usize {
        self.history.previous.len() + self.point.relation.parameter_dimension()
    }
    fn primal(&self, output: &mut [f64]) -> Result<(), Diagnostic> {
        self.point.relation.primal(output)
    }
    fn jvp(&self, tangent: RelationTangent<'_, f64>, output: &mut [f64]) -> Result<(), Diagnostic> {
        match tangent {
            RelationTangent::Unknown(unknown) => self
                .point
                .relation
                .jvp(RelationTangent::Unknown(unknown), output),
            RelationTangent::Parameter(values) => {
                let (previous, parameters) = self.split(values)?;
                self.point
                    .relation
                    .jvp(RelationTangent::Parameter(parameters), output)?;
                self.history_jvp(previous, output)
            }
            RelationTangent::Both { unknown, parameter } => {
                let (previous, parameter) = self.split(parameter)?;
                self.point
                    .relation
                    .jvp(RelationTangent::Both { unknown, parameter }, output)?;
                self.history_jvp(previous, output)
            }
        }
    }
    fn vjp(&self, input: &[f64], cotangent: RelationCotangent<'_, f64>) -> Result<(), Diagnostic> {
        match cotangent {
            RelationCotangent::Unknown(unknown) => self
                .point
                .relation
                .vjp(input, RelationCotangent::Unknown(unknown)),
            RelationCotangent::Parameter(parameter) => self.parameter_vjp(input, parameter),
            RelationCotangent::Both { unknown, parameter } => {
                self.point
                    .relation
                    .vjp(input, RelationCotangent::Unknown(unknown))?;
                self.parameter_vjp(input, parameter)
            }
        }
    }
}

impl DiscreteStepLinearization for CommonDiscreteStepLinearization<'_> {
    type StateCoordinate = SpatialStepCoordinate;
    type StateOutput = Self;
    fn state_output(&self) -> &Self {
        self
    }
    type ModelParameterCoordinate = SpatialDesignCoordinate;
    fn state_coordinates(&self) -> &[Self::StateCoordinate] {
        &self.history.coordinates
    }
    fn model_parameter_coordinates(&self) -> &[Self::ModelParameterCoordinate] {
        self.point.relation.design_coordinates()
    }
    fn model_parameter_values(&self) -> &[f64] {
        self.point.relation.design_values()
    }
    fn previous_state(&self) -> &[f64] {
        &self.history.previous
    }
    fn next_state(&self) -> &[f64] {
        self.history.output.values()
    }
    fn start_time(&self) -> f64 {
        self.history.interval[0]
    }
    fn end_time(&self) -> f64 {
        self.history.interval[1]
    }
    fn previous_state_parameter_dimension(&self) -> usize {
        self.history.previous.len()
    }
}

impl LinearizedOutput<f64> for CommonDiscreteStepLinearization<'_> {
    fn unknown_dimension(&self) -> usize {
        self.full_output().unknown_dimension()
    }
    fn parameter_dimension(&self) -> usize {
        self.full_output().parameter_dimension()
    }
    fn output_dimension(&self) -> usize {
        self.full_output().output_dimension()
    }
    fn primal(&self, output: &mut [f64]) -> Result<(), Diagnostic> {
        self.full_output().primal(output)
    }
    fn jvp(
        &self,
        unknown: &[f64],
        parameter: &[f64],
        output: &mut [f64],
    ) -> Result<(), Diagnostic> {
        self.full_output().jvp(unknown, parameter, output)
    }
    fn vjp(
        &self,
        input: &[f64],
        unknown: &mut [f64],
        parameter: &mut [f64],
    ) -> Result<(), Diagnostic> {
        self.full_output().vjp(input, unknown, parameter)
    }
}

impl LinearizedOutput<f64> for CommonDiscreteStepOutput<'_> {
    fn unknown_dimension(&self) -> usize {
        self.source.unknown_dimension()
    }
    fn parameter_dimension(&self) -> usize {
        self.previous_dimension + self.source.parameter_dimension()
    }
    fn output_dimension(&self) -> usize {
        self.source.output_dimension()
    }
    fn primal(&self, output: &mut [f64]) -> Result<(), Diagnostic> {
        self.source.primal(output)
    }
    fn jvp(
        &self,
        unknown: &[f64],
        parameter: &[f64],
        output: &mut [f64],
    ) -> Result<(), Diagnostic> {
        if parameter.len() != self.parameter_dimension()
            || parameter.iter().any(|value| !value.is_finite())
        {
            return Err(invalid(
                "step output direction requires exact finite coordinates",
            ));
        }
        self.source
            .jvp(unknown, &parameter[self.previous_dimension..], output)?;
        if let Some(matrix) = self.previous_jacobian {
            let mut action = vec![0.; output.len()];
            matrix.apply(&parameter[..self.previous_dimension], &mut action)?;
            for (value, action) in output.iter_mut().zip(action) {
                *value += action;
            }
            if output.iter().any(|value| !value.is_finite()) {
                return Err(invalid("interpolated output tangent is non-finite"));
            }
        }
        Ok(())
    }
    fn vjp(
        &self,
        input: &[f64],
        unknown: &mut [f64],
        parameter: &mut [f64],
    ) -> Result<(), Diagnostic> {
        if parameter.len() != self.parameter_dimension() {
            return Err(invalid("step output cotangent layout differs"));
        }
        let (previous, model) = parameter.split_at_mut(self.previous_dimension);
        self.source.vjp(input, unknown, model)?;
        match self.previous_jacobian {
            Some(matrix) => matrix.apply_oriented(
                LinearOperatorOrientation::ConjugateTransposed,
                input,
                previous,
            )?,
            None => previous.fill(0.),
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn selected_output_keeps_direct_history_separate_from_parameter_actions() {
        let source =
            eqiora_ir::DenseOutputLinearization::new(vec![0.], vec![2., 3.], vec![5.]).unwrap();
        let history = eqiora_assembly::CsrMatrix::from_sorted_csr(
            1,
            2,
            vec![0, 2],
            vec![0, 1],
            vec![7., -11.],
        )
        .unwrap();
        let output = CommonDiscreteStepOutput {
            source: &source,
            previous_dimension: 2,
            previous_jacobian: Some(&history),
        };
        let mut tangent = [0.];
        output
            .jvp(&[13., 17.], &[19., 23., 29.], &mut tangent)
            .unwrap();
        assert_eq!(tangent, [102.]);
        let mut unknown = [0.; 2];
        let mut inputs = [0.; 3];
        output.vjp(&[31.], &mut unknown, &mut inputs).unwrap();
        assert_eq!(unknown, [62., 93.]);
        assert_eq!(inputs, [217., -341., 155.]);
    }
}
