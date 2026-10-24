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

impl CommonDiscreteStepLinearization<'_> {
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
        self.point.relation.unknown_dimension()
    }
    fn parameter_dimension(&self) -> usize {
        LinearizedRelation::parameter_dimension(self)
    }
    fn output_dimension(&self) -> usize {
        self.history.output.output_dimension()
    }
    fn primal(&self, output: &mut [f64]) -> Result<(), Diagnostic> {
        self.history.output.primal(output)
    }
    fn jvp(
        &self,
        unknown: &[f64],
        parameter: &[f64],
        output: &mut [f64],
    ) -> Result<(), Diagnostic> {
        let (_, model) = self.split(parameter)?;
        self.history.output.jvp(unknown, model, output)
    }
    fn vjp(
        &self,
        input: &[f64],
        unknown: &mut [f64],
        parameter: &mut [f64],
    ) -> Result<(), Diagnostic> {
        if parameter.len() != LinearizedRelation::parameter_dimension(self) {
            return Err(invalid("step output cotangent layout differs"));
        }
        let (previous, model) = parameter.split_at_mut(self.history.previous.len());
        self.history.output.vjp(input, unknown, model)?;
        previous.fill(0.);
        Ok(())
    }
}
