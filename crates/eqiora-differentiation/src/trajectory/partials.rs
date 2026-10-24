//! Partial actions of the complete frozen trajectory, without implicit solves.
use super::*;

/// One concatenated residual/output linearization over individually accepted steps.
///
/// Unknowns and residuals concatenate step-local free coordinates in time order;
/// outputs concatenate each selected step projection. Parameters occur once.
/// The initial physical state is fixed. History between steps is reconstructed
/// through their complete-state projections, including direct boundary terms.
/// This performs no solve and introduces no global residual acceptance threshold.
#[derive(Debug)]
pub struct DiscreteTrajectoryLinearization<'a, 'point, R, O>
where
    R: DiscreteStepLinearization + ?Sized,
    O: LinearizedOutput<f64> + ?Sized,
{
    steps: &'a [AcceptedOutputLinearization<'point, R, O>],
    unknown_offsets: Vec<usize>,
    output_offsets: Vec<usize>,
    state_dimension: usize,
    parameters: usize,
}

#[derive(Clone, Copy)]
enum Projection {
    Residual,
    Output,
}

impl<'a, 'point, R, O> DiscreteTrajectoryLinearization<'a, 'point, R, O>
where
    R: DiscreteStepLinearization + ?Sized,
    O: LinearizedOutput<f64> + ?Sized,
{
    /// Bind one continuous sequence with its individual accepted output pairs.
    ///
    /// # Errors
    /// Rejects incompatible identities, values, times, checkpoints or dimensions.
    pub fn new(
        steps: &'a [AcceptedOutputLinearization<'point, R, O>],
        checkpoints: &[DiscreteTrajectoryCheckpoint],
    ) -> Result<Self, Diagnostic> {
        let relations = steps
            .iter()
            .map(|step| step.relation().relation())
            .collect::<Vec<_>>();
        validate_discrete_trajectory(&relations, checkpoints)?;
        let mut unknown_offsets = vec![0usize];
        let mut output_offsets = vec![0usize];
        for step in steps {
            unknown_offsets.push(
                unknown_offsets
                    .last()
                    .unwrap()
                    .checked_add(step.relation().relation().unknown_dimension())
                    .ok_or_else(|| {
                        invalid_linearization("trajectory unknown dimension overflows")
                    })?,
            );
            output_offsets.push(
                output_offsets
                    .last()
                    .unwrap()
                    .checked_add(step.output().output_dimension())
                    .ok_or_else(|| {
                        invalid_linearization("trajectory output dimension overflows")
                    })?,
            );
        }
        Ok(Self {
            steps,
            unknown_offsets,
            output_offsets,
            state_dimension: relations[0].state_coordinates().len(),
            parameters: relations[0].model_parameter_coordinates().len(),
        })
    }

    fn offsets(&self, projection: Projection) -> &[usize] {
        match projection {
            Projection::Residual => &self.unknown_offsets,
            Projection::Output => &self.output_offsets,
        }
    }

    fn primal_values(&self, projection: Projection, output: &mut [f64]) -> Result<(), Diagnostic> {
        let offsets = self.offsets(projection);
        require_size(output.len(), *offsets.last().unwrap())?;
        for (index, step) in self.steps.iter().enumerate() {
            let output = &mut output[offsets[index]..offsets[index + 1]];
            match projection {
                Projection::Residual => step.relation().relation().primal(output)?,
                Projection::Output => step.output().primal(output)?,
            }
        }
        validate_direction(output, output.len())
    }

    fn forward(
        &self,
        unknown: Option<&[f64]>,
        parameter: Option<&[f64]>,
        projection: Projection,
        output: &mut [f64],
    ) -> Result<(), Diagnostic> {
        if let Some(values) = unknown {
            validate_direction(values, *self.unknown_offsets.last().unwrap())?;
        }
        if let Some(values) = parameter {
            validate_direction(values, self.parameters)?;
        }
        let offsets = self.offsets(projection);
        require_size(output.len(), *offsets.last().unwrap())?;
        let mut direction = vec![0.; self.state_dimension + self.parameters];
        if let Some(parameter) = parameter {
            direction[self.state_dimension..].copy_from_slice(parameter);
        }
        for (index, step) in self.steps.iter().enumerate() {
            let range = self.unknown_offsets[index]..self.unknown_offsets[index + 1];
            let zeros;
            let current = match unknown {
                Some(unknown) => &unknown[range],
                None => {
                    zeros = vec![0.; range.len()];
                    &zeros
                }
            };
            let selected = &mut output[offsets[index]..offsets[index + 1]];
            match projection {
                Projection::Residual => step.relation().relation().jvp(
                    RelationTangent::Both {
                        unknown: current,
                        parameter: &direction,
                    },
                    selected,
                )?,
                Projection::Output => step.output().jvp(current, &direction, selected)?,
            }
            let mut next = vec![0.; self.state_dimension];
            step.relation()
                .relation()
                .state_output()
                .jvp(current, &direction, &mut next)?;
            validate_direction(&next, self.state_dimension)?;
            direction[..self.state_dimension].copy_from_slice(&next);
        }
        validate_direction(output, output.len())
    }

    fn reverse(
        &self,
        input: &[f64],
        mut unknown: Option<&mut [f64]>,
        mut parameter: Option<&mut [f64]>,
        projection: Projection,
    ) -> Result<(), Diagnostic> {
        let offsets = self.offsets(projection);
        validate_direction(input, *offsets.last().unwrap())?;
        if let Some(values) = &unknown {
            require_size(values.len(), *self.unknown_offsets.last().unwrap())?;
        }
        if let Some(values) = &parameter {
            require_size(values.len(), self.parameters)?;
        }
        if let Some(values) = &mut parameter {
            values.fill(0.);
        }
        let mut future = vec![0.; self.state_dimension];
        for (index, step) in self.steps.iter().enumerate().rev() {
            let relation = step.relation().relation();
            let mut current = vec![0.; relation.unknown_dimension()];
            let mut direct = vec![0.; relation.parameter_dimension()];
            let selected = &input[offsets[index]..offsets[index + 1]];
            match projection {
                Projection::Residual => relation.vjp(
                    selected,
                    RelationCotangent::Both {
                        unknown: &mut current,
                        parameter: &mut direct,
                    },
                )?,
                Projection::Output => step.output().vjp(selected, &mut current, &mut direct)?,
            }
            let mut inherited = vec![0.; current.len()];
            let mut history = vec![0.; direct.len()];
            relation
                .state_output()
                .vjp(&future, &mut inherited, &mut history)?;
            for (total, value) in current
                .iter_mut()
                .chain(&mut direct)
                .zip(inherited.iter().chain(&history))
            {
                *total += value;
            }
            validate_direction(&current, current.len())?;
            validate_direction(&direct, direct.len())?;
            if let Some(values) = &mut unknown {
                values[self.unknown_offsets[index]..self.unknown_offsets[index + 1]]
                    .copy_from_slice(&current);
            }
            if let Some(values) = &mut parameter {
                for (total, value) in values.iter_mut().zip(&direct[self.state_dimension..]) {
                    *total += value;
                }
                validate_direction(values, self.parameters)?;
            }
            future.copy_from_slice(&direct[..self.state_dimension]);
        }
        Ok(())
    }
}

fn require_size(actual: usize, expected: usize) -> Result<(), Diagnostic> {
    if actual != expected {
        return Err(invalid_linearization(
            "trajectory action has an incompatible buffer size",
        ));
    }
    Ok(())
}

impl<R, O> LinearizedRelation<f64> for DiscreteTrajectoryLinearization<'_, '_, R, O>
where
    R: DiscreteStepLinearization + ?Sized,
    O: LinearizedOutput<f64> + ?Sized,
{
    fn unknown_dimension(&self) -> usize {
        *self.unknown_offsets.last().unwrap()
    }
    fn residual_dimension(&self) -> usize {
        *self.unknown_offsets.last().unwrap()
    }
    fn parameter_dimension(&self) -> usize {
        self.parameters
    }
    fn primal(&self, output: &mut [f64]) -> Result<(), Diagnostic> {
        self.primal_values(Projection::Residual, output)
    }
    fn jvp(&self, tangent: RelationTangent<'_, f64>, output: &mut [f64]) -> Result<(), Diagnostic> {
        let (unknown, parameter) = match tangent {
            RelationTangent::Unknown(unknown) => (Some(unknown), None),
            RelationTangent::Parameter(parameter) => (None, Some(parameter)),
            RelationTangent::Both { unknown, parameter } => (Some(unknown), Some(parameter)),
        };
        self.forward(unknown, parameter, Projection::Residual, output)
    }
    fn vjp(&self, input: &[f64], cotangent: RelationCotangent<'_, f64>) -> Result<(), Diagnostic> {
        let (unknown, parameter) = match cotangent {
            RelationCotangent::Unknown(unknown) => (Some(unknown), None),
            RelationCotangent::Parameter(parameter) => (None, Some(parameter)),
            RelationCotangent::Both { unknown, parameter } => (Some(unknown), Some(parameter)),
        };
        self.reverse(input, unknown, parameter, Projection::Residual)
    }
}

impl<R, O> LinearizedOutput<f64> for DiscreteTrajectoryLinearization<'_, '_, R, O>
where
    R: DiscreteStepLinearization + ?Sized,
    O: LinearizedOutput<f64> + ?Sized,
{
    fn unknown_dimension(&self) -> usize {
        *self.unknown_offsets.last().unwrap()
    }
    fn parameter_dimension(&self) -> usize {
        self.parameters
    }
    fn output_dimension(&self) -> usize {
        *self.output_offsets.last().unwrap()
    }
    fn primal(&self, output: &mut [f64]) -> Result<(), Diagnostic> {
        self.primal_values(Projection::Output, output)
    }
    fn jvp(
        &self,
        unknown: &[f64],
        parameter: &[f64],
        output: &mut [f64],
    ) -> Result<(), Diagnostic> {
        self.forward(Some(unknown), Some(parameter), Projection::Output, output)
    }
    fn vjp(
        &self,
        input: &[f64],
        unknown: &mut [f64],
        parameter: &mut [f64],
    ) -> Result<(), Diagnostic> {
        self.reverse(input, Some(unknown), Some(parameter), Projection::Output)
    }
}
