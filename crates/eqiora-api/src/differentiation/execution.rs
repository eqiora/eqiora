//! Application dispatch over the existing point and trajectory product owners.
use super::*;
use eqiora_differentiation::{
    DiscreteTrajectoryLinearization, discrete_trajectory_adjoint, discrete_trajectory_sensitivity,
};
use eqiora_ir::DiscreteStepLinearization;
use eqiora_numerics::{
    CommonDifferentiationTrajectory, CommonDiscreteStepLinearization, CommonDiscreteStepOutput,
};
use eqiora_solver::SolveReport;
mod projection;
pub(super) use projection::OutputProjection;

/// Keep native history/report ownership intact rather than copying a final point.
#[derive(Debug, Clone)]
pub(super) enum NativeEvaluation {
    Point(Box<CommonDifferentiationPoint>),
    Trajectory(Box<CommonDifferentiationTrajectory>),
}

impl NativeEvaluation {
    pub(super) fn points(&self) -> &[CommonDifferentiationPoint] {
        match self {
            Self::Point(point) => std::slice::from_ref(point.as_ref()),
            Self::Trajectory(trajectory) => trajectory.steps(),
        }
    }

    pub(super) fn with_shared_receipt_binding(self, template: &Self) -> Result<Self, Diagnostic> {
        match (self, template) {
            (Self::Point(point), Self::Point(template)) if point.receipt().is_some() => (*point)
                .with_shared_receipt_binding(template)
                .map(|point| Self::Point(Box::new(point))),
            (value, _) => Ok(value),
        }
    }

    pub(super) fn parameter_values(&self) -> &[f64] {
        self.points()[0].linearization().relation().design_values()
    }

    pub(super) fn accepted_unknowns(&self) -> Vec<f64> {
        self.points()
            .iter()
            .flat_map(|point| {
                point
                    .linearization()
                    .relation()
                    .accepted_unknowns()
                    .iter()
                    .copied()
            })
            .collect()
    }

    pub(super) fn outputs(&self, projection: &OutputProjection) -> Result<Vec<f64>, Diagnostic> {
        let outputs = self
            .points()
            .iter()
            .map(|point| point.linearization().output_values())
            .collect::<Vec<_>>();
        projection.gather(&outputs)
    }

    pub(super) fn jvp(
        &self,
        parameter: &[f64],
        projection: &OutputProjection,
        solver: LinearSolveRequest<'_>,
    ) -> Result<(Vec<f64>, Vec<SolveReport>), Diagnostic> {
        let sensitivities = match self {
            Self::Point(point) => vec![forward_output_sensitivity(
                &accepted_point(point)?,
                parameter,
                point
                    .linearization()
                    .relation()
                    .state_jacobian()
                    .properties(),
                solver,
            )?],
            Self::Trajectory(trajectory) => with_trajectory(trajectory, |steps| {
                let relation = steps[0].relation().relation();
                discrete_trajectory_sensitivity(
                    steps,
                    &[],
                    &vec![0.; relation.state_coordinates().len()],
                    parameter,
                    trajectory.steps()[0]
                        .linearization()
                        .relation()
                        .state_jacobian()
                        .properties(),
                    solver,
                )
            })?,
        };
        let mut outputs = Vec::with_capacity(sensitivities.len());
        let mut reports = Vec::with_capacity(sensitivities.len());
        for sensitivity in sensitivities {
            let (state, output) = sensitivity.into_parts();
            let (_, report) = state.into_parts();
            outputs.push(output);
            reports.push(report);
        }
        Ok((
            projection.gather(&outputs.iter().map(Vec::as_slice).collect::<Vec<_>>())?,
            reports,
        ))
    }

    pub(super) fn vjp(
        &self,
        cotangent: &[f64],
        projection: &OutputProjection,
        solver: LinearSolveRequest<'_>,
    ) -> Result<(Vec<f64>, Vec<SolveReport>), Diagnostic> {
        let covectors = projection.scatter(cotangent)?;
        match self {
            Self::Point(point) => {
                let gradient = adjoint_output_gradient(
                    &accepted_point(point)?,
                    &covectors[0],
                    point
                        .linearization()
                        .relation()
                        .state_jacobian()
                        .properties(),
                    solver,
                )?;
                let (adjoint, gradient) = gradient.into_parts();
                let (_, report) = adjoint.into_parts();
                Ok((gradient, vec![report]))
            }
            Self::Trajectory(trajectory) => with_trajectory(trajectory, |steps| {
                let result = discrete_trajectory_adjoint(
                    steps,
                    &[],
                    &covectors.iter().map(Vec::as_slice).collect::<Vec<_>>(),
                    &vec![0.; self.parameter_values().len()],
                    trajectory.steps()[0]
                        .linearization()
                        .relation()
                        .state_jacobian()
                        .properties(),
                    solver,
                )?;
                Ok((
                    result.parameter_gradient().to_vec(),
                    result
                        .step_adjoints()
                        .iter()
                        .map(|solve| solve.report().clone())
                        .collect(),
                ))
            }),
        }
    }

    pub(super) fn partial<T>(
        &self,
        action: impl FnOnce(
            &dyn LinearizedRelation<f64>,
            &dyn LinearizedOutput<f64>,
        ) -> Result<T, Diagnostic>,
    ) -> Result<T, Diagnostic> {
        match self {
            Self::Point(point) => action(point.linearization().relation(), point.linearization()),
            Self::Trajectory(trajectory) => with_trajectory(trajectory, |steps| {
                let linearization = DiscreteTrajectoryLinearization::new(steps, &[])?;
                action(&linearization, &linearization)
            }),
        }
    }
}

fn accepted_point(
    point: &CommonDifferentiationPoint,
) -> Result<
    AcceptedOutputLinearization<
        '_,
        eqiora_numerics::common::AssembledLinearizedRelation,
        CommonDifferentiationLinearization,
    >,
    Diagnostic,
> {
    let linearization = point.linearization();
    AcceptedOutputLinearization::new_with_canonical_state_jacobian(
        linearization.relation(),
        linearization,
        linearization.relation().state_jacobian(),
        linearization.state_scalar_domain(),
        linearization.residual_target(),
    )
}

fn with_trajectory<T>(
    trajectory: &CommonDifferentiationTrajectory,
    action: impl FnOnce(
        &[AcceptedOutputLinearization<
            '_,
            CommonDiscreteStepLinearization<'_>,
            CommonDiscreteStepOutput<'_>,
        >],
    ) -> Result<T, Diagnostic>,
) -> Result<T, Diagnostic> {
    let relations = trajectory
        .steps()
        .iter()
        .map(|point| {
            point
                .linearization()
                .discrete_step()
                .ok_or_else(|| invalid("trajectory step lacks its history action"))
        })
        .collect::<Result<Vec<_>, _>>()?;
    let outputs = relations
        .iter()
        .map(CommonDiscreteStepLinearization::selected_output)
        .collect::<Vec<_>>();
    let accepted = relations
        .iter()
        .zip(&outputs)
        .zip(trajectory.steps())
        .map(|((relation, output), point)| {
            let point = point.linearization();
            AcceptedOutputLinearization::new_with_canonical_state_jacobian(
                relation,
                output,
                point.relation().state_jacobian(),
                point.state_scalar_domain(),
                point.residual_target(),
            )
        })
        .collect::<Result<Vec<_>, _>>()?;
    action(&accepted)
}
