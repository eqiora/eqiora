//! Frozen request meaning without an executable Program or Run closure.
use super::*;
use crate::differentiation::context::Context;
use eqiora_numerics::{CommonTimeRunRequest, RecordedTimeCheckpoint, ResolvedCommonPlan};

#[derive(Debug, Clone, PartialEq)]
pub(super) enum RecordedRequest {
    Evaluations {
        context: Box<Context<'static>>,
        program: Box<DifferentiableProgramIdentity>,
        points: Vec<DifferentiableValues>,
        shared: Vec<Parameter>,
        samples: Option<Vec<SampledParameterPoint>>,
    },
    Parameters {
        template: ResolvedCommonPlan,
        parameters: Vec<Parameter>,
        points: Vec<Vec<eqiora_core::ValueLiteral>>,
        policy: EvaluationMapExecutionPolicy,
        member_bytes_limit: usize,
    },
    Time {
        plan: ResolvedCommonPlan,
        requests: Vec<CommonTimeRunRequest>,
        checkpoints: Vec<Option<RecordedTimeCheckpoint>>,
    },
}

impl RecordedRequest {
    pub(super) fn capture(plan: &EvaluationMapPlan) -> Result<Self, Diagnostic> {
        Ok(match plan.inputs.as_ref() {
            Inputs::Evaluations {
                program,
                points,
                shared_inputs,
                samples,
            } => Self::Evaluations {
                context: Box::new(program.context().clone()),
                program: Box::new(program.identity().clone()),
                points: points.to_vec(),
                shared: shared_inputs.to_vec(),
                samples: samples.as_ref().map(|v| v.to_vec()),
            },
            Inputs::Parameters(inputs) => Self::Parameters {
                template: inputs.template.as_ref().clone(),
                parameters: inputs.parameters.to_vec(),
                points: inputs.points.to_vec(),
                policy: plan.policy(),
                member_bytes_limit: inputs.member_bytes_limit,
            },
            Inputs::Time(inputs) => Self::Time {
                plan: ResolvedCommonPlan::Time(Box::new(inputs.plan.as_ref().clone())),
                requests: inputs.requests.to_vec(),
                checkpoints: inputs
                    .checkpoints
                    .iter()
                    .map(|value| value.as_ref().map(|value| value.record()).transpose())
                    .collect::<Result<_, _>>()?,
            },
        })
    }
    pub(super) fn len(&self) -> usize {
        match self {
            Self::Evaluations { points, .. } => points.len(),
            Self::Parameters { points, .. } => points.len(),
            Self::Time { requests, .. } => requests.len(),
        }
    }
    pub(super) fn plan(&self) -> &ResolvedCommonPlan {
        match self {
            Self::Evaluations { context, .. } => context.plan(),
            Self::Time { plan, .. } => plan,
            Self::Parameters { template, .. } => template,
        }
    }
    pub(super) fn program_identity(&self) -> Option<&DifferentiableProgramIdentity> {
        match self {
            Self::Evaluations { program, .. } => Some(program),
            _ => None,
        }
    }
    pub(super) fn points(&self) -> Option<&[DifferentiableValues]> {
        match self {
            Self::Evaluations { points, .. } => Some(points),
            _ => None,
        }
    }
    pub(super) fn samples(&self) -> Option<&[SampledParameterPoint]> {
        match self {
            Self::Evaluations { samples, .. } => samples.as_deref(),
            _ => None,
        }
    }
    pub(super) fn time_requests(&self) -> Option<&[CommonTimeRunRequest]> {
        match self {
            Self::Time { requests, .. } => Some(requests),
            _ => None,
        }
    }
    pub(super) fn time_checkpoints(&self) -> Option<&[Option<RecordedTimeCheckpoint>]> {
        match self {
            Self::Time { checkpoints, .. } => Some(checkpoints),
            _ => None,
        }
    }
    pub(super) fn shared_inputs(&self) -> Option<&[Parameter]> {
        match self {
            Self::Evaluations { shared, .. } => Some(shared),
            _ => None,
        }
    }
    pub(super) fn validate(&self) -> Result<(), Diagnostic> {
        match self {
            Self::Evaluations {
                program,
                points,
                shared,
                samples,
                ..
            } => {
                if points
                    .iter()
                    .any(|point| point.signatures() != program.inputs())
                {
                    return Err(invalid(
                        "collection point signatures differ from their exact Program",
                    ));
                }
                partition::admit_shared(program, points.iter(), shared)?;
                if let Some(samples) = samples
                    && (samples.len() != points.len()
                        || samples.iter().zip(points).any(|(sample, point)| {
                            sample.program_identity() != program.as_ref()
                                || !super::super::exact_values(
                                    sample.point().coordinates(),
                                    point.coordinates(),
                                )
                        }))
                {
                    return Err(invalid(
                        "collection sampling attribution differs from requested points",
                    ));
                }
            }
            Self::Parameters {
                template,
                parameters: selected,
                points,
                policy,
                member_bytes_limit,
            } => {
                parameters::validate_template(template)?;
                let original = parameters::document(template).map_err(crate::first_diagnostic)?;
                parameters::validate_points(&original, selected, points)?;
                parameters::storage_charge(selected, points, *policy, *member_bytes_limit)?;
            }
            Self::Time {
                plan,
                requests,
                checkpoints,
            } => {
                if plan.as_time().is_none() || checkpoints.len() != requests.len() {
                    return Err(invalid(
                        "collection time request requires its exact Plan and checkpoint inventory",
                    ));
                }
                for (request, checkpoint) in requests.iter().zip(checkpoints) {
                    if request.plan().identity() != plan.identity()
                        || request.output_times_s() != requests[0].output_times_s()
                        || checkpoint
                            .as_ref()
                            .is_some_and(|value| value.request().identity() != request.identity())
                    {
                        return Err(invalid(
                            "collection time request or checkpoint differs from its original Run",
                        ));
                    }
                }
            }
        }
        Ok(())
    }
    pub(super) fn validate_outcome(
        &self,
        index: usize,
        outcome: &RecordedEvaluationOutcome,
    ) -> Result<(), Diagnostic> {
        match (self, outcome) {
            (
                Self::Evaluations {
                    program, points, ..
                },
                RecordedEvaluationOutcome::Accepted(record),
            ) => {
                if record.result().evidence().identity() != program.as_ref()
                    || !super::super::exact_values(
                        record.result().evidence().point().coordinates(),
                        points[index].coordinates(),
                    )
                {
                    return Err(invalid(
                        "collection member differs from its exact requested Program or point",
                    ));
                }
            }
            (
                Self::Parameters {
                    template,
                    parameters: selected,
                    points,
                    member_bytes_limit,
                    ..
                },
                RecordedEvaluationOutcome::RunAccepted(result),
            ) => {
                parameters::check_member_size(result, *member_bytes_limit)?;
                let original = parameters::document(template).map_err(crate::first_diagnostic)?;
                parameters::validate_result(&original, selected, &points[index], result)?;
            }
            (Self::Time { requests, .. }, RecordedEvaluationOutcome::RunAccepted(result)) => {
                time::validate_request(&requests[index], result)?
            }
            (
                Self::Time { requests, .. },
                RecordedEvaluationOutcome::Cancelled(Some(checkpoint)),
            ) => {
                if checkpoint.request().identity() != requests[index].identity() {
                    return Err(invalid(
                        "collection cancellation checkpoint belongs to a foreign request",
                    ));
                }
            }
            (_, RecordedEvaluationOutcome::Failed(diagnostics)) => {
                if diagnostics.is_empty()
                    || diagnostics.iter().any(|value| value.validate().is_err())
                {
                    return Err(invalid(
                        "collection failure requires nonempty bounded original diagnostics",
                    ));
                }
            }
            (
                _,
                RecordedEvaluationOutcome::Cancelled(None) | RecordedEvaluationOutcome::NotStarted,
            ) => {}
            _ => {
                return Err(invalid(
                    "collection outcome family differs from its exact request",
                ));
            }
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn recorded_parameter_request_rejects_a_time_template_before_outcome_admission() {
        let components =
            include_str!("../../../packages/Eqiora.Stochastic.Processes/src/processes.eqi");
        let composition = include_str!("../../../../../examples/stochastic-processes/src/main.eqi")
            .replace(
                "import Eqiora.Stochastic.Processes.processes as processes;",
                "",
            )
            .replace("processes.", "");
        let original = crate::ModelDocument::compile(
            "foreign-time-template.eqi",
            &format!("{components}\n{composition}"),
        )
        .unwrap();
        let model = eqiora_artifact::ModelEnvelope::from_program(original.program()).unwrap();
        let time =
            eqiora_numerics::CommonTimePlan::resolve_euler_maruyama(&model, original.program(), 3)
                .unwrap();
        let request = RecordedRequest::Parameters {
            template: ResolvedCommonPlan::Time(Box::new(time)),
            parameters: vec![],
            points: vec![vec![]],
            policy: EvaluationMapExecutionPolicy::retained(1 << 28),
            member_bytes_limit: 1 << 20,
        };
        let error = request
            .wire()
            .unwrap()
            .replay(&eqiora_solver::REFERENCE_LINEAR_SOLVER, &[])
            .unwrap_err();
        assert!(
            error[0].message().contains("require a stationary Plan"),
            "{error:?}"
        );
    }
}
