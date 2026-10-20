//! Frozen request meaning without an executable Program or Run closure.
use super::*;
use eqiora_numerics::{
    CommonInitialState, CommonTimeRunRequest, RecordedTimeCheckpoint, ResolvedCommonPlan,
};

#[derive(Debug, Clone, PartialEq)]
pub(super) enum RecordedRequest {
    Evaluations {
        plan: ResolvedCommonPlan,
        program: Box<DifferentiableProgramIdentity>,
        initial: Option<CommonInitialState<'static>>,
        points: Vec<DifferentiableValues>,
        shared: Vec<Parameter>,
        samples: Option<Vec<SampledParameterPoint>>,
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
                plan: program.plan().clone(),
                program: Box::new(program.identity().clone()),
                initial: program.initial_state().cloned(),
                points: points.to_vec(),
                shared: shared_inputs.to_vec(),
                samples: samples.as_ref().map(|v| v.to_vec()),
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
            Self::Time { requests, .. } => requests.len(),
        }
    }
    pub(super) fn plan(&self) -> &ResolvedCommonPlan {
        match self {
            Self::Evaluations { plan, .. } | Self::Time { plan, .. } => plan,
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
            (Self::Time { requests, .. }, RecordedEvaluationOutcome::TimeAccepted(result)) => {
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
