use super::super::outcomes::{CompleteMembers, Completed, Outcome, TerminalMembers};
use super::*;

#[derive(Debug, Clone)]
pub(in crate::evaluation_map) struct RunMember {
    pub(super) result: CommonResult,
    pub(super) fingerprint: String,
}

pub(in crate::evaluation_map) type RunOutcome = Outcome<RunMember, String, CommonTimeCheckpoint>;

impl Completed<RunMember, String> {
    pub(in crate::evaluation_map) fn validate(
        &self,
        plan: &EvaluationMapPlan,
    ) -> Result<usize, Diagnostic> {
        match self {
            Self::Retained(values) => {
                if values.len() != plan.len() {
                    return Err(invalid(
                        "complete Run map differs from its original occurrence inventory",
                    ));
                }
                for (index, member) in values.iter().enumerate() {
                    match plan.inputs.as_ref() {
                        Inputs::Time(inputs) => {
                            validate_request(&inputs.requests[index], &member.result)?
                        }
                        Inputs::Parameters(inputs) => parameters::validate_result(
                            &inputs.original,
                            &inputs.parameters,
                            &inputs.points[index],
                            &member.result,
                        )?,
                        _ => {
                            return Err(invalid(
                                "ordinary Run member requires its original requests",
                            ));
                        }
                    }
                }
                Ok(values.len())
            }
            Self::Released(values) => Ok(values.len()),
        }
    }
}

pub(in crate::evaluation_map) fn validate_request(
    request: &CommonTimeRunRequest,
    result: &CommonResult,
) -> Result<(), Diagnostic> {
    if result
        .trajectory()
        .and_then(|trajectory| trajectory.time_request())
        .is_none_or(|original| original.identity() != request.identity())
    {
        return Err(invalid(
            "time map Result differs from its exact requested Run",
        ));
    }
    Ok(())
}

pub(super) fn admit_result(
    request: &CommonTimeRunRequest,
    result: CommonResult,
    limit: usize,
) -> Result<RunMember, Vec<Diagnostic>> {
    validate_request(request, &result).map_err(|error| vec![error])?;
    admit_common_result(result, limit)
}

pub(in crate::evaluation_map) fn admit_common_result(
    result: CommonResult,
    limit: usize,
) -> Result<RunMember, Vec<Diagnostic>> {
    check_size(result.to_bytes().map_err(|error| vec![error])?.len(), limit)
        .map_err(|error| vec![error])?;
    // Wall time is presentation metadata, not mathematical acceptance evidence.
    // Use the existing canonical Result owner, never another JSON projection/codec.
    let fingerprint = acceptance_fingerprint(&result).map_err(|error| vec![error])?;
    Ok(RunMember {
        result,
        fingerprint,
    })
}

pub(in crate::evaluation_map) fn acceptance_fingerprint(
    result: &CommonResult,
) -> Result<String, Diagnostic> {
    Ok(result
        .clone()
        .with_elapsed_seconds(0.)?
        .identity()
        .to_owned())
}

pub(super) fn finish(
    plan: &EvaluationMapPlan,
    execution: ulid::Ulid,
    outcomes: Vec<RunOutcome>,
) -> Result<CompleteEvaluationMap, EvaluationMapTerminalReport> {
    let members =
        Completed::from_outcomes(outcomes, plan.policy().retention()).map_err(|outcomes| {
            EvaluationMapTerminalReport {
                execution,
                plan: plan.clone(),
                outcomes: TerminalMembers::Runs(outcomes),
            }
        })?;
    Ok(CompleteEvaluationMap {
        execution,
        plan: plan.clone(),
        members: CompleteMembers::Runs(members),
    })
}

pub(in crate::evaluation_map) fn occurrence(value: &RunOutcome) -> EvaluationMapOccurrence<'_> {
    match value {
        Outcome::Accepted(member) => EvaluationMapOccurrence::RunAccepted(&member.result),
        Outcome::Released(fingerprint) => EvaluationMapOccurrence::RunReleased(fingerprint),
        Outcome::Failed(errors) => EvaluationMapOccurrence::Failed(errors),
        Outcome::Cancelled(Some(checkpoint)) => EvaluationMapOccurrence::TimeCancelled(checkpoint),
        Outcome::Cancelled(None) => EvaluationMapOccurrence::Cancelled,
        Outcome::NotStarted => EvaluationMapOccurrence::NotStarted,
    }
}

fn recompute(
    plan: &EvaluationMapPlan,
    index: usize,
    expected: &str,
) -> Result<CommonResult, Vec<Diagnostic>> {
    let outcome = match plan.inputs.as_ref() {
        Inputs::Time(inputs) => inputs.evaluate(index),
        Inputs::Parameters(inputs) => inputs.evaluate(index),
        _ => {
            return Err(vec![invalid(
                "ordinary Run recomputation requires its original request",
            )]);
        }
    };
    match outcome {
        Outcome::Accepted(member) if member.fingerprint == expected => Ok(member.result),
        Outcome::Failed(errors) => Err(errors),
        _ => Err(vec![invalid(
            "recomputed Run occurrence changed its original acceptance or did not complete",
        )]),
    }
}

impl CompleteEvaluationMap {
    /// One retained ordinary Run Result; differentiable maps and released members return None.
    #[must_use]
    pub fn run_result(&self, index: usize) -> Option<&CommonResult> {
        match &self.members {
            CompleteMembers::Runs(Completed::Retained(values)) => {
                values.get(index).map(|member| &member.result)
            }
            _ => None,
        }
    }
    /// Original canonical Run acceptance fingerprint, excluding elapsed wall time.
    #[must_use]
    pub fn run_evidence(&self, index: usize) -> Option<&str> {
        match &self.members {
            CompleteMembers::Runs(Completed::Retained(values)) => {
                values.get(index).map(|member| member.fingerprint.as_str())
            }
            CompleteMembers::Runs(Completed::Released(values)) => {
                values.get(index).map(String::as_str)
            }
            _ => None,
        }
    }
    /// Reexecute one explicitly released ordinary Run against its original evidence.
    /// No fresh sample identity or implicit retry is introduced.
    pub fn recompute_run(&self, index: usize) -> Result<CommonResult, Vec<Diagnostic>> {
        if self.plan.policy().retention() != EvaluationMapRetention::Recompute {
            return Err(vec![invalid(
                "Run recomputation requires explicit Recompute retention",
            )]);
        }
        let expected = self
            .run_evidence(index)
            .ok_or_else(|| vec![invalid("Run occurrence is absent")])?;
        recompute(&self.plan, index, expected)
    }
}

impl EvaluationMapTerminalReport {
    /// Retained ordinary Result at its original position, without promoting incomplete membership.
    #[must_use]
    pub fn run_result(&self, index: usize) -> Option<&CommonResult> {
        match self.occurrence(index)? {
            EvaluationMapOccurrence::RunAccepted(result) => Some(result),
            _ => None,
        }
    }
    /// Atomic checkpoint from an interrupted time member, bound to its original request.
    #[must_use]
    pub fn time_checkpoint(&self, index: usize) -> Option<&CommonTimeCheckpoint> {
        match self.occurrence(index)? {
            EvaluationMapOccurrence::TimeCancelled(checkpoint) => Some(checkpoint),
            _ => None,
        }
    }
    /// Recompute a released ordinary Run while preserving original partial membership.
    pub fn recompute_run(&self, index: usize) -> Result<CommonResult, Vec<Diagnostic>> {
        let Some(EvaluationMapOccurrence::RunReleased(expected)) = self.occurrence(index) else {
            return Err(vec![invalid(
                "only a released Run occurrence can be recomputed",
            )]);
        };
        recompute(&self.plan, index, expected)
    }
}
