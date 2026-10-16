use super::super::outcomes::{CompleteMembers, Completed, Outcome, TerminalMembers};
use super::*;

#[derive(Debug, Clone)]
pub(in crate::evaluation_map) struct TimeMember {
    pub(super) result: CommonResult,
    pub(super) fingerprint: String,
}

pub(in crate::evaluation_map) type TimeOutcome = Outcome<TimeMember, String, CommonTimeCheckpoint>;

impl Completed<TimeMember, String> {
    pub(in crate::evaluation_map) fn validate(
        &self,
        plan: &EvaluationMapPlan,
    ) -> Result<usize, Diagnostic> {
        let inputs = inputs(plan)?;
        match self {
            Self::Retained(values) => {
                for (request, member) in inputs.requests.iter().zip(values) {
                    validate_request(request, &member.result)?;
                }
                Ok(values.len())
            }
            Self::Released(values) => Ok(values.len()),
        }
    }
}

fn validate_request(
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
) -> Result<TimeMember, Vec<Diagnostic>> {
    validate_request(request, &result).map_err(|error| vec![error])?;
    check_size(result.to_bytes().map_err(|error| vec![error])?.len(), limit)
        .map_err(|error| vec![error])?;
    // Wall time is presentation metadata, not mathematical acceptance evidence.
    // Use the existing canonical Result owner, never another JSON projection/codec.
    let fingerprint = result
        .clone()
        .with_elapsed_seconds(0.)
        .map_err(|error| vec![error])?
        .identity()
        .to_owned();
    Ok(TimeMember {
        result,
        fingerprint,
    })
}

pub(super) fn finish(
    plan: &EvaluationMapPlan,
    outcomes: Vec<TimeOutcome>,
) -> Result<CompleteEvaluationMap, EvaluationMapTerminalReport> {
    let members =
        Completed::from_outcomes(outcomes, plan.policy().retention()).map_err(|outcomes| {
            EvaluationMapTerminalReport {
                plan: plan.clone(),
                outcomes: TerminalMembers::Time(outcomes),
            }
        })?;
    Ok(CompleteEvaluationMap {
        plan: plan.clone(),
        members: CompleteMembers::Time(members),
    })
}

pub(in crate::evaluation_map) fn occurrence(value: &TimeOutcome) -> EvaluationMapOccurrence<'_> {
    match value {
        Outcome::Accepted(member) => EvaluationMapOccurrence::TimeAccepted(&member.result),
        Outcome::Released(fingerprint) => EvaluationMapOccurrence::TimeReleased(fingerprint),
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
    let inputs = inputs(plan).map_err(|error| vec![error])?;
    match inputs.evaluate(index) {
        Outcome::Accepted(member) if member.fingerprint == expected => Ok(member.result),
        Outcome::Failed(errors) => Err(errors),
        _ => Err(vec![invalid(
            "recomputed time occurrence changed its original acceptance or did not complete",
        )]),
    }
}

impl CompleteEvaluationMap {
    /// One retained time Result; steady maps and released occurrences return None.
    #[must_use]
    pub fn time_result(&self, index: usize) -> Option<&CommonResult> {
        match &self.members {
            CompleteMembers::Time(Completed::Retained(values)) => {
                values.get(index).map(|member| &member.result)
            }
            _ => None,
        }
    }
    /// Original canonical time acceptance fingerprint, excluding elapsed wall time.
    #[must_use]
    pub fn time_evidence(&self, index: usize) -> Option<&str> {
        match &self.members {
            CompleteMembers::Time(Completed::Retained(values)) => {
                values.get(index).map(|member| member.fingerprint.as_str())
            }
            CompleteMembers::Time(Completed::Released(values)) => {
                values.get(index).map(String::as_str)
            }
            _ => None,
        }
    }
    /// Reexecute one explicitly released time occurrence against its original evidence.
    /// No fresh sample identity or implicit retry is introduced.
    pub fn recompute_time(&self, index: usize) -> Result<CommonResult, Vec<Diagnostic>> {
        if self.plan.policy().retention() != EvaluationMapRetention::Recompute {
            return Err(vec![invalid(
                "time recomputation requires explicit Recompute retention",
            )]);
        }
        let expected = self
            .time_evidence(index)
            .ok_or_else(|| vec![invalid("time occurrence is absent")])?;
        recompute(&self.plan, index, expected)
    }
}

impl EvaluationMapTerminalReport {
    /// Retained time Result at its original position, without promoting incomplete membership.
    #[must_use]
    pub fn time_result(&self, index: usize) -> Option<&CommonResult> {
        match self.occurrence(index)? {
            EvaluationMapOccurrence::TimeAccepted(result) => Some(result),
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
    /// Recompute a released time member while keeping the original partial report partial.
    pub fn recompute_time(&self, index: usize) -> Result<CommonResult, Vec<Diagnostic>> {
        let Some(EvaluationMapOccurrence::TimeReleased(expected)) = self.occurrence(index) else {
            return Err(vec![invalid(
                "only a released time occurrence can be recomputed",
            )]);
        };
        recompute(&self.plan, index, expected)
    }
}
