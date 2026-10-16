//! Independent time Runs use the existing bounded map scheduler.
use super::*;
use eqiora_artifact::{CanonicalModelArtifact, ModelArtifactReference};
use eqiora_numerics::{CommonResult, CommonTimeCheckpoint, CommonTimePlan, CommonTimeRunRequest};
use std::{fmt, ops::ControlFlow};

mod members;
#[cfg(test)]
mod tests;
pub(super) use members::{Completed, TimeOutcome, occurrence};
use members::{TimeMember, admit_result};

type Run = dyn Fn(
        &CommonTimeRunRequest,
        Option<&CommonTimeCheckpoint>,
    ) -> Result<ControlFlow<CommonTimeCheckpoint, CommonResult>, Vec<Diagnostic>>
    + Send
    + Sync;

#[derive(Clone)]
pub(super) struct TimeInputs {
    pub(super) model: ModelArtifactReference,
    pub(super) plan: Arc<CommonTimePlan>,
    pub(super) requests: Arc<[CommonTimeRunRequest]>,
    pub(super) checkpoints: Arc<[Option<CommonTimeCheckpoint>]>,
    run: Arc<Run>,
    member_bytes_limit: usize,
}

impl fmt::Debug for TimeInputs {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("TimeInputs")
            .field("plan", &self.plan.identity())
            .field("requests", &self.requests)
            .field("checkpoints", &self.checkpoints)
            .field("member_bytes_limit", &self.member_bytes_limit)
            .finish_non_exhaustive()
    }
}

impl TimeInputs {
    pub(super) fn same_inputs(&self, other: &Self) -> bool {
        self.plan.identity() == other.plan.identity()
            && self.requests == other.requests
            && self.checkpoints == other.checkpoints
    }

    fn evaluate(&self, index: usize) -> TimeOutcome {
        let request = &self.requests[index];
        let result =
            (self.run)(request, self.checkpoints[index].as_ref()).and_then(
                |outcome| match outcome {
                    ControlFlow::Continue(result) => {
                        admit_result(request, result, self.member_bytes_limit)
                            .map(|member| outcomes::Outcome::Accepted(Box::new(member)))
                    }
                    ControlFlow::Break(checkpoint) => {
                        if checkpoint.request().identity() != request.identity() {
                            return Err(vec![invalid(
                                "time map checkpoint belongs to a foreign Run request",
                            )]);
                        }
                        check_size(
                            checkpoint.to_bytes().map_err(|error| vec![error])?.len(),
                            self.member_bytes_limit,
                        )
                        .map_err(|error| vec![error])?;
                        Ok(outcomes::Outcome::Cancelled(Some(Box::new(checkpoint))))
                    }
                },
            );
        result.unwrap_or_else(|mut errors| {
            if errors.is_empty() {
                errors.push(invalid("time Run failed without a diagnostic"));
            }
            outcomes::Outcome::Failed(errors)
        })
    }
}

impl EvaluationMapPlan {
    /// Admit complete independent time requests through the ordinary time Run adapter.
    ///
    /// Every request must use this exact Plan and the same retained output times.
    /// Initial States and internal steps/events may differ. No padding, interpolation,
    /// output retiming, failed-member filtering or trajectory derivative is introduced.
    /// The ordinary Run owns event-side sampling and complete-horizon acceptance.
    ///
    /// `run` executes one ordinary Run and returns its accepted Result or atomic
    /// checkpoint. It must create independent numerical state per call; the existing
    /// Plan/Run/Result owners admit its backend and history. The callback can observe
    /// its own cancellation token within a member; map cancellation is also checked
    /// before dispatch. Checkpoints, when supplied, bind the original complete requests.
    ///
    /// The storage charge includes request coordinates/output inventories, encoded
    /// checkpoints, indexed terminal records and the configured per-member encoded
    /// Result/checkpoint limit times the retention/worker count. Shared Plan storage,
    /// callback captures, allocator overhead, solver/history scratch, serialization
    /// buffers and OS thread stacks are excluded; this is not a heap or peak-memory
    /// bound. Each produced member is checked before collection acceptance.
    /// Recompute retains a canonical Result fingerprint excluding elapsed wall time.
    ///
    /// # Errors
    /// Rejects foreign Plans/checkpoints, inconsistent observation schedules and
    /// unaddressable or excessive storage charges before copying or running members.
    pub fn from_time_requests<F>(
        plan: Arc<CommonTimePlan>,
        requests: &[CommonTimeRunRequest],
        checkpoints: Option<&[Option<CommonTimeCheckpoint>]>,
        policy: EvaluationMapExecutionPolicy,
        member_bytes_limit: usize,
        run: F,
    ) -> Result<Self, Diagnostic>
    where
        F: Fn(
                &CommonTimeRunRequest,
                Option<&CommonTimeCheckpoint>,
            )
                -> Result<ControlFlow<CommonTimeCheckpoint, CommonResult>, Vec<Diagnostic>>
            + Send
            + Sync
            + 'static,
    {
        if member_bytes_limit == 0
            || checkpoints.is_some_and(|values| values.len() != requests.len())
        {
            return Err(invalid(
                "time map requires a positive member byte limit and complete checkpoint inventory",
            ));
        }
        let mut charge = 0usize;
        let add = |a: usize, b: usize| {
            a.checked_add(b)
                .ok_or_else(|| invalid("time map storage charge overflow"))
        };
        for (index, request) in requests.iter().enumerate() {
            if request.plan().identity() != plan.identity()
                || request.output_times_s() != requests[0].output_times_s()
            {
                return Err(invalid(
                    "time map requires one exact Plan and fixed observation times",
                ));
            }
            let coordinates = add(
                request.state().values().len(),
                resources::multiply(add(request.output_times_s().len(), 1)?, 3)?,
            )?;
            charge = add(
                charge,
                add(
                    size_of::<CommonTimeRunRequest>(),
                    resources::multiply(coordinates, size_of::<f64>())?,
                )?,
            )?;
            if let Some((_, identity)) = request.sampling() {
                for bytes in [
                    identity.sample_id().len(),
                    identity.channel().len(),
                    resources::multiply(identity.lineage().len(), size_of::<u64>())?,
                    identity.coupling_id().map_or(0, <[u8]>::len),
                    32,
                ] {
                    charge = add(charge, bytes)?;
                }
            }
            if let Some(checkpoint) = checkpoints.and_then(|values| values[index].as_ref()) {
                request.require_continuation()?;
                if checkpoint.request().identity() != request.identity() {
                    return Err(invalid(
                        "time map input checkpoint belongs to a foreign Run request",
                    ));
                }
                charge = add(charge, checkpoint.to_bytes()?.len())?;
            }
        }
        let count = requests.len();
        let chunk = count.min(policy.chunk_size());
        let workers = chunk.min(policy.workers());
        let numerical = match policy.retention() {
            EvaluationMapRetention::Retain => add(count, workers)?,
            EvaluationMapRetention::Recompute => add(chunk, workers)?,
        };
        charge = add(
            charge,
            resources::multiply(
                count,
                add(
                    add(
                        size_of::<TimeOutcome>(),
                        size_of::<Option<CommonTimeCheckpoint>>(),
                    )?,
                    add(size_of::<String>(), 64)?,
                )?,
            )?,
        )?;
        charge = add(charge, resources::multiply(numerical, member_bytes_limit)?)?;
        charge = add(
            charge,
            resources::multiply(
                workers,
                add(size_of::<TimeOutcome>(), 4 * size_of::<usize>())?,
            )?,
        )?;
        if charge > policy.storage_bytes_limit() || charge > isize::MAX as usize {
            return Err(invalid(
                "time map storage charge exceeds its limit or addressable storage",
            ));
        }
        let model = plan.model_artifact().artifact_reference()?;
        Ok(Self {
            inputs: Arc::new(Inputs::Time(TimeInputs {
                model,
                plan,
                requests: requests.to_vec().into(),
                checkpoints: checkpoints
                    .map_or_else(|| vec![None; count], <[_]>::to_vec)
                    .into(),
                run: Arc::new(run),
                member_bytes_limit,
            })),
            policy,
            estimated_storage_bytes: charge,
        })
    }

    /// The immutable time Plan, absent for steady evaluation maps.
    #[must_use]
    pub fn time_plan(&self) -> Option<&CommonTimePlan> {
        match self.inputs.as_ref() {
            Inputs::Time(inputs) => Some(&inputs.plan),
            _ => None,
        }
    }
    /// Complete original time requests in occurrence order, including repeats.
    #[must_use]
    pub fn time_requests(&self) -> Option<&[CommonTimeRunRequest]> {
        match self.inputs.as_ref() {
            Inputs::Time(inputs) => Some(&inputs.requests),
            _ => None,
        }
    }
    /// Explicit restart checkpoints in the same original occurrence order.
    #[must_use]
    pub fn time_checkpoints(&self) -> Option<&[Option<CommonTimeCheckpoint>]> {
        match self.inputs.as_ref() {
            Inputs::Time(inputs) => Some(&inputs.checkpoints),
            _ => None,
        }
    }
}

fn inputs(plan: &EvaluationMapPlan) -> Result<&TimeInputs, Diagnostic> {
    match plan.inputs.as_ref() {
        Inputs::Time(inputs) => Ok(inputs),
        _ => Err(invalid(
            "this map contains steady evaluations, not time Runs",
        )),
    }
}

fn check_size(bytes: usize, limit: usize) -> Result<(), Diagnostic> {
    if bytes > limit {
        return Err(invalid("time map member exceeds its encoded byte limit"));
    }
    Ok(())
}

pub(super) fn execute<C, D>(
    plan: &EvaluationMapPlan,
    inputs: &TimeInputs,
    should_cancel: &mut C,
    deliver: &mut D,
) -> Result<CompleteEvaluationMap, EvaluationMapTerminalReport>
where
    C: FnMut() -> bool,
    D: FnMut(usize, EvaluationMapOccurrence<'_>),
{
    let outcomes = schedule::execute_members(
        plan.len(),
        plan.policy(),
        &|index| inputs.evaluate(index),
        should_cancel,
        &mut |index, member: &TimeMember| {
            deliver(index, EvaluationMapOccurrence::TimeAccepted(&member.result))
        },
        &|member: &TimeMember| member.fingerprint.clone(),
        &mut |_| {},
    );
    members::finish(plan, outcomes)
}
