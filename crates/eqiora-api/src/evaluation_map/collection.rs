//! Read-only ordered executions with explicit groups and immutable request lineage.
use std::collections::{BTreeMap, BTreeSet};

use super::*;
use crate::{DiagnosticRecord, RecordedDifferentiableEvaluation};
use eqiora_artifact::ArtifactDigest;
use eqiora_numerics::{CommonResult, RecordedTimeCheckpoint};

mod artifact;
mod request;
mod request_artifact;
use request::RecordedRequest;

/// Historical outcome; accepted members of a partial group keep their own authority.
#[derive(Debug, Clone, PartialEq)]
pub enum RecordedEvaluationOutcome {
    Accepted(Arc<RecordedDifferentiableEvaluation>),
    TimeAccepted(Arc<CommonResult>),
    Failed(Vec<DiagnosticRecord>),
    Cancelled(Option<Box<RecordedTimeCheckpoint>>),
    NotStarted,
}

impl RecordedEvaluationOutcome {
    fn reference(&self) -> Option<&str> {
        match self {
            Self::Accepted(record) => Some(record.identity()),
            Self::TimeAccepted(result) => Some(result.identity()),
            _ => None,
        }
    }

    pub fn is_accepted(&self) -> bool {
        matches!(self, Self::Accepted(_) | Self::TimeAccepted(_))
    }
}

/// One exact requested inventory and its original execution outcomes.
/// Different Plans, signatures and meshes remain in explicitly separate groups.
#[derive(Debug, Clone, PartialEq)]
pub struct EvaluationCollectionGroup {
    execution: ulid::Ulid,
    request: RecordedRequest,
    outcomes: Vec<RecordedEvaluationOutcome>,
    estimators: Vec<crate::RecordedMomentEstimator>,
}

impl EvaluationCollectionGroup {
    /// Computed statistics attributed to this complete group, never an incomplete parent.
    pub fn estimators(&self) -> &[crate::RecordedMomentEstimator] {
        &self.estimators
    }
    pub fn execution_id(&self) -> String {
        self.execution.to_string()
    }
    pub fn len(&self) -> usize {
        self.outcomes.len()
    }
    pub fn is_empty(&self) -> bool {
        self.outcomes.is_empty()
    }
    pub fn is_complete(&self) -> bool {
        self.outcomes
            .iter()
            .all(RecordedEvaluationOutcome::is_accepted)
    }
    pub fn occurrence(
        &self,
        index: usize,
    ) -> Option<(EvaluationMapOccurrenceId, &RecordedEvaluationOutcome)> {
        self.outcomes.get(index).map(|outcome| {
            (
                EvaluationMapOccurrenceId::new(self.execution, index),
                outcome,
            )
        })
    }
    pub fn program_identity(&self) -> Option<&DifferentiableProgramIdentity> {
        self.request.program_identity()
    }
    pub fn points(&self) -> Option<&[DifferentiableValues]> {
        self.request.points()
    }
    pub fn samples(&self) -> Option<&[SampledParameterPoint]> {
        self.request.samples()
    }
    pub fn time_requests(&self) -> Option<&[eqiora_numerics::CommonTimeRunRequest]> {
        self.request.time_requests()
    }
    /// Original checkpoint content; restoration remains an explicit operation.
    pub fn time_checkpoints(&self) -> Option<&[Option<RecordedTimeCheckpoint>]> {
        self.request.time_checkpoints()
    }
    pub fn plan(&self) -> &eqiora_numerics::ResolvedCommonPlan {
        self.request.plan()
    }
    pub fn shared_inputs(&self) -> Option<&[Parameter]> {
        self.request.shared_inputs()
    }
}

/// A bounded content-addressed collection of explicitly grouped recorded executions.
///
/// This value has no executor, sampler, live Program, derivative action or implicit
/// statistical reduction. Repeated requests and sample IDs remain separate occurrences.
#[derive(Debug, Clone, PartialEq)]
pub struct EvaluationCollection {
    identity: String,
    groups: Vec<EvaluationCollectionGroup>,
}

impl EvaluationCollection {
    /// Attach a computed statistic to an explicitly selected complete execution group.
    /// Time estimators must match its exact ordered Results. A statistic on one group
    /// never represents other groups or an incomplete combined collection.
    pub fn with_estimator<T>(
        &self,
        group: usize,
        estimator: &crate::MomentEstimator<'_, T>,
    ) -> Result<Self, Diagnostic> {
        let mut groups = self.groups.clone();
        let group = groups
            .get_mut(group)
            .ok_or_else(|| invalid("statistic group is out of range"))?;
        group
            .estimators
            .push(crate::RecordedMomentEstimator::capture(group, estimator)?);
        Self::new(groups)
    }

    pub fn identity(&self) -> &str {
        &self.identity
    }
    pub fn groups(&self) -> &[EvaluationCollectionGroup] {
        &self.groups
    }
    pub fn is_complete(&self) -> bool {
        self.groups
            .iter()
            .all(EvaluationCollectionGroup::is_complete)
    }

    /// Combine explicitly associated groups, preserving their order and original IDs.
    /// A copied execution cannot appear twice, even if its values happen to differ.
    pub fn combine(collections: Vec<Self>) -> Result<Self, Diagnostic> {
        Self::new(
            collections
                .into_iter()
                .flat_map(|value| value.groups)
                .collect(),
        )
    }

    fn new(groups: Vec<EvaluationCollectionGroup>) -> Result<Self, Diagnostic> {
        let mut ids = BTreeSet::new();
        for group in &groups {
            group.request.validate()?;
            if !ids.insert(group.execution) || group.request.len() != group.outcomes.len() {
                return Err(invalid(
                    "collection has a duplicate execution or incomplete occurrence inventory",
                ));
            }
            if !group.is_complete()
                && !group.outcomes.iter().any(|state| {
                    matches!(
                        state,
                        RecordedEvaluationOutcome::Failed(_)
                            | RecordedEvaluationOutcome::Cancelled(_)
                    )
                })
            {
                return Err(invalid(
                    "incomplete collection has no original failure or cancellation",
                ));
            }
            let mut estimates = BTreeSet::new();
            for estimator in &group.estimators {
                estimator.validate(group)?;
                if !estimates.insert(estimator.identity()) {
                    return Err(invalid("duplicate recorded estimator content"));
                }
            }
            for (index, outcome) in group.outcomes.iter().enumerate() {
                group.request.validate_outcome(index, outcome)?;
            }
        }
        let mut collection = Self {
            identity: String::new(),
            groups,
        };
        collection.identity =
            ArtifactDigest::compute(artifact::SCHEMA.as_bytes(), &collection.to_bytes()?)
                .as_str()
                .to_owned();
        Ok(collection)
    }

    fn capture<'a>(
        plan: &EvaluationMapPlan,
        execution: ulid::Ulid,
        occurrences: impl Iterator<Item = EvaluationMapOccurrence<'a>>,
        mut delivered: BTreeMap<EvaluationMapOccurrenceId, RecordedEvaluationOutcome>,
    ) -> Result<Self, Diagnostic> {
        let request = RecordedRequest::capture(plan)?;
        let mut outcomes = Vec::with_capacity(plan.len());
        let mut records = BTreeMap::new();
        for (index, occurrence) in occurrences.enumerate() {
            let id = EvaluationMapOccurrenceId::new(execution, index);
            let value = match occurrence {
                EvaluationMapOccurrence::Released(evidence) => match delivered.remove(&id) {
                    Some(RecordedEvaluationOutcome::Accepted(record))
                        if record.result().evidence() == evidence =>
                    {
                        RecordedEvaluationOutcome::Accepted(record)
                    }
                    _ => {
                        return Err(invalid(
                            "released occurrence requires its exact originally delivered record",
                        ));
                    }
                },
                EvaluationMapOccurrence::TimeReleased(expected) => match delivered.remove(&id) {
                    Some(RecordedEvaluationOutcome::TimeAccepted(result))
                        if time::acceptance_fingerprint(&result)? == expected =>
                    {
                        RecordedEvaluationOutcome::TimeAccepted(result)
                    }
                    _ => {
                        return Err(invalid(
                            "released time occurrence requires its exact originally delivered Result",
                        ));
                    }
                },
                other => {
                    // Retained members have the same original payload; consume any captured copy.
                    delivered.remove(&id);
                    record_outcome(other)?
                }
            };
            let value = match value.reference() {
                Some(reference) => records.entry(reference.to_owned()).or_insert(value).clone(),
                None => value,
            };
            outcomes.push(value);
        }
        if !delivered.is_empty() {
            return Err(invalid(
                "delivery belongs to a different collection execution",
            ));
        }
        Self::new(vec![EvaluationCollectionGroup {
            execution,
            request,
            outcomes,
            estimators: Vec::new(),
        }])
    }
}

impl CompleteEvaluationMap {
    /// Capture retained original members; released values require capture during execution.
    pub fn record(&self) -> Result<EvaluationCollection, Diagnostic> {
        self.record_delivered(BTreeMap::new())
    }
    fn record_delivered(
        &self,
        delivered: BTreeMap<EvaluationMapOccurrenceId, RecordedEvaluationOutcome>,
    ) -> Result<EvaluationCollection, Diagnostic> {
        self.validate()?;
        let occurrences = match &self.members {
            outcomes::CompleteMembers::Evaluations(outcomes::Completed::Retained(members)) => {
                members
                    .iter()
                    .map(EvaluationMapOccurrence::Accepted)
                    .collect::<Vec<_>>()
            }
            outcomes::CompleteMembers::Evaluations(outcomes::Completed::Released(members)) => {
                members
                    .iter()
                    .map(EvaluationMapOccurrence::Released)
                    .collect()
            }
            outcomes::CompleteMembers::Time(outcomes::Completed::Retained(members)) => (0..members
                .len())
                .map(|i| {
                    EvaluationMapOccurrence::TimeAccepted(
                        self.time_result(i).expect("retained time member"),
                    )
                })
                .collect(),
            outcomes::CompleteMembers::Time(outcomes::Completed::Released(members)) => members
                .iter()
                .map(|s| EvaluationMapOccurrence::TimeReleased(s))
                .collect(),
        };
        EvaluationCollection::capture(
            &self.plan,
            self.execution,
            occurrences.into_iter(),
            delivered,
        )
    }
}

impl EvaluationMapTerminalReport {
    /// Preserve the complete requested inventory and all original terminal states.
    pub fn record(&self) -> Result<EvaluationCollection, Diagnostic> {
        self.record_delivered(BTreeMap::new())
    }
    fn record_delivered(
        &self,
        delivered: BTreeMap<EvaluationMapOccurrenceId, RecordedEvaluationOutcome>,
    ) -> Result<EvaluationCollection, Diagnostic> {
        EvaluationCollection::capture(
            &self.plan,
            self.execution,
            (0..self.plan.len()).map(|i| self.occurrence(i).expect("complete terminal inventory")),
            delivered,
        )
    }
}

impl EvaluationMapPlan {
    /// Execute through ordinary delivery and capture even explicitly released members.
    ///
    /// Returned collections can be incomplete; inspect `is_complete`. Recording owns
    /// additional payload storage outside the map's numerical-retention charge and
    /// enforces the shared encoded artifact bound. No member is implicitly rerun.
    pub fn execute_recorded<C: FnMut() -> bool>(
        &self,
        mut should_cancel: C,
    ) -> Result<EvaluationCollection, Diagnostic> {
        let mut delivered = BTreeMap::new();
        let mut charge = 0usize;
        let error = std::cell::RefCell::new(None);
        let result = self.execute_with_delivery(
            || error.borrow().is_some() || should_cancel(),
            |id, occurrence| {
                if error.borrow().is_some() {
                    return;
                }
                let capture = record_outcome(occurrence).and_then(|record| {
                    charge = charge
                        .checked_add(artifact::outcome_bytes(&record)?)
                        .ok_or_else(|| invalid("recorded collection byte charge overflow"))?;
                    if charge > eqiora_artifact::JsonLimits::EXECUTION.max_bytes {
                        return Err(invalid(
                            "recorded collection exceeds its artifact byte limit",
                        ));
                    }
                    delivered.insert(id, record);
                    Ok(())
                });
                if let Err(failure) = capture {
                    *error.borrow_mut() = Some(failure);
                }
            },
        );
        if let Some(error) = error.into_inner() {
            return Err(error);
        }
        match result {
            Ok(complete) => complete.record_delivered(delivered),
            Err(partial) => partial.record_delivered(delivered),
        }
    }
}

fn record_outcome(
    value: EvaluationMapOccurrence<'_>,
) -> Result<RecordedEvaluationOutcome, Diagnostic> {
    Ok(match value {
        EvaluationMapOccurrence::Accepted(value) => {
            RecordedEvaluationOutcome::Accepted(Arc::new(value.record()?))
        }
        EvaluationMapOccurrence::TimeAccepted(value) => {
            RecordedEvaluationOutcome::TimeAccepted(Arc::new(value.clone()))
        }
        EvaluationMapOccurrence::TimeCancelled(value) => {
            RecordedEvaluationOutcome::Cancelled(Some(Box::new(value.record()?)))
        }
        EvaluationMapOccurrence::Failed(errors) => RecordedEvaluationOutcome::Failed(
            errors
                .iter()
                .cloned()
                .map(DiagnosticRecord::from_kernel)
                .collect::<Result<_, _>>()
                .map_err(|()| {
                    invalid("collection diagnostic exceeds the shared diagnostic limits")
                })?,
        ),
        EvaluationMapOccurrence::Cancelled => RecordedEvaluationOutcome::Cancelled(None),
        EvaluationMapOccurrence::NotStarted => RecordedEvaluationOutcome::NotStarted,
        EvaluationMapOccurrence::Released(_) | EvaluationMapOccurrence::TimeReleased(_) => {
            return Err(invalid(
                "released occurrence has no original payload to record",
            ));
        }
    })
}

#[cfg(test)]
mod tests;
