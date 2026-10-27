use std::borrow::Cow;

use crate::DifferentiationEvidence;
use eqiora_core::{Diagnostic, diagnostic::codes};

use super::EvaluationMapOccurrenceId;
use super::{EvaluationMapPlan, EvaluationMapRetention, invalid, time, validate_member};
use crate::DifferentiableEvaluation;

#[derive(Debug, Clone)]
pub(super) enum Outcome<M = DifferentiableEvaluation, E = DifferentiationEvidence, S = ()> {
    Accepted(Box<M>),
    Released(Box<E>),
    Failed(Vec<Diagnostic>),
    Cancelled(Option<Box<S>>),
    NotStarted,
}

#[derive(Debug, Clone)]
pub(super) enum CompleteMembers {
    Evaluations(Completed<DifferentiableEvaluation, DifferentiationEvidence>),
    Runs(Completed<time::RunMember, String>),
}

/// Complete membership and its retention invariant have one owner for all maps.
#[derive(Debug, Clone)]
pub(super) enum Completed<M, E> {
    Retained(Vec<M>),
    Released(Vec<E>),
}

impl<M, E> Completed<M, E> {
    pub(super) fn from_outcomes<S>(
        outcomes: Vec<Outcome<M, E, S>>,
        retention: EvaluationMapRetention,
    ) -> Result<Self, Vec<Outcome<M, E, S>>> {
        if outcomes
            .iter()
            .any(|state| !matches!(state, Outcome::Accepted(_) | Outcome::Released(_)))
        {
            return Err(outcomes);
        }
        Ok(match retention {
            EvaluationMapRetention::Retain => Self::Retained(
                outcomes
                    .into_iter()
                    .map(|state| match state {
                        Outcome::Accepted(member) => *member,
                        _ => unreachable!("retained complete map"),
                    })
                    .collect(),
            ),
            EvaluationMapRetention::Recompute => Self::Released(
                outcomes
                    .into_iter()
                    .map(|state| match state {
                        Outcome::Released(evidence) => *evidence,
                        _ => unreachable!("released complete map"),
                    })
                    .collect(),
            ),
        })
    }
}

/// Complete ordered accepted membership, with an explicit numerical lifetime.
#[derive(Debug, Clone)]
pub struct CompleteEvaluationMap {
    pub(super) execution: ulid::Ulid,
    pub(super) plan: EvaluationMapPlan,
    pub(super) members: CompleteMembers,
}

impl CompleteEvaluationMap {
    /// Identity of this execution, separate from the admitted request meaning.
    #[must_use]
    pub fn execution_id(&self) -> String {
        self.execution.to_string()
    }

    /// Exact occurrence identity, also attached to its original delivery.
    #[must_use]
    pub fn occurrence_id(&self, index: usize) -> Option<EvaluationMapOccurrenceId> {
        (index < self.plan.len()).then(|| EvaluationMapOccurrenceId::new(self.execution, index))
    }

    pub(crate) fn validate(&self) -> Result<(), Diagnostic> {
        let count = match &self.members {
            CompleteMembers::Evaluations(Completed::Retained(members)) => {
                for (point, member) in self.plan.points()?.iter().zip(members) {
                    validate_member(&self.plan, point, member)?;
                }
                members.len()
            }
            CompleteMembers::Evaluations(Completed::Released(records)) => records.len(),
            CompleteMembers::Runs(members) => members.validate(&self.plan)?,
        };
        if count != self.plan.len() {
            return Err(invalid("complete map is missing an accepted occurrence"));
        }
        Ok(())
    }

    #[cfg(test)]
    pub(super) fn retained_members_mut(&mut self) -> &mut Vec<DifferentiableEvaluation> {
        match &mut self.members {
            CompleteMembers::Evaluations(Completed::Retained(members)) => members,
            _ => panic!("retained steady membership mutant"),
        }
    }
    pub(super) fn finish(
        plan: &EvaluationMapPlan,
        execution: ulid::Ulid,
        outcomes: Vec<Outcome>,
    ) -> Result<Self, EvaluationMapTerminalReport> {
        let members =
            Completed::from_outcomes(outcomes, plan.policy().retention()).map_err(|outcomes| {
                EvaluationMapTerminalReport {
                    execution,
                    plan: plan.clone(),
                    outcomes: TerminalMembers::Evaluations(outcomes),
                }
            })?;
        Ok(Self {
            execution,
            plan: plan.clone(),
            members: CompleteMembers::Evaluations(members),
        })
    }

    #[cfg(test)]
    pub(super) fn from_members(
        plan: &EvaluationMapPlan,
        members: Vec<DifferentiableEvaluation>,
    ) -> Result<Self, Diagnostic> {
        if members.len() != plan.len()
            || plan.policy().retention() != EvaluationMapRetention::Retain
        {
            return Err(invalid(
                "complete evaluation map requires exactly one retained member per occurrence",
            ));
        }
        for (point, member) in plan.points()?.iter().zip(&members) {
            validate_member(plan, point, member)?;
        }
        Ok(Self {
            execution: ulid::Ulid::generate(),
            plan: plan.clone(),
            members: CompleteMembers::Evaluations(Completed::Retained(members)),
        })
    }

    /// Exact admitted request, including empty membership and sampled inputs.
    #[must_use]
    pub const fn plan(&self) -> &EvaluationMapPlan {
        &self.plan
    }
    /// Number of accepted occurrences, independent of numerical retention.
    #[must_use]
    pub fn len(&self) -> usize {
        self.plan.len()
    }
    /// Whether the complete requested inventory is empty.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }
    /// Retained evaluations, or None under the explicit Recompute policy.
    #[must_use]
    pub fn members(&self) -> Option<&[DifferentiableEvaluation]> {
        match &self.members {
            CompleteMembers::Evaluations(Completed::Retained(values)) => Some(values),
            CompleteMembers::Evaluations(Completed::Released(_)) | CompleteMembers::Runs(_) => None,
        }
    }
    /// Select a retained occurrence without recomputation.
    #[must_use]
    pub fn evaluation(&self, index: usize) -> Option<&DifferentiableEvaluation> {
        self.members()?.get(index)
    }
    /// Original immutable acceptance evidence, even after numerical release.
    #[must_use]
    pub fn evidence(&self, index: usize) -> Option<Cow<'_, DifferentiationEvidence>> {
        match &self.members {
            CompleteMembers::Evaluations(Completed::Retained(values)) => values
                .get(index)
                .map(|member| Cow::Owned(member.primal_evidence())),
            CompleteMembers::Evaluations(Completed::Released(values)) => {
                values.get(index).map(Cow::Borrowed)
            }
            CompleteMembers::Runs(_) => None,
        }
    }
    /// Explicitly recompute one released occurrence from the frozen exact point.
    ///
    /// The ordinary evaluator must reproduce the entire original evidence. No
    /// sampler is called; a changed acceptance or numerical failure is an error,
    /// not an implicit retry. Retained maps expose their original member instead.
    ///
    /// # Errors
    /// Rejects retained policy, invalid index, failure or changed acceptance.
    pub fn recompute(&self, index: usize) -> Result<DifferentiableEvaluation, Vec<Diagnostic>> {
        if self.plan.policy().retention() != EvaluationMapRetention::Recompute {
            return Err(vec![invalid(
                "recomputation requires the explicit Recompute retention policy",
            )]);
        }
        let evidence = self
            .evidence(index)
            .ok_or_else(|| vec![invalid("evaluation map occurrence is out of bounds")])?;
        recompute(&self.plan, index, &evidence)
    }

    pub(crate) fn accepted_member(
        &self,
        index: usize,
    ) -> Result<Cow<'_, DifferentiableEvaluation>, Diagnostic> {
        if let Some(member) = self.evaluation(index) {
            return Ok(Cow::Borrowed(member));
        }
        self.recompute(index).map(Cow::Owned).map_err(|errors| {
            // Ordinary numerical diagnostics remain available through explicit
            // recompute; the first diagnostic owns a failed dense product.
            errors
                .into_iter()
                .next()
                .unwrap_or_else(|| invalid("map recomputation failed without a diagnostic"))
                .with_context(format!("map recomputation at point occurrence {index}"))
        })
    }
}

/// Borrowed indexed state; accepted members need not form a prefix.
#[derive(Debug)]
pub enum EvaluationMapOccurrence<'a> {
    /// Accepted numerical state retained and independently inspectable.
    Accepted(&'a DifferentiableEvaluation),
    /// Accepted state delivered and released; its original evidence is retained.
    Released(&'a DifferentiationEvidence),
    /// Accepted ordinary Result with its own exact Plan and original request association.
    RunAccepted(&'a eqiora_numerics::CommonResult),
    /// Canonical acceptance fingerprint excluding only elapsed wall time.
    RunReleased(&'a str),
    /// Atomic physical and solver/path checkpoint for this exact occurrence.
    TimeCancelled(&'a eqiora_numerics::CommonTimeCheckpoint),
    /// Original nonempty diagnostics for this exact failed occurrence.
    Failed(&'a [Diagnostic]),
    /// Cancellation observed before this occurrence was dispatched.
    Cancelled,
    /// Never dispatched after failure or cancellation.
    NotStarted,
}

/// Indexed incomplete membership; never convertible to a complete map.
///
/// In-flight workers can accept positions after a failure. Dense consumers must
/// propagate this report, not stack or normalize only its accepted members.
#[derive(Debug, Clone)]
pub struct EvaluationMapTerminalReport {
    pub(super) execution: ulid::Ulid,
    pub(super) plan: EvaluationMapPlan,
    pub(super) outcomes: TerminalMembers,
}

#[derive(Debug, Clone)]
pub(super) enum TerminalMembers {
    Evaluations(Vec<Outcome>),
    Runs(Vec<time::RunOutcome>),
}

impl EvaluationMapTerminalReport {
    /// Identity of this execution, separate from the admitted request meaning.
    #[must_use]
    pub fn execution_id(&self) -> String {
        self.execution.to_string()
    }

    /// Exact occurrence identity, also attached to its original delivery.
    #[must_use]
    pub fn occurrence_id(&self, index: usize) -> Option<EvaluationMapOccurrenceId> {
        (index < self.plan.len()).then(|| EvaluationMapOccurrenceId::new(self.execution, index))
    }

    /// Full requested inventory, including unstarted positions.
    #[must_use]
    pub const fn plan(&self) -> &EvaluationMapPlan {
        &self.plan
    }
    fn evaluations(&self) -> Option<&[Outcome]> {
        match &self.outcomes {
            TerminalMembers::Evaluations(values) => Some(values),
            TerminalMembers::Runs(_) => None,
        }
    }
    /// Retained steady evaluations paired with their original occurrence indices.
    pub fn accepted_members(&self) -> impl Iterator<Item = (usize, &DifferentiableEvaluation)> {
        self.evaluations()
            .into_iter()
            .flatten()
            .enumerate()
            .filter_map(|(index, state)| match state {
                Outcome::Accepted(member) => Some((index, member.as_ref())),
                _ => None,
            })
    }
    /// Smallest failed/cancelled position, not the accepted member count.
    #[must_use]
    pub fn stopped_index(&self) -> usize {
        (0..self.plan.len())
            .find(|&index| {
                matches!(
                    self.occurrence(index),
                    Some(
                        EvaluationMapOccurrence::Failed(_)
                            | EvaluationMapOccurrence::Cancelled
                            | EvaluationMapOccurrence::TimeCancelled(_)
                    )
                )
            })
            .expect("terminal reports retain an original failure or cancellation")
    }
    /// Original diagnostics at the primary terminal position.
    #[must_use]
    pub fn diagnostics(&self) -> &[Diagnostic] {
        match self
            .occurrence(self.stopped_index())
            .expect("terminal position")
        {
            EvaluationMapOccurrence::Failed(errors) => errors,
            EvaluationMapOccurrence::Cancelled | EvaluationMapOccurrence::TimeCancelled(_) => {
                CANCELLED.as_slice()
            }
            _ => unreachable!("terminal position"),
        }
    }
    /// Whether the primary terminal position was cancelled.
    #[must_use]
    pub fn is_cancelled(&self) -> bool {
        matches!(
            self.occurrence(self.stopped_index()),
            Some(EvaluationMapOccurrence::Cancelled | EvaluationMapOccurrence::TimeCancelled(_))
        )
    }
    /// Inspect one exact occurrence, including accepted holes and released state.
    #[must_use]
    pub fn occurrence(&self, index: usize) -> Option<EvaluationMapOccurrence<'_>> {
        let TerminalMembers::Evaluations(outcomes) = &self.outcomes else {
            let TerminalMembers::Runs(outcomes) = &self.outcomes else {
                unreachable!()
            };
            return outcomes.get(index).map(time::occurrence);
        };
        Some(match outcomes.get(index)? {
            Outcome::Accepted(value) => EvaluationMapOccurrence::Accepted(value),
            Outcome::Released(evidence) => EvaluationMapOccurrence::Released(evidence),
            Outcome::Failed(errors) => EvaluationMapOccurrence::Failed(errors),
            Outcome::Cancelled(_) => EvaluationMapOccurrence::Cancelled,
            Outcome::NotStarted => EvaluationMapOccurrence::NotStarted,
        })
    }
    /// Recompute a released accepted member without promoting partial membership.
    ///
    /// # Errors
    /// Rejects every non-released state, changed acceptance or numerical failure.
    pub fn recompute(&self, index: usize) -> Result<DifferentiableEvaluation, Vec<Diagnostic>> {
        let Some(Outcome::Released(evidence)) =
            self.evaluations().and_then(|values| values.get(index))
        else {
            return Err(vec![invalid(
                "only a released accepted occurrence can be recomputed",
            )]);
        };
        recompute(&self.plan, index, evidence)
    }
}

static CANCELLED: std::sync::LazyLock<[Diagnostic; 1]> = std::sync::LazyLock::new(|| {
    [Diagnostic::error(
        codes::EXECUTION_CANCELLED,
        "evaluation map cancelled at an occurrence boundary",
    )]
});

fn recompute(
    plan: &EvaluationMapPlan,
    index: usize,
    evidence: &DifferentiationEvidence,
) -> Result<DifferentiableEvaluation, Vec<Diagnostic>> {
    let point = &plan.points().map_err(|error| vec![error])?[index];
    let member = plan
        .program()
        .map_err(|error| vec![error])?
        .evaluate(point)?;
    validate_member(plan, point, &member).map_err(|error| vec![error])?;
    let member = plan
        .program()
        .map_err(|error| vec![error])?
        .prepare_map_member(member)
        .map_err(|error| vec![error])?;
    if &member.primal_evidence() != evidence {
        return Err(vec![invalid(format!(
            "recomputed occurrence {index} changed its original acceptance evidence"
        ))]);
    }
    Ok(member)
}

#[cfg(test)]
pub(super) fn verify_recompute_evidence_falsifier() {
    tests::recompute_rejects_substituted_original_evidence_without_repairing_membership();
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::evaluation_map::products::tests::FixtureProducts;
    use crate::evaluation_map::{EvaluationMapExecutionPolicy, tests::fixture};
    use eqiora_realization::Target;
    use std::{num::NonZeroUsize, sync::Arc};

    #[test]
    pub(super) fn recompute_rejects_substituted_original_evidence_without_repairing_membership() {
        let (_, program) = fixture();
        let program = Arc::new(program);
        let policy = EvaluationMapExecutionPolicy::new(
            2,
            Target::HostCpu {
                threads: NonZeroUsize::MIN,
            },
            EvaluationMapRetention::Recompute,
            1 << 30,
        )
        .unwrap();
        let plan = EvaluationMapPlan::new(
            program.clone(),
            &crate::evaluation_map::tests::typed_points(
                &program,
                &[&[3.0, 1.25, -0.25], &[2.0, 0.75, 0.5]],
            ),
            &[],
            policy,
        )
        .unwrap();
        let mut map = plan.execute().unwrap();
        let CompleteMembers::Evaluations(Completed::Released(records)) = &mut map.members else {
            panic!("released fixture")
        };
        records.swap(0, 1);
        let error = map.recompute(0).unwrap_err();
        assert!(
            error[0]
                .message()
                .contains("changed its original acceptance evidence")
        );
        assert!(map.members().is_none());
        let products = crate::EvaluationMapProducts::new(&map, &[2], &[], &[0], 1 << 30).unwrap();
        assert!(
            products
                .fixture_jvp(&[], &[1.0; 6])
                .unwrap_err()
                .message()
                .contains("changed its original acceptance evidence")
        );
        assert_eq!(map.len(), 2);
    }
}
