use std::borrow::Cow;

use crate::DifferentiationEvidence;
use eqiora_core::{Diagnostic, diagnostic::codes};

use super::{EvaluationMapPlan, EvaluationMapRetention, invalid, validate_member};
use crate::DifferentiableEvaluation;

#[derive(Debug, Clone)]
pub(super) enum Outcome {
    Accepted(Box<DifferentiableEvaluation>),
    Released(Box<DifferentiationEvidence>),
    Failed(Vec<Diagnostic>),
    Cancelled,
    NotStarted,
}

#[derive(Debug, Clone)]
enum CompleteMembers {
    Retained(Vec<DifferentiableEvaluation>),
    Released(Vec<DifferentiationEvidence>),
}

/// Complete ordered accepted membership, with an explicit numerical lifetime.
#[derive(Debug, Clone)]
pub struct CompleteEvaluationMap {
    plan: EvaluationMapPlan,
    members: CompleteMembers,
}

impl CompleteEvaluationMap {
    pub(super) fn validate(&self) -> Result<(), Diagnostic> {
        let count = match &self.members {
            CompleteMembers::Retained(members) => {
                for (point, member) in self.plan.points().iter().zip(members) {
                    validate_member(&self.plan, point, member)?;
                }
                members.len()
            }
            CompleteMembers::Released(records) => records.len(),
        };
        if count != self.plan.points().len() {
            return Err(invalid("complete map is missing an accepted occurrence"));
        }
        Ok(())
    }

    #[cfg(test)]
    pub(super) fn retained_members_mut(&mut self) -> &mut Vec<DifferentiableEvaluation> {
        match &mut self.members {
            CompleteMembers::Retained(members) => members,
            CompleteMembers::Released(_) => panic!("retained membership mutant"),
        }
    }
    pub(super) fn finish(
        plan: &EvaluationMapPlan,
        outcomes: Vec<Outcome>,
    ) -> Result<Self, EvaluationMapTerminalReport> {
        if outcomes
            .iter()
            .any(|state| !matches!(state, Outcome::Accepted(_) | Outcome::Released(_)))
        {
            return Err(EvaluationMapTerminalReport {
                plan: plan.clone(),
                outcomes,
            });
        }
        let members = match plan.policy().retention() {
            EvaluationMapRetention::Retain => CompleteMembers::Retained(
                outcomes
                    .into_iter()
                    .map(|state| match state {
                        Outcome::Accepted(member) => *member,
                        _ => unreachable!("retained complete map"),
                    })
                    .collect(),
            ),
            EvaluationMapRetention::Recompute => CompleteMembers::Released(
                outcomes
                    .into_iter()
                    .map(|state| match state {
                        Outcome::Released(evidence) => *evidence,
                        _ => unreachable!("released complete map"),
                    })
                    .collect(),
            ),
        };
        Ok(Self {
            plan: plan.clone(),
            members,
        })
    }

    #[cfg(test)]
    pub(super) fn from_members(
        plan: &EvaluationMapPlan,
        members: Vec<DifferentiableEvaluation>,
    ) -> Result<Self, Diagnostic> {
        if members.len() != plan.points().len()
            || plan.policy().retention() != EvaluationMapRetention::Retain
        {
            return Err(invalid(
                "complete evaluation map requires exactly one retained member per occurrence",
            ));
        }
        for (point, member) in plan.points().iter().zip(&members) {
            validate_member(plan, point, member)?;
        }
        Ok(Self {
            plan: plan.clone(),
            members: CompleteMembers::Retained(members),
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
        self.plan.points().len()
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
            CompleteMembers::Retained(values) => Some(values),
            CompleteMembers::Released(_) => None,
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
            CompleteMembers::Retained(values) => values
                .get(index)
                .map(|member| Cow::Owned(member.primal_evidence())),
            CompleteMembers::Released(values) => values.get(index).map(Cow::Borrowed),
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

    pub(super) fn product_member(
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
                .with_context(format!(
                    "mapped product recomputation at point occurrence {index}"
                ))
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
    plan: EvaluationMapPlan,
    outcomes: Vec<Outcome>,
}

impl EvaluationMapTerminalReport {
    /// Full requested inventory, including unstarted positions.
    #[must_use]
    pub const fn plan(&self) -> &EvaluationMapPlan {
        &self.plan
    }
    /// Retained accepted members paired with their original occurrence indices.
    pub fn accepted_members(&self) -> impl Iterator<Item = (usize, &DifferentiableEvaluation)> {
        self.outcomes
            .iter()
            .enumerate()
            .filter_map(|(index, state)| match state {
                Outcome::Accepted(member) => Some((index, member.as_ref())),
                _ => None,
            })
    }
    /// Smallest failed/cancelled position, not the accepted member count.
    #[must_use]
    pub fn stopped_index(&self) -> usize {
        self.outcomes
            .iter()
            .position(|state| matches!(state, Outcome::Failed(_) | Outcome::Cancelled))
            .expect("terminal reports always have a failed or cancelled occurrence")
    }
    /// Diagnostics for the primary terminal position; inspect every failure with occurrence().
    #[must_use]
    pub fn diagnostics(&self) -> &[Diagnostic] {
        match &self.outcomes[self.stopped_index()] {
            Outcome::Failed(errors) => errors,
            Outcome::Cancelled => CANCELLED.as_slice(),
            _ => unreachable!("terminal position"),
        }
    }
    /// Whether the primary terminal position was cancelled, rather than failed.
    #[must_use]
    pub fn is_cancelled(&self) -> bool {
        matches!(self.outcomes[self.stopped_index()], Outcome::Cancelled)
    }
    /// Inspect one exact occurrence, including accepted holes and released state.
    #[must_use]
    pub fn occurrence(&self, index: usize) -> Option<EvaluationMapOccurrence<'_>> {
        Some(match self.outcomes.get(index)? {
            Outcome::Accepted(value) => EvaluationMapOccurrence::Accepted(value),
            Outcome::Released(evidence) => EvaluationMapOccurrence::Released(evidence),
            Outcome::Failed(errors) => EvaluationMapOccurrence::Failed(errors),
            Outcome::Cancelled => EvaluationMapOccurrence::Cancelled,
            Outcome::NotStarted => EvaluationMapOccurrence::NotStarted,
        })
    }
    /// Recompute a released accepted member without promoting partial membership.
    ///
    /// # Errors
    /// Rejects every non-released state, changed acceptance or numerical failure.
    pub fn recompute(&self, index: usize) -> Result<DifferentiableEvaluation, Vec<Diagnostic>> {
        let Some(Outcome::Released(evidence)) = self.outcomes.get(index) else {
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
    let point = &plan.points()[index];
    let member = plan.program.evaluate(point.coordinates())?;
    validate_member(plan, point, &member).map_err(|error| vec![error])?;
    let member = plan
        .program
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
        let plan =
            EvaluationMapPlan::new(program, &[&[3.0, 1.25, -0.25], &[2.0, 0.75, 0.5]], policy)
                .unwrap();
        let mut map = plan.execute().unwrap();
        let CompleteMembers::Released(records) = &mut map.members else {
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
        let products =
            crate::EvaluationMapProducts::new(&map, &[], &[2], &[], &[0], 1 << 30).unwrap();
        assert!(
            products
                .jvp(&[], &[1.0; 6])
                .unwrap_err()
                .message()
                .contains("changed its original acceptance evidence")
        );
        assert_eq!(map.len(), 2);
    }
}
