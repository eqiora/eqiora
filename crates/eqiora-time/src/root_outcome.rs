//! Native smooth root-search prefixes, before the hybrid owner commits a reset.
use crate::diagnostic::time_solve_failed;
use crate::{AcceptedTimeHistory, RootProposal, TimeSolution};
use eqiora_core::Diagnostic;

/// Requested native samples and complete smooth history through the first root or horizon.
#[derive(Debug, Clone, PartialEq)]
pub struct TimeRootOutcome {
    kind: OutcomeKind,
}

#[derive(Debug, Clone, PartialEq)]
enum OutcomeKind {
    Horizon(TimeSolution),
    Root {
        proposal: RootProposal,
        history: AcceptedTimeHistory,
        samples: Option<TimeSolution>,
    },
}

impl TimeRootOutcome {
    /// Accept a completed smooth solve retaining native history.
    /// # Errors
    /// Rejects an output-only solution or history containing committed resets.
    pub fn horizon(solution: TimeSolution) -> Result<Self, Diagnostic> {
        if solution
            .history()
            .is_none_or(|history| !history.events().is_empty())
        {
            return Err(time_solve_failed(
                "root-search horizon requires smooth native history",
            ));
        }
        Ok(Self {
            kind: OutcomeKind::Horizon(solution),
        })
    }

    /// Accept a native prefix ending exactly at a localized, uncommitted root.
    ///
    /// Samples contain only requested times strictly before the root. A requested
    /// output at the root belongs to the hybrid owner's post-reset state.
    /// # Errors
    /// Rejects inconsistent root time/state/report or samples outside the prefix.
    pub fn localized(
        proposal: RootProposal,
        history: AcceptedTimeHistory,
        samples: Option<TimeSolution>,
    ) -> Result<Self, Diagnostic> {
        let first = &history.steps()[0];
        let last = history
            .steps()
            .last()
            .expect("accepted history is nonempty");
        if !history.events().is_empty()
            || history.dimension() != proposal.state().len()
            || last.end_time() != proposal.time()
            || last.end_state() != proposal.state()
            || samples.as_ref().is_some_and(|samples| {
                samples.dimension() != history.dimension()
                    || samples.report() != proposal.report()
                    || samples.times()[0] < first.start_time()
                    || *samples
                        .times()
                        .last()
                        .expect("accepted samples are nonempty")
                        >= proposal.time()
                    || samples.history().is_some()
            })
        {
            return Err(time_solve_failed(
                "root-search prefix differs from its exact localized proposal or requested samples",
            ));
        }
        Ok(Self {
            kind: OutcomeKind::Root {
                proposal,
                history,
                samples,
            },
        })
    }

    /// Uncommitted localized root, absent when the requested horizon was reached.
    #[must_use]
    pub const fn proposal(&self) -> Option<&RootProposal> {
        match &self.kind {
            OutcomeKind::Horizon(_) => None,
            OutcomeKind::Root { proposal, .. } => Some(proposal),
        }
    }

    /// Complete smooth native history, including a native stencil cut at the root.
    #[must_use]
    pub fn history(&self) -> &AcceptedTimeHistory {
        match &self.kind {
            OutcomeKind::Horizon(solution) => solution.history().expect("checked native history"),
            OutcomeKind::Root { history, .. } => history,
        }
    }

    /// Requested samples before a root, or all requested samples at the horizon.
    ///
    /// Absent when a root precedes the first requested output.
    #[must_use]
    pub const fn samples(&self) -> Option<&TimeSolution> {
        match &self.kind {
            OutcomeKind::Horizon(solution) => Some(solution),
            OutcomeKind::Root { samples, .. } => samples.as_ref(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        InitialConditionPolicy, RootRegistrationId, TimeBackendIdentity, TimeEquationClass,
        TimeExecutionReport, TimeHistoryStep, TimeMethod,
    };

    #[test]
    fn event_time_samples_and_mismatched_prefix_are_not_accepted() {
        let report = TimeExecutionReport::new(
            TimeBackendIdentity::new("test.roots", "1"),
            TimeMethod::Tsitouras45,
            TimeEquationClass::ExplicitOde,
            InitialConditionPolicy::Provided,
        );
        let root = RootProposal::accepted(
            RootRegistrationId::from_sha256([2; 32]),
            1.0,
            0,
            1,
            vec![0.0],
            1,
            report,
        )
        .unwrap();
        let history = AcceptedTimeHistory::accepted(
            1,
            vec![TimeHistoryStep::accepted(0.0, 1.0, vec![1.0], vec![0.5], vec![0.0]).unwrap()],
            vec![],
        )
        .unwrap();
        assert!(TimeRootOutcome::localized(root.clone(), history.clone(), None).is_ok());
        let at_event = TimeSolution::accepted(1, vec![1.0], vec![0.0], report).unwrap();
        assert!(TimeRootOutcome::localized(root.clone(), history.clone(), Some(at_event)).is_err());
        let wrong_history = AcceptedTimeHistory::accepted(
            1,
            vec![TimeHistoryStep::accepted(0.0, 1.0, vec![1.0], vec![0.5], vec![0.1]).unwrap()],
            vec![],
        )
        .unwrap();
        assert!(TimeRootOutcome::localized(root, wrong_history, None).is_err());
        let output_only = TimeSolution::accepted(1, vec![0.5], vec![0.5], report).unwrap();
        assert!(TimeRootOutcome::horizon(output_only).is_err());
    }
}
