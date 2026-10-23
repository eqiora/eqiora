//! Canonical event proposal receipts reconstructed against exact Plan roots.
use super::*;
use eqiora_time::{RootProposal, TimeEventDiscontinuity};

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct WireEvent {
    registration_sha256: [u8; 32],
    time: f64,
    root_index: u64,
    before_state: Vec<f64>,
    after_state: Vec<f64>,
    report: WireReport,
}
pub(crate) fn encode(events: &[TimeEventDiscontinuity]) -> Result<Vec<WireEvent>, Diagnostic> {
    events
        .iter()
        .map(|event| {
            let proposal = event.proposal();
            Ok(WireEvent {
                registration_sha256: proposal.registration().as_sha256(),
                time: proposal.time(),
                root_index: u64::try_from(proposal.root_index())
                    .map_err(|_| invalid("event root index exceeds u64"))?,
                before_state: proposal.state().to_vec(),
                after_state: event.after_state().to_vec(),
                report: WireReport::from_native(proposal.report())?,
            })
        })
        .collect()
}

pub(crate) fn replay(
    events: &[WireEvent],
    plan: &crate::CommonOdePlan,
) -> Result<Vec<TimeEventDiscontinuity>, Diagnostic> {
    if events.is_empty() {
        return Ok(Vec::new());
    }
    let policy = plan
        .event_policy()
        .ok_or_else(|| invalid("event history requires explicit ODE event policy"))?;
    if events.len() > policy.max_events() {
        return Err(invalid("event history exceeds the exact Plan event budget"));
    }
    let roots = plan
        .root_set()?
        .ok_or_else(|| invalid("event history requires the exact registered root set"))?;
    events
        .iter()
        .map(|event| {
            if event.registration_sha256 != roots.registration().as_sha256() {
                return Err(invalid(
                    "event receipt registration differs from the exact Plan root set",
                ));
            }
            let root_index = usize::try_from(event.root_index)
                .map_err(|_| invalid("event root index exceeds usize"))?;
            let proposal = RootProposal::accepted(
                roots.registration(),
                event.time,
                root_index,
                roots.events().len(),
                event.before_state.clone(),
                plan.state_dimensions().len(),
                event.report.replay(plan.backend())?,
            )?;
            TimeEventDiscontinuity::accepted(proposal, event.after_state.clone())
        })
        .collect()
}

#[cfg(test)]
mod tests;
