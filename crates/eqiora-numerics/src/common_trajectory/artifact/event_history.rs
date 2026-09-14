//! Canonical event proposal receipts reconstructed against exact Plan roots.
use super::*;
use eqiora_time::{
    InitialConditionPolicy, RootProposal, TimeEquationClass, TimeEventDiscontinuity,
    TimeExecutionReport, TimeMethod,
};

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub(super) struct WireEvent {
    registration_sha256: [u8; 32],
    time: f64,
    root_index: u64,
    before_state: Vec<f64>,
    after_state: Vec<f64>,
    report: WireReport,
}
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct WireReport {
    backend: String,
    backend_version: String,
    method: WireMethod,
    equation_class: WireEquationClass,
    initial_condition: WireInitialCondition,
}
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
enum WireMethod {
    Tsitouras45,
}
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
enum WireEquationClass {
    ExplicitOde,
}
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
enum WireInitialCondition {
    Provided,
}

impl WireReport {
    fn from_native(report: TimeExecutionReport) -> Result<Self, Diagnostic> {
        if report.method() != TimeMethod::Tsitouras45
            || report.equation_class() != TimeEquationClass::ExplicitOde
            || report.initial_condition() != InitialConditionPolicy::Provided
        {
            return Err(invalid(
                "event receipt requires admitted explicit Tsitouras45 execution with provided initial data",
            ));
        }
        Ok(Self {
            backend: report.backend().to_owned(),
            backend_version: report.backend_version().to_owned(),
            method: WireMethod::Tsitouras45,
            equation_class: WireEquationClass::ExplicitOde,
            initial_condition: WireInitialCondition::Provided,
        })
    }
    fn replay(&self, plan: &crate::CommonOdePlan) -> Result<TimeExecutionReport, Diagnostic> {
        if self.backend != plan.backend().id() || self.backend_version != plan.backend().version() {
            return Err(invalid(
                "event receipt backend differs from the exact ODE Plan",
            ));
        }
        Ok(TimeExecutionReport::new(
            plan.backend(),
            TimeMethod::Tsitouras45,
            TimeEquationClass::ExplicitOde,
            InitialConditionPolicy::Provided,
        ))
    }
}

pub(super) fn encode(events: &[TimeEventDiscontinuity]) -> Result<Vec<WireEvent>, Diagnostic> {
    events
        .iter()
        .map(|event| {
            let proposal = event.proposal();
            Ok(WireEvent {
                registration_sha256: proposal.registration().as_sha256(),
                time: proposal.time(),
                root_index: to_u64(proposal.root_index(), "event root index")?,
                before_state: proposal.state().to_vec(),
                after_state: event.after_state().to_vec(),
                report: WireReport::from_native(proposal.report())?,
            })
        })
        .collect()
}

pub(super) fn replay(
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
            let root_index = to_usize(event.root_index, "event root index")?;
            let proposal = RootProposal::accepted(
                roots.registration(),
                event.time,
                root_index,
                roots.events().len(),
                event.before_state.clone(),
                plan.field_dimensions().len(),
                event.report.replay(plan)?,
            )?;
            TimeEventDiscontinuity::accepted(proposal, event.after_state.clone())
        })
        .collect()
}

#[cfg(test)]
mod tests;
