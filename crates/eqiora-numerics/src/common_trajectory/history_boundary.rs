//! Validate outputs against retained native stencils and captured dense polynomials.
use super::*;

pub(crate) fn validate(
    history: &AcceptedTimeHistory,
    time: f64,
    values: &[f64],
) -> Result<(), Diagnostic> {
    let events = history.events();
    let event = events.get(events.partition_point(|event| event.proposal().time() < time));
    // Requested output at an event always observes the committed reset side,
    // including a reset at the terminal instant with no following interval.
    let known = if let Some(event) = event.filter(|event| event.proposal().time() == time) {
        Some(event.after_state().to_vec())
    } else {
        sample(history.steps(), time)?
    };
    if known.is_some_and(|known| known.as_slice() != values) {
        return Err(invalid(
            "ODE output State differs from the retained native history state at the same exact time",
        ));
    }
    Ok(())
}

pub(super) fn sample(
    steps: &[eqiora_time::TimeHistoryStep],
    time: f64,
) -> Result<Option<Vec<f64>>, Diagnostic> {
    let Some(step) = steps.get(steps.partition_point(|step| step.end_time() < time)) else {
        return Ok(None);
    };
    step.sample(time).map(Some)
}

#[cfg(test)]
mod tests;
