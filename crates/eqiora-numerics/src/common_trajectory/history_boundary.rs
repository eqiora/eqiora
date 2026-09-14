//! Check only retained native states; a quadrature stencil is not an interpolant.
use super::*;

pub(super) fn validate(
    history: &AcceptedTimeHistory,
    time: f64,
    values: &[f64],
) -> Result<(), Diagnostic> {
    let events = history.events();
    let event = events.get(events.partition_point(|event| event.proposal().time() < time));
    // Requested output at an event always observes the committed reset side,
    // including a reset at the terminal instant with no following interval.
    let known = if let Some(event) = event.filter(|event| event.proposal().time() == time) {
        Some(event.after_state())
    } else {
        history
            .steps()
            .get(
                history
                    .steps()
                    .partition_point(|step| step.end_time() < time),
            )
            .and_then(|step| {
                if time == step.start_time() {
                    Some(step.start_state())
                } else if time == step.end_time() {
                    Some(step.end_state())
                } else if time == step.start_time() + (step.end_time() - step.start_time()) * 0.5 {
                    Some(step.midpoint_state())
                } else {
                    None
                }
            })
    };
    if known.is_some_and(|known| known != values) {
        return Err(invalid(
            "ODE output State differs from the retained native history state at the same exact time",
        ));
    }
    Ok(())
}

#[cfg(test)]
mod tests;
