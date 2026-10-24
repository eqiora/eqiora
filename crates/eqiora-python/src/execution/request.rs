//! Shared Python schedule admission for ordinary Runs and differentiable Programs.
use super::*;
use eqiora_numerics::CommonState;

pub(crate) fn spatial_request(
    py: Python<'_>,
    plan: &ResolvedCommonPlan,
    state: &CommonState,
    until_s: Option<f64>,
    output_times_s: Option<Vec<f64>>,
    steps: Option<usize>,
    output_steps: Option<Vec<usize>>,
) -> PyResult<CommonTransientRunRequest> {
    match (until_s, output_times_s, steps, output_steps) {
        (Some(until), Some(outputs), None, None) => {
            CommonTransientRunRequest::from_times(plan.clone(), state.clone(), until, outputs)
        }
        (None, None, Some(steps), Some(outputs)) => {
            CommonTransientRunRequest::from_steps(plan.clone(), state.clone(), steps, outputs)
        }
        _ => return Err(PyTypeError::new_err(
            "transient Run requires exactly one complete until_s/output_times_s or steps/output_steps family",
        )),
    }
    .map_err(|diagnostic| validation_error(py, &[diagnostic]))
}
