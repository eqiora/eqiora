//! Exact Python references projected into the common native Program.
use super::*;
use crate::common_plan::PyPlan;
use crate::model::{PyModelFieldRef, PyModelParameterRef, PyObservableRef};
use crate::trajectory::PyState;
use eqiora::api::ModelEntityRef;
use eqiora::entity::kinds;
use pyo3::types::PySequence;

/// Compile one exact accepted-point differentiable program.
#[pyfunction(name = "_compile_differentiable")]
#[allow(clippy::too_many_arguments)]
#[pyo3(signature = (plan, /, *, inputs, outputs, state=None, quadrature_points=None, until_s=None, output_times_s=None, steps=None, output_steps=None))]
pub(crate) fn compile_differentiable(
    py: Python<'_>,
    plan: &PyPlan,
    inputs: &Bound<'_, PyAny>,
    outputs: &Bound<'_, PyAny>,
    state: Option<&PyState>,
    quadrature_points: Option<usize>,
    until_s: Option<f64>,
    output_times_s: Option<Vec<f64>>,
    steps: Option<usize>,
    output_steps: Option<Vec<usize>>,
) -> PyResult<PyDifferentiableProgram> {
    panic_boundary(py, || {
        let invalid = |message: &str| {
            validation_error(
                py,
                &[eqiora::Diagnostic::error(
                    eqiora::diagnostic::codes::INVALID_LINEARIZATION,
                    message,
                )],
            )
        };
        let sequence = inputs.cast::<PySequence>().map_err(|_| {
            invalid("differentiable inputs must be an ordered sequence of ParameterRef values")
        })?;
        let mut selected = Vec::with_capacity(sequence.len()?);
        for index in 0..sequence.len()? {
            let item = sequence.get_item(index)?;
            selected.push(
                item.extract::<PyRef<'_, PyModelParameterRef>>()?
                    .value
                    .clone(),
            );
        }
        let native_plan = plan.native().clone();
        let backend = crate::execution::resolved_linear_backend(
            native_plan
                .linear_solver_provider()
                .ok_or_else(|| invalid("differentiation requires a linear solver provider"))?,
        )
        .map_err(|diagnostics| diagnostic_error(py, &diagnostics))?;
        let request = if native_plan.is_temporal() {
            let state = state.and_then(PyState::common_native).ok_or_else(|| {
                pyo3::exceptions::PyTypeError::new_err(
                    "spatial Run differentiation requires state=State(...)",
                )
            })?;
            Some(crate::execution::spatial_request(
                py,
                &native_plan,
                state,
                until_s,
                output_times_s,
                steps,
                output_steps,
            )?)
        } else {
            if until_s.is_some()
                || output_times_s.is_some()
                || steps.is_some()
                || output_steps.is_some()
            {
                return Err(pyo3::exceptions::PyTypeError::new_err(
                    "stationary differentiation accepts no transient Run controls",
                ));
            }
            None
        };
        let initial = state
            .filter(|_| request.is_none())
            .map(|state| {
                state
                    .initial_state()
                    .map(eqiora_numerics::CommonInitialState::into_owned)
                    .ok_or_else(|| {
                        invalid("differentiation requires an exact native initial State")
                    })
            })
            .transpose()?;
        let model = native_plan.model_artifact();
        let sequence = outputs.cast::<PySequence>().map_err(|_| {
            invalid("differentiable outputs must be an ordered sequence of FieldRef or ObservableRef values")
        })?;
        let mut selected_outputs = Vec::with_capacity(sequence.len()?);
        for index in 0..sequence.len()? {
            let output = sequence.get_item(index)?;
            let (output, expected_model) =
                if let Ok(field) = output.extract::<PyRef<'_, PyModelFieldRef>>() {
                    (
                        ModelEntityRef::from_model_artifact(
                            model,
                            ulid::Ulid::from_string(field.exact_id())
                                .map(eqiora::Id::<kinds::Field>::from_ulid)
                                .map_err(|_| {
                                    invalid("differentiable FieldRef has an invalid identity")
                                })?,
                        )
                        .map_err(|diagnostics| diagnostic_error(py, &diagnostics))?
                        .erase(),
                        field.exact_model_digest().to_owned(),
                    )
                } else if let Ok(observable) = output.extract::<PyRef<'_, PyObservableRef>>() {
                    (
                        ModelEntityRef::from_model_artifact(model, observable.id)
                            .map_err(|diagnostics| diagnostic_error(py, &diagnostics))?
                            .erase(),
                        observable.model_digest.clone(),
                    )
                } else {
                    return Err(invalid(
                        "differentiable output must be an exact FieldRef or ObservableRef",
                    ));
                };
            if output.model().artifact().to_string() != expected_model {
                return Err(invalid(
                    "differentiable output belongs to another exact Model artifact",
                ));
            }
            selected_outputs.push(output);
        }
        let quadratures = match quadrature_points {
            None => std::collections::HashMap::new(),
            Some(points) => {
                let observables = selected_outputs
                    .iter()
                    .filter_map(|output| {
                        output.id().downcast::<eqiora::entity::kinds::Observable>()
                    })
                    .collect::<Vec<_>>();
                native_plan
                    .observation_quadratures(&observables, points)
                    .map_err(|diagnostic| diagnostic_error(py, &[diagnostic]))?
            }
        };
        let value = py
            .detach(move || match request {
                Some(request) => DifferentiableProgram::compile_run(
                    request.into(),
                    &selected,
                    &selected_outputs,
                    quadratures,
                    backend,
                ),
                None => DifferentiableProgram::compile(
                    native_plan,
                    &selected,
                    &selected_outputs,
                    quadratures,
                    initial,
                    backend,
                ),
            })
            .map_err(|diagnostics| diagnostic_error(py, &diagnostics))?;
        Ok(PyDifferentiableProgram {
            value: Arc::new(value),
        })
    })
}
