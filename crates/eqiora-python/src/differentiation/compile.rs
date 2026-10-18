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
#[pyo3(signature = (plan, /, *, inputs, outputs, state=None, quadrature_points=None))]
pub(crate) fn compile_differentiable(
    py: Python<'_>,
    plan: &PyPlan,
    inputs: &Bound<'_, PyAny>,
    outputs: &Bound<'_, PyAny>,
    state: Option<&PyState>,
    quadrature_points: Option<usize>,
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
        let initial = state
            .map(|state| {
                state
                    .algebraic_native
                    .clone()
                    .ok_or_else(|| invalid("differentiation requires a finite initial State"))
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
            .detach(move || {
                DifferentiableProgram::compile(
                    native_plan,
                    &selected,
                    &selected_outputs,
                    quadratures,
                    initial,
                    backend,
                )
            })
            .map_err(|diagnostics| diagnostic_error(py, &diagnostics))?;
        Ok(PyDifferentiableProgram {
            value: Arc::new(value),
        })
    })
}
