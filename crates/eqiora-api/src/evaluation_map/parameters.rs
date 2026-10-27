//! Independent stationary Runs bind every point against the same original Model.
use super::*;
use eqiora_artifact::{ModelEnvelope, WireValueLiteral};
use eqiora_core::ValueLiteral;
use eqiora_graph::{GraphStore, InMemoryGraphStore, Op, Precondition, Revision, Transaction};
use eqiora_numerics::{CommonResult, ResolvedCommonPlan};
use eqiora_solver::LinearSolverBackend;

#[derive(Debug, Clone)]
pub(super) struct ParameterInputs {
    pub(super) template: Arc<ResolvedCommonPlan>,
    pub(super) original: Arc<crate::ModelDocument>,
    pub(super) parameters: Arc<[Parameter]>,
    pub(super) points: Arc<[Vec<ValueLiteral>]>,
    pub(super) model: eqiora_artifact::ModelArtifactReference,
    pub(super) member_bytes_limit: usize,
    backend: Arc<dyn LinearSolverBackend + Send + Sync>,
}

pub(super) fn original_model(plan: &ResolvedCommonPlan) -> &ModelEnvelope {
    plan.as_algebraic()
        .and_then(|plan| plan.harmonic_original_model())
        .or_else(|| {
            plan.as_spatial()
                .and_then(|plan| plan.harmonic_original_model())
        })
        .unwrap_or_else(|| plan.model_artifact())
}

pub(super) fn document(plan: &ResolvedCommonPlan) -> Result<crate::ModelDocument, Vec<Diagnostic>> {
    let model = original_model(plan);
    let (transaction, model_id) = model.to_transaction()?;
    let store =
        InMemoryGraphStore::restore_snapshot(transaction, Revision(model.source_revision()))?;
    let mut geometries = Vec::new();
    if let Some(mesh) = plan.authenticated_mesh() {
        geometries.extend(mesh.geometry().cloned());
        geometries.extend_from_slice(mesh.model_geometries());
    }
    let geometry_refs = geometries.iter().collect::<Vec<_>>();
    let program = eqiora_sem::KernelProgram::from_snapshot_with_geometry(
        &store.snapshot(),
        model_id,
        &geometry_refs,
    )?;
    crate::ModelDocument::from_store(store, program, Default::default(), geometries, None)
}

pub(super) fn validate_template(template: &ResolvedCommonPlan) -> Result<(), Diagnostic> {
    if !(template.as_algebraic().is_some() || template.as_spatial().is_some())
        || template.backward_euler().is_some()
        || template.ode_temporal().is_some()
    {
        return Err(invalid("parameter maps require a stationary Plan"));
    }
    Ok(())
}

pub(super) fn validate_points(
    original: &crate::ModelDocument,
    parameters: &[Parameter],
    points: &[Vec<ValueLiteral>],
) -> Result<(), Diagnostic> {
    let mut seen = std::collections::BTreeSet::new();
    for parameter in parameters {
        if !seen.insert(parameter.ulid())
            || original.program().typed_value(parameter.erase()).is_none()
        {
            return Err(invalid(
                "stationary map requires distinct Parameters of its original Model",
            ));
        }
    }
    for point in points {
        if point.len() != parameters.len()
            || parameters.iter().zip(point).any(|(id, value)| {
                original
                    .program()
                    .typed_value(id.erase())
                    .is_none_or(|before| before.value_type() != value.value_type())
            })
        {
            return Err(invalid(
                "stationary map point differs from its complete Parameter types, units or order",
            ));
        }
    }
    Ok(())
}

/// The ordinary value-edit owner checks each changed value; one shared transaction commits the point.
pub(super) fn bind(
    original: &crate::ModelDocument,
    parameters: &[Parameter],
    point: &[ValueLiteral],
) -> Result<ModelEnvelope, Vec<Diagnostic>> {
    let mut transaction = Transaction::new("bind stationary map Parameter point");
    transaction.require(Precondition::RevisionIs(original.program().revision()));
    let mut changed = false;
    for (id, after) in parameters.iter().zip(point) {
        let before = original
            .program()
            .typed_value(id.erase())
            .expect("validated Parameter");
        if before == after {
            continue;
        }
        original
            .preview_value_edit(id.erase(), after.clone())
            .map_err(|error| vec![error])?;
        transaction.require(Precondition::ValueEquals {
            target: id.erase(),
            expected: before.clone(),
        });
        transaction.push(Op::SetValue {
            target: id.erase(),
            value: after.clone(),
        });
        changed = true;
    }
    if !changed {
        return ModelEnvelope::from_json(
            &original.canonical_json().map_err(|error| vec![error])?,
            Default::default(),
        )
        .map_err(|error| vec![error]);
    }
    let wire = crate::ModelTransactionEnvelope::from_transaction(&transaction)
        .map_err(|error| vec![error])?;
    let digest = wire.digest().map_err(|error| vec![error])?.to_string();
    let bound = original.replay_model_transaction(
        &wire,
        &digest,
        "stationary map transaction identity changed",
    )?;
    ModelEnvelope::from_json(
        &bound.canonical_json().map_err(|error| vec![error])?,
        Default::default(),
    )
    .map_err(|error| vec![error])
}

impl ParameterInputs {
    pub(super) fn same_inputs(&self, other: &Self) -> bool {
        self.template == other.template
            && self.parameters == other.parameters
            && self.points == other.points
    }
    pub(super) fn evaluate(&self, index: usize) -> time::RunOutcome {
        let result = (|| {
            let model = bind(&self.original, &self.parameters, &self.points[index])?;
            let plan = self
                .template
                .resolve_parameter_model(&model, self.backend.as_ref())
                .map_err(|error| vec![error])?;
            // Re-resolution retains the optional provider inventory. Execution follows the
            // newly resolved exact provider, including the built-in reference implementation.
            let execution_backend: &dyn LinearSolverBackend = match plan.linear_solver_provider() {
                Some(provider) if provider == eqiora_solver::REFERENCE_SOLVER_PROVIDER => {
                    &eqiora_solver::REFERENCE_LINEAR_SOLVER
                }
                Some(provider) if provider == self.backend.provider() => self.backend.as_ref(),
                _ => {
                    return Err(vec![invalid(
                        "stationary member selected an unavailable execution provider",
                    )]);
                }
            };
            let result = match &plan {
                ResolvedCommonPlan::Algebraic(plan) => plan
                    .initial_state(&[])
                    .and_then(|state| plan.run_result(&state, execution_backend)),
                ResolvedCommonPlan::Spatial(plan) => plan.run_result(None, execution_backend),
                _ => Err(invalid(
                    "stationary parameter map requires an ordinary stationary Plan",
                )),
            }
            .map_err(|error| vec![error])?;
            validate_result(
                &self.original,
                &self.parameters,
                &self.points[index],
                &result,
            )
            .map_err(|error| vec![error])?;
            check_member_size(&result, self.member_bytes_limit).map_err(|error| vec![error])?;
            time::admit_common_result(result, self.member_bytes_limit)
        })();
        match result {
            Ok(member) => outcomes::Outcome::Accepted(Box::new(member)),
            Err(mut errors) => {
                if errors.is_empty() {
                    errors.push(invalid("stationary Run failed without a diagnostic"));
                }
                outcomes::Outcome::Failed(errors)
            }
        }
    }
}

pub(super) fn validate_result(
    original: &crate::ModelDocument,
    parameters: &[Parameter],
    point: &[ValueLiteral],
    result: &CommonResult,
) -> Result<(), Diagnostic> {
    let expected = bind(original, parameters, point).map_err(crate::first_diagnostic)?;
    if original_model(result.plan()) != &expected || result.trajectory().is_some() {
        return Err(invalid(
            "stationary map Result belongs to a foreign original Model or requested Parameter point",
        ));
    }
    Ok(())
}

pub(super) fn check_member_size(result: &CommonResult, limit: usize) -> Result<(), Diagnostic> {
    let bytes = result
        .to_bytes()?
        .len()
        .checked_add(result.plan().to_bytes()?.len())
        .ok_or_else(|| invalid("stationary member encoded byte charge overflow"))?;
    if bytes > limit {
        return Err(invalid(
            "stationary map member Result and Plan exceed their encoded byte limit",
        ));
    }
    Ok(())
}

pub(super) fn storage_charge(
    parameters: &[Parameter],
    points: &[Vec<ValueLiteral>],
    policy: EvaluationMapExecutionPolicy,
    member_bytes_limit: usize,
) -> Result<usize, Diagnostic> {
    if member_bytes_limit == 0 {
        return Err(invalid(
            "stationary map requires a positive member byte limit",
        ));
    }
    let metadata = eqiora_artifact::JsonLimits::EXECUTION
        .encode(
            &points
                .iter()
                .map(|point| {
                    point
                        .iter()
                        .map(WireValueLiteral::encode)
                        .collect::<Result<Vec<_>, _>>()
                })
                .collect::<Result<Vec<_>, _>>()?,
        )
        .map_err(|_| invalid("cannot encode stationary map points"))?
        .len();
    let resident = match policy.retention() {
        EvaluationMapRetention::Retain => points.len(),
        EvaluationMapRetention::Recompute => points.len().min(policy.chunk_size()),
    };
    let charge = resources::multiply(
        resident
            .checked_add(policy.workers())
            .ok_or_else(|| invalid("stationary map storage charge overflow"))?,
        member_bytes_limit,
    )?
    .checked_add(metadata)
    .and_then(|v| v.checked_add(parameters.len().checked_mul(size_of::<Parameter>())?))
    .and_then(|v| {
        v.checked_add(
            points
                .len()
                .checked_mul(size_of::<time::RunOutcome>() + size_of::<String>() + 64)?,
        )
    })
    .ok_or_else(|| invalid("stationary map storage charge overflow"))?;
    if charge > policy.storage_bytes_limit() || charge > isize::MAX as usize {
        return Err(invalid(
            "stationary map storage charge exceeds its limit or addressable storage",
        ));
    }
    Ok(charge)
}

impl EvaluationMapPlan {
    /// Admit ordered, complete typed Parameter points for independent stationary Runs.
    /// Each point binds the retained original Model and repeats ordinary reduction and resolution.
    /// Repeated and unchanged points remain distinct occurrences. Numerical admission failures
    /// belong to their indexed member, not to this structural inventory admission.
    /// Storage charges encoded typed points, selected IDs, indexed outcomes and each resident
    /// member's encoded Result plus Plan limit. Shared template/Model storage, allocator overhead,
    /// numerical scratch and serialization buffers are excluded; this is not a heap bound.
    pub fn from_parameters(
        template: Arc<ResolvedCommonPlan>,
        parameters: &[Parameter],
        points: &[Vec<ValueLiteral>],
        policy: EvaluationMapExecutionPolicy,
        member_bytes_limit: usize,
        backend: Arc<dyn LinearSolverBackend + Send + Sync>,
    ) -> Result<Self, Diagnostic> {
        validate_template(&template)?;
        let original = Arc::new(document(&template).map_err(crate::first_diagnostic)?);
        validate_points(&original, parameters, points)?;
        let charge = storage_charge(parameters, points, policy, member_bytes_limit)?;
        let model = original.artifact_reference()?;
        Ok(Self {
            inputs: Arc::new(Inputs::Parameters(ParameterInputs {
                template,
                original,
                parameters: parameters.into(),
                points: points.into(),
                model,
                member_bytes_limit,
                backend,
            })),
            policy,
            estimated_storage_bytes: charge,
        })
    }
    /// Frozen typed points, absent for differentiable and time maps.
    pub fn parameter_points(&self) -> Option<&[Vec<ValueLiteral>]> {
        match self.inputs.as_ref() {
            Inputs::Parameters(inputs) => Some(&inputs.points),
            _ => None,
        }
    }
    /// Original selected Parameter identities in each point's declared order.
    pub fn parameters(&self) -> Option<&[Parameter]> {
        match self.inputs.as_ref() {
            Inputs::Parameters(inputs) => Some(&inputs.parameters),
            _ => None,
        }
    }
    /// Original stationary template; each accepted Result retains its own derived Plan.
    pub fn parameter_template(&self) -> Option<&ResolvedCommonPlan> {
        match self.inputs.as_ref() {
            Inputs::Parameters(inputs) => Some(&inputs.template),
            _ => None,
        }
    }
}

#[cfg(test)]
mod tests;
