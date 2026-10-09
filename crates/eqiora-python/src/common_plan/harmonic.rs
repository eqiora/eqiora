//! Projection of the retained harmonic lineage; reduction stays in numerics.
use super::*;

pub(crate) fn original_model(
    plan: &ResolvedCommonPlan,
) -> Option<&eqiora::artifact::ModelEnvelope> {
    match plan {
        ResolvedCommonPlan::Algebraic(plan) => plan.harmonic_original_model(),
        ResolvedCommonPlan::Linear(plan) => plan.harmonic_original_model(),
        _ => None,
    }
}

pub(super) fn amplitudes(py: Python<'_>, plan: &ResolvedCommonPlan) -> PyResult<Py<PyTuple>> {
    let Some(original) = original_model(plan) else {
        return Ok(PyTuple::empty(py).unbind());
    };
    let original_digest = original
        .artifact_reference()
        .map_err(|error| validation_error(py, &[error]))?
        .artifact()
        .to_string();
    let pairs: Vec<_> = match plan {
        ResolvedCommonPlan::Algebraic(plan) => plan.harmonic_amplitudes().collect(),
        ResolvedCommonPlan::Linear(plan) => plan.harmonic_amplitudes().collect(),
        _ => unreachable!("harmonic original owner"),
    };
    PyTuple::new(
        py,
        pairs.into_iter().map(|(name, original, amplitude)| {
            (
                name.to_owned(),
                PyModelFieldRef::from_exact(original_digest.clone(), original.ulid().to_string()),
                PyModelFieldRef::from_exact(
                    plan.model_digest().to_owned(),
                    amplitude.ulid().to_string(),
                ),
            )
        }),
    )
    .map(Bound::unbind)
}
