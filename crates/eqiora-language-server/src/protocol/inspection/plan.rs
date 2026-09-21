//! Thin, read-only projection of the existing canonical numerical Plan owner.
use eqiora::{
    api::ModelDocument,
    backends::{diffsol::DIFFSOL_TIME_BACKEND, faer::FaerLinearSolver},
};
use eqiora_numerics::ResolvedCommonPlan;
use serde_json::{Value, json};

pub(super) fn project(bytes: &[u8], selected: &ModelDocument) -> Result<Value, String> {
    // Decoding re-resolves admission and checks exact canonical bytes and provider
    // versions. It never executes the Plan or accepts renderer-authored controls.
    let plan = ResolvedCommonPlan::from_bytes(bytes, &FaerLinearSolver, DIFFSOL_TIME_BACKEND)
        .map_err(|error| format!("Cannot validate numerical Plan: {}", error.message()))?;
    let selected_digest = selected
        .digest()
        .map_err(|error| error.message().to_owned())?;
    let mut metadata: Value = serde_json::from_slice(bytes).map_err(|error| error.to_string())?;
    // The artifact codec owns the vocabulary. Omit embedded binary replay roots
    // from display; expose their accepted digests instead.
    if let Some(fields) = metadata.as_object_mut() {
        fields.retain(|name, _| !name.ends_with("_base64"));
    }
    Ok(json!({
        "identity": plan.identity(),
        "modelDigest": plan.model_digest(),
        "modelRevision": plan.model_revision(),
        "selectedModelDigest": selected_digest,
        "matchesSelectedModel": plan.model_digest() == selected_digest,
        "geometryDigest": plan.geometry_digest(),
        "meshDigest": plan.mesh_digest(),
        "solverBackend": plan.solver_backend(),
        "solverBackendVersion": plan.solver_backend_version(),
        "metadata": metadata,
    }))
}

#[cfg(test)]
mod tests {
    use super::*;
    use eqiora::artifact::{ModelDecoderLimits, ModelEnvelope};
    use eqiora::kernel::KernelNode;
    use eqiora_numerics::{CommonTsitouras45, CommonTsitourasTolerance, resolve_common_ode_plan};

    fn model(rate: u8) -> ModelDocument {
        ModelDocument::compile("decay.eqi", &format!("model Decay() {{ parameter rate: 1 / s = {rate}; state x: 1; initial {{ x = 1; }} relation decay {{ derivative(x) = -rate * x; }} }}")).unwrap()
    }

    fn artifact(model: &ModelDocument) -> Vec<u8> {
        let envelope = ModelEnvelope::from_json(
            &model.canonical_json().unwrap(),
            ModelDecoderLimits::default(),
        )
        .unwrap();
        let field = model
            .program()
            .nodes()
            .find_map(|node| match node {
                KernelNode::Field(field) => Some(field.id()),
                _ => None,
            })
            .unwrap();
        resolve_common_ode_plan(
            &envelope,
            model.program(),
            CommonTsitouras45::new(
                0.01,
                1e-6,
                vec![CommonTsitourasTolerance::new(field, 1e-9).unwrap()],
            )
            .unwrap(),
            DIFFSOL_TIME_BACKEND,
        )
        .unwrap()
        .to_bytes()
        .unwrap()
    }

    #[test]
    fn validated_projection_preserves_exact_binding_and_admitted_controls() {
        let selected = model(1);
        let bytes = artifact(&selected);
        let view = project(&bytes, &selected).unwrap();
        assert_eq!(view["matchesSelectedModel"], true);
        assert_eq!(view["modelDigest"], selected.digest().unwrap());
        assert_eq!(view["solverBackend"], DIFFSOL_TIME_BACKEND.id());
        assert_eq!(view["metadata"]["family"], "ode");
        assert_eq!(view["metadata"]["temporal"]["initial_step_s"], 0.01);
        assert_eq!(view["metadata"]["temporal"]["relative_tolerance"], 1e-6);
        assert_eq!(
            view["metadata"]["temporal"]["absolute_tolerances"][0]["value"],
            1e-9
        );
        assert!(view["metadata"].get("model_base64").is_none());
        assert_eq!(view["geometryDigest"], Value::Null);
        assert_eq!(view["meshDigest"], Value::Null);
        let different = project(&bytes, &model(2)).unwrap();
        assert_eq!(different["matchesSelectedModel"], false);
        assert_ne!(different["modelDigest"], different["selectedModelDigest"]);
    }

    #[test]
    fn invalid_noncanonical_and_tampered_plans_are_rejected() {
        let selected = model(1);
        assert!(
            project(b"{}", &selected)
                .unwrap_err()
                .contains("Cannot validate")
        );
        let bytes = artifact(&selected);
        let mut wire: Value = serde_json::from_slice(&bytes).unwrap();
        assert!(project(&serde_json::to_vec_pretty(&wire).unwrap(), &selected).is_err());
        wire["identity"] = json!("invented identity");
        assert!(project(&serde_json::to_vec(&wire).unwrap(), &selected).is_err());
    }
}
