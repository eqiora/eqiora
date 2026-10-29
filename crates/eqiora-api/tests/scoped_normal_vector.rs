//! Boundary support and parameter-only data survive the public saved-model path.
use eqiora_api::ModelDocument;
use eqiora_schema::kernel::{ExprNode, KernelNode};

fn source(boundary: &str, profile: &str) -> String {
    format!(
        r#"model M() {{
      domain body=box(0,2,0,3);
      domain left=boundary(body,axis=0,side=lower);
      domain top=boundary(body,axis=1,side=upper);
      variable u:vector<m/s,2> on body in smooth;
      parameter speed:m/s=2;
      relation prescribed on {boundary} {{ trace(u) = {profile}; }}
    }}"#
    )
}

#[test]
fn parameter_only_normal_keeps_exact_boundary_through_saved_model() {
    let model =
        ModelDocument::compile("normal.eqi", &source("left", "-normal_vector() * speed")).unwrap();
    let bytes = model.canonical_json().unwrap();
    let replay = ModelDocument::replay(&bytes).unwrap();
    assert_eq!(
        model.structural_fingerprint().unwrap(),
        replay.structural_fingerprint().unwrap()
    );
    assert_eq!(bytes, replay.canonical_json().unwrap());
    let boundary = replay
        .program()
        .nodes()
        .find_map(|node| match node {
            KernelNode::Relation(relation) => {
                relation
                    .expression()
                    .nodes()
                    .iter()
                    .find_map(|node| match node {
                        ExprNode::NormalVector { on } => Some(*on),
                        _ => None,
                    })
            }
            _ => None,
        })
        .expect("saved normal expression");
    assert!(matches!(
        replay.program().node(boundary.erase()),
        Some(KernelNode::Domain(_))
    ));
    let other =
        ModelDocument::compile("normal.eqi", &source("top", "-normal_vector() * speed")).unwrap();
    assert_ne!(
        model.structural_fingerprint().unwrap(),
        other.structural_fingerprint().unwrap()
    );
}

#[test]
fn scoped_normal_rejects_volume_scope_and_wrong_physical_units() {
    assert!(
        ModelDocument::compile("normal.eqi", &source("body", "normal_vector() * speed")).is_err()
    );
    assert!(
        ModelDocument::compile("normal.eqi", &source("left", "normal_vector() * 2[m]")).is_err()
    );
}

#[test]
fn immediate_previous_envelopes_are_rejected_without_fallback() {
    use eqiora_artifact::{ModelEnvelope, ModelTransactionEnvelope};
    let document =
        ModelDocument::compile("normal.eqi", &source("left", "normal_vector() * speed")).unwrap();
    let artifact = ModelEnvelope::from_program(document.program()).unwrap();
    let mut model: serde_json::Value =
        serde_json::from_slice(&artifact.canonical_json().unwrap()).unwrap();
    model["schema"] = serde_json::json!("eqiora.model-envelope/v49");
    assert!(ModelDocument::replay(&serde_json::to_vec(&model).unwrap()).is_err());
    let (transaction, _) = artifact.to_transaction().unwrap();
    let mut wire: serde_json::Value = serde_json::from_slice(
        &ModelTransactionEnvelope::from_transaction(&transaction)
            .unwrap()
            .canonical_json()
            .unwrap(),
    )
    .unwrap();
    wire["schema"] = serde_json::json!("eqiora.model-transaction-envelope/v49");
    let error = ModelTransactionEnvelope::from_json(
        &serde_json::to_vec(&wire).unwrap(),
        Default::default(),
    )
    .unwrap_err();
    assert!(
        error
            .message()
            .contains("unsupported eqiora.model-transaction-envelope/v50 schema")
    );
}

#[test]
fn authored_normal_vector_name_remains_a_parameter_not_a_builtin() {
    let source = r#"model M() { parameter normal_vector: m/s=2;
        variable x:m/s; relation r { x=normal_vector; } }"#;
    let model = ModelDocument::compile("collision.eqi", source).unwrap();
    assert!(model.program().nodes().all(|node| {
        match node {
            KernelNode::Relation(relation) => relation
                .expression()
                .nodes()
                .iter()
                .all(|node| !matches!(node, ExprNode::NormalVector { .. })),
            _ => true,
        }
    }));
    assert!(
        ModelDocument::compile(
            "normal.eqi",
            &self::source("left", "normal_vector(speed) * speed")
        )
        .is_err()
    );
}
