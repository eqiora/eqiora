//! Authored regularity is mandatory persisted meaning, not a numerical hint.
use eqiora_artifact::{ModelDecoderLimits, ModelEnvelope, StructuralSemanticFingerprint};
use eqiora_graph::{GraphStore, InMemoryGraphStore, Op, Transaction};
use eqiora_schema::kernel::{KernelNode, SpatialRegularity};
use eqiora_sem::KernelProgram;

#[test]
fn changing_only_field_regularity_changes_fingerprint_and_survives_replay() {
    let source = "model M() { domain body=box(0,1,0,1); variable u:1 on body; relation law on body { u=0; } }";
    let (original, model, _) = eqiora_compiler::compile("regularity.eqi", source)
        .unwrap()
        .remove(0)
        .into_parts();
    let mut fingerprints = Vec::new();
    for regularity in [
        SpatialRegularity::Unspecified,
        SpatialRegularity::L2,
        SpatialRegularity::H1,
        SpatialRegularity::Smooth,
    ] {
        let mut transaction = Transaction::new("change only authored Field regularity");
        for op in original.ops() {
            transaction.push(match op {
                Op::DefineKernelNode {
                    node: KernelNode::Field(field),
                } => Op::DefineKernelNode {
                    node: field.clone().with_spatial_regularity(regularity).into(),
                },
                other => other.clone(),
            });
        }
        let mut store = InMemoryGraphStore::new();
        store.commit(transaction).unwrap();
        let program = KernelProgram::from_snapshot(&store.snapshot(), model).unwrap();
        let fingerprint = StructuralSemanticFingerprint::from_program(&program).unwrap();
        assert!(!fingerprints.contains(&fingerprint));
        fingerprints.push(fingerprint.clone());
        let bytes = ModelEnvelope::from_program(&program)
            .unwrap()
            .canonical_json()
            .unwrap();
        let replay = ModelEnvelope::from_json(&bytes, ModelDecoderLimits::default())
            .unwrap()
            .to_program()
            .unwrap();
        assert_eq!(
            StructuralSemanticFingerprint::from_program(&replay).unwrap(),
            fingerprint
        );
        let replay_fields = replay
            .nodes()
            .filter_map(|node| match node {
                KernelNode::Field(field) => Some(field.spatial_regularity()),
                _ => None,
            })
            .collect::<Vec<_>>();
        assert_eq!(replay_fields, [regularity]);
        let previous = String::from_utf8(bytes)
            .unwrap()
            .replace("eqiora.model-envelope/v45", "eqiora.model-envelope/v43");
        assert!(
            ModelEnvelope::from_json(previous.as_bytes(), ModelDecoderLimits::default()).is_err()
        );
    }
}
