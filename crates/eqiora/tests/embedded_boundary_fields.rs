//! Boundary Fields retain intrinsic measure and ambient frame independently.
use eqiora::api::ModelDocument;
use eqiora::kernel::KernelNode;

const LINE: &str = "model Line() {
  domain body=box(0,2,0,3);
  domain wall=boundary(body,axis=0,side=lower);
  variable density:kg/m on wall;
  relation retain on wall {density=1[kg/m];}
  observable mass:kg=integral(density,measure(wall));
}";

#[test]
fn physical_line_field_and_intrinsic_measure_survive_model_replay() {
    let model = ModelDocument::compile("line-field.eqi", LINE).unwrap();
    let replay = ModelDocument::replay(&model.canonical_json().unwrap()).unwrap();
    assert_eq!(
        model.canonical_json().unwrap(),
        replay.canonical_json().unwrap()
    );
    for program in [model.program(), replay.program()] {
        let relation = program
            .nodes()
            .find_map(|node| match node {
                KernelNode::Relation(value) => Some(value.id()),
                _ => None,
            })
            .unwrap();
        let typed = program.typed_relation_residual(relation).unwrap();
        assert!(
            typed
                .node_types()
                .iter()
                .any(|value| value.support.is_some())
        );
        assert!(
            typed
                .node_types()
                .iter()
                .filter_map(|value| value.support.as_ref())
                .all(|support| support.intrinsic_dimensions() == 1
                    && support.ambient_dimensions() == Some(2))
        );
    }
}

#[test]
fn oblique_geometry_line_retains_exact_parent_and_artifact_authority() {
    use eqiora::artifact::AcceptedModelArtifact;
    use eqiora::compiler::StaticBindingValue;
    use eqiora::geometry::{
        CanonicalGeometryV1, EDGE_DIMENSION, FACE_DIMENSION, NamedEntitySet, PlanarFace,
        PlanarRegion,
    };
    use eqiora::graph::{GraphStore, InMemoryGraphStore};
    let region = PlanarRegion::new(
        vec![
            [0.0, 0.0],
            [3.0, 0.0],
            [0.0, 4.0],
            [10.0, 0.0],
            [13.0, 0.0],
            [10.0, 4.0],
        ],
        vec![
            PlanarFace::new(vec![0, 1, 2], vec![]),
            PlanarFace::new(vec![3, 4, 5], vec![]),
        ],
        vec![
            NamedEntitySet::new("body", FACE_DIMENSION, vec![0]),
            NamedEntitySet::new("wall", EDGE_DIMENSION, vec![1]),
            NamedEntitySet::new("foreign", EDGE_DIMENSION, vec![4]),
            NamedEntitySet::new("perimeter", EDGE_DIMENSION, vec![0, 1, 2]),
        ],
        0.0001,
    )
    .unwrap();
    let edge = region.entity_set("wall").unwrap().members()[0];
    let outer = region.faces()[0].outer();
    assert_eq!(region.vertices()[outer[edge]], [3.0, 0.0]);
    assert_eq!(
        region.vertices()[outer[(edge + 1) % outer.len()]],
        [0.0, 4.0]
    );
    let geometry = CanonicalGeometryV1::from_region(&region).unwrap();
    let body = geometry.entity_set("body").unwrap();
    let wall = geometry.entity_set("wall").unwrap();
    assert!(
        geometry.cartesian_boundary_embedding(wall, body).is_none(),
        "the triangle must exercise the non-box embedding path"
    );
    let source = "model Line(support body:volume(ambient_dimension=2),support wall:boundary(parent=body)) { variable density:kg/m on wall; relation retain on wall {density=1[kg/m];} observable mass:kg=integral(density,measure(wall)); }";
    let document = ModelDocument::compile_selected(
        "oblique-line.eqi",
        source,
        "Line",
        &[
            (
                "body",
                StaticBindingValue::GeometrySupport {
                    geometry: &geometry,
                    selection: body,
                    parent: None,
                },
            ),
            (
                "wall",
                StaticBindingValue::GeometrySupport {
                    geometry: &geometry,
                    selection: wall,
                    parent: Some(body),
                },
            ),
        ],
    )
    .unwrap();
    let bytes = document.canonical_json().unwrap();
    assert!(
        ModelDocument::replay(&bytes).is_err(),
        "a Model reference does not manufacture Geometry authority"
    );
    let artifact = AcceptedModelArtifact::from_json(&bytes, Default::default()).unwrap();
    let (transaction, model) = artifact.to_transaction().unwrap();
    let store =
        InMemoryGraphStore::restore_snapshot(transaction, document.program().revision()).unwrap();
    let replay = eqiora::sem::KernelProgram::from_snapshot_with_geometry(
        &store.snapshot(),
        model,
        &[&geometry],
    )
    .unwrap();
    assert_eq!(&replay, document.program());
    // The foreign segment exists in the same exact artifact but belongs to the other face.
    for selection in ["foreign", "perimeter"] {
        let (transaction, model) = artifact.to_transaction().unwrap();
        let mut foreign = eqiora::graph::Transaction::new("foreign boundary Field");
        for op in transaction.ops() {
            foreign.push(match op {
                eqiora::graph::Op::DefineKernelNode {
                    node: KernelNode::Domain(domain),
                } if matches!(
                    domain.kind(),
                    eqiora::kernel::DomainKind::GeometryBoundary { .. }
                ) =>
                {
                    eqiora::graph::Op::DefineKernelNode {
                        node: eqiora::kernel::DomainDef::geometry_boundary(domain.id(), selection)
                            .unwrap()
                            .into(),
                    }
                }
                other => other.clone(),
            });
        }
        let store =
            InMemoryGraphStore::restore_snapshot(foreign, document.program().revision()).unwrap();
        let errors = eqiora::sem::KernelProgram::from_snapshot_with_geometry(
            &store.snapshot(),
            model,
            &[&geometry],
        )
        .unwrap_err();
        assert!(
            errors
                .iter()
                .any(|error| error.message().contains("Field spatial support")
                    && error.message().contains("affine boundary embedding")),
            "{errors:?}"
        );
    }
}

#[test]
fn boundary_field_rejects_wrong_measure_frame_foreign_support_and_volume_gradient() {
    for (source, gate) in [
        (
            LINE.replace("observable mass:kg", "observable mass:kg*m"),
            "expression and measure",
        ),
        (
            LINE.replace(
                "variable density:kg/m on wall",
                "variable density:vector<kg/m,1> on wall",
            ),
            "ambient",
        ),
        (
            LINE.replace(
                "relation retain on wall",
                "domain other=boundary(body,axis=0,side=upper); relation retain on other",
            ),
            "support",
        ),
        (
            LINE.replace("density=1[kg/m]", "grad(density)=grad(density)"),
            "gradient",
        ),
    ] {
        let errors = ModelDocument::compile("invalid-line.eqi", &source).unwrap_err();
        assert!(
            errors.iter().any(|error| error.message().contains(gate)),
            "{gate}: {errors:?}"
        );
    }
}

#[test]
fn boundary_field_component_forwarding_preserves_ambient_vector_frame() {
    let source = "component BoundaryLaw(support body:volume(ambient_dimension=2),support wall:boundary(parent=body)) { variable u:vector<m,2> on wall; relation retain on wall {u=u;} } model M() { domain body=box(0,2,0,3); domain wall=boundary(body,axis=0,side=lower); instance law:BoundaryLaw(body=body,wall=wall); }";
    let model = ModelDocument::compile("boundary-component.eqi", source).unwrap();
    let field = model
        .program()
        .nodes()
        .find_map(|node| match node {
            KernelNode::Field(field) => Some(field),
            _ => None,
        })
        .unwrap();
    assert_eq!(field.frame(), eqiora::ValueFrame::SpatialCartesian);
    assert_eq!(field.shape().extents()[0].get(), 2);
    let replay = ModelDocument::replay(&model.canonical_json().unwrap()).unwrap();
    assert_eq!(replay.program(), model.program());
}

#[test]
fn borrowed_boundary_field_requires_the_exact_bound_support() {
    let source = "component BoundaryLaw(support body:volume(ambient_dimension=2),support wall:boundary(parent=body),variable displacement:vector<m,2> on wall) { relation retain on wall {displacement=displacement;} } model M() { domain body=box(0,2,0,3); domain wall=boundary(body,axis=0,side=lower); variable u:vector<m,2> on wall; instance law:BoundaryLaw(body=body,wall=wall,displacement=u); }";
    let model = ModelDocument::compile("borrowed-boundary.eqi", source).unwrap();
    assert_eq!(
        model
            .program()
            .nodes()
            .filter(|node| matches!(node, KernelNode::Field(_)))
            .count(),
        1,
        "the component borrows the caller Field instead of allocating another unknown"
    );
    let foreign = source.replace(
        "variable u:vector<m,2> on wall;",
        "domain other=boundary(body,axis=0,side=upper); variable u:vector<m,2> on other;",
    );
    let errors = ModelDocument::compile("foreign-borrowed-boundary.eqi", &foreign).unwrap_err();
    assert!(
        errors
            .iter()
            .any(|error| error.message().contains("support")),
        "{errors:?}"
    );
}
