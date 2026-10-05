//! Shear maps bind an exact polygon, including points outside its bounding box fiction.
use eqiora_api::ModelDocument;
use eqiora_artifact::AcceptedModelArtifact;
use eqiora_compiler::StaticBindingValue;
use eqiora_geometry::{CanonicalGeometryV1, NamedEntitySet, PlanarFace, PlanarRegion};
use eqiora_graph::GraphStore;
use eqiora_schema::kernel::KernelNode;

#[test]
fn sheared_image_membership_survives_exact_geometry_replay() {
    // Image of [0,1]^2 under x=2xi+eta,y=3eta. Its area is 6,
    // while its axis-aligned bounding box has area 9.
    let region = PlanarRegion::new(
        vec![[0.0, 0.0], [2.0, 0.0], [3.0, 3.0], [1.0, 3.0]],
        vec![PlanarFace::new(vec![0, 1, 2, 3], vec![])],
        vec![NamedEntitySet::new("body", 2, vec![0])],
        1e-12,
    )
    .unwrap();
    let geometry = CanonicalGeometryV1::from_region(&region).unwrap();
    for (mapped_x, accepted) in [("2*xi+eta", true), ("xi", false)] {
        let source = format!(
            r#"model M(support body:volume(ambient_dimension=2)) {{
            domain reference=box(0,1,0,1);
            coordinate xi:m on reference from reference[0];
            coordinate eta:m on reference from reference[1];
            coordinate x:m on body from body[0];
            coordinate y:m on body from body[1];
            variable anchor:1;
            relation r {{anchor=0;}}
            observable sample:m^2=evaluate(
                pullback(x*x+x*y,from=(xi,eta),at=(x={mapped_x},y=3*eta)),
                at=(xi=0.25[m],eta=0.5[m]));
        }}"#
        );
        let document = ModelDocument::compile_selected(
            "shear.eqi",
            &source,
            "M",
            &[(
                "body",
                StaticBindingValue::GeometrySupport {
                    geometry: &geometry,
                    selection: geometry.entity_set("body").unwrap(),
                    parent: None,
                },
            )],
        )
        .unwrap();
        let bytes = document.canonical_json().unwrap();
        assert!(
            ModelDocument::replay(&bytes).is_err(),
            "geometry cannot be omitted"
        );
        let artifact = AcceptedModelArtifact::from_json(&bytes, Default::default()).unwrap();
        let (transaction, model) = artifact.to_transaction().unwrap();
        let store = eqiora_graph::InMemoryGraphStore::restore_snapshot(
            transaction,
            document.program().revision(),
        )
        .unwrap();
        let replay = eqiora_sem::KernelProgram::from_snapshot_with_geometry(
            &store.snapshot(),
            model,
            &[&geometry],
        )
        .unwrap();
        assert_eq!(&replay, document.program());
        let observable = replay
            .nodes()
            .find_map(|node| match node {
                KernelNode::Observable(value) => Some(value.id()),
                _ => None,
            })
            .unwrap();
        let result = replay.evaluate_finite_observable(observable, &mut |_| None);
        if accepted {
            assert_eq!(result.unwrap().real_scalar_value().unwrap().value(), 2.5);
        } else {
            // (1/4,3/2) lies inside [0,3]^2 but outside the sheared image:
            // its inverse coordinate xi=(x-y/3)/2=-1/8.
            assert!(
                result
                    .unwrap_err()
                    .message()
                    .contains("outside its exact geometry support")
            );
        }
    }
}
