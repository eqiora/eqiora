//! Full document admission of the actual imported cylinder example.
use eqiora_api::package::{SourceImportContext, compile_source};
use eqiora_artifact::AcceptedModelArtifact;
use eqiora_compiler::StaticBindingValue;
use eqiora_core::{DimExponents, ScalarDomain, ValueLiteral, ValueType};
use eqiora_geometry::{CanonicalGeometryV1, NamedEntitySet};
use eqiora_graph::GraphStore;
use eqiora_schema::kernel::KernelNode;

fn cylinder(radius: f64) -> CanonicalGeometryV1 {
    CanonicalGeometryV1::from_circular_hole(
        [[0.0, 2.2], [0.0, 0.41]],
        [0.2, 0.2],
        radius,
        vec![
            NamedEntitySet::new("fluid", 2, vec![0]),
            NamedEntitySet::new("inlet", 1, vec![0]),
            NamedEntitySet::new("outlet", 1, vec![1]),
            NamedEntitySet::new("walls", 1, vec![2, 3]),
            NamedEntitySet::new("cylinder", 1, vec![4]),
        ],
    )
    .unwrap()
}
fn value(exponents: [i32; 7], value: f64) -> ValueLiteral {
    ValueLiteral::from_real(
        ValueType::scalar(
            ScalarDomain::Real,
            DimExponents::from_integers(exponents).unwrap(),
        )
        .unwrap(),
        value,
    )
    .unwrap()
}
#[test]
fn imported_cylinder_reaches_full_model_document_admission() {
    let geometry = cylinder(0.05);
    let viscosity = value([1, -1, -1, 0, 0, 0, 0], 0.001);
    let speed = value([0, 1, -1, 0, 0, 0, 0], 0.3);
    let height = value([0, 1, 0, 0, 0, 0, 0], 0.41);
    let mut bindings = ["fluid", "inlet", "outlet", "walls", "cylinder"]
        .map(|name| {
            (
                name,
                StaticBindingValue::GeometrySupport {
                    geometry: &geometry,
                    selection: geometry.entity_set(name).unwrap(),
                    parent: (name != "fluid").then(|| geometry.entity_set("fluid").unwrap()),
                },
            )
        })
        .to_vec();
    bindings.extend([
        ("dynamic_viscosity", StaticBindingValue::Value(&viscosity)),
        ("inlet_speed", StaticBindingValue::Value(&speed)),
        ("channel_height", StaticBindingValue::Value(&height)),
    ]);
    let result = compile_source(
        "steady-flow-past-cylinder.eqi",
        include_str!("../../../examples/steady-flow-past-cylinder.eqi"),
        Some("SteadyFlowPastCylinder"),
        &bindings,
        SourceImportContext::Bundled,
    )
    .unwrap_or_else(|errors| panic!("{errors:?}"));
    let document = result.model();
    let bytes = document.canonical_json().unwrap();
    // A bare replay cannot fabricate the separately authenticated Geometry.
    assert!(eqiora_api::ModelDocument::replay(&bytes).is_err());
    let artifact = AcceptedModelArtifact::from_json(&bytes, Default::default()).unwrap();
    let (transaction, model) = artifact.to_transaction().unwrap();
    let store = eqiora_graph::InMemoryGraphStore::restore_snapshot(
        transaction,
        eqiora_graph::Revision(document.program().revision().0),
    )
    .unwrap();
    let replay = eqiora_sem::KernelProgram::from_snapshot_with_geometry(
        &store.snapshot(),
        model,
        &[&geometry],
    )
    .unwrap();
    assert_eq!(&replay, document.program());
    let different_geometry = cylinder(0.04);
    assert_ne!(geometry.digest_bytes(), different_geometry.digest_bytes());
    assert!(
        eqiora_sem::KernelProgram::from_snapshot_with_geometry(
            &store.snapshot(),
            model,
            &[&different_geometry],
        )
        .is_err()
    );
    let incomplete = include_str!("../../../examples/steady-flow-past-cylinder.eqi").replace(
        "boundaries(inlet, outlet, walls, cylinder)",
        "boundaries(inlet, outlet, walls)",
    );
    let errors = compile_source(
        "incomplete-cylinder.eqi",
        &incomplete,
        Some("SteadyFlowPastCylinder"),
        &bindings,
        SourceImportContext::Bundled,
    )
    .unwrap_err();
    assert!(
        errors
            .diagnostics()
            .unwrap()
            .iter()
            .any(|error| error.message().contains("omits an entity")),
        "{errors:?}"
    );
    for program in [document.program(), &replay] {
        let connections = program
            .nodes()
            .filter_map(|node| match node {
                KernelNode::Connection(connection) => Some(connection.id()),
                _ => None,
            })
            .collect::<Vec<_>>();
        assert_eq!(connections.len(), 4);
        for connection in connections {
            let junction = program
                .compose_boundary_physical_junction(connection)
                .unwrap();
            assert!(matches!(
                junction.geometry(),
                eqiora_sem::BoundaryJunctionGeometry::Coincident
            ));
            assert_eq!(junction.dag().roots().len(), 2);
        }
    }
}
