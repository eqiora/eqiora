//! Physical Cartesian blocks retain exact identities inside dimensioned products.
use eqiora::api::ModelDocument;
use eqiora::compiler::StaticBindingValue;
use eqiora::kernel::{AxisBounds, DomainKind, KernelNode, typing::SpatialSupport};
use eqiora::{DimExponents, DynQuantity};

fn velocity() -> StaticBindingValue<'static> {
    let speed = DimExponents::from_integers([0, 1, -1, 0, 0, 0, 0]).unwrap();
    StaticBindingValue::CoordinateInterval(
        AxisBounds::new(DynQuantity::new(-2.0, speed), DynQuantity::new(2.0, speed)).unwrap(),
    )
}

#[test]
fn physical_blocks_have_intrinsic_measure_without_an_ambient_product_frame() {
    for (axes, bounds) in [(1, "0,2"), (2, "0,2,0,3"), (3, "0,2,0,3,0,4")] {
        // Independently: d^n x dv has units m^(n+1)/s; its reciprocal integrates to 1.
        let source = format!(
            "model Distribution(support velocity:interval(m/s)) {{
            domain position=box({bounds}); support phase:product(position,velocity);
            variable f:s/m^{} on phase;
            relation retain on phase {{f=0[s/m^{}];}}
            observable count:1=integral(f,measure(phase));
        }}",
            axes + 1,
            axes + 1
        );
        let document = ModelDocument::compile_selected(
            "physical-product.eqi",
            &source,
            "Distribution",
            &[("velocity", velocity())],
        )
        .unwrap();
        let replay = ModelDocument::replay(&document.canonical_json().unwrap()).unwrap();
        assert_eq!(
            document.canonical_json().unwrap(),
            replay.canonical_json().unwrap()
        );
        let position = document
            .program()
            .nodes()
            .find_map(|node| match node {
                KernelNode::Domain(domain)
                    if matches!(domain.kind(), DomainKind::CartesianBox { .. }) =>
                {
                    Some(domain.id().erase())
                }
                _ => None,
            })
            .unwrap();
        for program in [document.program(), replay.program()] {
            let relation = program
                .nodes()
                .find_map(|node| match node {
                    KernelNode::Relation(value) => Some(value.id()),
                    _ => None,
                })
                .unwrap();
            let typed = program.typed_relation_residual(relation).unwrap();
            let supports = typed
                .node_types()
                .iter()
                .filter_map(|value| value.support.as_ref())
                .collect::<Vec<_>>();
            assert!(!supports.is_empty());
            for support in supports {
                let SpatialSupport::Coordinates { factors, .. } = support else {
                    panic!("{support:?}")
                };
                assert_eq!(factors.len(), 2);
                assert_eq!(factors[0].0, position);
                assert_eq!(factors[0].2, axes);
                assert_eq!(factors[1].2, 1);
                assert_eq!(support.intrinsic_dimensions(), axes + 1);
                assert_eq!(support.ambient_dimensions(), None);
            }
        }
        let wrong = source.replace("count:1", "count:m");
        let errors = ModelDocument::compile_selected(
            "wrong-measure.eqi",
            &wrong,
            "Distribution",
            &[("velocity", velocity())],
        )
        .unwrap_err();
        assert!(
            errors
                .iter()
                .any(|error| error.message().contains("expression and measure")),
            "{errors:?}"
        );
    }
}

#[test]
fn geometry_factor_requires_exact_cartesian_region_authority_on_replay() {
    use eqiora::artifact::AcceptedModelArtifact;
    use eqiora::geometry::{CanonicalGeometryV1, NamedEntitySet, PlanarFace, PlanarRegion};
    use eqiora::graph::{GraphStore, InMemoryGraphStore};
    let geometry = CanonicalGeometryV1::from_region(
        &PlanarRegion::new(
            vec![[0.0, 0.0], [2.0, 0.0], [2.0, 3.0], [0.0, 3.0]],
            vec![PlanarFace::new(vec![0, 1, 2, 3], vec![])],
            vec![NamedEntitySet::new("position", 2, vec![0])],
            1e-9,
        )
        .unwrap(),
    )
    .unwrap();
    let source = "model Distribution(support position:volume(ambient_dimension=2), support velocity:interval(m/s)) {
        support phase:product(position,velocity); variable f:s/m^3 on phase;
        relation retain on phase {f=0[s/m^3];} observable count:1=integral(f,measure(phase));
    }";
    let document = ModelDocument::compile_selected(
        "geometry-product.eqi",
        source,
        "Distribution",
        &[
            (
                "position",
                StaticBindingValue::GeometrySupport {
                    geometry: &geometry,
                    selection: geometry.entity_set("position").unwrap(),
                    parent: None,
                },
            ),
            ("velocity", velocity()),
        ],
    )
    .unwrap();
    let bytes = document.canonical_json().unwrap();
    let errors = ModelDocument::replay(&bytes).unwrap_err();
    assert!(
        errors.iter().any(|error| error
            .message()
            .contains("exact admitted Cartesian Geometry")),
        "{errors:?}"
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
    // A triangular region has the same ambient dimension and bounds as the rectangle,
    // but cannot stand in for a Cartesian factor.
    let triangle = CanonicalGeometryV1::from_region(
        &PlanarRegion::new(
            vec![[0.0, 0.0], [2.0, 0.0], [0.0, 3.0]],
            vec![PlanarFace::new(vec![0, 1, 2], vec![])],
            vec![NamedEntitySet::new("position", 2, vec![0])],
            1e-9,
        )
        .unwrap(),
    )
    .unwrap();
    let errors = ModelDocument::compile_selected(
        "triangle-product.eqi",
        source,
        "Distribution",
        &[
            (
                "position",
                StaticBindingValue::GeometrySupport {
                    geometry: &triangle,
                    selection: triangle.entity_set("position").unwrap(),
                    parent: None,
                },
            ),
            ("velocity", velocity()),
        ],
    )
    .unwrap_err();
    assert!(
        errors.iter().any(|error| error
            .message()
            .contains("exact admitted Cartesian Geometry")),
        "{errors:?}"
    );
    assert!(
        eqiora::sem::KernelProgram::from_snapshot_with_geometry(
            &store.snapshot(),
            model,
            &[&triangle]
        )
        .is_err()
    );
}

#[test]
fn component_products_forward_exact_physical_blocks_and_reject_wrong_contexts() {
    let source = "component Distribution(support position:volume(ambient_dimension=2),support velocity:interval(m/s)) {
        support phase:product(position,velocity); variable f:s/m^3 on phase;
        relation retain on phase {f=0[s/m^3];} observable count:1=integral(f,measure(phase));
    } model M(support velocity:interval(m/s)) {
        domain position=box(0,2,0,3); instance law:Distribution(position=position,velocity=velocity);
    }";
    let document = ModelDocument::compile_selected(
        "component-product.eqi",
        source,
        "M",
        &[("velocity", velocity())],
    )
    .unwrap();
    let replay = ModelDocument::replay(&document.canonical_json().unwrap()).unwrap();
    assert_eq!(document.program(), replay.program());
    for (source, gate) in [
        (
            source.replace(
                "product(position,velocity)",
                "product(position,position,velocity)",
            ),
            "repeats an exact factor",
        ),
        (source.replace("f=0[s/m^3]", "grad(f)=grad(f)"), "gradient"),
        (
            source.replace("variable f:s/m^3", "variable f:vector<s/m^3,3>"),
            "ambient",
        ),
        (
            source.replace(
                "position=position,velocity=velocity",
                "position=velocity,velocity=velocity",
            ),
            "factor kind",
        ),
    ] {
        let errors = ModelDocument::compile_selected(
            "invalid-component-product.eqi",
            &source,
            "M",
            &[("velocity", velocity())],
        )
        .unwrap_err();
        assert!(
            errors.iter().any(|error| error.message().contains(gate)),
            "{gate}: {errors:?}"
        );
    }
}

#[test]
fn position_radius_distribution_preserves_same_unit_factor_identity() {
    let length = DimExponents::from_integers([0, 1, 0, 0, 0, 0, 0]).unwrap();
    let radius = StaticBindingValue::CoordinateInterval(
        AxisBounds::new(
            DynQuantity::new(0.001, length),
            DynQuantity::new(0.01, length),
        )
        .unwrap(),
    );
    let source = "model Population(support radius:interval(m)) {
        domain position=box(0,2); support population:product(position,radius);
        support reversed:product(radius,position);
        variable density:1/m^2 on population;
        relation retain on population {density=1[1/m^2];}
        observable count:1=integral(density,measure(population));
    }";
    let document = ModelDocument::compile_selected(
        "particle-population.eqi",
        source,
        "Population",
        &[("radius", radius)],
    )
    .unwrap();
    let replay = ModelDocument::replay(&document.canonical_json().unwrap()).unwrap();
    assert_eq!(document.program(), replay.program());
    // Both factors have units m, but reversing their exact support cannot silently
    // exchange position with particle radius, even though the measure still has units m².
    let foreign = source.replace(
        "relation retain on population",
        "relation retain on reversed",
    );
    let errors = ModelDocument::compile_selected(
        "reversed-population.eqi",
        &foreign,
        "Population",
        &[("radius", radius)],
    )
    .unwrap_err();
    assert!(
        errors
            .iter()
            .any(|error| error.message().contains("support")),
        "{errors:?}"
    );
}
