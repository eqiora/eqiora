use std::process::Command;

use eqiora::api::StructuralSemanticFingerprint;
use eqiora::artifact::ModelEnvelope;
use eqiora::diagnostic::codes;
use eqiora::geometry::{
    CanonicalGeometryLimits, CanonicalGeometryV1, EDGE_DIMENSION, FACE_DIMENSION, NamedEntitySet,
    PlanarFace, PlanarRegion, VERTEX_DIMENSION,
};
use eqiora::graph::{EdgeKind, GraphStore, InMemoryGraphStore, Op, Transaction};
use eqiora::kernel::typing::SpatialSupport;
use eqiora::kernel::{
    ActivationDef, BoundaryPairing, BoundaryPhysicalConnector, DomainDef, ExprDagBuilder, FieldDef,
    GeometryDigest, KernelNode, PortDef, RelationDef, RepresentationDef, SymbolRef,
};
use eqiora::ontology::{Model, ModelView, OntologyId};
use eqiora::sem::KernelProgram;
use eqiora::{Diagnostic, DimExponents, DynQuantity, Id, RawId, ValueShape, kinds};
use eqiora_core::ValueFrame;

const FROZEN_DIGEST: &str = "e6f8e17ac215ef37ca3c9de07b9979e34f13412a5de11dc9240ea1def8130030";

fn square_with_hole() -> CanonicalGeometryV1 {
    let region = PlanarRegion::new(
        vec![
            [0.0, 0.0],
            [1.0, 0.0],
            [1.0, 1.0],
            [0.0, 1.0],
            [0.25, 0.25],
            [0.75, 0.25],
            [0.75, 0.75],
            [0.25, 0.75],
        ],
        vec![PlanarFace::new(vec![0, 1, 2, 3], vec![vec![4, 5, 6, 7]])],
        vec![
            NamedEntitySet::new("exterior", EDGE_DIMENSION, vec![0, 1, 2, 3]),
            NamedEntitySet::new("hole", EDGE_DIMENSION, vec![4, 5, 6, 7]),
            NamedEntitySet::new("fluid", FACE_DIMENSION, vec![0]),
        ],
        0.0625,
    )
    .expect("precommitted square-with-hole geometry");
    CanonicalGeometryV1::from_region(&region).expect("canonical geometry")
}

fn filled_square() -> CanonicalGeometryV1 {
    let region = PlanarRegion::new(
        vec![[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]],
        vec![PlanarFace::new(vec![0, 1, 2, 3], Vec::new())],
        vec![
            NamedEntitySet::new("foreign-only", EDGE_DIMENSION, vec![0]),
            NamedEntitySet::new("body2", FACE_DIMENSION, vec![0]),
        ],
        0.0625,
    )
    .expect("precommitted filled square");
    CanonicalGeometryV1::from_region(&region).expect("canonical geometry")
}

fn two_face_names_square() -> CanonicalGeometryV1 {
    let region = PlanarRegion::new(
        vec![[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]],
        vec![PlanarFace::new(vec![0, 1, 2, 3], Vec::new())],
        vec![
            NamedEntitySet::new("body-a", FACE_DIMENSION, vec![0]),
            NamedEntitySet::new("body-b", FACE_DIMENSION, vec![0]),
        ],
        0.0625,
    )
    .expect("two aliases for one canonical face");
    CanonicalGeometryV1::from_region(&region).expect("canonical geometry")
}

fn hex_digest(bytes: [u8; 32]) -> String {
    bytes.iter().map(|byte| format!("{byte:02x}")).collect()
}

fn circular_hole_sets() -> Vec<NamedEntitySet> {
    vec![
        NamedEntitySet::new("fluid", FACE_DIMENSION, vec![0, 0]),
        NamedEntitySet::new("walls", EDGE_DIMENSION, vec![3, 2]),
        NamedEntitySet::new("inlet", EDGE_DIMENSION, vec![0]),
        NamedEntitySet::new("cylinder", EDGE_DIMENSION, vec![4]),
        NamedEntitySet::new("outlet", EDGE_DIMENSION, vec![1]),
    ]
}

fn circular_hole_with(
    bounds: [[f64; 2]; 2],
    center: [f64; 2],
    radius: f64,
    sets: Vec<NamedEntitySet>,
    tolerance: f64,
) -> Result<CanonicalGeometryV1, Diagnostic> {
    CanonicalGeometryV1::from_circular_hole(bounds, center, radius, sets, tolerance)
}

fn circular_hole_witness() -> CanonicalGeometryV1 {
    circular_hole_with(
        [[0.0, 2.2], [0.0, 0.41]],
        [0.2, 0.2],
        0.05,
        circular_hole_sets(),
        1e-12,
    )
    .expect("exact DFG-shaped circular-hole geometry")
}

fn committed_model(
    label: &str,
    mut nodes: Vec<KernelNode>,
    edges: impl IntoIterator<Item = (RawId, RawId, EdgeKind)>,
) -> (InMemoryGraphStore, OntologyId<Model>) {
    let mut edges = edges.into_iter().collect::<Vec<_>>();
    if !nodes
        .iter()
        .any(|node| matches!(node, KernelNode::Relation(_)))
    {
        let relation = Id::new();
        let activation = Id::new();
        let mut expression = ExprDagBuilder::new();
        let zero = expression
            .constant(DynQuantity::new(0.0, DimExponents::DIMENSIONLESS))
            .expect("constant");
        nodes.extend([
            KernelNode::from(
                RelationDef::new(
                    relation,
                    expression.finish([zero, zero]).expect("constant residual"),
                )
                .unwrap(),
            ),
            KernelNode::from(ActivationDef::continuous(activation)),
        ]);
        edges.push((activation.erase(), relation.erase(), EdgeKind::Activates));
    }
    let model = OntologyId::new();
    let members = nodes.iter().map(KernelNode::id).collect::<Vec<_>>();
    let mut transaction = Transaction::new(label);
    for node in nodes {
        transaction.push(Op::DefineKernelNode { node });
    }
    for (from, to, edge) in edges {
        transaction.push(Op::Connect { from, to, edge });
    }
    transaction.push(Op::DefineOntologyView {
        view: ModelView::new(model, members, None)
            .expect("closed test Model")
            .into(),
    });
    let mut store = InMemoryGraphStore::new();
    store.commit(transaction).expect("commit test Model");
    (store, model)
}

#[derive(Clone, Copy)]
struct PositiveIds {
    region: Id<kinds::Domain>,
    boundary: Id<kinds::Domain>,
    representation: Id<kinds::Representation>,
    field: Id<kinds::Field>,
    relation: Id<kinds::Relation>,
    activation: Id<kinds::Activation>,
}

fn positive_model(
    geometry_digest: [u8; 32],
    region_set: &str,
    boundary_set: &str,
    field_extent: u32,
) -> (InMemoryGraphStore, OntologyId<Model>, PositiveIds) {
    let ids = PositiveIds {
        region: Id::new(),
        boundary: Id::new(),
        representation: Id::new(),
        field: Id::new(),
        relation: Id::new(),
        activation: Id::new(),
    };
    let mut expression = ExprDagBuilder::new();
    let velocity = expression
        .symbol(SymbolRef::Field(ids.field))
        .expect("field symbol");
    let divergence = expression.divergence(velocity).expect("divergence node");
    let nodes = vec![
        KernelNode::from(
            DomainDef::geometry_region(
                ids.region,
                GeometryDigest::new(geometry_digest),
                region_set,
            )
            .expect("named region"),
        ),
        KernelNode::from(
            DomainDef::geometry_boundary(ids.boundary, boundary_set)
                .expect("named geometry boundary"),
        ),
        KernelNode::from(RepresentationDef::continuum(ids.representation)),
        KernelNode::from(FieldDef::new(
            ids.field,
            eqiora_core::ValueType::shaped(
                eqiora_core::ScalarDomain::Real,
                DimExponents::DIMENSIONLESS,
                ValueShape::new([field_extent]).expect("vector extent"),
                ValueFrame::SpatialCartesian,
            )
            .expect("spatial-vector Field"),
            eqiora::kernel::FieldRole::Variable,
        )),
        KernelNode::from(
            RelationDef::new(
                ids.relation,
                {
                    let equation_zero_0 = expression
                        .constant(
                            eqiora_core::ValueLiteral::from_real(
                                eqiora_core::ValueType::scalar(
                                    eqiora_core::ScalarDomain::Real,
                                    DimExponents::from_integers([0, -1, 0, 0, 0, 0, 0]).unwrap(),
                                )
                                .expect("valid scalar type"),
                                0.0,
                            )
                            .unwrap(),
                        )
                        .unwrap();
                    expression.finish([divergence, equation_zero_0])
                }
                .expect("closed residual DAG"),
            )
            .unwrap(),
        ),
        KernelNode::from(ActivationDef::continuous(ids.activation)),
    ];
    let edges = [
        (
            ids.boundary.erase(),
            ids.region.erase(),
            EdgeKind::BoundaryOf,
        ),
        (ids.field.erase(), ids.region.erase(), EdgeKind::DefinedOn),
        (
            ids.field.erase(),
            ids.representation.erase(),
            EdgeKind::DefinedOn,
        ),
        (
            ids.relation.erase(),
            ids.region.erase(),
            EdgeKind::AppliesOn,
        ),
        (ids.relation.erase(), ids.field.erase(), EdgeKind::DependsOn),
        (
            ids.activation.erase(),
            ids.relation.erase(),
            EdgeKind::Activates,
        ),
    ];
    let (store, model) = committed_model("geometry-backed positive Model", nodes, edges);
    (store, model, ids)
}

fn boundary_relation_model(
    geometry_digest: [u8; 32],
    region_set: &str,
    boundary_set: &str,
    extra_regions: impl IntoIterator<Item = ([u8; 32], &'static str)>,
) -> (InMemoryGraphStore, OntologyId<Model>, PositiveIds) {
    let ids = PositiveIds {
        region: Id::new(),
        boundary: Id::new(),
        representation: Id::new(),
        field: Id::new(),
        relation: Id::new(),
        activation: Id::new(),
    };
    let mut expression = ExprDagBuilder::new();
    let field_value = expression
        .symbol(SymbolRef::Field(ids.field))
        .expect("Field symbol");
    let trace = expression.trace(field_value).expect("boundary trace");
    let mut nodes = vec![
        KernelNode::from(
            DomainDef::geometry_region(
                ids.region,
                GeometryDigest::new(geometry_digest),
                region_set,
            )
            .expect("geometry region"),
        ),
        KernelNode::from(
            DomainDef::geometry_boundary(ids.boundary, boundary_set).expect("geometry boundary"),
        ),
        KernelNode::from(RepresentationDef::continuum(ids.representation)),
        KernelNode::from(FieldDef::new(
            ids.field,
            eqiora_core::ValueType::shaped(
                eqiora_core::ScalarDomain::Real,
                DimExponents::DIMENSIONLESS,
                ValueShape::new([2]).expect("two-component vector"),
                ValueFrame::SpatialCartesian,
            )
            .expect("spatial Field"),
            eqiora::kernel::FieldRole::Variable,
        )),
        KernelNode::from(
            RelationDef::new(
                ids.relation,
                {
                    let equation_zero_0 = expression
                        .constant(
                            eqiora_core::ValueLiteral::from_real(
                                eqiora_core::ValueType::shaped(
                                    eqiora_core::ScalarDomain::Real,
                                    DimExponents::DIMENSIONLESS,
                                    ValueShape::new([2]).unwrap(),
                                    ValueFrame::SpatialCartesian,
                                )
                                .unwrap(),
                                0.0,
                            )
                            .unwrap(),
                        )
                        .unwrap();
                    expression.finish([trace, equation_zero_0])
                }
                .expect("trace residual"),
            )
            .unwrap(),
        ),
        KernelNode::from(ActivationDef::continuous(ids.activation)),
    ];
    nodes.extend(extra_regions.into_iter().map(|(digest, set)| {
        KernelNode::from(
            DomainDef::geometry_region(Id::new(), GeometryDigest::new(digest), set)
                .expect("extra geometry region"),
        )
    }));
    let (store, model) = committed_model(
        "geometry boundary Relation",
        nodes,
        [
            (
                ids.boundary.erase(),
                ids.region.erase(),
                EdgeKind::BoundaryOf,
            ),
            (ids.field.erase(), ids.region.erase(), EdgeKind::DefinedOn),
            (
                ids.field.erase(),
                ids.representation.erase(),
                EdgeKind::DefinedOn,
            ),
            (
                ids.relation.erase(),
                ids.boundary.erase(),
                EdgeKind::AppliesOn,
            ),
            (ids.relation.erase(), ids.field.erase(), EdgeKind::DependsOn),
            (
                ids.activation.erase(),
                ids.relation.erase(),
                EdgeKind::Activates,
            ),
        ],
    );
    (store, model, ids)
}

fn declaration_model(
    regions: impl IntoIterator<Item = ([u8; 32], &'static str)>,
) -> (
    InMemoryGraphStore,
    OntologyId<Model>,
    Vec<Id<kinds::Domain>>,
) {
    let mut ids = Vec::new();
    let nodes = regions
        .into_iter()
        .map(|(digest, set)| {
            let id = Id::new();
            ids.push(id);
            KernelNode::from(
                DomainDef::geometry_region(id, GeometryDigest::new(digest), set)
                    .expect("named declaration"),
            )
        })
        .collect();
    let (store, model) = committed_model("declaration-only geometry Model", nodes, []);
    (store, model, ids)
}

fn geometry_free_model() -> (InMemoryGraphStore, OntologyId<Model>) {
    let relation = Id::new();
    let activation = Id::new();
    let mut expression = ExprDagBuilder::new();
    let zero = expression
        .constant(DynQuantity::new(0.0, DimExponents::DIMENSIONLESS))
        .expect("constant");
    committed_model(
        "geometry-free Model",
        vec![
            KernelNode::from(
                RelationDef::new(
                    relation,
                    expression.finish([zero, zero]).expect("constant residual"),
                )
                .unwrap(),
            ),
            KernelNode::from(ActivationDef::continuous(activation)),
        ],
        [(activation.erase(), relation.erase(), EdgeKind::Activates)],
    )
}

fn assert_diagnostic_at(diagnostic: &Diagnostic, id: RawId) {
    assert_eq!(
        diagnostic
            .graph_path()
            .expect("diagnostic must identify its Domain")
            .to_string(),
        format!("semantic.{:?}.{id}", id.kind())
    );
}

#[test]
fn closed_artifact_admission_reconstructs_support_for_typing_and_later_queries() {
    let geometry = square_with_hole();
    assert_eq!(hex_digest(geometry.digest_bytes()), FROZEN_DIGEST);
    let geometry_ref = &geometry;
    assert_eq!(geometry_ref.ambient_dimension(), 2);
    assert_eq!(geometry_ref.topological_dimension(), 2);
    assert_eq!(geometry_ref.entity_set_dimension("fluid"), Some(2));
    assert_eq!(geometry_ref.entity_set_dimension("hole"), Some(1));
    assert_eq!(geometry_ref.entity_set_dimension("absent"), None);
    let debug = format!("{geometry_ref:?}");
    assert!(!debug.contains("vertices"));
    assert!(!debug.contains("canonical_bytes"));

    let (store, model, ids) = positive_model(geometry.digest_bytes(), "fluid", "hole", 2);
    let program =
        KernelProgram::from_snapshot_with_geometry(&store.snapshot(), model, &[geometry_ref])
            .expect("exact artifact bundle admits geometry-backed spatial meaning");

    let typed = program
        .typed_relation_residual(ids.relation)
        .expect("stored support is reused without resupplying the artifact");
    assert_eq!(
        typed.node_types()[0].support,
        Some(SpatialSupport::Volume {
            domain: ids.region.erase(),
            dimensions: 2,
        })
    );
    assert_eq!(
        typed.node_types()[1].support,
        Some(SpatialSupport::Volume {
            domain: ids.region.erase(),
            dimensions: 2,
        })
    );

    let artifact_free = KernelProgram::from_snapshot(&store.snapshot(), model)
        .expect_err("geometry-backed consumers require artifact admission");
    assert!(artifact_free.iter().any(|diagnostic| {
        diagnostic.code() == codes::INVALID_KERNEL_DEFINITION
            && diagnostic.message().contains("requires artifact admission")
    }));
}

#[test]
fn independent_exact_circular_hole_identity_enters_the_same_admission_seam() {
    let oracle = Command::new("python3")
        .arg(
            std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
                .join("../../verify/geometry/exact-circular-hole-geometry/oracle.py"),
        )
        .output()
        .expect("independent Python identity oracle must execute");
    assert!(
        oracle.status.success(),
        "independent oracle failed: {}",
        String::from_utf8_lossy(&oracle.stderr)
    );
    let oracle_output = String::from_utf8(oracle.stdout).expect("oracle emits UTF-8");
    let mut oracle_lines = oracle_output.lines();
    let oracle_bytes = oracle_lines
        .next()
        .expect("oracle emits complete canonical JSON")
        .as_bytes();
    assert!(oracle_output.contains("bytes=511"));
    assert!(
        oracle_output
            .contains("sha256=b00123472a596e8289820cabaee20d52cdf81b5572fa9ce58ff17cdaa00046d9")
    );

    let geometry = circular_hole_witness();
    assert_eq!(geometry.canonical_bytes(), oracle_bytes);
    assert_eq!(geometry.canonical_bytes().len(), 511);
    assert_eq!(
        hex_digest(geometry.digest_bytes()),
        "b00123472a596e8289820cabaee20d52cdf81b5572fa9ce58ff17cdaa00046d9"
    );

    let geometry_ref = &geometry;
    let (store, model, ids) = positive_model(geometry.digest_bytes(), "fluid", "cylinder", 2);
    let program =
        KernelProgram::from_snapshot_with_geometry(&store.snapshot(), model, &[geometry_ref])
            .expect("the circular kind enters unchanged semantic admission");
    let typed = program
        .typed_relation_residual(ids.relation)
        .expect("derived support survives without retaining geometry");
    assert_eq!(
        typed.node_types()[0].support,
        Some(SpatialSupport::Volume {
            domain: ids.region.erase(),
            dimensions: 2,
        })
    );

    let artifact_free = KernelProgram::from_snapshot(&store.snapshot(), model)
        .expect_err("geometry-backed consumers still require artifact admission");
    assert!(artifact_free.iter().any(|diagnostic| {
        diagnostic.code() == codes::INVALID_KERNEL_DEFINITION
            && diagnostic.message().contains("requires artifact admission")
    }));
}

#[test]
fn exact_circular_hole_reference_projects_only_one_supported_constant_normal() {
    let oracle = Command::new("python3")
        .arg(
            std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join(
                "../../verify/geometry/geometry-boundary-relation-scope/oracle/boundary_scope_oracle.py",
            ),
        )
        .output()
        .expect("independent boundary-scope oracle executes");
    assert!(
        oracle.status.success(),
        "independent boundary-scope oracle failed: {}",
        String::from_utf8_lossy(&oracle.stderr)
    );
    let oracle_output = String::from_utf8(oracle.stdout).expect("oracle emits UTF-8");
    assert!(oracle_output.contains("registration=registered"));
    assert!(oracle_output.ends_with("OK\n"));

    let geometry = circular_hole_witness();
    let reference = &geometry;
    let normal_bits = |name| {
        reference
            .constant_parent_outward_normal(name)
            .map(|normal| normal.map(f64::to_bits))
    };
    assert_eq!(
        normal_bits("inlet"),
        Some([(-1.0f64).to_bits(), 0.0f64.to_bits()])
    );
    assert_eq!(
        normal_bits("outlet"),
        Some([1.0f64.to_bits(), 0.0f64.to_bits()])
    );
    assert_eq!(reference.constant_parent_outward_normal("walls"), None);
    assert_eq!(reference.constant_parent_outward_normal("cylinder"), None);
    assert_eq!(reference.constant_parent_outward_normal("fluid"), None);
    assert_eq!(reference.constant_parent_outward_normal("absent"), None);

    let multi_edge_inlet = circular_hole_with(
        [[0.0, 2.2], [0.0, 0.41]],
        [0.2, 0.2],
        0.05,
        vec![NamedEntitySet::new("inlet", EDGE_DIMENSION, vec![0, 2])],
        1.0e-12,
    )
    .expect("two-member inlet remains valid exact geometry");
    assert_eq!(
        multi_edge_inlet.constant_parent_outward_normal("inlet"),
        None
    );

    let renamed_sides = circular_hole_with(
        [[0.0, 2.2], [0.0, 0.41]],
        [0.2, 0.2],
        0.05,
        vec![
            NamedEntitySet::new("left", EDGE_DIMENSION, vec![0]),
            NamedEntitySet::new("right", EDGE_DIMENSION, vec![1]),
            NamedEntitySet::new("cylinder", EDGE_DIMENSION, vec![0]),
            NamedEntitySet::new("wall-lower", EDGE_DIMENSION, vec![2]),
            NamedEntitySet::new("wall-upper", EDGE_DIMENSION, vec![3]),
        ],
        1.0e-12,
    )
    .expect("exact side identity does not depend on an author-supplied set name");
    let renamed_reference = &renamed_sides;
    assert_eq!(
        renamed_reference
            .constant_parent_outward_normal("left")
            .map(|normal| normal.map(f64::to_bits)),
        Some([(-1.0f64).to_bits(), 0.0f64.to_bits()])
    );
    assert_eq!(
        renamed_reference
            .constant_parent_outward_normal("right")
            .map(|normal| normal.map(f64::to_bits)),
        Some([1.0f64.to_bits(), 0.0f64.to_bits()])
    );
    assert_eq!(
        renamed_reference
            .constant_parent_outward_normal("cylinder")
            .map(|normal| normal.map(f64::to_bits)),
        Some([(-1.0f64).to_bits(), 0.0f64.to_bits()])
    );
    assert_eq!(
        renamed_reference.constant_parent_outward_normal("wall-lower"),
        None
    );
    assert_eq!(
        renamed_reference.constant_parent_outward_normal("wall-upper"),
        None
    );

    let witness_names_off_the_x_sides = circular_hole_with(
        [[0.0, 2.2], [0.0, 0.41]],
        [0.2, 0.2],
        0.05,
        vec![
            NamedEntitySet::new("inlet", EDGE_DIMENSION, vec![2]),
            NamedEntitySet::new("outlet", EDGE_DIMENSION, vec![4]),
        ],
        1.0e-12,
    )
    .expect("the witness names carry no side identity of their own");
    let off_x_reference = &witness_names_off_the_x_sides;
    assert_eq!(
        off_x_reference.constant_parent_outward_normal("inlet"),
        None
    );
    assert_eq!(
        off_x_reference.constant_parent_outward_normal("outlet"),
        None
    );

    let vertex_set = circular_hole_with(
        [[0.0, 2.2], [0.0, 0.41]],
        [0.2, 0.2],
        0.05,
        vec![NamedEntitySet::new("pin", VERTEX_DIMENSION, vec![0])],
        1.0e-12,
    )
    .expect("a named rectangle corner remains valid exact geometry");
    assert_eq!(vertex_set.constant_parent_outward_normal("pin"), None);

    let straight_edged = square_with_hole();
    let straight_reference = &straight_edged;
    assert_eq!(
        straight_reference.constant_parent_outward_normal("exterior"),
        None
    );
    assert_eq!(
        straight_reference.constant_parent_outward_normal("hole"),
        None
    );

    let straight_inlet = PlanarRegion::new(
        vec![[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]],
        vec![PlanarFace::new(vec![0, 1, 2, 3], Vec::new())],
        vec![
            NamedEntitySet::new("inlet", EDGE_DIMENSION, vec![0]),
            NamedEntitySet::new("fluid", FACE_DIMENSION, vec![0]),
        ],
        0.0625,
    )
    .expect("straight-edged single-member inlet remains valid");
    let straight_inlet =
        CanonicalGeometryV1::from_region(&straight_inlet).expect("canonical straight inlet");
    // The complete lower side of [0,1]^2 has outward normal -e_y,
    // independently of the arbitrary selection name and admission tolerance.
    assert_eq!(
        straight_inlet.constant_parent_outward_normal("inlet"),
        Some([0.0, -1.0]),
    );
}

#[test]
fn exact_circular_hole_external_admission_and_falsifiers_are_registered_evidence() {
    let geometry = circular_hole_witness();
    let bytes = geometry.canonical_bytes();
    let defaults = CanonicalGeometryLimits::default();
    let decoded = CanonicalGeometryV1::decode_circular_hole_canonical(bytes, defaults)
        .expect("canonical bytes reconstruct exactly");
    assert_eq!(decoded, geometry);
    assert!(
        geometry.region().is_none(),
        "exact circular meaning must not fabricate a chordal PlanarRegion"
    );
    assert_eq!(
        geometry.circular_hole_bounds(),
        Some(&[[0.0, 2.2], [0.0, 0.41]])
    );
    assert_eq!(geometry.circular_hole_center(), Some([0.2, 0.2]));
    assert_eq!(geometry.circular_hole_radius_m(), Some(0.05));

    let straight = square_with_hole();
    assert!(straight.region().is_some());
    assert_eq!(straight.circular_hole_bounds(), None);
    assert_eq!(straight.circular_hole_center(), None);
    assert_eq!(straight.circular_hole_radius_m(), None);
    assert!(
        CanonicalGeometryV1::decode_canonical(bytes, defaults).is_err(),
        "the straight decoder must remain closed to the circular wire"
    );
    assert!(
        CanonicalGeometryV1::decode_circular_hole_canonical(straight.canonical_bytes(), defaults,)
            .is_err(),
        "the circular decoder must remain closed to the straight wire"
    );

    let positive_zero = circular_hole_witness();
    let negative_zero = circular_hole_with(
        [[-0.0, 2.2], [-0.0, 0.41]],
        [0.2, 0.2],
        0.05,
        circular_hole_sets(),
        1e-12,
    )
    .expect("signed zero normalizes");
    assert_eq!(
        negative_zero.canonical_bytes(),
        positive_zero.canonical_bytes()
    );
    assert_eq!(negative_zero.digest_bytes(), positive_zero.digest_bytes());
    let positive_center_zero = circular_hole_with(
        [[-1.0, 1.0], [-1.0, 1.0]],
        [0.0, 0.0],
        0.1,
        circular_hole_sets(),
        1e-12,
    )
    .expect("positive center zero is valid");
    let negative_center_zero = circular_hole_with(
        [[-1.0, 1.0], [-1.0, 1.0]],
        [-0.0, -0.0],
        0.1,
        circular_hole_sets(),
        1e-12,
    )
    .expect("negative center zero normalizes");
    assert_eq!(
        negative_center_zero.canonical_bytes(),
        positive_center_zero.canonical_bytes()
    );
    assert_eq!(
        negative_center_zero.digest_bytes(),
        positive_center_zero.digest_bytes()
    );

    for limits in [
        CanonicalGeometryLimits {
            max_bytes: bytes.len() - 1,
            ..defaults
        },
        CanonicalGeometryLimits {
            max_vertices: 3,
            ..defaults
        },
        CanonicalGeometryLimits {
            max_faces: 0,
            ..defaults
        },
        CanonicalGeometryLimits {
            max_loop_indices: 3,
            ..defaults
        },
        CanonicalGeometryLimits {
            max_entity_sets: 4,
            ..defaults
        },
        CanonicalGeometryLimits {
            max_entity_set_members: 5,
            ..defaults
        },
    ] {
        assert!(
            CanonicalGeometryV1::decode_circular_hole_canonical(bytes, limits).is_err(),
            "every applicable decoder budget must fail closed"
        );
    }

    let text = String::from_utf8(bytes.to_vec()).expect("canonical JSON is UTF-8");
    let prefix = concat!(
        "{\"schema\":\"eqiora.planar-circular-hole-envelope/v1\",",
        "\"encoding\":\"eqiora.canonical-json/v1\""
    );
    let reordered_prefix = concat!(
        "{\"encoding\":\"eqiora.canonical-json/v1\",",
        "\"schema\":\"eqiora.planar-circular-hole-envelope/v1\""
    );
    let duplicate_schema = concat!(
        "{\"schema\":\"eqiora.planar-circular-hole-envelope/v1\",",
        "\"schema\":\"eqiora.planar-circular-hole-envelope/v1\""
    );
    let wire_falsifiers = [
        text.replace("\"radius_m\":0.05", "\"radius_m\":0.050"),
        format!("{text} "),
        text.replace(prefix, reordered_prefix),
        text.replace(prefix, duplicate_schema),
        text.replace("\"entity_sets\":", "\"unknown\":0,\"entity_sets\":"),
        text.replace(
            "eqiora.planar-circular-hole-envelope/v1",
            "eqiora.planar-circular-hole-envelope/v2",
        ),
        text.replace("eqiora.canonical-json/v1", "eqiora.canonical-json/v2"),
        text.replace("axis-aligned-rectangle-with-circular-hole-v1", "other-kind"),
        text.replace("\"length_unit\":\"metre\"", "\"length_unit\":\"foot\""),
        "{".to_owned(),
    ];
    for falsifier in wire_falsifiers {
        assert!(
            CanonicalGeometryV1::decode_circular_hole_canonical(falsifier.as_bytes(), defaults,)
                .is_err(),
            "malformed, unknown, or noncanonical wire spelling must fail closed"
        );
    }

    let invalid_geometry = [
        ([[0.0, 0.0], [0.0, 1.0]], [0.5, 0.5], 0.1, 1e-12),
        ([[1.0, 0.0], [0.0, 1.0]], [0.5, 0.5], 0.1, 1e-12),
        (
            [[f64::NEG_INFINITY, 1.0], [0.0, 1.0]],
            [0.5, 0.5],
            0.1,
            1e-12,
        ),
        ([[-f64::MAX, f64::MAX], [0.0, 1.0]], [0.0, 0.5], 0.1, 1e-12),
        ([[0.0, 1.0], [0.0, 1.0]], [f64::NAN, 0.5], 0.1, 1e-12),
        ([[0.0, 1.0], [0.0, 1.0]], [0.5, 0.5], 0.0, 1e-12),
        ([[0.0, 1.0], [0.0, 1.0]], [0.5, 0.5], -0.1, 1e-12),
        ([[0.0, 1.0], [0.0, 1.0]], [0.5, 0.5], f64::NAN, 1e-12),
        ([[0.0, 1.0], [0.0, 1.0]], [0.5, 0.5], 0.1, 0.0),
        ([[0.0, 1.0], [0.0, 1.0]], [0.5, 0.5], 0.1, -1e-12),
        ([[0.0, 1.0], [0.0, 1.0]], [0.5, 0.5], 0.1, f64::NAN),
        ([[0.0, 1.0], [0.0, 1.0]], [0.5, 0.5], f64::MAX, f64::MAX),
        (
            [[-f64::MAX, -f64::MAX / 2.0], [0.0, 1.0]],
            [f64::MAX, 0.5],
            0.1,
            1e-12,
        ),
        ([[0.0, 1.0], [0.0, 1.0]], [0.1, 0.5], 0.1, 1e-12),
        ([[0.0, 1.0], [0.0, 1.0]], [0.1875, 0.5], 0.125, 0.0625),
    ];
    for (bounds, center, radius, tolerance) in invalid_geometry {
        let diagnostic =
            circular_hole_with(bounds, center, radius, circular_hole_sets(), tolerance)
                .expect_err("invalid circle predicate must fail closed");
        assert_eq!(diagnostic.code(), codes::INVALID_ARTIFACT);
    }

    let invalid_sets = [
        vec![NamedEntitySet::new(" ", FACE_DIMENSION, vec![0])],
        vec![
            NamedEntitySet::new("same", FACE_DIMENSION, vec![0]),
            NamedEntitySet::new("same", EDGE_DIMENSION, vec![0]),
        ],
        vec![NamedEntitySet::new("bad-dimension", 3, vec![0])],
        vec![NamedEntitySet::new("empty", FACE_DIMENSION, Vec::new())],
        vec![NamedEntitySet::new("out-of-range", EDGE_DIMENSION, vec![5])],
    ];
    for sets in invalid_sets {
        let diagnostic = circular_hole_with([[0.0, 1.0], [0.0, 1.0]], [0.5, 0.5], 0.1, sets, 1e-12)
            .expect_err("invalid exact entity set must fail closed");
        assert_eq!(diagnostic.code(), codes::INVALID_ARTIFACT);
    }
}

#[test]
fn every_exact_circular_hole_identity_field_is_sensitivity_evidence() {
    let witness = circular_hole_witness();
    let expected_digest = witness.digest_bytes();
    let expected_bytes = witness.canonical_bytes();
    let changed = [
        circular_hole_with(
            [[0.0, 2.3], [0.0, 0.41]],
            [0.2, 0.2],
            0.05,
            circular_hole_sets(),
            1e-12,
        ),
        circular_hole_with(
            [[0.0, 2.2], [0.0, 0.41]],
            [0.21, 0.2],
            0.05,
            circular_hole_sets(),
            1e-12,
        ),
        circular_hole_with(
            [[0.0, 2.2], [0.0, 0.41]],
            [0.2, 0.2],
            0.051,
            circular_hole_sets(),
            1e-12,
        ),
        circular_hole_with(
            [[0.0, 2.2], [0.0, 0.41]],
            [0.2, 0.2],
            0.05,
            circular_hole_sets(),
            2e-12,
        ),
        circular_hole_with(
            [[0.0, 2.2], [0.0, 0.41]],
            [0.2, 0.2],
            0.05,
            {
                let mut sets = circular_hole_sets();
                sets[3] = NamedEntitySet::new("obstacle", EDGE_DIMENSION, vec![4]);
                sets
            },
            1e-12,
        ),
        circular_hole_with(
            [[0.0, 2.2], [0.0, 0.41]],
            [0.2, 0.2],
            0.05,
            {
                let mut sets = circular_hole_sets();
                sets[0] = NamedEntitySet::new("fluid", EDGE_DIMENSION, vec![0]);
                sets
            },
            1e-12,
        ),
        circular_hole_with(
            [[0.0, 2.2], [0.0, 0.41]],
            [0.2, 0.2],
            0.05,
            {
                let mut sets = circular_hole_sets();
                sets[1] = NamedEntitySet::new("walls", EDGE_DIMENSION, vec![2]);
                sets
            },
            1e-12,
        ),
    ];
    for variant in changed {
        let variant = variant.expect("single-field perturbation remains valid");
        assert_ne!(variant.canonical_bytes(), expected_bytes);
        assert_ne!(variant.digest_bytes(), expected_digest);
    }
}

#[test]
fn artifact_bundle_is_exact_closed_and_independent_of_caller_order() {
    let hole = square_with_hole();
    let filled = filled_square();
    let (store, model, _) = declaration_model([
        (hole.digest_bytes(), "fluid"),
        (filled.digest_bytes(), "body2"),
    ]);
    let hole_ref = &hole;
    let filled_ref = &filled;
    let forward = KernelProgram::from_snapshot_with_geometry(
        &store.snapshot(),
        model,
        &[hole_ref, filled_ref],
    )
    .expect("complete bundle");
    let reverse = KernelProgram::from_snapshot_with_geometry(
        &store.snapshot(),
        model,
        &[filled_ref, hole_ref],
    )
    .expect("permuted complete bundle");
    assert_eq!(forward, reverse);

    let missing =
        KernelProgram::from_snapshot_with_geometry(&store.snapshot(), model, &[filled_ref])
            .expect_err("missing required digest");
    assert_eq!(missing.len(), 1);
    assert_eq!(missing[0].code(), codes::INVALID_ARTIFACT);
    assert!(missing[0].message().contains("missing canonical geometry"));
    assert!(missing[0].graph_path().is_some());

    let (alias_store, alias_model, alias_ids) = declaration_model([
        (hole.digest_bytes(), "fluid"),
        (hole.digest_bytes(), "fluid"),
    ]);
    let alias_missing =
        KernelProgram::from_snapshot_with_geometry(&alias_store.snapshot(), alias_model, &[])
            .expect_err("one missing digest can have multiple referencing regions");
    assert_eq!(alias_missing.len(), 1);
    let first_region = alias_ids
        .iter()
        .map(|id| id.erase())
        .min()
        .expect("two regions");
    assert_diagnostic_at(&alias_missing[0], first_region);

    let duplicate = KernelProgram::from_snapshot_with_geometry(
        &store.snapshot(),
        model,
        &[hole_ref, hole_ref, filled_ref],
    )
    .expect_err("duplicate supplied digest");
    assert_eq!(duplicate.len(), 1);
    assert_eq!(duplicate[0].code(), codes::INVALID_ARTIFACT);
    assert!(
        duplicate[0]
            .message()
            .contains("duplicate canonical geometry")
    );

    let (free_store, free_model) = geometry_free_model();
    let ordinary = KernelProgram::from_snapshot(&free_store.snapshot(), free_model)
        .expect("ordinary geometry-free program");
    let explicitly_empty =
        KernelProgram::from_snapshot_with_geometry(&free_store.snapshot(), free_model, &[])
            .expect("empty closed bundle");
    assert_eq!(ordinary, explicitly_empty);
    let extra =
        KernelProgram::from_snapshot_with_geometry(&free_store.snapshot(), free_model, &[hole_ref])
            .expect_err("geometry-free Model rejects an artifact");
    assert_eq!(extra.len(), 1);
    assert_eq!(extra[0].code(), codes::INVALID_ARTIFACT);
    assert!(
        extra[0]
            .message()
            .contains("unreferenced canonical geometry")
    );
}

#[test]
fn declaration_only_geometry_still_participates_in_bundle_closure_and_identity() {
    let geometry = square_with_hole();
    let (store, model, _) = declaration_model([(geometry.digest_bytes(), "fluid")]);
    let artifact_free = KernelProgram::from_snapshot(&store.snapshot(), model)
        .expect("a declaration can be replayed without artifact facts");
    let missing = KernelProgram::from_snapshot_with_geometry(&store.snapshot(), model, &[])
        .expect_err("the explicit admission path requires its declaration artifact");
    assert_eq!(missing.len(), 1);
    assert!(missing[0].message().contains("missing canonical geometry"));

    let admitted =
        KernelProgram::from_snapshot_with_geometry(&store.snapshot(), model, &[&geometry])
            .expect("declaration entity set is proven");
    assert_ne!(admitted, artifact_free);

    let admitted_model = ModelEnvelope::from_program(&admitted).expect("current Model");
    let artifact_free_model =
        ModelEnvelope::from_program(&artifact_free).expect("artifact-free current Model");
    assert_eq!(
        admitted_model
            .canonical_json()
            .expect("canonical admitted Model"),
        artifact_free_model
            .canonical_json()
            .expect("canonical artifact-free Model")
    );
    assert_eq!(
        StructuralSemanticFingerprint::from_program(&admitted).expect("admitted fingerprint"),
        StructuralSemanticFingerprint::from_program(&artifact_free)
            .expect("artifact-free fingerprint")
    );
}

#[test]
fn entity_set_dimension_and_parent_artifact_are_fail_closed() {
    let hole = square_with_hole();
    let hole_ref = &hole;

    let (region_store, region_model, region_ids) =
        declaration_model([(hole.digest_bytes(), "hole")]);
    let region_error = KernelProgram::from_snapshot_with_geometry(
        &region_store.snapshot(),
        region_model,
        &[hole_ref],
    )
    .expect_err("an edge set cannot be admitted as a volume");
    assert_eq!(region_error.len(), 1);
    assert!(
        region_error[0]
            .message()
            .contains("dimension 1, expected 2")
    );
    assert_diagnostic_at(&region_error[0], region_ids[0].erase());

    let (boundary_store, boundary_model, boundary_ids) =
        boundary_declarations(&hole, "fluid", "fluid");
    let boundary_error = KernelProgram::from_snapshot_with_geometry(
        &boundary_store.snapshot(),
        boundary_model,
        &[hole_ref],
    )
    .expect_err("a face set cannot be admitted as a boundary");
    assert_eq!(boundary_error.len(), 1);
    assert!(
        boundary_error[0]
            .message()
            .contains("dimension 2, expected 1")
    );
    assert_diagnostic_at(&boundary_error[0], boundary_ids.1.erase());

    let filled = filled_square();
    let (cross_store, cross_model, b1, b2) = cross_artifact_boundaries(&hole, &filled);
    let forward = KernelProgram::from_snapshot_with_geometry(
        &cross_store.snapshot(),
        cross_model,
        &[&hole, &filled],
    )
    .expect_err("B1 must resolve only against its own parent artifact");
    let reverse = KernelProgram::from_snapshot_with_geometry(
        &cross_store.snapshot(),
        cross_model,
        &[&filled, &hole],
    )
    .expect_err("artifact order cannot change diagnostics");
    assert_eq!(forward, reverse);
    assert_eq!(forward.len(), 1);
    assert!(
        forward[0]
            .message()
            .contains("absent from its parent artifact")
    );
    assert_diagnostic_at(&forward[0], b1.erase());
    assert_ne!(
        forward[0].graph_path().expect("B1 path").to_string(),
        format!("semantic.Domain.{}", b2.erase())
    );
}

fn boundary_declarations(
    geometry: &CanonicalGeometryV1,
    region_set: &str,
    boundary_set: &str,
) -> (
    InMemoryGraphStore,
    OntologyId<Model>,
    (Id<kinds::Domain>, Id<kinds::Domain>),
) {
    let region = Id::new();
    let boundary = Id::new();
    let (store, model) = committed_model(
        "geometry boundary declarations",
        vec![
            KernelNode::from(
                DomainDef::geometry_region(
                    region,
                    GeometryDigest::new(geometry.digest_bytes()),
                    region_set,
                )
                .expect("region declaration"),
            ),
            KernelNode::from(
                DomainDef::geometry_boundary(boundary, boundary_set).expect("boundary declaration"),
            ),
        ],
        [(boundary.erase(), region.erase(), EdgeKind::BoundaryOf)],
    );
    (store, model, (region, boundary))
}

fn cross_artifact_boundaries(
    hole: &CanonicalGeometryV1,
    filled: &CanonicalGeometryV1,
) -> (
    InMemoryGraphStore,
    OntologyId<Model>,
    Id<kinds::Domain>,
    Id<kinds::Domain>,
) {
    let r1 = Id::new();
    let r2 = Id::new();
    let b1 = Id::new();
    let b2 = Id::new();
    let (store, model) = committed_model(
        "parent-relative cross-artifact selection",
        vec![
            KernelNode::from(
                DomainDef::geometry_region(r1, GeometryDigest::new(hole.digest_bytes()), "fluid")
                    .expect("R1"),
            ),
            KernelNode::from(
                DomainDef::geometry_region(r2, GeometryDigest::new(filled.digest_bytes()), "body2")
                    .expect("R2"),
            ),
            KernelNode::from(DomainDef::geometry_boundary(b1, "foreign-only").expect("B1")),
            KernelNode::from(DomainDef::geometry_boundary(b2, "foreign-only").expect("B2")),
        ],
        [
            (b1.erase(), r1.erase(), EdgeKind::BoundaryOf),
            (b2.erase(), r2.erase(), EdgeKind::BoundaryOf),
        ],
    );
    (store, model, b1, b2)
}

#[test]
fn one_artifact_can_prove_multiple_region_aliases() {
    let geometry = two_face_names_square();
    let (store, model, _) = declaration_model([
        (geometry.digest_bytes(), "body-a"),
        (geometry.digest_bytes(), "body-b"),
    ]);
    KernelProgram::from_snapshot_with_geometry(&store.snapshot(), model, &[&geometry])
        .expect("bundle closure is over distinct digests, not Domain count");
}

#[test]
fn domain_topology_fault_suppresses_secondary_entity_set_admission() {
    let geometry = square_with_hole();
    let invalid_region = Id::new();
    let valid_region = Id::new();
    let (store, model) = committed_model(
        "invalid region topology",
        vec![
            KernelNode::from(
                DomainDef::geometry_region(
                    invalid_region,
                    GeometryDigest::new(geometry.digest_bytes()),
                    "absent",
                )
                .expect("invalid region declaration"),
            ),
            KernelNode::from(
                DomainDef::geometry_region(
                    valid_region,
                    GeometryDigest::new(geometry.digest_bytes()),
                    "fluid",
                )
                .expect("valid region declaration"),
            ),
        ],
        [(
            invalid_region.erase(),
            valid_region.erase(),
            EdgeKind::BoundaryOf,
        )],
    );
    let diagnostics =
        KernelProgram::from_snapshot_with_geometry(&store.snapshot(), model, &[&geometry])
            .expect_err("a region cannot have a BoundaryOf parent");
    assert_eq!(diagnostics.len(), 1);
    assert!(
        diagnostics[0]
            .message()
            .contains("only a boundary Domain may have a BoundaryOf edge")
    );
    assert!(!diagnostics[0].message().contains("absent"));
    assert_diagnostic_at(&diagnostics[0], invalid_region.erase());
}

#[test]
fn geometry_boundary_port_requires_an_embedding_contract_even_after_admission() {
    let geometry = circular_hole_witness();
    let region = Id::new();
    let boundary = Id::new();
    let connector = Id::new();
    let port = Id::new();
    let boundary_contract = BoundaryPhysicalConnector::new(
        eqiora_core::ValueType::scalar(
            eqiora_core::ScalarDomain::Real,
            DimExponents::DIMENSIONLESS,
        )
        .expect("valid scalar type"),
        eqiora_core::ValueType::scalar(
            eqiora_core::ScalarDomain::Real,
            DimExponents::DIMENSIONLESS,
        )
        .expect("valid scalar type"),
        BoundaryPairing::EuclideanBoundaryDuality,
    )
    .expect("boundary connector");
    let (store, model) = committed_model(
        "geometry boundary Port",
        vec![
            KernelNode::from(
                DomainDef::geometry_region(
                    region,
                    GeometryDigest::new(geometry.digest_bytes()),
                    "fluid",
                )
                .expect("region"),
            ),
            KernelNode::from(DomainDef::geometry_boundary(boundary, "cylinder").expect("boundary")),
            KernelNode::from(DomainDef::boundary_physical(connector, boundary_contract)),
            KernelNode::from(PortDef::boundary_physical(port, connector, boundary)),
        ],
        [(boundary.erase(), region.erase(), EdgeKind::BoundaryOf)],
    );
    let diagnostics =
        KernelProgram::from_snapshot_with_geometry(&store.snapshot(), model, &[&geometry])
            .expect_err("entity-set dimension does not invent boundary embedding facts");
    let embedding = diagnostics
        .iter()
        .find(|diagnostic| {
            diagnostic.message()
                == "boundary-physical Port support on a geometry Domain requires a non-Cartesian boundary embedding contract"
        })
        .expect("the admission boundary, not an incidental network fault, rejects the Port");
    assert_eq!(embedding.code(), codes::INVALID_KERNEL_DEFINITION);
    assert_diagnostic_at(embedding, port.erase());
}

#[test]
fn admitted_geometry_boundary_support_accepts_relation_scope_only() {
    let circular = circular_hole_witness();
    for boundary_set in ["cylinder", "walls"] {
        let (store, model, ids) = boundary_relation_model(
            circular.digest_bytes(),
            "fluid",
            boundary_set,
            std::iter::empty(),
        );
        let program =
            KernelProgram::from_snapshot_with_geometry(&store.snapshot(), model, &[&circular])
                .expect("artifact admission proves the exact circular boundary Relation scope");
        program
            .typed_relation_residual(ids.relation)
            .expect("the admitted circular boundary support remains available to typing");
    }

    let straight = square_with_hole();
    let (straight_store, straight_model, straight_ids) =
        boundary_relation_model(straight.digest_bytes(), "fluid", "hole", std::iter::empty());
    let straight_program = KernelProgram::from_snapshot_with_geometry(
        &straight_store.snapshot(),
        straight_model,
        &[&straight],
    )
    .expect("the straight-edged kind admits the same Relation scope");
    straight_program
        .typed_relation_residual(straight_ids.relation)
        .expect("straight-edged derived support remains available to typing");

    let (store, model, ids) = boundary_relation_model(
        circular.digest_bytes(),
        "fluid",
        "cylinder",
        std::iter::empty(),
    );
    let diagnostics = KernelProgram::from_snapshot(&store.snapshot(), model)
        .expect_err("artifact-free boundary Relations remain rejected");
    let admission = diagnostics
        .iter()
        .find(|diagnostic| {
            diagnostic.message()
                == "Relation spatial scope from a geometry Domain requires artifact admission"
        })
        .expect("the artifact-free entry keeps its exact diagnostic");
    assert_diagnostic_at(admission, ids.relation.erase());

    let region = Id::new();
    let relation = Id::new();
    let activation = Id::new();
    let mut expression = ExprDagBuilder::new();
    let x = expression
        .spatial_coordinate(0)
        .expect("region coordinate residual");
    let (region_store, region_model) = committed_model(
        "geometry region Relation without a Field",
        vec![
            KernelNode::from(
                DomainDef::geometry_region(
                    region,
                    GeometryDigest::new(circular.digest_bytes()),
                    "fluid",
                )
                .expect("region"),
            ),
            KernelNode::from(
                RelationDef::new(
                    relation,
                    {
                        let equation_zero_0 = expression
                            .constant(
                                eqiora_core::ValueLiteral::from_real(
                                    eqiora_core::ValueType::scalar(
                                        eqiora_core::ScalarDomain::Real,
                                        DimExponents::from_integers([0, 1, 0, 0, 0, 0, 0]).unwrap(),
                                    )
                                    .expect("valid scalar type"),
                                    0.0,
                                )
                                .unwrap(),
                            )
                            .unwrap();
                        expression.finish([x, equation_zero_0])
                    }
                    .expect("closed residual DAG"),
                )
                .unwrap(),
            ),
            KernelNode::from(ActivationDef::continuous(activation)),
        ],
        [
            (relation.erase(), region.erase(), EdgeKind::AppliesOn),
            (activation.erase(), relation.erase(), EdgeKind::Activates),
        ],
    );
    KernelProgram::from_snapshot_with_geometry(
        &region_store.snapshot(),
        region_model,
        &[&circular],
    )
    .expect("artifact admission also proves the parent region Relation scope");
    let region_diagnostics = KernelProgram::from_snapshot(&region_store.snapshot(), region_model)
        .expect_err("an artifact-free region Relation remains fail-closed");
    let region_admission = region_diagnostics
        .iter()
        .find(|diagnostic| {
            diagnostic.message()
                == "Relation spatial scope from a geometry Domain requires artifact admission"
        })
        .expect("the region-scoped Relation itself keeps the exact admission diagnostic");
    assert_diagnostic_at(region_admission, relation.erase());
}

#[test]
fn geometry_boundary_relation_falsifiers_reach_the_claimed_consumer() {
    let circular = circular_hole_witness();
    for (boundary_set, expected) in [
        (
            "absent",
            "geometry boundary entity set `absent` is absent from its parent artifact",
        ),
        (
            "fluid",
            "geometry boundary entity set `fluid` has dimension 2, expected 1",
        ),
    ] {
        let (store, model, ids) = boundary_relation_model(
            circular.digest_bytes(),
            "fluid",
            boundary_set,
            std::iter::empty(),
        );
        let diagnostics =
            KernelProgram::from_snapshot_with_geometry(&store.snapshot(), model, &[&circular])
                .expect_err("invalid parent-relative entity set must reject the scoped Relation");
        let diagnostic = diagnostics
            .iter()
            .find(|diagnostic| diagnostic.message() == expected)
            .expect("the exact entity-set diagnostic reaches the Relation fixture");
        assert_eq!(diagnostic.code(), codes::INVALID_KERNEL_DEFINITION);
        assert_diagnostic_at(diagnostic, ids.boundary.erase());
    }

    let foreign = filled_square();
    let (foreign_store, foreign_model, foreign_ids) = boundary_relation_model(
        circular.digest_bytes(),
        "fluid",
        "foreign-only",
        [(foreign.digest_bytes(), "body2")],
    );
    let foreign_diagnostics = KernelProgram::from_snapshot_with_geometry(
        &foreign_store.snapshot(),
        foreign_model,
        &[&foreign, &circular],
    )
    .expect_err("same-named membership in a foreign artifact cannot prove the parent selection");
    let reverse_foreign_diagnostics = KernelProgram::from_snapshot_with_geometry(
        &foreign_store.snapshot(),
        foreign_model,
        &[&circular, &foreign],
    )
    .expect_err("artifact order cannot change the parent-relative rejection");
    assert_eq!(foreign_diagnostics, reverse_foreign_diagnostics);
    let foreign_diagnostic = foreign_diagnostics
        .iter()
        .find(|diagnostic| {
            diagnostic.message()
                == "geometry boundary entity set `foreign-only` is absent from its parent artifact"
        })
        .expect("foreign membership is checked against the exact parent");
    assert_diagnostic_at(foreign_diagnostic, foreign_ids.boundary.erase());

    let (missing_store, missing_model, missing_ids) = boundary_relation_model(
        circular.digest_bytes(),
        "fluid",
        "cylinder",
        std::iter::empty(),
    );
    let missing =
        KernelProgram::from_snapshot_with_geometry(&missing_store.snapshot(), missing_model, &[])
            .expect_err("the scoped Relation cannot bypass a missing parent artifact");
    let expected_missing = format!(
        "missing canonical geometry artifact {}",
        hex_digest(circular.digest_bytes())
    );
    let missing_diagnostic = missing
        .iter()
        .find(|diagnostic| diagnostic.message() == expected_missing)
        .expect("the exact missing-artifact diagnostic reaches the Relation fixture");
    assert_eq!(missing_diagnostic.code(), codes::INVALID_ARTIFACT);
    assert_diagnostic_at(missing_diagnostic, missing_ids.region.erase());

    let parent = Id::new();
    let boundary = Id::new();
    let relation = Id::new();
    let activation = Id::new();
    let mut expression = ExprDagBuilder::new();
    let zero = expression
        .constant(DynQuantity::new(0.0, DimExponents::DIMENSIONLESS))
        .expect("constant residual");
    let (wrong_store, wrong_model) = committed_model(
        "geometry boundary Relation with wrong parent kind",
        vec![
            KernelNode::from(DomainDef::new(parent)),
            KernelNode::from(
                DomainDef::geometry_boundary(boundary, "cylinder").expect("geometry boundary"),
            ),
            KernelNode::from(
                RelationDef::new(
                    relation,
                    expression.finish([zero, zero]).expect("closed residual"),
                )
                .unwrap(),
            ),
            KernelNode::from(ActivationDef::continuous(activation)),
        ],
        [
            (boundary.erase(), parent.erase(), EdgeKind::BoundaryOf),
            (relation.erase(), boundary.erase(), EdgeKind::AppliesOn),
            (activation.erase(), relation.erase(), EdgeKind::Activates),
        ],
    );
    let wrong_parent =
        KernelProgram::from_snapshot_with_geometry(&wrong_store.snapshot(), wrong_model, &[])
            .expect_err("a geometry boundary Relation requires a geometry-region parent");
    let wrong_parent_diagnostic = wrong_parent
        .iter()
        .find(|diagnostic| {
            diagnostic.message() == "geometry boundary parent must be a geometry region Domain"
        })
        .expect("the exact parent-kind diagnostic reaches the Relation fixture");
    assert_diagnostic_at(wrong_parent_diagnostic, boundary.erase());
}

#[test]
fn admitted_geometry_boundary_field_keeps_its_embedding_diagnostic() {
    let geometry = circular_hole_witness();
    let region = Id::new();
    let boundary = Id::new();
    let representation = Id::new();
    let field = Id::new();
    let (store, model) = committed_model(
        "geometry boundary Field",
        vec![
            KernelNode::from(
                DomainDef::geometry_region(
                    region,
                    GeometryDigest::new(geometry.digest_bytes()),
                    "fluid",
                )
                .expect("geometry region"),
            ),
            KernelNode::from(
                DomainDef::geometry_boundary(boundary, "cylinder").expect("geometry boundary"),
            ),
            KernelNode::from(RepresentationDef::continuum(representation)),
            KernelNode::from(FieldDef::new(
                field,
                eqiora_core::ValueType::shaped(
                    eqiora_core::ScalarDomain::Real,
                    DimExponents::DIMENSIONLESS,
                    ValueShape::new([2]).expect("two-component vector"),
                    ValueFrame::SpatialCartesian,
                )
                .expect("spatial Field"),
                eqiora::kernel::FieldRole::Variable,
            )),
        ],
        [
            (boundary.erase(), region.erase(), EdgeKind::BoundaryOf),
            (field.erase(), boundary.erase(), EdgeKind::DefinedOn),
            (field.erase(), representation.erase(), EdgeKind::DefinedOn),
        ],
    );
    let diagnostics =
        KernelProgram::from_snapshot_with_geometry(&store.snapshot(), model, &[&geometry])
            .expect_err("a Field still requires a non-Cartesian boundary embedding contract");
    let embedding = diagnostics
        .iter()
        .find(|diagnostic| {
            diagnostic.message()
                == "Field spatial support on a geometry boundary Domain requires a non-Cartesian boundary embedding contract"
        })
        .expect("the live Field branch keeps its exact diagnostic");
    assert_diagnostic_at(embedding, field.erase());
}

#[test]
fn admitted_spatial_cartesian_field_extent_must_match_geometry_dimension() {
    let geometry = square_with_hole();
    let (store, model, ids) = positive_model(geometry.digest_bytes(), "fluid", "hole", 3);
    let diagnostics =
        KernelProgram::from_snapshot_with_geometry(&store.snapshot(), model, &[&geometry])
            .expect_err("3-vector cannot claim a two-dimensional spatial frame");
    let extent = diagnostics
        .iter()
        .find(|diagnostic| {
            diagnostic
                .message()
                .contains("extents must equal its admitted Domain ambient dimension")
        })
        .expect("geometry-backed extent diagnostic");
    assert_diagnostic_at(extent, ids.field.erase());
}
