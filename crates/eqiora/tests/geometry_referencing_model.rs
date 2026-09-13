use eqiora::api::{SemanticFingerprintGeneration, StructuralSemanticFingerprint};
use eqiora::artifact::{ModelDecoderLimits, ModelEnvelope, ModelTransactionEnvelope};
use eqiora::diagnostic::codes;
use eqiora::graph::{EdgeKind, GraphStore, InMemoryGraphStore, Op, Transaction};
use eqiora::kernel::{
    ActivationDef, AxisBounds, BoundaryPairing, BoundaryPhysicalConnector, BoundarySide, DomainDef,
    ExprDagBuilder, FieldDef, GeometryDigest, KernelNode, PortDef, RelationDef,
};
use eqiora::ontology::{Model, ModelView, OntologyId};
use eqiora::sem::KernelProgram;
use eqiora::{DimExponents, DynQuantity, Id, kinds};
use serde_json::Value;

#[derive(Clone, Copy)]
struct FixtureIds {
    region_a: Id<kinds::Domain>,
    region_b: Id<kinds::Domain>,
    boundary: Id<kinds::Domain>,
    cartesian: Id<kinds::Domain>,
    connector: Id<kinds::Domain>,
    field: Id<kinds::Field>,
    relation: Id<kinds::Relation>,
    activation: Id<kinds::Activation>,
    port: Id<kinds::Port>,
    model: OntologyId<Model>,
}

impl FixtureIds {
    fn fresh() -> Self {
        Self {
            region_a: Id::new(),
            region_b: Id::new(),
            boundary: Id::new(),
            cartesian: Id::new(),
            connector: Id::new(),
            field: Id::new(),
            relation: Id::new(),
            activation: Id::new(),
            port: Id::new(),
            model: OntologyId::new(),
        }
    }
}

#[derive(Clone, Copy)]
struct GeometryMeaning<'a> {
    geometry: [u8; 32],
    region_a_set: &'a str,
    region_b_set: &'a str,
    boundary_set: Option<&'a str>,
    boundary_parent: BoundaryParent,
}

impl Default for GeometryMeaning<'static> {
    fn default() -> Self {
        Self {
            geometry: [0x11; 32],
            region_a_set: "body",
            region_b_set: "peer",
            boundary_set: Some("wall"),
            boundary_parent: BoundaryParent::RegionA,
        }
    }
}

#[derive(Clone, Copy)]
enum BoundaryParent {
    None,
    RegionA,
    RegionB,
    BothRegions,
    Cartesian,
}

struct Fixture {
    program: KernelProgram,
    transaction: ModelTransactionEnvelope,
}

#[test]
fn geometry_domains_validate_and_round_trip_through_the_current_wire() {
    let ids = FixtureIds::fresh();
    let region_only = build_fixture(
        ids,
        GeometryMeaning {
            boundary_set: None,
            boundary_parent: BoundaryParent::None,
            ..GeometryMeaning::default()
        },
    );
    assert_eq!(region_only.program.nodes().count(), 5);

    let fixture = build_fixture(ids, GeometryMeaning::default());
    let model = ModelEnvelope::from_program(&fixture.program).unwrap();
    let model_bytes = model.canonical_json().unwrap();
    let model_digest = model.digest().unwrap();
    let replayed = ModelEnvelope::from_json(&model_bytes, ModelDecoderLimits::default()).unwrap();
    assert_eq!(replayed.canonical_json().unwrap(), model_bytes);
    assert_eq!(replayed.digest().unwrap(), model_digest);
    let replayed_program = replayed.to_program().unwrap();
    assert_eq!(replayed_program.model(), fixture.program.model());
    assert_eq!(
        StructuralSemanticFingerprint::from_program(&replayed_program).unwrap(),
        StructuralSemanticFingerprint::from_program(&fixture.program).unwrap()
    );

    let transaction_bytes = fixture.transaction.canonical_json().unwrap();
    let transaction_digest = fixture.transaction.digest().unwrap();
    let replayed_transaction =
        ModelTransactionEnvelope::from_json(&transaction_bytes, ModelDecoderLimits::default())
            .unwrap();
    let reencoded_transaction =
        ModelTransactionEnvelope::from_transaction(&replayed_transaction.to_transaction().unwrap())
            .unwrap();
    assert_eq!(
        reencoded_transaction.canonical_json().unwrap(),
        transaction_bytes
    );
    assert_eq!(reencoded_transaction.digest().unwrap(), transaction_digest);

    let mut malformed: Value = serde_json::from_slice(&model_bytes).unwrap();
    let original = "11".repeat(32);
    assert_eq!(
        replace_exact_string(&mut malformed, &original, "not-a-digest"),
        2
    );
    let error = ModelEnvelope::from_json(
        &serde_json::to_vec(&malformed).unwrap(),
        ModelDecoderLimits::default(),
    )
    .unwrap_err();
    assert_eq!(error.code(), codes::INVALID_ARTIFACT);
}

#[test]
fn geometry_identity_names_and_topology_are_fingerprint_meaning() {
    let ids = FixtureIds::fresh();
    let baseline = build_fixture(ids, GeometryMeaning::default());
    let baseline_model = ModelEnvelope::from_program(&baseline.program).unwrap();
    let baseline_fingerprint =
        StructuralSemanticFingerprint::from_program(&baseline.program).unwrap();
    assert_eq!(
        baseline_fingerprint.generation(),
        SemanticFingerprintGeneration::V21
    );
    assert!(
        baseline_fingerprint
            .to_string()
            .starts_with("eqiora.structural-semantic-fingerprint/v21:")
    );

    let mut changed_digest = GeometryMeaning::default();
    changed_digest.geometry[31] ^= 1;
    assert_distinct_model_and_fingerprint(
        &baseline_model,
        &baseline_fingerprint,
        &build_fixture(ids, changed_digest),
    );
    assert_distinct_model_and_fingerprint(
        &baseline_model,
        &baseline_fingerprint,
        &build_fixture(
            ids,
            GeometryMeaning {
                region_a_set: "body-renamed",
                ..GeometryMeaning::default()
            },
        ),
    );
    assert_distinct_model_and_fingerprint(
        &baseline_model,
        &baseline_fingerprint,
        &build_fixture(
            ids,
            GeometryMeaning {
                boundary_set: Some("wall-renamed"),
                ..GeometryMeaning::default()
            },
        ),
    );
    assert_distinct_model_and_fingerprint(
        &baseline_model,
        &baseline_fingerprint,
        &build_fixture(
            ids,
            GeometryMeaning {
                boundary_parent: BoundaryParent::RegionB,
                ..GeometryMeaning::default()
            },
        ),
    );

    let fresh = build_fixture(FixtureIds::fresh(), GeometryMeaning::default());
    assert_ne!(
        baseline_model.digest().unwrap(),
        ModelEnvelope::from_program(&fresh.program)
            .unwrap()
            .digest()
            .unwrap()
    );
    assert_eq!(
        baseline_fingerprint,
        StructuralSemanticFingerprint::from_program(&fresh.program).unwrap()
    );
}

#[test]
fn geometry_parent_rules_and_spatial_support_fail_closed() {
    for meaning in [
        GeometryMeaning {
            boundary_parent: BoundaryParent::None,
            ..GeometryMeaning::default()
        },
        GeometryMeaning {
            boundary_parent: BoundaryParent::BothRegions,
            ..GeometryMeaning::default()
        },
        GeometryMeaning {
            boundary_parent: BoundaryParent::Cartesian,
            ..GeometryMeaning::default()
        },
    ] {
        let ids = FixtureIds::fresh();
        assert_invalid_kernel(
            build_transaction(ids, meaning, ExtraMeaning::None, false),
            ids.model,
        );
    }

    let ids = FixtureIds::fresh();
    assert_invalid_kernel(
        build_transaction(ids, GeometryMeaning::default(), ExtraMeaning::None, true),
        ids.model,
    );
    let (transaction, model) = cartesian_boundary_with_geometry_parent();
    assert_invalid_kernel(transaction, model);

    for extra in [
        ExtraMeaning::FieldSupport,
        ExtraMeaning::RelationSupport,
        ExtraMeaning::BoundaryPortSupport,
    ] {
        let ids = FixtureIds::fresh();
        let diagnostics = invalid_kernel(
            build_transaction(ids, GeometryMeaning::default(), extra, false),
            ids.model,
        );
        let admission = diagnostics
            .iter()
            .find(|diagnostic| diagnostic.message().contains("requires artifact admission"))
            .unwrap_or_else(|| {
                panic!(
                    "geometry support must remain closed before artifact admission: {diagnostics:?}"
                )
            });
        assert_eq!(admission.code(), codes::INVALID_KERNEL_DEFINITION);
    }
}

fn build_fixture(ids: FixtureIds, meaning: GeometryMeaning<'_>) -> Fixture {
    let transaction = build_transaction(ids, meaning, ExtraMeaning::None, false);
    let encoded = ModelTransactionEnvelope::from_transaction(&transaction).unwrap();
    let mut store = InMemoryGraphStore::new();
    store.commit(transaction).unwrap();
    let program = KernelProgram::from_snapshot(&store.snapshot(), ids.model).unwrap();
    Fixture {
        program,
        transaction: encoded,
    }
}

#[derive(Clone, Copy)]
enum ExtraMeaning {
    None,
    FieldSupport,
    RelationSupport,
    BoundaryPortSupport,
}

fn build_transaction(
    ids: FixtureIds,
    meaning: GeometryMeaning<'_>,
    extra: ExtraMeaning,
    region_has_parent: bool,
) -> Transaction {
    let length = DimExponents::from_integers([0, 1, 0, 0, 0, 0, 0]).expect("bounded dimension");
    let mut expression = ExprDagBuilder::new();
    let root = expression
        .constant(DynQuantity::new(0.0, DimExponents::DIMENSIONLESS))
        .unwrap();
    let mut nodes = vec![
        KernelNode::from(
            DomainDef::geometry_region(
                ids.region_a,
                GeometryDigest::new(meaning.geometry),
                meaning.region_a_set,
            )
            .unwrap(),
        ),
        KernelNode::from(
            DomainDef::geometry_region(
                ids.region_b,
                GeometryDigest::new(meaning.geometry),
                meaning.region_b_set,
            )
            .unwrap(),
        ),
        KernelNode::from(
            DomainDef::cartesian_box(
                ids.cartesian,
                vec![
                    AxisBounds::new(DynQuantity::new(0.0, length), DynQuantity::new(1.0, length))
                        .unwrap(),
                ],
            )
            .unwrap(),
        ),
        KernelNode::from(
            RelationDef::new(ids.relation, expression.finish([root, root]).unwrap()).unwrap(),
        ),
        KernelNode::from(ActivationDef::continuous(ids.activation)),
    ];
    if let Some(entity_set) = meaning.boundary_set {
        nodes.push(KernelNode::from(
            DomainDef::geometry_boundary(ids.boundary, entity_set).unwrap(),
        ));
    }

    match extra {
        ExtraMeaning::None => {}
        ExtraMeaning::FieldSupport => nodes.push(KernelNode::from(FieldDef::new(
            ids.field,
            eqiora_core::ValueType::scalar(
                eqiora_core::ScalarDomain::Real,
                DimExponents::DIMENSIONLESS,
            )
            .expect("valid scalar type"),
            eqiora::kernel::FieldRole::Variable,
        ))),
        ExtraMeaning::RelationSupport => {}
        ExtraMeaning::BoundaryPortSupport => {
            let connector = BoundaryPhysicalConnector::new(
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
            .unwrap();
            nodes.extend([
                KernelNode::from(DomainDef::boundary_physical(ids.connector, connector)),
                KernelNode::from(PortDef::boundary_physical(
                    ids.port,
                    ids.connector,
                    ids.boundary,
                )),
            ]);
        }
    }

    let members = nodes.iter().map(KernelNode::id).collect::<Vec<_>>();
    let mut transaction = Transaction::new("geometry-referencing model");
    for node in nodes {
        transaction.push(Op::DefineKernelNode { node });
    }
    match meaning.boundary_parent {
        BoundaryParent::None => {}
        BoundaryParent::RegionA => {
            transaction.push(Op::Connect {
                from: ids.boundary.erase(),
                to: ids.region_a.erase(),
                edge: EdgeKind::BoundaryOf,
            });
        }
        BoundaryParent::RegionB => {
            transaction.push(Op::Connect {
                from: ids.boundary.erase(),
                to: ids.region_b.erase(),
                edge: EdgeKind::BoundaryOf,
            });
        }
        BoundaryParent::BothRegions => {
            for parent in [ids.region_a, ids.region_b] {
                transaction.push(Op::Connect {
                    from: ids.boundary.erase(),
                    to: parent.erase(),
                    edge: EdgeKind::BoundaryOf,
                });
            }
        }
        BoundaryParent::Cartesian => {
            transaction.push(Op::Connect {
                from: ids.boundary.erase(),
                to: ids.cartesian.erase(),
                edge: EdgeKind::BoundaryOf,
            });
        }
    }
    if region_has_parent {
        transaction.push(Op::Connect {
            from: ids.region_a.erase(),
            to: ids.region_b.erase(),
            edge: EdgeKind::BoundaryOf,
        });
    }
    transaction.push(Op::Connect {
        from: ids.activation.erase(),
        to: ids.relation.erase(),
        edge: EdgeKind::Activates,
    });
    match extra {
        ExtraMeaning::None | ExtraMeaning::BoundaryPortSupport => {}
        ExtraMeaning::FieldSupport => {
            transaction.push(Op::Connect {
                from: ids.field.erase(),
                to: ids.region_a.erase(),
                edge: EdgeKind::DefinedOn,
            });
        }
        ExtraMeaning::RelationSupport => {
            transaction.push(Op::Connect {
                from: ids.relation.erase(),
                to: ids.region_a.erase(),
                edge: EdgeKind::AppliesOn,
            });
        }
    }
    transaction.push(Op::DefineOntologyView {
        view: ModelView::new(ids.model, members, None).unwrap().into(),
    });
    transaction
}

fn cartesian_boundary_with_geometry_parent() -> (Transaction, OntologyId<Model>) {
    let ids = FixtureIds::fresh();
    let mut expression = ExprDagBuilder::new();
    let root = expression
        .constant(DynQuantity::new(0.0, DimExponents::DIMENSIONLESS))
        .unwrap();
    let nodes = [
        KernelNode::from(
            DomainDef::geometry_region(ids.region_a, GeometryDigest::new([0x11; 32]), "body")
                .unwrap(),
        ),
        KernelNode::from(DomainDef::cartesian_boundary(
            ids.boundary,
            0,
            BoundarySide::Lower,
        )),
        KernelNode::from(
            RelationDef::new(ids.relation, expression.finish([root, root]).unwrap()).unwrap(),
        ),
        KernelNode::from(ActivationDef::continuous(ids.activation)),
    ];
    let mut transaction = Transaction::new("Cartesian boundary with geometry parent");
    for node in &nodes {
        transaction.push(Op::DefineKernelNode { node: node.clone() });
    }
    transaction
        .push(Op::Connect {
            from: ids.boundary.erase(),
            to: ids.region_a.erase(),
            edge: EdgeKind::BoundaryOf,
        })
        .push(Op::Connect {
            from: ids.activation.erase(),
            to: ids.relation.erase(),
            edge: EdgeKind::Activates,
        })
        .push(Op::DefineOntologyView {
            view: ModelView::new(ids.model, nodes.iter().map(KernelNode::id), None)
                .unwrap()
                .into(),
        });
    (transaction, ids.model)
}

fn invalid_kernel(transaction: Transaction, model: OntologyId<Model>) -> Vec<eqiora::Diagnostic> {
    let mut store = InMemoryGraphStore::new();
    store.commit(transaction).unwrap();
    KernelProgram::from_snapshot(&store.snapshot(), model).unwrap_err()
}

fn assert_invalid_kernel(transaction: Transaction, model: OntologyId<Model>) {
    let diagnostics = invalid_kernel(transaction, model);
    assert!(
        diagnostics
            .iter()
            .any(|diagnostic| diagnostic.code() == codes::INVALID_KERNEL_DEFINITION),
        "expected EQ0302: {diagnostics:?}"
    );
}

fn assert_distinct_model_and_fingerprint(
    baseline_model: &ModelEnvelope,
    baseline_fingerprint: &StructuralSemanticFingerprint,
    changed: &Fixture,
) {
    assert_ne!(
        baseline_model.digest().unwrap(),
        ModelEnvelope::from_program(&changed.program)
            .unwrap()
            .digest()
            .unwrap()
    );
    assert_ne!(
        baseline_fingerprint,
        &StructuralSemanticFingerprint::from_program(&changed.program).unwrap()
    );
}

fn replace_exact_string(value: &mut Value, before: &str, after: &str) -> usize {
    match value {
        Value::String(string) if string == before => {
            *string = after.to_owned();
            1
        }
        Value::Array(values) => values
            .iter_mut()
            .map(|value| replace_exact_string(value, before, after))
            .sum(),
        Value::Object(values) => values
            .values_mut()
            .map(|value| replace_exact_string(value, before, after))
            .sum(),
        _ => 0,
    }
}
