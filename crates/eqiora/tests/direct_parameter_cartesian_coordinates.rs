use eqiora::api::{ModelDocument, SemanticFingerprintGeneration, StructuralSemanticFingerprint};
use eqiora::artifact::{ModelDecoderLimits, ModelEnvelope, ModelTransactionEnvelope};
use eqiora::graph::{EdgeKind, GraphStore, InMemoryGraphStore, Op, Precondition, Transaction};
use eqiora::kernel::{
    AxisBounds, CartesianAxisDefinition, CartesianCoordinateSource, DomainDef, DomainKind,
    KernelNode,
};
use eqiora::ontology::ModelView;
use eqiora::sem::KernelProgram;
use eqiora::{Diagnostic, DimExponents, DynQuantity, Id, diagnostic::codes, kinds};
use serde::Deserialize;

const SOURCE: &str = include_str!(
    "../../../verify/geometry/direct-parameter-cartesian-coordinates/models/parameter-box.eqi"
);
const PERMUTED: &str = include_str!(
    "../../../verify/geometry/direct-parameter-cartesian-coordinates/models/permuted.eqi"
);
const ORACLE: &[u8] = include_bytes!(
    "../../../verify/geometry/direct-parameter-cartesian-coordinates/expected/oracle.json"
);

#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
struct Oracle {
    schema: String,
    base: RevisionOracle,
    second_revision: RevisionOracle,
}

#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
struct RevisionOracle {
    parameter_m: f64,
    bounds_m: [[f64; 2]; 3],
    extents_m: [f64; 3],
    volume_m3: f64,
}

#[test]
fn direct_sources_resolve_once_and_match_both_precommitted_revisions() {
    let oracle: Oracle = serde_json::from_slice(ORACLE).unwrap();
    assert_eq!(
        oracle.schema,
        "eqiora.verify.direct-parameter-cartesian-coordinates-oracle/v1"
    );
    let (mut store, base, body, parameter) = compile_program(SOURCE);

    assert_recipe_and_dependency(&base, body, parameter);
    assert_revision(&base, body, parameter, &oracle.base);
    assert_eq!(
        StructuralSemanticFingerprint::from_program(&base)
            .unwrap()
            .generation(),
        SemanticFingerprintGeneration::V21
    );

    let before = base.value(parameter.erase()).unwrap();
    let mut update = Transaction::new("independent second immutable revision");
    update
        .require(Precondition::RevisionIs(base.revision()))
        .require(Precondition::ValueEquals {
            target: parameter.erase(),
            expected: before.try_into().unwrap(),
        })
        .push(Op::SetValue {
            target: parameter.erase(),
            value: DynQuantity::new(oracle.second_revision.parameter_m, length_dimension())
                .try_into()
                .unwrap(),
        });
    store.commit(update).unwrap();
    let second = KernelProgram::from_snapshot(&store.snapshot(), base.model()).unwrap();

    assert_recipe_and_dependency(&second, body, parameter);
    assert_revision(&second, body, parameter, &oracle.second_revision);
    assert_eq!(
        base.nodes().collect::<Vec<_>>(),
        second.nodes().collect::<Vec<_>>()
    );
    assert_eq!(base.edges(), second.edges());
}

#[test]
fn declaration_permutations_preserve_canonical_source_occurrences() {
    let (base_transaction, base_program) = compiled_parts(SOURCE);
    let (permuted_transaction, permuted_program) = compiled_parts(PERMUTED);

    let base_model = ModelEnvelope::from_program(&base_program).unwrap();
    let permuted_model = ModelEnvelope::from_program(&permuted_program).unwrap();
    assert_eq!(
        StructuralSemanticFingerprint::from_program(&base_program).unwrap(),
        StructuralSemanticFingerprint::from_program(&permuted_program).unwrap()
    );
    assert_eq!(
        base_model.digest().unwrap(),
        permuted_model.digest().unwrap()
    );
    let base_ids = base_program
        .nodes()
        .map(KernelNode::id)
        .collect::<std::collections::BTreeSet<_>>();
    let permuted_ids = permuted_program
        .nodes()
        .map(KernelNode::id)
        .collect::<std::collections::BTreeSet<_>>();
    assert_eq!(base_ids, permuted_ids);

    let base_edit = ModelTransactionEnvelope::from_transaction(&base_transaction).unwrap();
    let permuted_edit = ModelTransactionEnvelope::from_transaction(&permuted_transaction).unwrap();
    assert_eq!(base_edit.digest().unwrap(), permuted_edit.digest().unwrap());

    let model_bytes = base_model.canonical_json().unwrap();
    let replayed = ModelEnvelope::from_json(&model_bytes, ModelDecoderLimits::default()).unwrap();
    assert_eq!(replayed.to_program().unwrap(), base_program);
    let transaction_bytes = base_edit.canonical_json().unwrap();
    let replayed =
        ModelTransactionEnvelope::from_json(&transaction_bytes, ModelDecoderLimits::default())
            .unwrap();
    let replayed = replayed.to_transaction().unwrap();
    assert_eq!(replayed.label(), base_transaction.label());
    assert_eq!(replayed.preconditions(), base_transaction.preconditions());
    assert_eq!(replayed.ops(), base_transaction.ops());

    let mut malformed: serde_json::Value = serde_json::from_slice(&model_bytes).unwrap();
    assert!(replace_parameter_source_with_unknown(&mut malformed));
    assert!(
        ModelEnvelope::from_json(
            &serde_json::to_vec(&malformed).unwrap(),
            ModelDecoderLimits::default()
        )
        .is_err()
    );
}

#[test]
fn closed_language_and_whole_model_invariants_fail_before_exposure() {
    assert_compile_rejected_with(
        "model m() { domain body = box(0, missing); relation r on body { coordinate(0) - coordinate(0) = 0; } }",
        codes::LANGUAGE_TYPE_ERROR,
        "unresolved Cartesian coordinate Parameter `missing`",
    );
    assert_compile_rejected_with(
        "model m() { parameter extent: s = 1; domain body = box(0, extent); relation r on body { coordinate(0) - coordinate(0) = 0; } }",
        codes::LANGUAGE_TYPE_ERROR,
        "Cartesian coordinate Parameter `extent` is not a real scalar length",
    );
    assert_compile_rejected_with(
        "model m() { parameter extent: m = 1; domain body = box(0, extent + 1); relation r on body { coordinate(0) - coordinate(0) = 0; } }",
        codes::SYNTAX_ERROR,
        "after Cartesian bounds",
    );
    assert_compile_rejected_with(
        "model m() { parameter extent: m = 7; domain body = box(-1, extent, extent, 6); relation r on body { coordinate(0) - coordinate(0) = 0; } }",
        codes::INVALID_KERNEL_DEFINITION,
        "resolves to non-finite, equal, or reversed bounds",
    );
    assert_compile_rejected_with(
        "model m() { parameter extent: m = 1; domain a = box(0, extent); domain b = box(0, extent); relation r on a { coordinate(0) - coordinate(0) = 0; } }",
        codes::INVALID_KERNEL_DEFINITION,
        "is already owned by Domain",
    );

    let (transaction, model, symbols) = compiled(SOURCE);
    let body = symbols.get("body").unwrap();
    let parameter = symbols.get("extent").unwrap();
    let mut missing_dependency = Transaction::new(transaction.label());
    for op in transaction.ops() {
        if !matches!(
            op,
            Op::Connect {
                from,
                to,
                edge: EdgeKind::DependsOn,
            } if *from == body && *to == parameter
        ) {
            missing_dependency.push(op.clone());
        }
    }
    let mut store = InMemoryGraphStore::new();
    store.commit(missing_dependency).unwrap();
    assert!(KernelProgram::from_snapshot(&store.snapshot(), model).is_err());

    let invalid_axis = CartesianAxisDefinition::new(
        CartesianCoordinateSource::Fixed(DynQuantity::new(0.0, DimExponents::DIMENSIONLESS)),
        CartesianCoordinateSource::fixed(DynQuantity::new(1.0, length_dimension())).unwrap(),
    );
    assert!(DomainDef::cartesian_box_from_sources(Id::new(), vec![invalid_axis]).is_err());
}

#[test]
fn current_wire_defers_cross_node_recipe_checks_to_whole_model_replay() {
    let fixed = SOURCE.replace("box(-1, extent, extent, 6", "box(-1, 2, 2, 6");
    let document = ModelDocument::compile("fixed-with-parameter.eqi", &fixed).unwrap();
    let body = domain(&document, "body");
    let parameter: Id<kinds::Parameter> = document.aliases()["extent"].downcast().unwrap();
    let mut wire: serde_json::Value =
        serde_json::from_slice(&document.canonical_json().unwrap()).unwrap();
    wire["edges"]
        .as_array_mut()
        .unwrap()
        .push(serde_json::json!({
            "from": {"kind": "domain", "ulid": body.ulid().to_string()},
            "to": {"kind": "parameter", "ulid": parameter.ulid().to_string()},
            "kind": "depends-on"
        }));
    let bytes = serde_json::to_vec(&wire).unwrap();
    let admitted = ModelEnvelope::from_json(&bytes, ModelDecoderLimits::default()).unwrap();
    assert!(admitted.to_program().is_err());
}

#[test]
fn an_extra_dependency_to_an_unreferenced_in_model_parameter_fails_whole_model_validation() {
    let source = SOURCE.replace(
        "parameter extent: m = 2;",
        "parameter extent: m = 2;\n  parameter spare: m = 3;",
    );
    let (transaction, model, symbols) = compiled(&source);
    let body = symbols.get("body").unwrap();
    let spare = symbols.get("spare").unwrap();

    let mut accepted = InMemoryGraphStore::new();
    accepted.commit(clone_transaction(&transaction)).unwrap();
    KernelProgram::from_snapshot(&accepted.snapshot(), model)
        .expect("an in-Model Parameter that no coordinate references is valid on its own");

    let mut extra = clone_transaction(&transaction);
    extra.push(Op::Connect {
        from: body,
        to: spare,
        edge: EdgeKind::DependsOn,
    });
    let mut store = InMemoryGraphStore::new();
    store.commit(extra).unwrap();
    assert!(KernelProgram::from_snapshot(&store.snapshot(), model).is_err());
}

#[test]
fn a_non_cartesian_domain_cannot_carry_a_parameter_dependency() {
    let (transaction, model, symbols) = compiled(SOURCE);
    let parameter = symbols.get("extent").unwrap();
    let abstract_domain = Id::<kinds::Domain>::new();
    let mut mutated = Transaction::new(transaction.label());
    for precondition in transaction.preconditions() {
        mutated.require(precondition.clone());
    }
    for op in transaction.ops() {
        match op {
            Op::DefineOntologyView { view } => {
                mutated.push(Op::DefineKernelNode {
                    node: DomainDef::new(abstract_domain).into(),
                });
                let members = view
                    .members()
                    .iter()
                    .copied()
                    .chain([abstract_domain.erase()]);
                mutated.push(Op::DefineOntologyView {
                    view: ModelView::new(model, members, view.boundary().iter().copied())
                        .unwrap()
                        .into(),
                });
            }
            _ => {
                mutated.push(op.clone());
            }
        }
    }
    mutated.push(Op::Connect {
        from: abstract_domain.erase(),
        to: parameter,
        edge: EdgeKind::DependsOn,
    });

    let mut store = InMemoryGraphStore::new();
    store.commit(mutated).unwrap();
    let diagnostics = KernelProgram::from_snapshot(&store.snapshot(), model).unwrap_err();
    assert!(diagnostics.iter().any(|diagnostic| {
        diagnostic.code() == codes::INVALID_KERNEL_DEFINITION
            && diagnostic
                .message()
                .contains("non-Cartesian Domain must not depend on Parameters")
    }));
}

#[test]
fn incomplete_edit_paths_reject_a_geometry_driving_parameter() {
    let document = ModelDocument::compile("parameter-box.eqi", SOURCE).unwrap();
    let parameter = document.aliases()["extent"];
    let value_error = document
        .preview_value_edit(
            parameter,
            eqiora_core::ValueLiteral::from_real(
                document
                    .program()
                    .typed_value(parameter)
                    .unwrap()
                    .value_type()
                    .clone(),
                3.5,
            )
            .unwrap(),
        )
        .unwrap_err();
    assert_eq!(value_error.code(), codes::INVALID_OPERATION);
    assert_eq!(
        value_error.message(),
        "value edit cannot target a Cartesian coordinate Parameter; the geometry regeneration owner currently accepts one 3D Domain"
    );

    let body = domain(&document, "body");
    let edit_errors = document
        .preview_cartesian_domain_edit(body, [(0, axis_bounds(-2.0, 3.5))])
        .unwrap_err();
    assert!(edit_errors.iter().any(|diagnostic| {
        diagnostic.code() == codes::INVALID_OPERATION
            && diagnostic.message()
                == "direct Cartesian Domain edit does not admit a Parameter-backed coordinate"
    }));
}

#[test]
fn absent_parameter_definition_and_non_finite_value_fail_whole_model_resolution() {
    assert!(
        eqiora_core::ValueLiteral::try_from(DynQuantity::new(f64::NAN, length_dimension()))
            .is_err()
    );

    // A defined Parameter always seeds its revision-local value and no public
    // operation clears it. The constructible absence is therefore a coordinate
    // reference to a Parameter outside the selected Model.
    let (transaction, model, symbols) = compiled(SOURCE);
    let outside = symbols.get("extent").unwrap();
    let mut narrowed = Transaction::new(transaction.label());
    for precondition in transaction.preconditions() {
        narrowed.require(precondition.clone());
    }
    for op in transaction.ops() {
        let retained = match op {
            Op::DefineOntologyView { view } => Op::DefineOntologyView {
                view: ModelView::new(
                    model,
                    view.members()
                        .iter()
                        .copied()
                        .filter(|member| *member != outside),
                    view.boundary().iter().copied(),
                )
                .unwrap()
                .into(),
            },
            _ => op.clone(),
        };
        narrowed.push(retained);
    }
    let mut store = InMemoryGraphStore::new();
    store.commit(narrowed).unwrap();
    assert!(KernelProgram::from_snapshot(&store.snapshot(), model).is_err());
}

#[test]
fn current_decoding_rejects_wrong_kind_and_foreign_coordinate_parameter_ids() {
    let (_, base, body, _) = compile_program(SOURCE);
    let wire = current_wire(&base);
    decode_current(&wire).expect("the unmutated current model decodes");

    let mut wrong_kind = wire.clone();
    assert_eq!(
        retarget_coordinate_parameters(&mut wrong_kind, &wire_id("domain", body.ulid())),
        2
    );
    assert!(decode_current(&wrong_kind).is_err());

    let mut foreign = wire.clone();
    assert_eq!(
        retarget_coordinate_parameters(
            &mut foreign,
            &wire_id("parameter", Id::<kinds::Parameter>::new().ulid())
        ),
        2
    );
    assert!(decode_current(&foreign).is_err());
}

#[test]
fn current_decoding_rejects_duplicate_and_forged_dependency_edges() {
    let (_, base, body, parameter) = compile_program(SOURCE);
    let wire = current_wire(&base);
    let edge = dependency_edge_index(&wire, body, parameter);

    let mut duplicated = wire.clone();
    let repeated = duplicated["edges"][edge].clone();
    duplicated["edges"].as_array_mut().unwrap().push(repeated);
    assert!(decode_current(&duplicated).is_err());

    // The recipe still names the real Parameter; only the persisted
    // dependency claims an identity the Model does not contain.
    let mut forged = wire.clone();
    forged["edges"][edge]["to"] = wire_id("parameter", Id::<kinds::Parameter>::new().ulid());
    assert!(decode_current(&forged).is_err());

    let mut wrong_kind = wire.clone();
    wrong_kind["edges"][edge]["to"] = node_id(&wire, "relation");
    assert!(decode_current(&wrong_kind).is_err());
}

#[test]
fn current_decoding_rejects_a_parameter_definition_that_omits_its_mandatory_value() {
    let (_, base, _, _) = compile_program(SOURCE);
    let wire = current_wire(&base);
    decode_current(&wire).expect("the unmutated current model decodes");

    let mut omitted = wire.clone();
    assert!(remove_parameter_definition_value(&mut omitted));
    assert!(decode_current(&omitted).is_err());
}

fn compile_program(
    source: &str,
) -> (
    InMemoryGraphStore,
    KernelProgram,
    Id<kinds::Domain>,
    Id<kinds::Parameter>,
) {
    let (transaction, model, symbols) = compiled(source);
    let body = symbols.get("body").unwrap().downcast().unwrap();
    let parameter = symbols.get("extent").unwrap().downcast().unwrap();
    let mut store = InMemoryGraphStore::new();
    store.commit(transaction).unwrap();
    let program = KernelProgram::from_snapshot(&store.snapshot(), model).unwrap();
    (store, program, body, parameter)
}

fn compiled_parts(source: &str) -> (Transaction, KernelProgram) {
    let (transaction, model, _) = compiled(source);
    let retained = clone_transaction(&transaction);
    let mut store = InMemoryGraphStore::new();
    store.commit(transaction).unwrap();
    let program = KernelProgram::from_snapshot(&store.snapshot(), model).unwrap();
    (retained, program)
}

fn compiled(
    source: &str,
) -> (
    Transaction,
    eqiora::ontology::OntologyId<eqiora::ontology::Model>,
    eqiora::compiler::ModelSymbols,
) {
    let mut models = eqiora::compiler::compile("parameter-box.eqi", source).unwrap();
    models.remove(0).into_parts()
}

fn clone_transaction(transaction: &Transaction) -> Transaction {
    let mut retained = Transaction::new(transaction.label());
    for precondition in transaction.preconditions() {
        retained.require(precondition.clone());
    }
    for op in transaction.ops() {
        retained.push(op.clone());
    }
    retained
}

fn assert_compile_rejected_with(source: &str, code: eqiora::Code, message_fragment: &str) {
    let diagnostics = ModelDocument::compile("rejected.eqi", source).unwrap_err();
    assert!(
        diagnostics.iter().any(|diagnostic| {
            diagnostic.code() == code && diagnostic.message().contains(message_fragment)
        }),
        "expected {code} containing {message_fragment:?}, got {diagnostics:#?}"
    );
}

fn assert_recipe_and_dependency(
    program: &KernelProgram,
    body: Id<kinds::Domain>,
    parameter: Id<kinds::Parameter>,
) {
    let Some(KernelNode::Domain(domain)) = program.node(body.erase()) else {
        panic!("body must be a Domain");
    };
    let DomainKind::CartesianBox { coordinates } = domain.kind() else {
        panic!("body must be Cartesian");
    };
    assert!(matches!(
        coordinates[0].upper(),
        CartesianCoordinateSource::Parameter(target) if target == parameter
    ));
    assert!(matches!(
        coordinates[1].lower(),
        CartesianCoordinateSource::Parameter(target) if target == parameter
    ));
    assert_eq!(
        program
            .edges()
            .iter()
            .filter(|edge| {
                edge.from() == body.erase()
                    && edge.to() == parameter.erase()
                    && edge.kind() == EdgeKind::DependsOn
            })
            .count(),
        1
    );
}

fn assert_revision(
    program: &KernelProgram,
    body: Id<kinds::Domain>,
    parameter: Id<kinds::Parameter>,
    oracle: &RevisionOracle,
) {
    assert_eq!(
        program.value(parameter.erase()).unwrap().value(),
        oracle.parameter_m
    );
    let bounds = program.resolved_cartesian_bounds(body).unwrap();
    let actual_bounds: [[f64; 2]; 3] = bounds
        .iter()
        .map(|axis| [axis.lower().value(), axis.upper().value()])
        .collect::<Vec<_>>()
        .try_into()
        .unwrap();
    assert_eq!(actual_bounds, oracle.bounds_m);
    let extents = actual_bounds.map(|axis| axis[1] - axis[0]);
    assert_eq!(extents, oracle.extents_m);
    assert_eq!(extents.into_iter().product::<f64>(), oracle.volume_m3);
}

fn domain(document: &ModelDocument, name: &str) -> Id<kinds::Domain> {
    document.aliases()[name].downcast().unwrap()
}

fn length_dimension() -> DimExponents {
    DimExponents::from_integers([0, 1, 0, 0, 0, 0, 0]).expect("bounded dimension")
}

fn axis_bounds(lower: f64, upper: f64) -> AxisBounds {
    AxisBounds::new(
        DynQuantity::new(lower, length_dimension()),
        DynQuantity::new(upper, length_dimension()),
    )
    .unwrap()
}

fn current_wire(program: &KernelProgram) -> serde_json::Value {
    let bytes = ModelEnvelope::from_program(program)
        .unwrap()
        .canonical_json()
        .unwrap();
    serde_json::from_slice(&bytes).unwrap()
}

fn decode_current(wire: &serde_json::Value) -> Result<ModelEnvelope, Diagnostic> {
    ModelEnvelope::from_json(
        &serde_json::to_vec(wire).unwrap(),
        ModelDecoderLimits::default(),
    )
}

fn wire_id(kind: &str, ulid: impl std::fmt::Display) -> serde_json::Value {
    serde_json::json!({"kind": kind, "ulid": ulid.to_string()})
}

fn node_id(wire: &serde_json::Value, kind: &str) -> serde_json::Value {
    wire["nodes"]
        .as_array()
        .unwrap()
        .iter()
        .map(|node| node["id"].clone())
        .find(|id| id["kind"] == kind)
        .unwrap_or_else(|| panic!("the proving Model defines one {kind} node"))
}

fn dependency_edge_index(
    wire: &serde_json::Value,
    body: Id<kinds::Domain>,
    parameter: Id<kinds::Parameter>,
) -> usize {
    wire["edges"]
        .as_array()
        .unwrap()
        .iter()
        .position(|edge| {
            edge["kind"] == "depends-on"
                && edge["from"] == wire_id("domain", body.ulid())
                && edge["to"] == wire_id("parameter", parameter.ulid())
        })
        .expect("current persists one Domain DependsOn Parameter edge")
}

fn remove_parameter_definition_value(wire: &mut serde_json::Value) -> bool {
    wire["nodes"]
        .as_array_mut()
        .unwrap()
        .iter_mut()
        .find(|node| node["id"]["kind"] == "parameter")
        .expect("the proving Model defines one Parameter node")["definition"]
        .as_object_mut()
        .unwrap()
        .remove("value")
        .is_some()
}

fn retarget_coordinate_parameters(
    value: &mut serde_json::Value,
    replacement: &serde_json::Value,
) -> usize {
    match value {
        serde_json::Value::Object(object) => {
            if object.get("source").and_then(serde_json::Value::as_str) == Some("parameter") {
                object.insert("parameter".to_owned(), replacement.clone());
                return 1;
            }
            object
                .values_mut()
                .map(|value| retarget_coordinate_parameters(value, replacement))
                .sum()
        }
        serde_json::Value::Array(values) => values
            .iter_mut()
            .map(|value| retarget_coordinate_parameters(value, replacement))
            .sum(),
        _ => 0,
    }
}

fn replace_parameter_source_with_unknown(value: &mut serde_json::Value) -> bool {
    match value {
        serde_json::Value::Object(object) => {
            if object.get("source").and_then(serde_json::Value::as_str) == Some("parameter") {
                object.insert(
                    "source".to_owned(),
                    serde_json::Value::String("future-coordinate".to_owned()),
                );
                return true;
            }
            object
                .values_mut()
                .any(replace_parameter_source_with_unknown)
        }
        serde_json::Value::Array(values) => {
            values.iter_mut().any(replace_parameter_source_with_unknown)
        }
        _ => false,
    }
}
