use eqiora::api::{ModelDocument, SemanticFingerprintGeneration, StructuralSemanticFingerprint};
use eqiora::graph::{EdgeKind, GraphStore, InMemoryGraphStore, Op, Transaction};
use eqiora::kernel::{
    ActivationDef, ExprDagBuilder, FieldDef, PortDef, RelationDef, SignalDirection, SymbolRef,
};
use eqiora::language::{
    DraftConservingConnection, DraftConservingPort, DraftExpression, DraftField, DraftParameter,
    DraftPhysicalDomain, DraftRelation, Module,
};
use eqiora::ontology::{Model, ModelView, OntologyId};
use eqiora::sem::KernelProgram;
use eqiora::{DimExponents, Id, kinds};

const DECAY: &str =
    include_str!("../../../verify/interfaces/structural-semantic-fingerprint/models/decay.eqi");

const PHYSICAL: &str =
    include_str!("../../../verify/interfaces/structural-semantic-fingerprint/models/resistor.eqi");

#[test]
fn rational_dimension_meaning_survives_canonical_model_replay() {
    let source =
        "model Wave() { parameter amplitude: m ^ (-1 / 2) = 1; relation r { amplitude = 0; } }";
    let model = ModelDocument::compile("wave.eqi", source).unwrap();
    let equivalent =
        ModelDocument::compile("equal.eqi", &source.replace("-1 / 2", "-2 / 4")).unwrap();
    let changed =
        ModelDocument::compile("changed.eqi", &source.replace("-1 / 2", "-1 / 3")).unwrap();
    assert!(model.structurally_equivalent(&equivalent).unwrap());
    assert!(!model.structurally_equivalent(&changed).unwrap());
    let bytes = model.canonical_json().unwrap();
    let replay = ModelDocument::replay(&bytes).unwrap();
    assert_eq!(replay.canonical_json().unwrap(), bytes);
    assert!(model.structurally_equivalent(&replay).unwrap());
    let current_schema = String::from_utf8(bytes).unwrap();
    let old_schema =
        current_schema.replace("eqiora.model-envelope/v26", "eqiora.model-envelope/v10");
    assert_ne!(old_schema, current_schema);
    assert!(ModelDocument::replay(old_schema.as_bytes()).is_err());
    assert_eq!(
        model.structural_fingerprint().unwrap().generation(),
        SemanticFingerprintGeneration::V21
    );
}

#[test]
fn rational_physical_relations_typecheck_and_replay_together() {
    let source = r#"
model RationalQuantities() {
  parameter amplitude: m ^ (-1 / 2) = 1;
  parameter width: m = 1;
  parameter area: m ^ 2 = 4;
  parameter spectral_amplitude: Hz ^ (-1 / 2) = 1;
  variable probability: 1;
  variable length: m;
  variable time_root: s ^ (1 / 2);
  relation dimensions {
    probability = amplitude * amplitude * width;
    length = math.sqrt(area);
    time_root = spectral_amplitude;
  }
}
"#;
    // For a real constant amplitude on an interval, |psi|² dx has
    // dimension L^(-1/2 - 1/2 + 1) = 1. Hz^(-1/2) has T^(1/2).
    let model = ModelDocument::compile("rational-quantities.eqi", source).unwrap();
    let bytes = model.canonical_json().unwrap();
    let replay = ModelDocument::replay(&bytes).unwrap();
    assert_eq!(replay.canonical_json().unwrap(), bytes);
    assert!(model.structurally_equivalent(&replay).unwrap());

    for (original, incompatible) in [
        ("amplitude: m ^ (-1 / 2)", "amplitude: m ^ -1"),
        ("length: m;", "length: m ^ 2;"),
        (
            "spectral_amplitude: Hz ^ (-1 / 2)",
            "spectral_amplitude: Hz ^ (1 / 2)",
        ),
    ] {
        assert!(
            ModelDocument::compile("incompatible.eqi", &source.replace(original, incompatible))
                .is_err(),
            "incompatible dimension must reject: {incompatible}",
        );
    }
}

#[test]
fn current_generation_is_independent_of_coordinate_vocabulary() {
    let fixed = ModelDocument::compile(
        "fixed.eqi",
        "model m() { parameter length: m = 1; domain body = box(0, 1); relation r on body { coordinate(0) - coordinate(0) = 0; } }",
    )
    .unwrap();
    let referenced = ModelDocument::compile(
        "referenced.eqi",
        "model m() { parameter length: m = 1; domain body = box(0, length); relation r on body { coordinate(0) - coordinate(0) = 0; } }",
    )
    .unwrap();
    for model in [&fixed, &referenced] {
        assert_eq!(
            model.structural_fingerprint().unwrap().generation(),
            SemanticFingerprintGeneration::V21
        );
    }
    // Equal endpoint values do not erase the nominal Parameter dependency.
    assert!(!fixed.structurally_equivalent(&referenced).unwrap());
}

#[test]
fn source_native_codec_and_allocation_routes_share_only_structural_identity() {
    let source = ModelDocument::compile("decay.eqi", DECAY).unwrap();
    let independently_compiled = ModelDocument::compile(
        "renamed.eqi",
        "model renamed() { parameter r: 1/s=1; state state: 1; initial { state = 1; } relation balance { derivative(state)+r*state=0; } }",
    )
    .unwrap();
    let native = ModelDocument::compile_module(&native_decay(false), None, &[]).unwrap();
    let reordered_native = ModelDocument::compile_module(&native_decay(true), None, &[]).unwrap();

    for equivalent in [&independently_compiled, &native, &reordered_native] {
        assert!(source.structurally_equivalent(equivalent).unwrap());
        assert_eq!(
            source.structural_fingerprint().unwrap(),
            equivalent.structural_fingerprint().unwrap()
        );
        assert_ne!(
            source.artifact_reference().unwrap(),
            equivalent.artifact_reference().unwrap()
        );
    }
    let fingerprint = source.structural_fingerprint().unwrap();
    assert_eq!(fingerprint.generation(), SemanticFingerprintGeneration::V21);
    assert_eq!(fingerprint.digest().len(), 64);

    let replay = eqiora::api::ModelDocument::replay(&source.canonical_json().unwrap()).unwrap();
    assert_eq!(
        replay.artifact_reference().unwrap(),
        source.artifact_reference().unwrap()
    );
    assert!(source.structurally_equivalent(&replay).unwrap());

    let child = source
        .commit_value_edit(
            source
                .preview_value_edit(
                    source.aliases()["rate"],
                    eqiora_core::ValueLiteral::from_real(
                        source
                            .program()
                            .typed_value(source.aliases()["rate"])
                            .unwrap()
                            .value_type()
                            .clone(),
                        2.0,
                    )
                    .unwrap(),
                )
                .unwrap(),
        )
        .unwrap();
    assert!(!source.structurally_equivalent(child.document()).unwrap());
}

#[test]
fn nominal_identity_graph_wiring_values_and_operators_remain_meaning() {
    let source = ModelDocument::compile("resistor.eqi", PHYSICAL).unwrap();
    let native = ModelDocument::compile_module(&native_resistor(false), None, &[]).unwrap();
    let reordered_native =
        ModelDocument::compile_module(&native_resistor(true), None, &[]).unwrap();
    assert!(source.structurally_equivalent(&native).unwrap());
    assert!(source.structurally_equivalent(&reordered_native).unwrap());
    assert_ne!(source.digest().unwrap(), native.digest().unwrap());

    let changed_source = PHYSICAL.replace("= 2;", "= 3;");
    assert_ne!(
        changed_source, PHYSICAL,
        "the value probe must change its input"
    );
    let changed_value = ModelDocument::compile("value.eqi", &changed_source).unwrap();
    let operator_source = PHYSICAL.replace(
        "positive.voltage - negative.voltage",
        "positive.voltage + negative.voltage",
    );
    assert_ne!(
        operator_source, PHYSICAL,
        "the operator probe must change its input"
    );
    let changed_operator = ModelDocument::compile("operator.eqi", &operator_source).unwrap();
    assert!(!source.structurally_equivalent(&changed_value).unwrap());
    assert!(!source.structurally_equivalent(&changed_operator).unwrap());

    let distinct_domains = physical_domain_aliasing(false);
    let shared_domain = physical_domain_aliasing(true);
    assert!(
        !distinct_domains
            .structurally_equivalent(&shared_domain)
            .unwrap()
    );
}

#[test]
fn mathematical_signed_zero_is_normalized_without_weakening_other_values() {
    let positive = ModelDocument::compile(
        "positive.eqi",
        "model zero() { parameter p: 1 = 0.0; relation r { p = 0; } }",
    )
    .unwrap();
    let negative = ModelDocument::compile(
        "negative.eqi",
        "model zero() { parameter p: 1 = -0.0; relation r { p = 0; } }",
    )
    .unwrap();
    let nonzero = ModelDocument::compile(
        "nonzero.eqi",
        "model zero() { parameter p: 1 = 0.0000000000000001; relation r { p = 0; } }",
    )
    .unwrap();
    assert!(positive.structurally_equivalent(&negative).unwrap());
    assert!(!positive.structurally_equivalent(&nonzero).unwrap());
}

#[test]
fn semantic_types_support_and_model_time_are_fingerprint_meaning() {
    let scalar = ModelDocument::compile(
        "scalar.eqi",
        "model m() { variable value: 1; relation r { value = 0; } }",
    )
    .unwrap();
    let dimensioned = ModelDocument::compile(
        "dimensioned.eqi",
        "model m() { variable value: m; relation r { value = 0; } }",
    )
    .unwrap();
    assert!(!scalar.structurally_equivalent(&dimensioned).unwrap());

    let scalar_spatial = ModelDocument::compile(
        "scalar-spatial.eqi",
        "model m() { domain body = box(0, 1, 0, 1); variable value: m on body; relation r on body { value = 0; } }",
    )
    .unwrap();
    let vector_spatial = ModelDocument::compile(
        "vector-spatial.eqi",
        "model m() { domain body = box(0, 1, 0, 1); variable value: vector<m, 2> on body; relation r on body { value = 0; } }",
    )
    .unwrap();
    assert!(
        !scalar_spatial
            .structurally_equivalent(&vector_spatial)
            .unwrap()
    );

    let support_a = ModelDocument::compile(
        "support-a.eqi",
        "model m() { domain a = box(0, 1); domain b = box(0, 2); variable value: 1 on a; relation r on a { value = 0; } }",
    )
    .unwrap();
    let support_b = ModelDocument::compile(
        "support-b.eqi",
        "model m() { domain a = box(0, 1); domain b = box(0, 2); variable value: 1 on b; relation r on b { value = 0; } }",
    )
    .unwrap();
    assert!(!support_a.structurally_equivalent(&support_b).unwrap());

    let slow_clock = ModelDocument::compile(
        "slow.eqi",
        "model m() { state x: 1 at tick; initial { x = 0; } clock tick = periodic(1[s] / 10, phase = 0[s] / 1); relation update at tick { next(x) - x = 0; } }",
    )
    .unwrap();
    let fast_clock = ModelDocument::compile(
        "fast.eqi",
        "model m() { state x: 1 at tick; initial { x = 0; } clock tick = periodic(1[s] / 20, phase = 0[s] / 1); relation update at tick { next(x) - x = 0; } }",
    )
    .unwrap();
    assert!(!slow_clock.structurally_equivalent(&fast_clock).unwrap());
}

#[test]
fn expression_allocation_and_model_boundary_have_exact_public_projection_semantics() {
    let first = manually_allocated_program(false, false);
    let reordered_expression = manually_allocated_program(true, false);
    let exposed_port = manually_allocated_program(false, true);

    assert_eq!(
        StructuralSemanticFingerprint::from_program(&first).unwrap(),
        StructuralSemanticFingerprint::from_program(&reordered_expression).unwrap()
    );
    assert_ne!(
        StructuralSemanticFingerprint::from_program(&first).unwrap(),
        StructuralSemanticFingerprint::from_program(&exposed_port).unwrap()
    );
}

#[test]
fn pathological_default_projection_fails_without_a_partial_identity() {
    let mut source = String::from("model symmetric() {");
    for index in 0..258 {
        source.push_str(&format!(" parameter p{index}: 1 = 1;"));
    }
    source.push_str(" relation balance { 0 = 0; } }");
    let model = ModelDocument::compile("symmetric.eqi", &source).unwrap();

    let error = model
        .structural_fingerprint()
        .expect_err("fixed public limits must reject pathological exact labelling");
    assert_eq!(error.code().0, "EQ0901");
    assert!(error.message().contains("individualization-depth limit"));
}

fn native_decay(reversed: bool) -> Module {
    let field = DraftField::new(
        "state",
        eqiora_core::ValueType::scalar(
            eqiora_core::ScalarDomain::Real,
            DimExponents::DIMENSIONLESS,
        )
        .expect("valid scalar type"),
        eqiora::language::FieldRoleSyntax::State,
    );
    let rate = DraftParameter::new(
        "coefficient",
        eqiora_core::ValueLiteral::from_real(
            eqiora_core::ValueType::scalar(
                eqiora_core::ScalarDomain::Real,
                DimExponents::from_integers([0, 0, -1, 0, 0, 0, 0]).expect("bounded dimension"),
            )
            .expect("valid scalar type"),
            1.0,
        )
        .unwrap(),
    );
    let relation = DraftRelation::continuous(
        "balance",
        [(
            DraftExpression::derivative(&field) + rate.expression() * field.expression(),
            DraftExpression::constant(
                eqiora::language::DecimalLiteral::from_f64(0.0).expect("finite zero"),
            ),
        )],
    );
    let initial = eqiora::language::DraftDeclaration::Initial(vec![(
        field.expression(),
        DraftExpression::constant(
            eqiora::language::DecimalLiteral::from_f64(1.0).expect("finite fixture literal"),
        ),
    )]);
    let declarations = if reversed {
        vec![relation.into(), rate.into(), field.into(), initial]
    } else {
        vec![field.into(), rate.into(), relation.into(), initial]
    };
    Module::new("native_decay", declarations).unwrap()
}

fn native_resistor(reversed: bool) -> Module {
    let electrical = DraftPhysicalDomain::new(
        "pin",
        "voltage",
        eqiora_core::ValueType::scalar(
            eqiora_core::ScalarDomain::Real,
            DimExponents::from_integers([1, 2, -3, -1, 0, 0, 0]).expect("bounded dimension"),
        )
        .expect("valid scalar type"),
        "current",
        eqiora_core::ValueType::scalar(
            eqiora_core::ScalarDomain::Real,
            DimExponents::from_integers([0, 0, 0, 1, 0, 0, 0]).expect("bounded dimension"),
        )
        .expect("valid scalar type"),
    );
    let positive = DraftConservingPort::new("p", &electrical);
    let negative = DraftConservingPort::new("n", &electrical);
    let tap = DraftConservingPort::new("t", &electrical);
    let resistance = DraftParameter::new(
        "r",
        eqiora_core::ValueLiteral::from_real(
            eqiora_core::ValueType::scalar(
                eqiora_core::ScalarDomain::Real,
                DimExponents::from_integers([1, 2, -3, -2, 0, 0, 0]).expect("bounded dimension"),
            )
            .expect("valid scalar type"),
            2.0,
        )
        .unwrap(),
    );
    let law = DraftRelation::continuous(
        "law",
        [
            (
                DraftExpression::across(&positive)
                    - DraftExpression::across(&negative)
                    - resistance.expression() * DraftExpression::through(&positive),
                DraftExpression::constant(
                    eqiora::language::DecimalLiteral::from_f64(0.0).expect("finite zero"),
                ),
            ),
            (
                DraftExpression::through(&positive)
                    + DraftExpression::through(&negative)
                    + DraftExpression::through(&tap),
                DraftExpression::constant(
                    eqiora::language::DecimalLiteral::from_f64(0.0).expect("finite zero"),
                ),
            ),
        ],
    );
    let connection = DraftConservingConnection::new([&positive, &negative, &tap]);
    let declarations = if reversed {
        vec![
            connection.into(),
            law.into(),
            tap.into(),
            resistance.into(),
            negative.into(),
            positive.into(),
            electrical.into(),
        ]
    } else {
        vec![
            electrical.into(),
            positive.into(),
            negative.into(),
            tap.into(),
            resistance.into(),
            law.into(),
            connection.into(),
        ]
    };
    Module::new("native_resistor", declarations).unwrap()
}

fn physical_domain_aliasing(shared: bool) -> ModelDocument {
    let second_support = if shared { "first" } else { "second" };
    let source = format!(
        r#"
model network() {{
  domain first = scalar_physical(across potential: 1, through flow: 1);
  domain second = scalar_physical(across potential: 1, through flow: 1);
  port a1: first;
  port a2: first;
  port b1: {second_support};
  port b2: {second_support};
  relation a {{ a1.potential-a2.potential=0; a1.flow+a2.flow=0; }}
  relation b {{ b1.potential-b2.potential=0; b1.flow+b2.flow=0; }}
  connect a1, a2;
  connect b1, b2;
}}
"#
    );
    ModelDocument::compile("aliasing.eqi", &source).unwrap()
}

fn manually_allocated_program(reverse_expression: bool, expose_port: bool) -> KernelProgram {
    let left = Id::<kinds::Field>::new();
    let right = Id::<kinds::Field>::new();
    let relation = Id::<kinds::Relation>::new();
    let activation = Id::<kinds::Activation>::new();
    let port = Id::<kinds::Port>::new();
    let model = OntologyId::<Model>::new();
    let mut expression = ExprDagBuilder::new();
    let (left_value, right_value) = if reverse_expression {
        let right_value = expression.symbol(SymbolRef::Field(right)).unwrap();
        let left_value = expression.symbol(SymbolRef::Field(left)).unwrap();
        (left_value, right_value)
    } else {
        let left_value = expression.symbol(SymbolRef::Field(left)).unwrap();
        let right_value = expression.symbol(SymbolRef::Field(right)).unwrap();
        (left_value, right_value)
    };
    let root = expression.add(left_value, right_value).unwrap();
    let expression = {
        let equation_zero_0 = expression
            .constant(
                eqiora_core::ValueLiteral::from_real(
                    eqiora_core::ValueType::scalar(
                        eqiora_core::ScalarDomain::Real,
                        DimExponents::DIMENSIONLESS,
                    )
                    .expect("valid scalar type"),
                    0.0,
                )
                .unwrap(),
            )
            .unwrap();
        expression.finish([root, equation_zero_0])
    }
    .unwrap();
    let members = [
        left.erase(),
        right.erase(),
        relation.erase(),
        activation.erase(),
        port.erase(),
    ];
    let boundary = expose_port.then_some(port.erase());
    let view = ModelView::new(model, members, boundary).unwrap();
    let mut transaction = Transaction::new("manual expression allocation");
    transaction
        .push(Op::DefineKernelNode {
            node: FieldDef::new(
                left,
                eqiora_core::ValueType::scalar(
                    eqiora_core::ScalarDomain::Real,
                    DimExponents::DIMENSIONLESS,
                )
                .expect("valid scalar type"),
                eqiora::kernel::FieldRole::Variable,
            )
            .into(),
        })
        .push(Op::DefineKernelNode {
            node: FieldDef::new(
                right,
                eqiora_core::ValueType::scalar(
                    eqiora_core::ScalarDomain::Real,
                    DimExponents::DIMENSIONLESS,
                )
                .expect("valid scalar type"),
                eqiora::kernel::FieldRole::Variable,
            )
            .into(),
        })
        .push(Op::DefineKernelNode {
            node: RelationDef::new(relation, expression).unwrap().into(),
        })
        .push(Op::DefineKernelNode {
            node: ActivationDef::continuous(activation).into(),
        })
        .push(Op::DefineKernelNode {
            node: PortDef::signal(
                port,
                SignalDirection::Output,
                eqiora_core::ValueType::scalar(
                    eqiora_core::ScalarDomain::Real,
                    DimExponents::DIMENSIONLESS,
                )
                .expect("valid scalar type"),
            )
            .into(),
        })
        .push(Op::Connect {
            from: relation.erase(),
            to: left.erase(),
            edge: EdgeKind::DependsOn,
        })
        .push(Op::Connect {
            from: relation.erase(),
            to: right.erase(),
            edge: EdgeKind::DependsOn,
        })
        .push(Op::Connect {
            from: activation.erase(),
            to: relation.erase(),
            edge: EdgeKind::Activates,
        })
        .push(Op::DefineOntologyView { view: view.into() });
    let mut store = InMemoryGraphStore::new();
    store.commit(transaction).unwrap();
    KernelProgram::from_snapshot(&store.snapshot(), model).unwrap()
}
