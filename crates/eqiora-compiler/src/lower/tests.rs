mod extrema;
mod runtime_integer;
use super::*;
use crate::compile;
use eqiora_lang::parse;

#[test]
fn source_signal_types_round_trip_and_reject_narrowing_or_shape_changes() {
    for (output, input, accepted) in [
        ("m", "complex<m>", true),
        ("array<m, 3>", "array<complex<m>, 3>", true),
        ("complex<m>", "m", false),
        ("array<complex<m>, 3>", "array<complex<m>, 2>", false),
        ("array<m, 3>", "m", false),
        ("m", "complex<s>", false),
        ("array<m, 3>", "vector<m, 3>", false),
    ] {
        for source in [
            format!(
                "model M() {{
                port out: signal output {output};
                port sink: signal input {input};
                connect out -> sink;
                relation r {{ out - out = 0; sink - sink = 0; }}
            }}"
            ),
            format!(
                "dimension Length = m;
                component Sender(output out: {output}) {{
                    relation r {{ out - out = 0; }} }}
                component Receiver(input sink: {input}) {{
                    relation r {{ sink - sink = 0; }} }}
                model M() {{ instance a: Sender(); instance b: Receiver();
                    connect a.out -> b.sink; }}"
            )
            .replace("<m", "<Length"),
        ] {
            let result = compile("signal-types.eqi", &source);
            if accepted {
                let compiled = result.unwrap();
                assert!(compiled[0].transaction().ops().iter().any(|op| matches!(op,
                    Op::DefineKernelNode { node: KernelNode::Port(port) }
                    if port.signal_contract().is_some_and(|(_, ty)| ty.scalar_domain() == eqiora_core::ScalarDomain::Complex)
                )));
                let document = parse("signal-types.eqi", &source).into_document().unwrap();
                let canonical = eqiora_lang::format(&document);
                compile("canonical-signal.eqi", &canonical).unwrap();
                assert_eq!(
                    eqiora_lang::format(
                        &parse("canonical-signal.eqi", &canonical)
                            .into_document()
                            .unwrap()
                    ),
                    canonical
                );
            } else {
                let errors = result.unwrap_err();
                assert!(
                    errors
                        .iter()
                        .any(|error| error.code() == codes::LANGUAGE_TYPE_ERROR
                            && error.source_span().is_some()),
                    "{errors:?}"
                );
            }
        }
    }
}

#[test]
fn annotated_let_aliases_preserve_complete_types() {
    for source in [
        "model M() { let x: complex<m> = 2[m]; relation r { x - x = 0; } }",
        "model M() { parameter p: array<m, 3> = 0; let x: array<complex<m>, 3> = p; relation r { x - x = 0; } }",
        "dimension Length = m; model M() { let x: array<complex<Length>, 3> = 0; relation r { x - x = 0; } }",
    ] {
        compile("typed-let.eqi", source).unwrap();
    }
    for source in [
        "model M() { parameter p: complex<m> = 0; let x: m = p; relation r { x - x = 0; } }",
        "model M() { parameter p: array<m, 3> = 0; let x: array<m, 2> = p; relation r { x - x = 0; } }",
    ] {
        let errors = compile("typed-let.eqi", source).unwrap_err();
        assert!(
            errors
                .iter()
                .any(|error| error.code() == codes::LANGUAGE_TYPE_ERROR
                    && error.source_span().is_some()),
            "{errors:?}"
        );
    }
}

#[test]
fn typed_literal_lowering_preserves_type_through_detachment_and_zero_negation() {
    use eqiora_core::{ScalarDomain, ValueLiteral, ValueType};
    let scalar = ValueType::scalar(ScalarDomain::Complex, DimExponents::DIMENSIONLESS)
        .expect("admitted numeric scalar type");
    for value_type in [scalar.clone(), scalar.array(3).unwrap()] {
        let literal = LoweringExpression::literal(
            ValueLiteral::from_real(value_type.clone(), -0.0).unwrap(),
            TextRange::new(0, 1),
        );
        let literal = LoweringExpression::neg(literal.detached_clone(), TextRange::new(0, 1));
        assert_eq!(expression::lowering_integer_literal(&literal), None);
        let model = LoweringModel {
            structural_dependencies: Default::default(),
            name: "M".into(),
            range: TextRange::new(0, 1),
            items: vec![LoweringItem::Relation {
                name: "r".into(),
                activation: ActivationSyntax::Continuous,
                domain: None,
                initial: false,
                range: TextRange::new(0, 1),
                body: vec![LoweringEquation {
                    kind: eqiora_schema::kernel::RelationConditionKind::Equality,
                    left: literal,
                    right: LoweringExpression::number(
                        eqiora_lang::DecimalLiteral::parse("0").unwrap(),
                        TextRange::new(0, 1),
                    ),
                    contextual_left_zero: false,
                    contextual_right_zero: true,
                    range: TextRange::new(0, 1),
                }]
                .into(),
            }],
        };
        let compiled =
            lower_typed_model("literal.eqi", &model, &mut FreshLoweringIdentities).unwrap();
        let constant = compiled
            .transaction()
            .ops()
            .iter()
            .find_map(|op| match op {
                Op::DefineKernelNode {
                    node: KernelNode::Relation(relation),
                } => relation
                    .expression()
                    .nodes()
                    .iter()
                    .find_map(|node| match node {
                        eqiora_schema::kernel::ExprNode::Constant(value) => Some(value),
                        _ => None,
                    }),
                _ => None,
            })
            .unwrap();
        assert_eq!(constant.value_type(), &value_type);
        assert_eq!(
            constant.component(0).unwrap().0.to_bits(),
            0.0_f64.to_bits()
        );
    }
}

#[test]
fn signed_parameter_quantities_preserve_the_declared_dimension() {
    compile(
        "signed.eqi",
        "component C(parameter length: m = -2[m]) {

        relation r { length + 1[m] = 0; }
    } model M() { instance c: C(); }",
    )
    .unwrap();
}

#[test]
fn component_parameters_preserve_complex_and_array_literals() {
    for syntax in [
        "complex<m>",
        "array<complex<m>, 3>",
        "array<array<m, 2>, 3>",
    ] {
        let source = format!(
            "component C(parameter x: {syntax} = 0) {{
            relation r {{ x - x = 0; }}
        }} model M() {{ instance c: C(); }}"
        );
        let compiled = compile("component-types.eqi", &source).unwrap();
        assert!(compiled[0].transaction().ops().iter().any(|op| match op {
            Op::DefineKernelNode {
                node: KernelNode::Relation(relation),
            } => relation.expression().nodes().iter().any(|node| {
                match node {
                    eqiora_schema::kernel::ExprNode::Constant(value) => {
                        eqiora_lang::ValueTypeSyntax::from_checked(value.value_type(), |_| None)
                            .unwrap()
                            .to_source()
                            == syntax
                    }
                    _ => false,
                }
            }),
            _ => false,
        }));
    }
}

#[test]
fn component_real_to_complex_binding_keeps_the_real_parameter_identity() {
    let compiled = compile(
        "embedding.eqi",
        "component C(parameter z: complex<m>) {

        relation r { z - z = 0; }
    } model M() { parameter p: m = 2[m]; instance c: C(z = p); }",
    )
    .unwrap();
    let nodes = compiled[0]
        .transaction()
        .ops()
        .iter()
        .filter_map(|op| match op {
            Op::DefineKernelNode { node } => Some(node),
            _ => None,
        })
        .collect::<Vec<_>>();
    assert_eq!(
        nodes
            .iter()
            .filter(|node| matches!(node, KernelNode::Parameter(_)))
            .count(),
        1
    );
    assert!(nodes.iter().any(|node| match node {
        KernelNode::Parameter(parameter) =>
            parameter.value_type().scalar_domain() == eqiora_core::ScalarDomain::Real,
        _ => false,
    }));
    assert!(nodes.iter().any(|node| match node {
        KernelNode::Relation(relation) =>
            relation.expression().nodes().iter().any(|node| match node {
                eqiora_schema::kernel::ExprNode::Constant(value) =>
                    value.value_type().scalar_domain() == eqiora_core::ScalarDomain::Complex
                        && value.value_type().dimension() == DimExponents::DIMENSIONLESS
                        && value.component(0).unwrap().0 == 1.0,
                _ => false,
            }),
        _ => false,
    }));
}

#[test]
fn component_array_embedding_and_complex_narrowing_follow_declared_types() {
    compile(
        "array-embedding.eqi",
        "component C(parameter x: array<complex<m>, 3>) {

        relation r { x - x = 0; }
    } model M() { parameter p: array<m, 3> = 0; instance c: C(x = p); }",
    )
    .unwrap();
    let errors = compile(
        "narrowing.eqi",
        "component Sink(parameter x: m) {

        relation r { x - x = 0; }
    } component C(parameter x: complex<m>) {

        instance sink: Sink(x = x);
    } model M() { instance c: C(x = 0); }",
    )
    .unwrap_err();
    assert!(
        errors.iter().any(
            |error| error.message().contains("requires a real scalar type")
                && error.source_span().is_some()
        ),
        "{errors:?}"
    );
}

#[test]
fn component_array_binding_rejects_extent_mismatch() {
    let errors = compile(
        "array-binding.eqi",
        "component C(parameter x: array<complex<m>, 2>) {

        relation r { x - x = 0; }
    } model M() { parameter p: array<m, 3> = 0; instance c: C(x = p); }",
    )
    .unwrap_err();
    assert!(
        errors
            .iter()
            .any(|error| error.message().contains("incompatible types")
                && error.source_span().is_some()),
        "{errors:?}"
    );
}

#[test]
fn typed_lowering_keeps_parameter_domains_and_array_roles() {
    use eqiora_core::{ScalarDomain, ValueType};
    let scalar = ValueType::scalar(ScalarDomain::Complex, DimExponents::DIMENSIONLESS)
        .expect("admitted numeric scalar type");
    for value_type in [scalar.clone(), scalar.array(3).unwrap()] {
        let syntax = eqiora_lang::ValueTypeSyntax::from_checked(&value_type, |_| None).unwrap();
        let source = format!(
            "model M() {{ parameter p: {} = 0; relation r {{ p - p = 0; }} }}",
            syntax.to_source(),
        );
        let compiled = compile("typed.eqi", &source).unwrap();
        let parameter = compiled[0]
            .transaction()
            .ops()
            .iter()
            .find_map(|op| match op {
                Op::DefineKernelNode {
                    node: KernelNode::Parameter(parameter),
                } => Some(parameter),
                _ => None,
            })
            .unwrap();
        assert_eq!(parameter.value_type(), &value_type);
        assert_eq!(parameter.value().component(0).unwrap().0, 0.0);
    }
}

#[test]
fn source_parameter_literals_preserve_domains_and_reject_nonzero_shapes() {
    let compiled = compile(
        "typed.eqi",
        "model M() { parameter p: complex<m> = 2[m]; relation r { p - p = 0; } }",
    )
    .unwrap();
    let parameter = compiled[0]
        .transaction()
        .ops()
        .iter()
        .find_map(|op| match op {
            Op::DefineKernelNode {
                node: KernelNode::Parameter(parameter),
            } => Some(parameter),
            _ => None,
        })
        .unwrap();
    assert_eq!(
        parameter.value_type().scalar_domain(),
        eqiora_core::ScalarDomain::Complex
    );
    assert_eq!(
        parameter.value_type().dimension(),
        crate::dimensions::length_dimension()
    );
    assert_eq!(parameter.value().component(0).unwrap().0, 2.0);
    for syntax in ["array<m, 3>", "array<complex<m>, 3>"] {
        let errors = compile(
            "typed.eqi",
            &format!("model M() {{ parameter p: {syntax} = 2[m]; relation r {{ p - p = 0; }} }}"),
        )
        .unwrap_err();
        assert!(
            errors.iter().any(|error| {
                error.message().contains("incompatible types") && error.source_span().is_some()
            }),
            "{errors:?}"
        );
    }
}

#[test]
fn source_parameter_aliases_do_not_erase_complex_or_array_types() {
    for syntax in ["complex<m>", "array<m, 3>"] {
        let errors = compile(
            "typed.eqi",
            &format!(
                "component C(parameter x: m) {{  }} model M() {{
                parameter p: {syntax} = 0;
                let alias = p;
                instance c: C(x = alias);
            }}"
            ),
        )
        .unwrap_err();
        assert!(
            errors.iter().any(
                |error| error.message().contains("requires a real scalar type")
                    && error.source_span().is_some()
            ),
            "{errors:?}"
        );
    }
}

#[test]
fn typed_cartesian_coordinates_require_real_scalar_lengths() {
    use eqiora_core::{ScalarDomain, ValueType};
    let length = DimExponents::from_integers([0, 1, 0, 0, 0, 0, 0]).unwrap();
    for value_type in [
        ValueType::scalar(ScalarDomain::Complex, length).expect("admitted numeric scalar type"),
        ValueType::scalar(ScalarDomain::Real, length)
            .expect("admitted numeric scalar type")
            .array(1)
            .unwrap(),
    ] {
        let syntax = eqiora_lang::ValueTypeSyntax::from_checked(&value_type, |_| None).unwrap();
        let source = format!(
            "model M() {{ parameter extent: {} = 0; domain body = box(0, extent); relation r {{ extent - extent = 0; }} }}",
            syntax.to_source(),
        );
        let errors = compile("coordinate.eqi", &source).unwrap_err();
        assert!(
            errors
                .iter()
                .any(|error| error.message().contains("not a real scalar length")),
            "{errors:?}"
        );
    }
}

#[test]
fn entity_symbols_are_ordered_and_exact() {
    let alpha = Id::<kinds::Field>::new().erase();
    let zeta = Id::<kinds::Parameter>::new().erase();
    let symbols = ModelSymbols::from_map(BTreeMap::from([
        ("zeta".to_owned(), zeta),
        ("alpha".to_owned(), alpha),
    ]));

    assert_eq!(symbols.get("alpha"), Some(alpha));
    assert_eq!(symbols.get("zeta"), Some(zeta));
    assert_eq!(symbols.get("missing"), None);
    assert_eq!(
        symbols.iter().collect::<Vec<_>>(),
        vec![("alpha", alpha), ("zeta", zeta)]
    );
}

#[test]
fn executable_compile_still_requires_a_model_entry() {
    let source = "public component Resistor() {} public connector Pin {\n  across voltage: 1;\n  through current: A;\n}";
    let diagnostics = compile("library.eqi", source)
        .expect_err("a declarations-only library is not an executable model");

    assert_eq!(diagnostics.len(), 1);
    assert_eq!(diagnostics[0].code(), codes::SYNTAX_ERROR);
    assert_eq!(
        diagnostics[0].message(),
        "source requires at least one `model` declaration"
    );
    let span = diagnostics[0]
        .source_span()
        .expect("missing Model remains a source diagnostic");
    assert_eq!(span.file, "library.eqi");
    assert_eq!(span.start, u32::try_from(source.len()).unwrap());
    assert_eq!(span.end, span.start);
}

struct AssignedTestIdentities {
    model: OntologyId<Model>,
    domain: Id<kinds::Domain>,
    ports: BTreeMap<String, Id<kinds::Port>>,
    relation: Id<kinds::Relation>,
    activation: Id<kinds::Activation>,
    connection: Id<kinds::Connection>,
}

impl LoweringIdentities for AssignedTestIdentities {
    fn model(&mut self, _name: &str) -> OntologyId<Model> {
        self.model
    }

    fn domain(&mut self, _name: &str) -> Id<kinds::Domain> {
        self.domain
    }

    fn representation(&mut self, _name: &str) -> Id<kinds::Representation> {
        panic!("fixture has no Representation")
    }

    fn field(&mut self, _name: &str) -> Id<kinds::Field> {
        panic!("fixture has no Field")
    }

    fn parameter(&mut self, _name: &str) -> Id<kinds::Parameter> {
        panic!("fixture has no Parameter")
    }

    fn port(&mut self, name: &str) -> Id<kinds::Port> {
        self.ports[name]
    }

    fn clock(&mut self, _name: &str) -> Id<kinds::ClockDomain> {
        panic!("fixture has no ClockDomain")
    }

    fn observable(&mut self, _name: &str) -> Id<kinds::Observable> {
        Id::new()
    }

    fn activation(&mut self, _name: &str) -> Id<kinds::Activation> {
        panic!("fixture has no Event")
    }

    fn relation(&mut self, _name: &str) -> (Id<kinds::Relation>, Id<kinds::Activation>) {
        (self.relation, self.activation)
    }

    fn connection(&mut self) -> Id<kinds::Connection> {
        self.connection
    }
}

fn voltage_dimension() -> DimExponents {
    DimExponents::from_integers([1, 2, -3, -1, 0, 0, 0]).expect("bounded dimension")
}

fn current_dimension() -> DimExponents {
    DimExponents::from_integers([0, 0, 0, 1, 0, 0, 0]).expect("bounded dimension")
}

fn resistance_dimension() -> DimExponents {
    DimExponents::from_integers([1, 2, -3, -2, 0, 0, 0]).expect("bounded dimension")
}

#[test]
fn staged_identity_source_controls_every_lowered_identity() {
    let source = r#"
model assigned() {
  domain electrical = scalar_physical(across potential: 1, through flow: 1);
  port positive: electrical;
  port negative: electrical;
  relation equal { positive.potential - negative.potential = 0; }
  connect positive, negative;
}
"#;
    let document = parse("assigned.eqi", source)
        .into_document()
        .expect("fixture parses");
    let positive = Id::new();
    let negative = Id::new();
    let mut identities = AssignedTestIdentities {
        model: OntologyId::new(),
        domain: Id::new(),
        ports: BTreeMap::from([
            ("negative".to_owned(), negative),
            ("positive".to_owned(), positive),
        ]),
        relation: Id::new(),
        activation: Id::new(),
        connection: Id::new(),
    };

    let declaration = &document.models()[0];
    let domain = declaration
        .items()
        .iter()
        .find_map(|item| match item {
            eqiora_lang::Item::Domain(value) => Some(value),
            _ => None,
        })
        .unwrap();
    let relation = declaration
        .items()
        .iter()
        .find_map(|item| match item {
            eqiora_lang::Item::Relation(value) => Some(value),
            _ => None,
        })
        .unwrap();
    let mut items = vec![LoweringItem::Domain {
        name: "electrical".into(),
        contract: LoweringDomainContract::Source(domain.syntax().clone()),
        range: domain.range(),
    }];
    for name in ["positive", "negative"] {
        items.push(LoweringItem::Port {
            name: name.into(),
            contract: LoweringPortContract::Source(PortSyntax::ScalarPhysical {
                domain: "electrical".into(),
            }),
            range: declaration.range(),
        });
    }
    items.push(LoweringItem::Relation {
        name: "equal".into(),
        activation: ActivationSyntax::Continuous,
        domain: None,
        // This unit exercises staged Kernel IDs, below lexical source lookup.
        // The frontend already resolves the two declared members to these roles.
        body: vec![LoweringEquation::rewritten(
            &relation.conditions().unwrap()[0],
            LoweringExpression::binary(
                eqiora_lang::BinaryOp::Sub,
                LoweringExpression::call(
                    "across".to_owned(),
                    LoweringExpression::name("positive".to_owned(), relation.range()),
                    relation.range(),
                ),
                LoweringExpression::call(
                    "across".to_owned(),
                    LoweringExpression::name("negative".to_owned(), relation.range()),
                    relation.range(),
                ),
                relation.range(),
            ),
            LoweringExpression::number(
                eqiora_lang::DecimalLiteral::parse("0").unwrap(),
                relation.range(),
            ),
        )]
        .into(),
        initial: false,
        range: relation.range(),
    });
    items.push(LoweringItem::Connection {
        syntax: ConnectionSyntax::Conserving,
        ports: vec!["positive".into(), "negative".into()],
        range: declaration.range(),
    });
    let model = LoweringModel {
        structural_dependencies: Default::default(),
        name: "assigned".into(),
        range: declaration.range(),
        items,
    };
    let compiled = lower_typed_model("assigned.eqi", &model, &mut identities)
        .expect("assigned identities lower");

    assert_eq!(compiled.model(), identities.model);
    assert_eq!(
        compiled.symbols().get("electrical"),
        Some(identities.domain.erase())
    );
    assert_eq!(compiled.symbols().get("positive"), Some(positive.erase()));
    assert_eq!(compiled.symbols().get("negative"), Some(negative.erase()));
    assert_eq!(
        compiled.symbols().get("equal"),
        Some(identities.relation.erase())
    );
    assert!(compiled.transaction().ops().iter().any(|operation| {
        matches!(
            operation,
            Op::DefineKernelNode {
                node: KernelNode::Activation(definition),
            } if definition.id() == identities.activation
        )
    }));
    assert!(compiled.transaction().ops().iter().any(|operation| {
        matches!(
            operation,
            Op::DefineKernelNode {
                node: KernelNode::Connection(definition),
            } if definition.id() == identities.connection
        )
    }));
}

#[test]
fn compiler_rejects_dimensionally_invalid_residual_at_source_span() {
    let source = r#"
model invalid() {
  variable temperature: K; initial { temperature = 293[K]; }
  parameter tau: s = 10[s];
  relation bad {
    temperature + tau = 0;
  }
}
"#;
    let diagnostics = compile("invalid.eqi", source).expect_err("K + s is invalid");

    assert!(diagnostics.iter().any(|diagnostic| {
        diagnostic.code() == codes::LANGUAGE_TYPE_ERROR
            && diagnostic.source_span().is_some_and(|span| {
                source[span.start as usize..span.end as usize].contains("temperature + tau")
            })
    }));
}

#[test]
fn compiler_rejects_unresolved_named_activation() {
    let source =
        "model m() { state x: 1; initial { x = 0; } relation r at missing { next(x) = 0; } }";
    let diagnostics = compile("missing.eqi", source).expect_err("named activation is unresolved");

    assert!(
        diagnostics
            .iter()
            .any(|diagnostic| diagnostic.message().contains("ClockDomain or Event"))
    );
}

#[test]
fn compiler_rejects_discrete_symbols_in_continuous_relations() {
    let source = "model m() { state x: 1; initial { x = 0; } relation r { next(x) = 0; } }";
    let diagnostics =
        compile("activation.eqi", source).expect_err("Next requires an admitted reset activation");

    assert!(
        diagnostics.iter().any(|diagnostic| {
            diagnostic
                .message()
                .contains("state evolution requires its exact clock")
        }),
        "{diagnostics:?}"
    );
}

#[test]
fn compiler_rejects_spatial_boundary_unit_mismatch() {
    let source = r#"
model bar() {
  domain body = box(0, 1);
  domain loaded = boundary(body, axis = 0, side = upper);
  variable u: m on body; initial { u = 0; }
  parameter stiffness: kg * m / s ^ 2 = 10[kg * m / s ^ 2];
  parameter wrong_load: m = 1[m];
  relation load on loaded {
    normal(stiffness * grad(u)) - wrong_load = 0;
  }
}
"#;
    let diagnostics = compile("bar.eqi", source).expect_err("force and length conflict");

    assert!(diagnostics.iter().any(|diagnostic| {
        diagnostic.code() == codes::LANGUAGE_TYPE_ERROR
            && diagnostic.message().contains("addition/subtraction")
    }));
}

#[test]
fn compiler_requires_dimensionless_trigonometric_arguments() {
    let source = r#"
model invalid() {
  domain interval = box(0, 1);
  variable u: 1 on interval; initial { u = 0; }
  relation balance on interval {
    -div(grad(u)) - math.sin(coordinate(0)) = 0;
  }
}

"#;
    let diagnostics = compile("invalid-sin.eqi", source)
        .expect_err("a physical coordinate is not an angle by itself");

    assert!(diagnostics.iter().any(|diagnostic| {
        diagnostic.code() == codes::LANGUAGE_TYPE_ERROR
            && diagnostic.message().contains("dimensionless scalar")
    }));
}

#[test]
fn compiler_owns_the_scalar_mathematics_namespace() {
    let valid = r#"
model valid() {
  domain interval = box(0, 1);
  variable u: 1 on interval; initial { u = 0; }
  relation balance on interval {
    u - math.sin(math.pi) = 0;
  }
}
"#;
    let compiled = compile("math-sin.eqi", valid).expect("the compiler-owned sine compiles");
    let relation = compiled[0]
        .transaction()
        .ops()
        .iter()
        .find_map(|operation| match operation {
            Op::DefineKernelNode {
                node: KernelNode::Relation(relation),
            } if !relation.is_initial() => Some(relation),
            _ => None,
        })
        .expect("canonical Relation");
    assert!(relation.expression().nodes().iter().any(|node| matches!(
        node,
        eqiora_schema::kernel::ExprNode::UnaryMath(UnaryMathFunction::Sin, _)
    )));
    assert!(relation.expression().nodes().iter().any(|node| matches!(
        node,
        eqiora_schema::kernel::ExprNode::Constant(value)
            if value.component(0).unwrap().0.to_bits() == 0x4009_21fb_5444_2d18
    )));

    for (source, expected) in [
        (
            "model invalid() { domain d = box(0, 1); variable u: 1 on d; initial { u = 0; } relation law on d { u - sin(0) = 0; } }",
            "bare `sin` is not language vocabulary",
        ),
        (
            "model invalid() { domain d = box(0, 1); variable u: 1 on d; initial { u = 0; } relation law on d { u - math.cos(0) = 0; } }",
            "unknown compiler-owned scalar mathematics member `math.cos`",
        ),
        (
            "model invalid() { domain d = box(0, 1); variable u: 1 on d; initial { u = 0; } relation law on d { u - math.tau = 0; } }",
            "unknown compiler-owned scalar mathematics member `math.tau`",
        ),
        (
            "model invalid() { parameter math: 1 = 1; }",
            "identifier `math` is reserved for compiler-owned scalar mathematics",
        ),
        (
            "model math() { relation law { 0 = 0; } }",
            "identifier `math` is reserved for compiler-owned scalar mathematics",
        ),
        (
            "dimension math = m; model invalid() { relation law { 0 = 0; } }",
            "identifier `math` is reserved for compiler-owned scalar mathematics",
        ),
    ] {
        let diagnostics = compile("invalid-math.eqi", source).expect_err(expected);
        assert!(
            diagnostics
                .iter()
                .any(|diagnostic| diagnostic.message().contains(expected)),
            "missing diagnostic containing {expected:?}: {diagnostics:#?}"
        );
    }
}

#[test]
fn compiler_lowers_canonical_tensor_structure_without_a_physics_node() {
    let source = r#"
model elastic_relation() {
  domain body = box(0, 1, 0, 1);
  variable displacement: vector<m, 2> on body;
  parameter mu: kg / (m * s ^ 2) = 2[kg / (m * s ^ 2)];
  parameter lambda: kg / (m * s ^ 2) = 3[kg / (m * s ^ 2)];
  relation balance on body {
    -div(
      2 * mu * symmetric_part(grad(displacement))
      + lambda * isotropic_lift(div(displacement))
    ) = 0;
  }
}
"#;
    let compiled = compile("elastic-relation.eqi", source).expect("typed tensor relation");
    let relation = compiled[0]
        .transaction()
        .ops()
        .iter()
        .find_map(|operation| match operation {
            Op::DefineKernelNode {
                node: KernelNode::Relation(relation),
            } if !relation.is_initial() => Some(relation),
            _ => None,
        })
        .expect("canonical Relation");
    assert!(
        relation
            .expression()
            .nodes()
            .iter()
            .any(|node| matches!(node, eqiora_schema::kernel::ExprNode::SymmetricPart(_)))
    );
    assert!(
        relation
            .expression()
            .nodes()
            .iter()
            .any(|node| matches!(node, eqiora_schema::kernel::ExprNode::IsotropicLift(_)))
    );
}

#[test]
fn flat_semantic_typing_distinguishes_scalar_gradients_from_vector_strain() {
    let scalar = r#"
model scalar_poisson() {
  domain body = box(0, 1, 0, 1);
  variable potential: 1 on body;
  relation balance on body { -div(grad(potential)) = 0; }
}
"#;
    compile("scalar-poisson.eqi", scalar).expect("a scalar gradient remains admissible");

    let wrong_displacement =
        scalar.replace("-div(grad(potential))", "symmetric_part(grad(potential))");
    let diagnostics = compile("wrong-strain.eqi", &wrong_displacement)
        .expect_err("symmetric strain requires a spatial-vector Field");
    assert!(diagnostics.iter().any(|diagnostic| {
        diagnostic.code() == codes::LANGUAGE_TYPE_ERROR
            && diagnostic
                .message()
                .contains("symmetric_part requires an exact")
    }));
}

#[test]
fn source_pure_operators_admit_complex_fields_without_real_narrowing() {
    let source =
        "public operator dyadic(input left: spatial[1], input right: spatial[1]): spatial[2]
        = component(left, 0) * component(right, 1);
    model M() {
        domain body = box(0, 1, 0, 1);
        variable left: vector<complex<1>, 2> on body;
        variable right: vector<1, 2> on body;
        relation r on body { div(div(dyadic(left=left, right=right))) = 0; }
    }";
    compile("complex-operator.eqi", source).unwrap();
}

#[test]
fn compiler_lowers_source_declared_pure_operator_as_one_generic_application() {
    let source = r#"
public operator dyadic(input left: spatial[1], input right: spatial[1]): spatial[2]
  = component(left, 0) * component(right, 1);

model generic_operator() {
  domain body = box(0, 1, 0, 1);
  variable left: vector<1, 2> on body;
  variable right: vector<1, 2> on body;
  relation balance on body {
    div(div(dyadic(left=left, right=right))) = 0;
  }
}
"#;
    let compiled = compile("generic-operator.eqi", source).expect("typed pure operator");
    let relation = compiled[0]
        .transaction()
        .ops()
        .iter()
        .find_map(|operation| match operation {
            Op::DefineKernelNode {
                node: KernelNode::Relation(relation),
            } if !relation.is_initial() => Some(relation),
            _ => None,
        })
        .expect("canonical Relation");
    let dag = relation.expression();
    let applications = dag
        .nodes()
        .iter()
        .filter_map(|node| match node {
            eqiora_schema::kernel::ExprNode::PureOperatorApplication(application) => {
                Some(application)
            }
            _ => None,
        })
        .collect::<Vec<_>>();
    assert_eq!(applications.len(), 1);
    assert_eq!(applications[0].arguments().len(), 2);
    assert_eq!(dag.definitions().len(), 1);
    assert_eq!(
        applications[0].definition(),
        eqiora_schema::kernel::pure_operator::PureOperatorDefinition::dyadic_product()
            .expect("standard definition")
            .digest()
    );
    let (transaction, _, _) = compiled
        .into_iter()
        .next()
        .expect("compiled Model")
        .into_parts();
    eqiora_graph::GraphStore::commit(&mut eqiora_graph::InMemoryGraphStore::new(), transaction)
        .expect("generic application passes whole-model admission");
}

#[test]
fn pure_operator_arity_and_exact_value_class_fail_before_lowering() {
    let prefix = r#"
public operator dyadic(input left: spatial[1], input right: spatial[1]): spatial[2]
  = component(left, 0) * component(right, 1);
model invalid() {
  domain body = box(0, 1, 0, 1);
  variable scalar: 1 on body; initial { scalar = 0; }
  variable vector: vector<1, 2> on body;
  relation balance on body {
"#;
    for (residual, expected) in [
        ("dyadic(left=vector) = 0;", "missing operator argument"),
        ("dyadic(left=scalar, right=vector) = 0;", "exact type rule"),
    ] {
        let source = format!("{prefix}{residual}\n  }}\n}}\n");
        let diagnostics = compile("invalid-pure-operator.eqi", &source)
            .expect_err("invalid application must fail closed");
        assert!(
            diagnostics
                .iter()
                .any(|diagnostic| diagnostic.message().contains(expected)),
            "expected `{expected}`, got {diagnostics:#?}"
        );
    }
}

#[test]
fn compiler_requires_a_literal_coordinate_axis() {
    let source = r#"
model invalid() {
  domain interval = box(0, 1);
  variable u: m on interval; initial { u = 0; }
  relation identity on interval { u - coordinate(u) = 0; }
}
"#;
    let diagnostics = compile("invalid-coordinate.eqi", source)
        .expect_err("coordinate axis selection must remain structural");

    assert!(diagnostics.iter().any(|diagnostic| {
        diagnostic.code() == codes::LANGUAGE_TYPE_ERROR
            && diagnostic.message().contains("integer literal axis")
    }));
}

#[test]
fn native_lowering_replaces_synthetic_ranges_with_declaration_paths() {
    let temperature = eqiora_lang::DraftField::new(
        "temperature",
        eqiora_core::ValueType::scalar(
            eqiora_core::ScalarDomain::Real,
            DimExponents::from_integers([0, 0, 0, 0, 1, 0, 0]).expect("bounded dimension"),
        )
        .expect("admitted numeric scalar type"),
        eqiora_lang::FieldRoleSyntax::State,
    );
    let duration = eqiora_lang::DraftParameter::new(
        "duration",
        eqiora_core::ValueLiteral::from_real(
            eqiora_core::ValueType::scalar(
                eqiora_core::ScalarDomain::Real,
                DimExponents::from_integers([0, 0, 1, 0, 0, 0, 0]).expect("bounded dimension"),
            )
            .expect("admitted numeric scalar type"),
            1.0,
        )
        .unwrap(),
    );
    let relation = eqiora_lang::DraftRelation::continuous(
        "invalid",
        [(
            temperature.expression() + duration.expression(),
            eqiora_lang::DraftExpression::constant(
                eqiora_lang::DecimalLiteral::parse("0").unwrap(),
            ),
        )],
    );
    let draft = Module::new(
        "thermal",
        [temperature.into(), duration.into(), relation.into()],
    )
    .unwrap();

    let diagnostics = lower_module(&draft, None, &[]).unwrap_err();
    assert_eq!(diagnostics[0].code(), codes::LANGUAGE_TYPE_ERROR);
    assert_eq!(
        diagnostics[0].graph_path().unwrap().to_string(),
        "thermal.invalid"
    );
    assert!(diagnostics[0].source_span().is_none());
}

#[test]
fn native_field_types_survive_direct_lowering() {
    use eqiora_core::{ScalarDomain, ValueFrame, ValueShape, ValueType};
    use eqiora_lang::{DraftField, DraftRelation, DraftSpatialDomain};
    let domain = DraftSpatialDomain::cartesian_box("body", [(0.0, 1.0), (0.0, 1.0)]);
    let value_type = ValueType::shaped(
        ScalarDomain::Complex,
        DimExponents::DIMENSIONLESS,
        ValueShape::new([2]).unwrap(),
        ValueFrame::SpatialCartesian,
    )
    .unwrap()
    .array(3)
    .unwrap();
    let field = DraftField::spatial(
        "channels",
        &domain,
        value_type.clone(),
        eqiora_lang::FieldRoleSyntax::Variable,
    );
    let relation = DraftRelation::continuous_on(
        "balance",
        &domain,
        [(field.expression(), field.expression())],
    );
    let draft = Module::new("M", [domain.into(), field.into(), relation.into()]).unwrap();
    let compiled = lower_module(&draft, None, &[]).unwrap();
    let (transaction, _, _) = compiled.into_parts();
    let field = transaction
        .ops()
        .iter()
        .find_map(|op| match op {
            Op::DefineKernelNode {
                node: KernelNode::Field(field),
            } => Some(field),
            _ => None,
        })
        .unwrap();
    assert_eq!(field.value_type(), &value_type);
    assert_eq!(field.role(), eqiora_schema::kernel::FieldRole::Variable);
}

#[test]
fn source_and_native_physical_models_lower_to_the_same_normalized_semantics() {
    let source = r#"
model resistor() {
  domain electrical = scalar_physical(across voltage: kg * m ^ 2 / (s ^ 3 * A), through current: A);
  port positive: electrical;
  port negative: electrical;
  port tap: electrical;
  parameter resistance: kg * m ^ 2 / (s ^ 3 * A ^ 2) = 2[kg * m ^ 2 / (s ^ 3 * A ^ 2)];
  relation law {
    positive.voltage - negative.voltage - resistance * positive.current = 0;
    positive.current + negative.current + tap.current = 0;
  }
  connect positive, negative, tap;
}
"#;
    let source_model = compile("resistor.eqi", source).unwrap().remove(0);
    let renamed = compile(
        "resistor.eqi",
        &source
            .replace("voltage", "potential")
            .replace("current", "flow"),
    )
    .unwrap()
    .remove(0);
    // Quantity spelling participates in authored identity, not the role-valued
    // laws, complete types, nominal domain references, or connection membership.
    assert_ne!(source_model.model(), renamed.model());
    assert_eq!(
        normalized_physical_semantics(&source_model),
        normalized_physical_semantics(&renamed)
    );
    let documented = compile(
        "resistor.eqi",
        &format!("/// A documented resistor.\n{}", source.trim_start()),
    )
    .unwrap()
    .remove(0);
    assert_eq!(source_model.model(), documented.model());
    assert_eq!(
        normalized_physical_semantics(&source_model),
        normalized_physical_semantics(&documented)
    );

    let electrical = eqiora_lang::DraftPhysicalDomain::new(
        "electrical",
        "voltage",
        eqiora_core::ValueType::scalar(eqiora_core::ScalarDomain::Real, voltage_dimension())
            .expect("admitted numeric scalar type"),
        "current",
        eqiora_core::ValueType::scalar(eqiora_core::ScalarDomain::Real, current_dimension())
            .expect("admitted numeric scalar type"),
    );
    let positive = eqiora_lang::DraftConservingPort::new("positive", &electrical);
    let negative = eqiora_lang::DraftConservingPort::new("negative", &electrical);
    let tap = eqiora_lang::DraftConservingPort::new("tap", &electrical);
    let resistance = eqiora_lang::DraftParameter::new(
        "resistance",
        eqiora_core::ValueLiteral::from_real(
            eqiora_core::ValueType::scalar(eqiora_core::ScalarDomain::Real, resistance_dimension())
                .expect("admitted numeric scalar type"),
            2.0,
        )
        .unwrap(),
    );
    let law = eqiora_lang::DraftRelation::continuous(
        "law",
        [
            (
                eqiora_lang::DraftExpression::across(&positive)
                    - eqiora_lang::DraftExpression::across(&negative)
                    - resistance.expression() * eqiora_lang::DraftExpression::through(&positive),
                eqiora_lang::DraftExpression::constant(
                    eqiora_lang::DecimalLiteral::parse("0").unwrap(),
                ),
            ),
            (
                eqiora_lang::DraftExpression::through(&positive)
                    + eqiora_lang::DraftExpression::through(&negative)
                    + eqiora_lang::DraftExpression::through(&tap),
                eqiora_lang::DraftExpression::constant(
                    eqiora_lang::DecimalLiteral::parse("0").unwrap(),
                ),
            ),
        ],
    );
    let draft = Module::new(
        "resistor",
        [
            electrical.into(),
            positive.clone().into(),
            negative.clone().into(),
            tap.clone().into(),
            resistance.into(),
            law.into(),
            eqiora_lang::DraftConservingConnection::new([&positive, &negative, &tap]).into(),
        ],
    )
    .unwrap();
    let native_model = lower_module(&draft, None, &[]).unwrap();

    assert_eq!(
        normalized_physical_semantics(&source_model),
        normalized_physical_semantics(&native_model)
    );

    let (transaction, _, _) = native_model.into_parts();
    let mut store = eqiora_graph::InMemoryGraphStore::new();
    eqiora_graph::GraphStore::commit(&mut store, transaction)
        .expect("the shared compiler transaction must pass full graph admission");
}

#[test]
fn native_physical_projection_is_insensitive_to_declaration_and_net_permutation() {
    let electrical = eqiora_lang::DraftPhysicalDomain::new(
        "electrical",
        "voltage",
        eqiora_core::ValueType::scalar(eqiora_core::ScalarDomain::Real, voltage_dimension())
            .expect("admitted numeric scalar type"),
        "current",
        eqiora_core::ValueType::scalar(eqiora_core::ScalarDomain::Real, current_dimension())
            .expect("admitted numeric scalar type"),
    );
    let positive = eqiora_lang::DraftConservingPort::new("positive", &electrical);
    let negative = eqiora_lang::DraftConservingPort::new("negative", &electrical);
    let relation = eqiora_lang::DraftRelation::continuous(
        "balance",
        [
            (
                eqiora_lang::DraftExpression::across(&positive)
                    - eqiora_lang::DraftExpression::across(&negative),
                eqiora_lang::DraftExpression::constant(
                    eqiora_lang::DecimalLiteral::parse("0").unwrap(),
                ),
            ),
            (
                eqiora_lang::DraftExpression::through(&positive)
                    + eqiora_lang::DraftExpression::through(&negative),
                eqiora_lang::DraftExpression::constant(
                    eqiora_lang::DecimalLiteral::parse("0").unwrap(),
                ),
            ),
        ],
    );
    let forward = Module::new(
        "permuted",
        [
            electrical.clone().into(),
            positive.clone().into(),
            negative.clone().into(),
            relation.clone().into(),
            eqiora_lang::DraftConservingConnection::new([&positive, &negative]).into(),
        ],
    )
    .unwrap();
    let reversed = Module::new(
        "permuted",
        [
            eqiora_lang::DraftConservingConnection::new([&negative, &positive]).into(),
            relation.into(),
            negative.into(),
            positive.into(),
            electrical.into(),
        ],
    )
    .unwrap();

    assert_eq!(
        normalized_physical_semantics(&lower_module(&forward, None, &[]).unwrap()),
        normalized_physical_semantics(&lower_module(&reversed, None, &[]).unwrap())
    );
}

#[test]
fn direct_flat_physical_fragments_normalize_before_kernel_lowering() {
    let nary = r#"
model network() {
  domain physical = scalar_physical(across potential: 1, through flow: 1);
  port a: physical;
  port b: physical;
  port c: physical;
  relation owner {
    a.potential - b.potential = 0;
    b.potential - c.potential = 0;
    a.flow + b.flow + c.flow = 0;
  }
  connect a, b, c;
}
"#;
    let chain = r#"
model network() {
  domain physical = scalar_physical(across potential: 1, through flow: 1);
  port a: physical;
  port b: physical;
  port c: physical;
  relation owner {
    a.potential - b.potential = 0;
    b.potential - c.potential = 0;
    a.flow + b.flow + c.flow = 0;
  }
  connect a, b;
  connect b, c;
}
"#;
    let nary = compile("nary.eqi", nary).unwrap().remove(0);
    let chain = compile("chain.eqi", chain).unwrap().remove(0);

    assert_eq!(
        normalized_physical_semantics(&nary),
        normalized_physical_semantics(&chain)
    );
    assert_eq!(
        chain
            .transaction()
            .ops()
            .iter()
            .filter(|operation| matches!(
                operation,
                Op::DefineKernelNode {
                    node: KernelNode::Connection(_),
                }
            ))
            .count(),
        1
    );

    let (transaction, _, _) = chain.into_parts();
    let mut store = eqiora_graph::InMemoryGraphStore::new();
    eqiora_graph::GraphStore::commit(&mut store, transaction).unwrap();
}

#[test]
fn physical_ports_require_nominal_declarations_not_value_types() {
    for (marker, expected) in [
        ("A", "unresolved scalar physical Domain `A`"),
        ("1", "physical Domain or Connector name"),
        ("m / s", "`;` after Port declaration"),
    ] {
        let source = format!("model M() {{ port p: {marker}; relation r {{ p = 0; }} }}");
        let errors = compile("marker.eqi", &source).unwrap_err();
        assert!(
            errors
                .iter()
                .any(|error| error.message().contains(expected) && error.source_span().is_some()),
            "{errors:?}"
        );
    }
}

#[test]
fn compiler_rejects_dimension_coincidence_across_nominal_domains() {
    let source = r#"
model crossed_types() {
  domain electrical_a = scalar_physical(across voltage: kg * m ^ 2 / (s ^ 3 * A), through current: A);
  domain electrical_b = scalar_physical(across voltage: kg * m ^ 2 / (s ^ 3 * A), through current: A);
  port a: electrical_a;
  port b: electrical_b;
  relation owner_a { a.voltage = 0; }
  relation owner_b { b.voltage = 0; }
  connect a, b;
}
"#;
    let diagnostics = compile("crossed-types.eqi", source)
        .expect_err("equal dimensions never erase nominal Domain identity");

    assert!(
        diagnostics.iter().any(|diagnostic| {
            diagnostic.source_span().is_some()
                && diagnostic
                    .message()
                    .contains("exact same nominal Connector or Domain")
        }),
        "{diagnostics:?}"
    );
}

#[test]
fn flat_lowering_consumes_the_shared_scalar_connection_contract() {
    let cases = [
        (
            "dimension mismatch",
            "model m() { port out: signal output m; port sink: signal input s; connect out -> sink; }",
            "dimension-matched inputs",
        ),
        (
            "source direction",
            "model m() { port out: signal output 1; port sink: signal input 1; connect sink -> out; }",
            "source before `->`",
        ),
        (
            "mixed conserving families",
            "model m() { domain d = scalar_physical(across potential: 1, through flow: 1); port causal: signal input 1; port physical: d; connect causal, physical; }",
            "cannot mix",
        ),
    ];
    for (name, source, message) in cases {
        let diagnostics = compile("connections.eqi", source).expect_err(name);
        assert!(
            diagnostics.iter().any(|diagnostic| {
                diagnostic.source_span().is_some() && diagnostic.message().contains(message)
            }),
            "{name}: {diagnostics:?}"
        );
    }
}

#[test]
fn compiler_rejects_non_physical_domains_and_unqualified_physical_ports() {
    let wrong_domain = r#"
model wrong_domain() {
  domain space = box(0, 1);
  port p: space;
  relation owner { p.potential = 0; }
}
"#;
    let diagnostics = compile("wrong-domain.eqi", wrong_domain)
        .expect_err("spatial Domains cannot type physical Ports");
    assert!(diagnostics.iter().any(|diagnostic| {
        diagnostic.source_span().is_some() && diagnostic.message().contains("not scalar physical")
    }));

    let unqualified = r#"
model unqualified() {
  domain electrical = scalar_physical(across potential: 1, through flow: 1);
  port p: electrical;
  relation owner { p = 0; }
}
"#;
    let diagnostics = compile("unqualified.eqi", unqualified)
        .expect_err("physical variables require an explicit declared member");
    assert!(diagnostics.iter().any(|diagnostic| {
        diagnostic.source_span().is_some()
            && diagnostic
                .message()
                .contains("requires a declared quantity member")
    }));
}

#[test]
fn retired_physical_role_calls_are_rejected_at_source_lookup() {
    let malformed = r#"
model malformed() {
  domain electrical = scalar_physical(across potential: 1, through flow: 1);
  port p: electrical;
  relation owner { across(p + 1) = 0; }
}
"#;
    let diagnostics =
        compile("malformed.eqi", malformed).expect_err("retired accessor is not a pure operator");
    assert!(
        diagnostics.iter().any(|diagnostic| {
            diagnostic.source_span().is_some()
                && diagnostic
                    .message()
                    .contains("unresolved pure operator `across`")
        }),
        "{diagnostics:?}"
    );

    let signal = r#"
model signal_accessor() {
  port p: signal input 1;
  relation owner { through(p) = 0; }
}
"#;
    let diagnostics =
        compile("signal-accessor.eqi", signal).expect_err("signal Ports have no through variable");
    assert!(
        diagnostics.iter().any(|diagnostic| {
            diagnostic.source_span().is_some()
                && diagnostic
                    .message()
                    .contains("unresolved pure operator `through`")
        }),
        "{diagnostics:?}"
    );
}

fn normalized_physical_semantics(model: &CompiledModel) -> Vec<String> {
    use eqiora_schema::kernel::{ActivationKind, DomainKind, ExprNode, PortPayload, SymbolRef};

    let names = model
        .symbols()
        .iter()
        .map(|(name, id)| (id, name.to_owned()))
        .collect::<BTreeMap<_, _>>();
    let mut signatures = Vec::new();
    let mut connections = BTreeMap::<RawId, Vec<String>>::new();
    let mut activations = BTreeMap::new();

    for operation in model.transaction().ops() {
        match operation {
            Op::DefineKernelNode {
                node: KernelNode::Domain(domain),
            } => {
                if let DomainKind::ScalarPhysical {
                    across_type,
                    through_type,
                } = domain.kind()
                {
                    signatures.push(format!(
                        "domain:{}:{across_type:?}:{through_type:?}",
                        named(&names, domain.id().erase())
                    ));
                }
            }
            Op::DefineKernelNode {
                node: KernelNode::Parameter(parameter),
            } => signatures.push(format!(
                "parameter:{}:{:016x}:{:?}",
                named(&names, parameter.id().erase()),
                parameter.value().component(0).unwrap().0.to_bits(),
                parameter.value_type().dimension()
            )),
            Op::DefineKernelNode {
                node: KernelNode::Port(port),
            } => {
                if let PortPayload::ScalarPhysical { domain } = port.payload() {
                    signatures.push(format!(
                        "port:{}:{}",
                        named(&names, port.id().erase()),
                        named(&names, domain.erase())
                    ));
                }
            }
            Op::DefineKernelNode {
                node: KernelNode::Relation(relation),
            } => {
                signatures.push(format!(
                    "relation:{}:{}",
                    named(&names, relation.id().erase()),
                    normalize_dag(relation.expression(), &names)
                ));
            }
            Op::DefineKernelNode {
                node: KernelNode::Activation(activation),
            } => {
                let kind = match activation.kind() {
                    ActivationKind::Continuous => "continuous",
                    ActivationKind::Periodic => "periodic",
                    ActivationKind::Event { .. } => "event",
                    ActivationKind::Guard { .. } => "guard",
                    _ => "newer",
                };
                activations.insert(activation.id().erase(), kind);
            }
            Op::DefineKernelNode {
                node: KernelNode::Connection(connection),
            } => {
                signatures.push(format!("connection-kind:{:?}", connection.semantics()));
                connections.entry(connection.id().erase()).or_default();
            }
            Op::Connect {
                from,
                to,
                edge: EdgeKind::Connects,
            } => connections
                .entry(*from)
                .or_default()
                .push(named(&names, *to).to_owned()),
            Op::Connect {
                from,
                to,
                edge: EdgeKind::DependsOn | EdgeKind::HasPort,
            } => signatures.push(format!(
                "edge:{:?}:{}:{}",
                operation_edge(operation),
                named(&names, *from),
                named(&names, *to)
            )),
            Op::Connect {
                from,
                to,
                edge: EdgeKind::Activates,
            } => signatures.push(format!(
                "activation:{}:{}",
                activations.get(from).copied().unwrap_or("missing"),
                named(&names, *to)
            )),
            _ => {}
        }
    }

    for mut members in connections.into_values() {
        members.sort();
        signatures.push(format!("connection-members:{members:?}"));
    }
    signatures.sort();

    fn normalize_dag(
        dag: &eqiora_schema::kernel::ExprDag,
        names: &BTreeMap<RawId, String>,
    ) -> String {
        let nodes = dag
            .nodes()
            .iter()
            .map(|node| match node {
                ExprNode::Constant(value) => format!(
                    "constant({:016x},{:?})",
                    value.component(0).unwrap().0.to_bits(),
                    value.value_type()
                ),
                ExprNode::Symbol(symbol) => normalize_symbol(*symbol, names),
                ExprNode::Neg(value) => format!("neg({})", value.index()),
                ExprNode::Add(left, right) => {
                    format!("add({},{})", left.index(), right.index())
                }
                ExprNode::Sub(left, right) => {
                    format!("sub({},{})", left.index(), right.index())
                }
                ExprNode::Mul(left, right) => {
                    format!("mul({},{})", left.index(), right.index())
                }
                ExprNode::Div(left, right) => {
                    format!("div({},{})", left.index(), right.index())
                }
                other => format!("{other:?}"),
            })
            .collect::<Vec<_>>();
        let roots = dag
            .roots()
            .iter()
            .map(|root| root.index())
            .collect::<Vec<_>>();
        format!("{nodes:?}:{roots:?}")
    }

    fn normalize_symbol(symbol: SymbolRef, names: &BTreeMap<RawId, String>) -> String {
        match symbol {
            SymbolRef::Field(id) => format!("field({})", named(names, id.erase())),
            SymbolRef::Derivative(id) => {
                format!("derivative({})", named(names, id.erase()))
            }
            SymbolRef::Pre(id) => format!("pre({})", named(names, id.erase())),
            SymbolRef::Next(id) => format!("next({})", named(names, id.erase())),
            SymbolRef::Parameter(id) => format!("parameter({})", named(names, id.erase())),
            SymbolRef::Port(id) => format!("port({})", named(names, id.erase())),
            SymbolRef::Across(id) => format!("across({})", named(names, id.erase())),
            SymbolRef::Through(id) => format!("through({})", named(names, id.erase())),
            SymbolRef::Time => "time".to_owned(),
            _ => "newer-symbol".to_owned(),
        }
    }

    fn operation_edge(operation: &Op) -> EdgeKind {
        let Op::Connect { edge, .. } = operation else {
            unreachable!("only Connect operations reach this helper");
        };
        *edge
    }

    fn named(names: &BTreeMap<RawId, String>, id: RawId) -> &str {
        names.get(&id).map_or("<anonymous>", String::as_str)
    }

    signatures
}
#[test]
fn declaration_literals_inherit_units_but_general_expressions_and_bindings_do_not() {
    for source in [
        "model M() { variable p: m; initial { p = 2[s]; } relation r { p - p = 0; } }",
        "model M() { variable p: array<m, 2>; initial { p = 0[s]; } relation r { p - p = 0; } }",
        "model M() { variable p: m; initial { p = 2; } relation r { p - 2 = 0; } }",
        "model M() { let p: m = 1 + 1; relation r { p - p = 0; } }",
        "model M() { let n = 2; let p: m = n; relation r { p - p = 0; } }",
        "component C(parameter p: m = 1 + 1) {  relation r { p - p = 0; } } model M() { instance c: C(); }",
        "component C(parameter p: m) {  relation r { p - p = 0; } } model M() { instance c: C(p = -2); }",
    ] {
        let errors = compile("literal-units.eqi", source).unwrap_err();
        assert!(
            errors
                .iter()
                .any(|error| error.code() == codes::LANGUAGE_TYPE_ERROR
                    && error.source_span().is_some()),
            "{source}: {errors:?}"
        );
    }
    let errors = compile(
        "literal-shape.eqi",
        "model M() { variable p: array<m, 2>; initial { p = 2; } relation r { p - p = 0; } }",
    )
    .unwrap_err();
    assert!(errors.iter().any(|error| {
        error.code() == codes::LANGUAGE_TYPE_ERROR
            && error.message().contains("incompatible types")
            && error.source_span().is_some()
    }));
    for source in [
        "model M() { variable p: m; initial { p = 2[m]; } relation r { p - p = 0; } }",
        "model M() { variable p: complex<m>; initial { p = 0; } relation r { p - p = 0; } }",
        "component C() { variable p: m; initial { p = 2[m]; } relation r { p - p = 0; } } model M() { instance c: C(); }",
        "model M() { parameter p: m = 2; relation r { p - p = 0; } }",
        "model M() { parameter p: complex<m> = 2; relation r { p - p = 0; } }",
        "model M() { let p: m = 2; relation r { p - p = 0; } }",
        "model M() { let p: complex<m> = -2; relation r { p - p = 0; } }",
        "component C(parameter p: m = 2) {  relation r { p - p = 0; } } model M() { instance c: C(); }",
        "model M() { variable p: m; initial { p = 0; } relation r { p - p = 0; } }",
        "model M() { variable p: m; initial { p = -2[mm]; } relation r { p - p = 0; } }",
        "model M() { variable p: array<complex<m>, 2>; initial { p = 0; } relation r { p - p = 0; } }",
        "component C() { variable p: m; initial { p = 2[m]; } relation r { p - p = 0; } } model M() { instance c: C(); }",
        "model M() { parameter p: m = 0; relation r { p - p = 0; } }",
        "model M() { parameter p: complex<m> = 2[m]; relation r { p - p = 0; } }",
        "model M() { let p: m = -2[m]; relation r { p - p = 0; } }",
        "model M() { let p: complex<1> = -2; relation r { p - p = 0; } }",
        "component C(parameter p: m = 2[m]) {  relation r { p - p = 0; } } model M() { instance c: C(); }",
        "component C(parameter p: m) {  relation r { p - p = 0; } } model M() { instance c: C(p = -2[m]); }",
    ] {
        compile("literal-units.eqi", source).unwrap_or_else(|error| panic!("{source}: {error:?}"));
    }
}

#[test]
fn source_physical_domains_and_connectors_keep_complex_scalar_types() {
    use eqiora_schema::kernel::DomainKind;
    for source in [
        "model M() { domain electrical = scalar_physical(across potential: complex<V>, through flow: complex<A>); port p: electrical; port n: electrical; relation r { p.potential - n.potential = 0; p.flow + n.flow = 0; } connect p, n; }",
        "connector Pin {\n  across potential: complex<V>;\n  through flow: complex<A>;\n} component C(port p: Pin) {  relation r { p.potential = 0; p.flow = 0; } } model M() { instance a: C(); instance b: C(); connect a.p, b.p; }",
    ] {
        let document = eqiora_lang::parse("physical.eqi", source)
            .into_document()
            .unwrap();
        let emitted = eqiora_lang::format(&document);
        assert!(emitted.contains("complex<"));
        let reparsed = eqiora_lang::parse("emitted.eqi", &emitted)
            .into_document()
            .unwrap();
        assert_eq!(eqiora_lang::format(&reparsed), emitted);
        let models = compile("complex-physical.eqi", source).unwrap();
        let domains = models[0]
            .transaction()
            .ops()
            .iter()
            .filter_map(|op| match op {
                Op::DefineKernelNode {
                    node: KernelNode::Domain(domain),
                } => match domain.kind() {
                    DomainKind::ScalarPhysical {
                        across_type,
                        through_type,
                    } => Some((across_type, through_type)),
                    _ => None,
                },
                _ => None,
            })
            .collect::<Vec<_>>();
        assert!(!domains.is_empty());
        for (across, through) in domains {
            assert_eq!(across.scalar_domain(), eqiora_core::ScalarDomain::Complex);
            assert_eq!(through.scalar_domain(), eqiora_core::ScalarDomain::Complex);
            assert_eq!(across.dimension(), voltage_dimension());
            assert_eq!(through.dimension(), current_dimension());
        }
    }
    for kind in ["array<complex<V>, 1>", "vector<V, 2>"] {
        for declaration in ["domain", "connector"] {
            let source = if declaration == "domain" {
                format!(
                    "model M() {{ domain electrical = scalar_physical(across voltage: {kind}, through current: A); }}"
                )
            } else {
                format!(
                    "connector Pin {{\n  across voltage: {kind};\n  through current: A;\n}} model M() {{ variable x: 1; }}"
                )
            };
            let errors = compile("physical-shape.eqi", &source).unwrap_err();
            assert!(
                errors.iter().any(
                    |error| error.message().contains("scalar mathematical types")
                        && error.source_span().is_some()
                ),
                "{errors:?}"
            );
        }
    }
}

#[test]
fn field_initial_units_normalize_and_report_the_exact_literal() {
    let source =
        "model M() { variable p: m; initial { p = -2500[mm]; } relation r { p - p = 0; } }";
    let compiled = compile("field-initial.eqi", source).unwrap();
    let initial = compiled[0]
        .transaction()
        .ops()
        .iter()
        .find_map(|op| match op {
            Op::DefineKernelNode {
                node: KernelNode::Relation(relation),
            } if relation.is_initial() => Some(relation),
            _ => None,
        })
        .unwrap();
    assert!(initial.expression().nodes().iter().any(|node| matches!(node, eqiora_schema::kernel::ExprNode::Constant(value) if value.component(0).unwrap().0 == 2.5)));
    assert!(
        initial
            .expression()
            .nodes()
            .iter()
            .any(|node| matches!(node, eqiora_schema::kernel::ExprNode::Neg(_)))
    );
    for literal in ["2[s]", "0[s]"] {
        let source = format!(
            "model M() {{ variable p: m; initial {{ p = {literal}; }} relation r {{ p - p = 0; }} }}"
        );
        let errors = compile("field-initial.eqi", &source).unwrap_err();
        assert!(
            errors.iter().any(|error| {
                error.code() == codes::LANGUAGE_TYPE_ERROR
                    && error.source_span().is_some_and(|span| {
                        source[span.start as usize..span.end as usize].contains(literal)
                    })
            }),
            "{errors:?}"
        );
    }
}
