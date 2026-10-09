use crate::{
    BinaryOp, BoundaryPairingSyntax, ComponentItem, FrameSyntax, Item, SignalDirectionSyntax,
    format, parse,
};

use super::*;

use crate::cartesian::CartesianCoordinateSyntax;

fn range(start: u32, end: u32) -> TextRange {
    TextRange::new(start, end)
}

fn dimension() -> Expr {
    SourceAstFactory::expression(
        ExprKind::Number(crate::DecimalLiteral::parse("1.0").expect("exact literal")),
        range(0, 0),
    )
    .expect("dimension")
}

fn path(segments: &[&str]) -> NamePath {
    NamePath::from_segments(segments.iter().copied(), range(0, 0)).expect("path")
}

fn private_model(name: &str, items: Vec<Item>) -> crate::ModelDecl {
    SourceAstFactory::model(VisibilitySyntax::Private, name, vec![], items, range(0, 0))
        .expect("model")
}

#[test]
fn owned_flat_model_formats_and_parses_identically() {
    let domain = SourceAstFactory::domain(
        "body",
        DomainSyntax::CartesianBox(vec![(
            CartesianCoordinateSyntax::fixed(0.0, TextRange::new(0, 0)),
            CartesianCoordinateSyntax::fixed(1.0, TextRange::new(0, 0)),
        )]),
        range(0, 0),
    )
    .expect("Domain");
    let field = SourceAstFactory::field(
        "temperature",
        Some("body".to_owned()),
        crate::FieldRoleSyntax::Variable,
        eqiora_schema::kernel::SpatialRegularity::Unspecified,
        ActivationSyntax::Continuous,
        crate::ValueTypeSyntax::real(dimension()),
        range(0, 0),
    )
    .expect("Field");
    let parameter = SourceAstFactory::parameter(
        "gain",
        crate::ValueTypeSyntax::real(dimension()),
        SourceAstFactory::expression(
            ExprKind::Number(crate::DecimalLiteral::parse("2.0").expect("exact literal")),
            range(0, 0),
        )
        .unwrap(),
        range(0, 0),
    )
    .expect("Parameter");
    let output = SourceAstFactory::port(
        "output",
        PortSyntax::Signal {
            domain: None,
            activation: ActivationSyntax::Continuous,
            direction: SignalDirectionSyntax::Output,
            value_type: crate::ValueTypeSyntax::real(dimension()),
        },
        range(0, 0),
    )
    .expect("output Port");
    let input = SourceAstFactory::port(
        "input",
        PortSyntax::Signal {
            domain: None,
            activation: ActivationSyntax::Continuous,
            direction: SignalDirectionSyntax::Input,
            value_type: crate::ValueTypeSyntax::real(dimension()),
        },
        range(0, 0),
    )
    .expect("input Port");
    let seconds = |number: &str| {
        SourceAstFactory::expression(
            ExprKind::Quantity {
                value: crate::DecimalLiteral::parse(number).unwrap(),
                unit: Box::new(
                    SourceAstFactory::expression(ExprKind::Name("s".into()), range(0, 0)).unwrap(),
                ),
            },
            range(0, 0),
        )
        .unwrap()
    };
    let clock = SourceAstFactory::clock("sample", seconds("0.1"), seconds("0"), range(0, 0))
        .expect("Clock");
    let residual =
        SourceAstFactory::expression(ExprKind::Name("temperature".to_owned()), range(0, 0))
            .expect("residual");
    let relation = SourceAstFactory::relation(
        "balance",
        ActivationSyntax::Continuous,
        Some("body".to_owned()),
        vec![
            SourceAstFactory::equation(
                residual,
                SourceAstFactory::expression(
                    ExprKind::Number(crate::DecimalLiteral::parse("0.0").expect("exact literal")),
                    range(0, 0),
                )
                .unwrap(),
                range(0, 0),
            )
            .unwrap(),
        ],
        range(0, 0),
    )
    .expect("Relation");
    let connection = SourceAstFactory::connection(
        ConnectionSyntax::Signal,
        None,
        vec![
            SourceAstFactory::expression(ExprKind::Path(path(&["output"])), range(0, 0)).unwrap(),
            SourceAstFactory::expression(ExprKind::Path(path(&["input"])), range(0, 0)).unwrap(),
        ],
        range(0, 0),
    )
    .expect("Connection");
    let binding = SourceAstFactory::named_binding(
        "gain",
        SourceAstFactory::expression(
            ExprKind::Number(crate::DecimalLiteral::parse("3.0").expect("exact literal")),
            range(0, 0),
        )
        .expect("binding value"),
        range(0, 0),
    )
    .expect("binding");
    let instance = SourceAstFactory::instance(
        "nested",
        path(&["Reusable"]),
        None,
        vec![binding],
        range(0, 0),
    )
    .expect("instance");
    let model = SourceAstFactory::model(
        VisibilitySyntax::Private,
        "constructed",
        vec![],
        vec![
            Item::Domain(domain),
            Item::Field(field),
            Item::Parameter(parameter),
            Item::Port(output),
            Item::Port(input),
            Item::Clock(clock),
            Item::Relation(relation),
            Item::Connection(connection),
            Item::Instance(instance),
        ],
        range(0, 0),
    )
    .expect("model");
    let document = SourceAstFactory::flat_document(vec![model]).expect("document");

    let source = format(&document);
    let reparsed = parse("constructed.eqi", &source)
        .into_document()
        .expect("factory output parses");

    assert_eq!(format(&reparsed), source);
}

#[test]
fn owned_declaration_only_document_preserves_package_visibility() {
    let connector = SourceAstFactory::connector(
        VisibilitySyntax::Public,
        "Pin",
        ConnectorSyntax::ScalarPhysical {
            across_name: "potential".to_owned(),
            across_type: crate::ValueTypeSyntax::real(dimension()),
            through_name: "flow".to_owned(),
            through_type: crate::ValueTypeSyntax::real(dimension()),
        },
        range(0, 0),
    )
    .expect("Connector");
    let resistance = SourceAstFactory::component_parameter(
        VisibilitySyntax::Public,
        "resistance",
        crate::ValueTypeSyntax::real(dimension()),
        Some(dimension()),
        range(0, 0),
    )
    .expect("component Parameter");
    let component = SourceAstFactory::component(
        VisibilitySyntax::Public,
        "Resistor",
        vec![crate::SignatureItem::Parameter(resistance)],
        vec![],
        range(0, 0),
    )
    .expect("component");
    let document =
        SourceAstFactory::document(Vec::new(), vec![connector], vec![component], Vec::new())
            .expect("declaration-only document");

    let source = format(&document);
    let reparsed = parse("library.eqi", &source)
        .into_document()
        .expect("factory output parses");
    assert!(reparsed.models().is_empty());
    assert_eq!(
        reparsed.connectors()[0].visibility(),
        VisibilitySyntax::Public
    );
    assert_eq!(
        reparsed.components()[0].visibility(),
        VisibilitySyntax::Public
    );
    assert_eq!(format(&reparsed), source);
    assert!(SourceAstFactory::document(Vec::new(), Vec::new(), Vec::new(), Vec::new()).is_err());
}

#[test]
fn factory_constructs_exact_pure_operator_documents_without_weakening_legacy_document_api() {
    let rank = SourceAstFactory::exact_integer("01", range(10, 12)).expect("rank");
    let formal = SourceAstFactory::pure_operator_formal(
        "x",
        PureValueClassSyntax::Spatial { rank },
        range(8, 12),
    )
    .expect("formal");
    let body = SourceAstFactory::expression(
        ExprKind::Call {
            callee: NamePath::from_segments(["component"], range(17, 26)).unwrap(),
            arguments: crate::CallArguments::Positional(vec![
                SourceAstFactory::expression(ExprKind::Name("x".to_owned()), range(27, 28))
                    .unwrap(),
                SourceAstFactory::expression(
                    ExprKind::Number(crate::DecimalLiteral::parse("00").unwrap()),
                    range(30, 32),
                )
                .unwrap(),
            ]),
        },
        range(17, 33),
    )
    .unwrap();
    let operator = SourceAstFactory::pure_operator(
        VisibilitySyntax::Public,
        "identity",
        vec![formal],
        PureValueClassSyntax::Spatial {
            rank: SourceAstFactory::exact_integer("1", range(14, 15)).expect("result rank"),
        },
        body,
        range(0, 34),
    )
    .expect("operator");
    let document = SourceAstFactory::document_with_pure_operators(
        Vec::new(),
        Vec::new(),
        Vec::new(),
        vec![operator],
        Vec::new(),
    )
    .expect("declaration-only document");

    let source = format(&document);
    assert_eq!(
        source,
        "public operator identity(input x: spatial[1]): spatial[1] = component(x, 0);\n"
    );
    assert!(parse("factory-pure.eqi", &source).into_document().is_ok());
    assert!(
        SourceAstFactory::document_with_pure_operators(
            Vec::new(),
            Vec::new(),
            Vec::new(),
            Vec::new(),
            Vec::new()
        )
        .is_err()
    );
}

#[test]
fn owned_support_slots_and_bindings_format_and_parse_identically() {
    let body = SourceAstFactory::support_slot(
        VisibilitySyntax::Public,
        "body",
        SupportSlotSyntax::Volume {
            ambient_dimension: 2,
        },
        range(0, 0),
    )
    .expect("volume support slot");
    let interface = SourceAstFactory::support_slot(
        VisibilitySyntax::Public,
        "interface",
        SupportSlotSyntax::Boundary {
            parent: "body".to_owned(),
        },
        range(0, 0),
    )
    .expect("boundary support slot");
    let component = SourceAstFactory::component(
        VisibilitySyntax::Private,
        "BoundaryState",
        vec![
            crate::SignatureItem::Support(body),
            crate::SignatureItem::Support(interface),
        ],
        vec![],
        range(0, 0),
    )
    .expect("component");
    let support = SourceAstFactory::named_binding(
        "body",
        SourceAstFactory::expression(ExprKind::Name("fluid".into()), range(0, 0)).unwrap(),
        range(0, 0),
    )
    .expect("support binding");
    let instance = SourceAstFactory::instance(
        "probe",
        path(&["BoundaryState"]),
        None,
        vec![support],
        range(0, 0),
    )
    .expect("support-aware instance");
    let model = private_model("coupled", vec![Item::Instance(instance)]);
    let document = SourceAstFactory::document(Vec::new(), vec![], vec![component], vec![model])
        .expect("document");

    let source = format(&document);
    let reparsed = parse("supports.eqi", &source)
        .into_document()
        .expect("factory support source parses");

    assert_eq!(format(&reparsed), source);
    let Item::Instance(instance) = &reparsed.models()[0].items()[0] else {
        panic!("model member is an instance");
    };
    assert_eq!(instance.bindings().len(), 1);
    assert!(
        matches!(instance.bindings()[0].value().kind(), ExprKind::Name(name) if name == "fluid")
    );
}

#[test]
fn owned_field_slots_and_bindings_format_and_parse_identically() {
    let body = SourceAstFactory::support_slot(
        VisibilitySyntax::Public,
        "body",
        SupportSlotSyntax::Volume {
            ambient_dimension: 2,
        },
        range(0, 0),
    )
    .expect("volume support slot");
    let state = SourceAstFactory::field(
        "state",
        Some("body".to_owned()),
        crate::FieldRoleSyntax::Variable,
        eqiora_schema::kernel::SpatialRegularity::Unspecified,
        ActivationSyntax::Continuous,
        SourceAstFactory::value_type(
            crate::ValueTypeSyntaxKind::Vector {
                scalar: Box::new(crate::ValueTypeSyntax::real(dimension())),
                extent: 2,
            },
            range(0, 0),
        )
        .unwrap(),
        range(0, 0),
    )
    .expect("Field slot");
    let component = SourceAstFactory::component(
        VisibilitySyntax::Private,
        "StateLaw",
        vec![
            crate::SignatureItem::Support(body),
            crate::SignatureItem::Field(state),
        ],
        vec![],
        range(0, 0),
    )
    .expect("component");
    let support = SourceAstFactory::named_binding(
        "body",
        SourceAstFactory::expression(ExprKind::Name("region".into()), range(0, 0)).unwrap(),
        range(0, 0),
    )
    .expect("support binding");
    let field = SourceAstFactory::named_binding(
        "state",
        SourceAstFactory::expression(ExprKind::Name("temperature".into()), range(0, 0)).unwrap(),
        range(0, 0),
    )
    .expect("Field binding");
    let instance = SourceAstFactory::instance(
        "law",
        path(&["StateLaw"]),
        None,
        vec![support, field],
        range(0, 0),
    )
    .expect("slot-aware instance");
    let model = private_model("coupled", vec![Item::Instance(instance)]);
    let document = SourceAstFactory::document(Vec::new(), vec![], vec![component], vec![model])
        .expect("document");

    let source = format(&document);
    let reparsed = parse("field-slots.eqi", &source)
        .into_document()
        .expect("factory Field-slot source parses");

    assert_eq!(format(&reparsed), source);
    let crate::SignatureItem::Field(slot) = &reparsed.components()[0].signature()[1] else {
        panic!("second component member is a Field slot");
    };
    assert_eq!(slot.domain(), Some("body"));
    let Item::Instance(instance) = &reparsed.models()[0].items()[0] else {
        panic!("model member is an instance");
    };
    assert!(
        matches!(instance.bindings()[1].value().kind(), ExprKind::Name(name) if name == "temperature")
    );
}

#[test]
fn name_rewrite_preserves_expression_tree_and_ranges() {
    let bare = SourceAstFactory::expression(ExprKind::Name("x".to_owned()), range(10, 11))
        .expect("bare name");
    let qualified_path =
        NamePath::from_segments(["plant", "terminal"], range(20, 34)).expect("qualified");
    let qualified = SourceAstFactory::expression(ExprKind::Path(qualified_path), range(20, 34))
        .expect("qualified name");
    let call = SourceAstFactory::expression(
        ExprKind::Call {
            callee: path(&["across"]),
            arguments: crate::CallArguments::Positional(vec![qualified]),
        },
        range(13, 35),
    )
    .expect("call");
    let expression = SourceAstFactory::expression(
        ExprKind::Binary {
            op: BinaryOp::Add,
            left: Box::new(bare),
            right: Box::new(call),
        },
        range(10, 35),
    )
    .expect("expression");

    let rewritten = expression.rewrite_name_paths(|name| match name.as_str() {
        "x" => Some(NamePath::from_segments(["scope", "x"], range(100, 200)).expect("replacement")),
        "plant.terminal" => {
            Some(NamePath::from_segments(["terminal"], range(300, 400)).expect("replacement"))
        }
        "across" => Some(
            NamePath::from_segments(["operators", "across"], range(500, 600)).expect("replacement"),
        ),
        _ => None,
    });

    assert_eq!(rewritten.range(), expression.range());
    let ExprKind::Binary { left, right, .. } = rewritten.kind() else {
        panic!("binary topology is preserved");
    };
    assert_eq!(left.range(), range(10, 11));
    let ExprKind::Path(left_path) = left.kind() else {
        panic!("bare name was rewritten to a qualified path");
    };
    assert_eq!(left_path.as_str(), "scope.x");
    assert_eq!(left_path.range(), range(10, 11));
    assert_eq!(right.range(), range(13, 35));
    let ExprKind::Call { callee, arguments } = right.kind() else {
        panic!("Call topology is preserved");
    };
    assert_eq!(callee.as_str(), "operators.across");
    assert_eq!(callee.range(), range(0, 0));
    assert_eq!(arguments.positional().unwrap()[0].range(), range(20, 34));
    assert!(matches!(
        arguments.positional().unwrap()[0].kind(),
        ExprKind::Name(name) if name == "terminal"
    ));
}

#[test]
fn factory_constructs_closed_field_physical_source_shapes() {
    let trace =
        SourceAstFactory::connector_quantity("velocity", dimension()).expect("trace quantity");
    let flux =
        SourceAstFactory::connector_quantity("traction", dimension()).expect("flux quantity");
    let connector = SourceAstFactory::connector(
        VisibilitySyntax::Public,
        "MechanicalBoundary",
        ConnectorSyntax::FieldPhysical {
            trace,
            flux,
            shape: ValueShapeSyntax::Exact(vec![2]),
            frame: FrameSyntax::Spatial,
            pairing: BoundaryPairingSyntax::EuclideanBoundaryDuality,
        },
        range(0, 0),
    )
    .expect("field-physical Connector");
    let field = SourceAstFactory::field(
        "velocity",
        None,
        crate::FieldRoleSyntax::Variable,
        eqiora_schema::kernel::SpatialRegularity::Unspecified,
        ActivationSyntax::Continuous,
        SourceAstFactory::value_type(
            crate::ValueTypeSyntaxKind::Array {
                element: Box::new(crate::ValueTypeSyntax::real(dimension())),
                extent: SourceAstFactory::expression(
                    ExprKind::Number(crate::DecimalLiteral::parse("2").unwrap()),
                    range(0, 0),
                )
                .unwrap(),
            },
            range(0, 0),
        )
        .unwrap(),
        range(0, 0),
    )
    .expect("shaped Field");
    let port = SourceAstFactory::port(
        "interface",
        PortSyntax::FieldPhysical {
            connector: path(&["MechanicalBoundary"]),
            support: "wall".to_owned(),
        },
        range(0, 0),
    )
    .expect("field-physical Port");
    let model = SourceAstFactory::model(
        VisibilitySyntax::Private,
        "coupled",
        vec![],
        vec![Item::Field(field), Item::Port(port)],
        range(0, 0),
    )
    .expect("model");
    let document = SourceAstFactory::document(Vec::new(), vec![connector], Vec::new(), vec![model])
        .expect("document");
    let source = format(&document);

    assert_eq!(
        format(
            &parse("factory-boundary.eqi", &source)
                .into_document()
                .expect("factory source parses")
        ),
        source
    );
    assert!(SourceAstFactory::connector_quantity("not-valid", dimension()).is_err());
    assert!(
        SourceAstFactory::value_type(
            crate::ValueTypeSyntaxKind::Array {
                element: Box::new(crate::ValueTypeSyntax::real(dimension())),
                extent: SourceAstFactory::expression(
                    ExprKind::Number(crate::DecimalLiteral::parse("0").unwrap()),
                    range(0, 0)
                )
                .unwrap(),
            },
            range(0, 0)
        )
        .is_err()
    );
}

#[test]
fn factory_constructs_complete_exterior_families_and_roundtrips() {
    let body = SourceAstFactory::support_slot(
        VisibilitySyntax::Public,
        "body",
        SupportSlotSyntax::Volume {
            ambient_dimension: 2,
        },
        range(0, 0),
    )
    .expect("body support");
    let exterior = SourceAstFactory::support_slot(
        VisibilitySyntax::Public,
        "exterior",
        SupportSlotSyntax::CompleteExterior {
            parent: "body".to_owned(),
        },
        range(0, 0),
    )
    .expect("complete exterior support");
    let binder = SourceAstFactory::boundary_family_binder("boundary", "exterior", range(0, 0))
        .expect("family binder");
    let port = SourceAstFactory::component_port(
        VisibilitySyntax::Public,
        "mechanical",
        PortSyntax::FieldPhysical {
            connector: path(&["MechanicalBoundary"]),
            support: "boundary".to_owned(),
        },
        range(0, 0),
    )
    .expect("component Port");
    let port_family =
        SourceAstFactory::component_port_family(port, binder.clone()).expect("Port family");
    let selector = SourceAstFactory::boundary_port_selector("boundary", "boundary", range(0, 0))
        .expect("selector");
    let selected_port = SourceAstFactory::expression(
        ExprKind::BoundaryPortSelection {
            port: Box::new(path(&["mechanical"])),
            selector: Box::new(selector.clone()),
        },
        range(0, 0),
    )
    .expect("selected Port expression");
    let residual = SourceAstFactory::expression(
        ExprKind::Member {
            value: Box::new(selected_port),
            member: "traction".to_owned(),
        },
        range(0, 0),
    )
    .expect("family residual");
    let relation = SourceAstFactory::relation(
        "natural",
        ActivationSyntax::Continuous,
        Some("boundary".to_owned()),
        vec![
            SourceAstFactory::equation(
                residual,
                SourceAstFactory::expression(
                    ExprKind::Number(crate::DecimalLiteral::parse("0.0").expect("exact literal")),
                    range(0, 0),
                )
                .unwrap(),
                range(0, 0),
            )
            .unwrap(),
        ],
        range(0, 0),
    )
    .expect("Relation");
    let relation_family =
        SourceAstFactory::relation_family(relation, binder.clone()).expect("Relation family");
    let left = SourceAstFactory::boundary_port_reference(
        path(&["child", "mechanical"]),
        Some(selector.clone()),
    )
    .expect("left family Port");
    let right = SourceAstFactory::boundary_port_reference(path(&["mechanical"]), Some(selector))
        .expect("right family Port");
    let connection =
        SourceAstFactory::boundary_connection(Some(binder), vec![left, right], range(0, 0))
            .expect("pointwise Connection");
    let component = SourceAstFactory::component(
        VisibilitySyntax::Private,
        "BoundaryLaw",
        vec![
            crate::SignatureItem::Support(body),
            crate::SignatureItem::Support(exterior),
            crate::SignatureItem::PortFamily(port_family),
        ],
        vec![
            ComponentItem::RelationFamily(relation_family),
            ComponentItem::BoundaryConnection(connection),
        ],
        range(0, 0),
    )
    .expect("component");

    let members = ["x_lower", "x_upper", "y_lower", "y_upper"]
        .into_iter()
        .map(|name| SourceAstFactory::expression(ExprKind::Name(name.into()), range(0, 0)).unwrap())
        .collect();
    let exterior = SourceAstFactory::expression(
        ExprKind::Call {
            callee: path(&["boundaries"]),
            arguments: crate::CallArguments::Positional(members),
        },
        range(0, 0),
    )
    .unwrap();
    let instance = SourceAstFactory::instance(
        "law",
        path(&["BoundaryLaw"]),
        None,
        vec![
            SourceAstFactory::named_binding(
                "body",
                SourceAstFactory::expression(ExprKind::Name("fluid".into()), range(0, 0)).unwrap(),
                range(0, 0),
            )
            .unwrap(),
            SourceAstFactory::named_binding("exterior", exterior, range(0, 0)).unwrap(),
        ],
        range(0, 0),
    )
    .unwrap();
    let model = private_model("coupled", vec![Item::Instance(instance)]);
    let document = SourceAstFactory::document(Vec::new(), Vec::new(), vec![component], vec![model])
        .expect("document");

    let source = format(&document);
    assert!(source.contains("mechanical[boundary = boundary].traction"));
    let reparsed = parse("complete-exterior-factory.eqi", &source)
        .into_document()
        .expect("factory boundary-family source parses");
    assert_eq!(format(&reparsed), source);
    let Item::Instance(instance) = &reparsed.models()[0].items()[0] else {
        panic!("model member is an instance");
    };
    assert!(
        matches!(instance.bindings()[1].value().kind(), ExprKind::Call { arguments, .. } if arguments.expressions().len() == 4)
    );

    let signal_port = SourceAstFactory::component_port(
        VisibilitySyntax::Public,
        "signal",
        PortSyntax::Signal {
            domain: None,
            activation: ActivationSyntax::Continuous,
            direction: SignalDirectionSyntax::Input,
            value_type: crate::ValueTypeSyntax::real(dimension()),
        },
        range(0, 0),
    )
    .expect("signal Port");
    let binder =
        SourceAstFactory::boundary_family_binder("b", "exterior", range(0, 0)).expect("binder");
    assert!(SourceAstFactory::component_port_family(signal_port, binder).is_err());
}

#[test]
fn construction_rejects_unrepresentable_source_shapes() {
    assert!(NamePath::from_segments(Vec::<String>::new(), range(0, 0)).is_err());
    assert!(NamePath::from_segments(["not-valid"], range(0, 0)).is_err());
    assert!(
        SourceAstFactory::connection(
            ConnectionSyntax::Conserving,
            None,
            vec![
                SourceAstFactory::expression(ExprKind::Path(path(&["only_one"])), range(0, 0))
                    .unwrap()
            ],
            range(0, 0),
        )
        .is_err()
    );
    assert!(
        SourceAstFactory::expression(
            ExprKind::Number(crate::DecimalLiteral::parse("1.0").expect("exact literal")),
            range(2, 1)
        )
        .is_err()
    );
    assert!(
        SourceAstFactory::support_slot(
            VisibilitySyntax::Public,
            "boundary",
            SupportSlotSyntax::Boundary {
                parent: "not-valid".to_owned(),
            },
            range(0, 0),
        )
        .is_err()
    );
    assert!(
        SourceAstFactory::named_binding(
            "body",
            crate::Expr {
                resolved_enum: None,
                resolved_nominal: None,
                kind: ExprKind::Name("not-valid".into()),
                range: range(0, 0)
            },
            range(0, 0)
        )
        .is_err()
    );
}

#[test]
fn spatial_periodic_connection_is_closed_model_only() {
    let ports = ["lower", "upper"]
        .into_iter()
        .map(|name| {
            SourceAstFactory::boundary_port_reference(path(&[name]), None)
                .expect("periodic Port reference")
        })
        .collect();
    let connection = SourceAstFactory::spatial_periodic_boundary_connection(ports, range(0, 0))
        .expect("closed-model periodic Connection shape");

    let error = SourceAstFactory::component(
        VisibilitySyntax::Private,
        "InvalidPeriodicComponent",
        vec![],
        vec![ComponentItem::BoundaryConnection(connection)],
        range(0, 0),
    )
    .expect_err("periodic Connection must not enter reusable Component syntax");

    assert_eq!(
        error.to_string(),
        "a spatial-periodic Connection belongs only to a closed Model"
    );
}
