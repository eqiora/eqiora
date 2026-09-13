use super::*;

#[test]
fn parser_retains_exact_pure_operator_syntax_and_qualified_applications() {
    let source = r#"public operator dyadic(input left: spatial[01], input right: spatial[1]): spatial[2] =
  component(left, 0) * component(right, 1) + rational(03, 4) * delta(0, 1);

model coupled() {
  variable u: 1;
  variable v: 1;
  relation law { ops.dyadic(left = u, right = v) = 0; }
}"#;
    let document = parse("pure-operator.eqi", source)
        .into_document()
        .expect("bounded pure operator source");

    let operator = &document.pure_operators()[0];
    assert_eq!(operator.visibility(), VisibilitySyntax::Public);
    assert_eq!(operator.name(), "dyadic");
    assert_eq!(operator.formals().len(), 2);
    assert_eq!(operator.formals()[0].name(), "left");
    let PureValueClassSyntax::Spatial { rank } = operator.formals()[0].value_class() else {
        panic!("first formal is spatial");
    };
    assert_eq!(rank.value(), 1);
    assert_eq!(rank.spelling(), "01");
    assert_eq!(
        &source[rank.range().start() as usize..rank.range().end() as usize],
        "01"
    );
    assert_eq!(
        &source[operator.range().start() as usize..operator.range().end() as usize],
        &source[..source.find("\n\nmodel").expect("model separator")]
    );

    let ExprKind::Binary { left, right, .. } = operator.body().kind() else {
        panic!("addition is the pure body root");
    };
    assert!(matches!(
        left.kind(),
        ExprKind::Binary {
            op: BinaryOp::Mul,
            ..
        }
    ));
    let ExprKind::Binary {
        op: BinaryOp::Mul,
        left: rational,
        right: delta,
    } = right.kind()
    else {
        panic!("right term retains exact rational and delta nodes");
    };
    let ExprKind::Call { callee, arguments } = rational.kind() else {
        panic!("rational call");
    };
    assert_eq!(callee.as_str(), "rational");
    let ExprKind::Number(numerator) = arguments.positional().unwrap()[0].kind() else {
        panic!("exact number");
    };
    assert_eq!(numerator.canonical_text(), "3");
    assert!(matches!(delta.kind(), ExprKind::Call { callee, .. } if callee.as_str() == "delta"));

    let Item::Relation(relation) = &document.models()[0].items()[2] else {
        panic!("model relation retained");
    };
    let ExprKind::Call { callee, arguments } = relation.conditions().unwrap()[0].left().kind()
    else {
        panic!("residual is a qualified application");
    };
    assert_eq!(callee.segments().collect::<Vec<_>>(), ["ops", "dyadic"]);
    assert_eq!(arguments.expressions().len(), 2);
    assert!(
        matches!(arguments.named().unwrap()[0].value().kind(), ExprKind::Name(name) if name == "u")
    );
    assert!(
        matches!(arguments.named().unwrap()[1].value().kind(), ExprKind::Name(name) if name == "v")
    );
}

#[test]
fn parser_rejects_displaced_operator_syntax() {
    for source in [
        "pure operator bad(input x: scalar): scalar = x;",
        "operator bad(x: scalar): scalar = x;",
        "operator bad(input x: scalar) -> scalar = x;",
        "operator bad(): scalar = 1;",
    ] {
        assert!(
            parse("invalid-operator.eqi", source)
                .into_document()
                .is_err(),
            "{source}"
        );
    }
}

#[test]
fn parser_represents_scalar_component_selection_with_zero_axes() {
    let source = "operator negate(input s: scalar): scalar = -component(s);";
    let document = parse("scalar-component.eqi", source)
        .into_document()
        .expect("scalar component selection");
    let ExprKind::Unary {
        op: UnaryOp::Neg,
        value,
    } = document.pure_operators()[0].body().kind()
    else {
        panic!("negation retained");
    };
    assert!(matches!(
        value.kind(),
        ExprKind::Call { callee, arguments } if callee.as_str() == "component" && arguments.positional().unwrap().len() == 1
    ));
}

#[test]
fn parser_requires_pure_operators_before_models_and_nonempty_calls() {
    let late = parse(
        "late-operator.eqi",
        "model M() {} operator identity(input x: scalar): scalar = component(x, 0);",
    );
    assert!(late.into_document().is_err());

    let empty_call = parse(
        "empty-call.eqi",
        "model M() { relation r { ops.identity() = 0; } }",
    );
    assert!(empty_call.into_document().is_err());
}

#[test]
fn parser_builds_continuous_and_periodic_relations() {
    let source = r#"
model thermal() {
  state temperature: K;
  state command: 1 at control;
  initial { temperature = 293[K]; command = 0; }
  clock control = periodic(1[s] / 10, phase = 0[s] / 1);
  relation plant {
derivative(temperature) - command = 0;
  }
  relation controller at control {
next(command) - pre(command) = 0;
  }
}
"#;
    let result = parse("thermal.eqi", source);
    let document = result.into_document().expect("valid source");

    assert_eq!(document.models().len(), 1);
    assert_eq!(document.models()[0].items().len(), 6);
}

#[test]
fn parser_recovers_after_an_invalid_item() {
    let source = "model m() { nonsense; variable x: 1; }";
    let result = parse("recovery.eqi", source);

    assert!(!result.diagnostics().is_empty());
    assert_eq!(
        result.document().expect("recovered").models()[0]
            .items()
            .len(),
        1
    );
}

#[test]
fn parser_retains_scalar_physical_contracts_and_source_ranges() {
    let source = r#"model circuit() {
  domain electrical = scalar_physical(across voltage: kg * m ^ 2 / (s ^ 3 * A), through current: A);
  port terminal: electrical;
  relation component { terminal.voltage = 0; }
}"#;
    let document = parse("circuit.eqi", source)
        .into_document()
        .expect("valid physical source");
    let items = document.models()[0].items();

    let Item::Domain(domain) = &items[0] else {
        panic!("first item is the physical Domain");
    };
    let DomainSyntax::ScalarPhysical {
        across_type,
        through_type,
        ..
    } = domain.syntax()
    else {
        panic!("Domain retains the scalar physical contract");
    };
    assert_eq!(
        &source[domain.range().start() as usize..domain.range().end() as usize],
        "domain electrical = scalar_physical(across voltage: kg * m ^ 2 / (s ^ 3 * A), through current: A);"
    );
    assert_eq!(
        &source[across_type.range().start() as usize..across_type.range().end() as usize],
        "kg * m ^ 2 / (s ^ 3 * A)"
    );
    assert_eq!(
        &source[through_type.range().start() as usize..through_type.range().end() as usize],
        "A"
    );

    let Item::Port(port) = &items[1] else {
        panic!("second item is the physical Port");
    };
    assert!(matches!(
        port.syntax(),
        PortSyntax::ScalarPhysical { domain } if domain == "electrical"
    ));
    assert_eq!(
        &source[port.range().start() as usize..port.range().end() as usize],
        "port terminal: electrical;"
    );
}

#[test]
fn parser_builds_typed_component_interfaces_instances_and_paths() {
    let source = r#"
connector Pin {
  across voltage: kg * m ^ 2 / (s ^ 3 * A);
  through current: A;
}

component Pair(parameter resistance: kg * m ^ 2 / (s ^ 3 * A ^ 2), port positive: Pin) {

  parameter scale: 1 = 2;

  port command: signal input 1;
  instance inner: Catalog.Resistor(resistance = resistance * scale);
  relation law { inner.positive.voltage - resistance = 0; }
  connect inner.positive, positive;
}

model parallel(port positive: Pin) {
  instance r2: Pair(resistance = 2);
  instance r4: Pair(resistance = 4);
  connect r2.positive, r4.positive, positive;
}
"#;
    let document = parse("components.eqi", source)
        .into_document()
        .expect("component syntax is valid");

    assert_eq!(document.connectors().len(), 1);
    assert_eq!(document.connectors()[0].name(), "Pin");
    assert!(matches!(
        document.connectors()[0].syntax(),
        ConnectorSyntax::ScalarPhysical { .. }
    ));

    let component = &document.components()[0];
    assert_eq!(component.name(), "Pair");
    let crate::SignatureItem::Parameter(resistance) = &component.signature()[0] else {
        panic!("first member is the public Parameter");
    };
    assert_eq!(resistance.visibility(), VisibilitySyntax::Public);
    assert!(resistance.default().is_none());
    let ComponentItem::Parameter(scale) = &component.items()[0] else {
        panic!("second member is the private Parameter");
    };
    assert_eq!(scale.visibility(), VisibilitySyntax::Private);
    assert!(scale.default().is_some());
    let crate::SignatureItem::Port(positive) = &component.signature()[1] else {
        panic!("third member is the public Port");
    };
    assert_eq!(positive.visibility(), VisibilitySyntax::Public);
    let PortSyntax::ScalarPhysicalConnector { connector } = positive.syntax() else {
        panic!("physical component Ports retain nominal Connector syntax");
    };
    assert_eq!(connector.segments().collect::<Vec<_>>(), ["Pin"]);

    let ComponentItem::Instance(inner) = &component.items()[2] else {
        panic!("fifth member is the nested instance");
    };
    assert_eq!(
        inner.definition().segments().collect::<Vec<_>>(),
        ["Catalog", "Resistor"]
    );
    assert_eq!(inner.bindings()[0].name(), "resistance");

    let ComponentItem::Relation(relation) = &component.items()[3] else {
        panic!("sixth member is the Relation");
    };
    let ExprKind::Binary { left, .. } = relation.conditions().unwrap()[0].left().kind() else {
        panic!("Relation retains its subtraction");
    };
    let ExprKind::Path(path) = left.kind() else {
        panic!("quantity selection is a structured path");
    };
    assert_eq!(
        path.segments().collect::<Vec<_>>(),
        ["inner", "positive", "voltage"]
    );
    assert_eq!(
        &source[path.range().start() as usize..path.range().end() as usize],
        "inner.positive.voltage"
    );

    let Item::Connection(connection) = &document.models()[0].items()[2] else {
        panic!("third model member is a Connection");
    };
    let ExprKind::Path(path) = connection.port_expressions()[0].kind() else {
        panic!("qualified connection endpoint retains its path");
    };
    assert_eq!(path.segments().collect::<Vec<_>>(), ["r2", "positive"]);
    let crate::SignatureItem::Port(port) = &document.models()[0].signature()[0] else {
        panic!("public physical endpoint belongs to signature");
    };
    assert_eq!(port.name(), "positive");
}

#[test]
fn parser_retains_component_support_requirements_and_mixed_bindings() {
    let source = r#"component BoundaryState(
  support body: volume(ambient_dimension = 2),
  support interface: boundary(parent = body)
) {
  variable state: 1 on body;
}

model coupled() {
  domain fluid = box(0, 1, 0, 1);
  domain wall = boundary(fluid, axis = 0, side = lower);
  instance probe: BoundaryState(gain = 2, body = fluid, interface = wall);
}"#;
    let document = parse("supports.eqi", source)
        .into_document()
        .expect("support-slot syntax is valid");
    let component = &document.components()[0];

    let crate::SignatureItem::Support(body) = &component.signature()[0] else {
        panic!("first member is the volume support slot");
    };
    assert_eq!(body.visibility(), VisibilitySyntax::Public);
    assert!(matches!(
        body.syntax(),
        SupportSlotSyntax::Volume {
            ambient_dimension: 2
        }
    ));
    assert_eq!(
        &source[body.range().start() as usize..body.range().end() as usize],
        "support body: volume(ambient_dimension = 2)"
    );

    let crate::SignatureItem::Support(interface) = &component.signature()[1] else {
        panic!("second member is the boundary support slot");
    };
    assert!(matches!(
        interface.syntax(),
        SupportSlotSyntax::Boundary { parent } if parent == "body"
    ));
    let ComponentItem::Field(field) = &component.items()[0] else {
        panic!("third member is the private variable");
    };
    assert_eq!(field.name(), "state");
    let Item::Instance(instance) = &document.models()[0].items()[2] else {
        panic!("third model member is the component instance");
    };
    assert_eq!(
        instance
            .bindings()
            .iter()
            .map(|b| b.name())
            .collect::<Vec<_>>(),
        ["gain", "body", "interface"]
    );
    assert!(matches!(instance.bindings()[1].value().kind(), ExprKind::Name(n) if n == "fluid"));
    assert!(matches!(instance.bindings()[2].value().kind(), ExprKind::Name(n) if n == "wall"));
}

#[test]
fn parser_retains_occurrence_bound_field_slots_and_bindings() {
    let source = r#"component IsotropicBalance2d(
  support body: volume(ambient_dimension = 2),
  variable displacement: vector<m, 2> on body,
  variable load: vector<kg / (m * s ^ 2), 2> on body, parameter mu: kg / (m * s ^ 2)) {

}

model Main() {
  domain body = box(0, 1, 0, 1);
  variable u: vector<m, 2> on body;
  variable f: vector<kg / (m * s ^ 2), 2> on body;
  instance law: IsotropicBalance2d(mu = 3, body = body, displacement = u, load = f);
}"#;
    let document = parse("field-slots.eqi", source)
        .into_document()
        .expect("Field-slot syntax is valid");
    let component = &document.components()[0];

    let crate::SignatureItem::Field(displacement) = &component.signature()[1] else {
        panic!("second member is the displacement Field slot");
    };
    assert_eq!(displacement.name(), "displacement");
    assert_eq!(displacement.domain(), Some("body"));
    assert!(matches!(
        displacement.value_type().kind(),
        crate::ValueTypeSyntaxKind::Vector { extent: 2, .. }
    ));
    assert_eq!(
        &source[displacement.range().start() as usize..displacement.range().end() as usize],
        "variable displacement: vector<m, 2> on body"
    );

    let Item::Instance(instance) = &document.models()[0].items()[3] else {
        panic!("fourth model member is the component instance");
    };
    assert_eq!(
        instance
            .bindings()
            .iter()
            .map(|b| b.name())
            .collect::<Vec<_>>(),
        ["mu", "body", "displacement", "load"]
    );
    assert!(matches!(instance.bindings()[2].value().kind(), ExprKind::Name(n) if n == "u"));
    assert!(matches!(instance.bindings()[3].value().kind(), ExprKind::Name(n) if n == "f"));
    let formatted = crate::format(&document);
    assert_eq!(
        crate::format(&parse("again.eqi", &formatted).into_document().unwrap()),
        formatted
    );
}

#[test]
fn field_discriminator_does_not_reserve_the_parameter_name_field() {
    let document = parse(
        "field-parameter.eqi",
        "component C(parameter field: 1) {  } model m() { instance c: C(field = 1); }",
    )
    .into_document()
    .expect("`field = expression` remains a Parameter binding");
    let Item::Instance(instance) = &document.models()[0].items()[0] else {
        panic!("model member is an instance");
    };

    assert_eq!(instance.bindings()[0].name(), "field");
    assert_eq!(instance.bindings().len(), 1);
}

#[test]
fn parser_rejects_retired_body_field_slots() {
    let private = parse(
        "private-slot.eqi",
        "component C() { field slot state on body as continuum: 1; }",
    );
    assert!(private.into_document().is_err());

    let discrete = parse(
        "discrete-slot.eqi",
        "component C() { public field slot state on body as discrete: 1; }",
    );
    assert!(discrete.into_document().is_err());
}

#[test]
fn support_discriminator_does_not_reserve_the_parameter_name_support() {
    let document = parse(
        "support-parameter.eqi",
        "component C(parameter support: 1) {  } model m() { instance c: C(support = 1); }",
    )
    .into_document()
    .expect("`support = expression` remains a Parameter binding");
    let Item::Instance(instance) = &document.models()[0].items()[0] else {
        panic!("model member is an instance");
    };

    assert_eq!(instance.bindings()[0].name(), "support");
    assert_eq!(instance.bindings().len(), 1);
}

#[test]
fn parser_accepts_visibility_typed_declaration_only_documents() {
    let source = r#"public connector Pin {
  across voltage: 1;
  through current: A;
}
private component Internal() {}
public component Resistor() {}"#;
    let document = parse("library.eqi", source)
        .into_document()
        .expect("library declarations parse without a Model");

    assert!(document.models().is_empty());
    assert_eq!(
        document.connectors()[0].visibility(),
        VisibilitySyntax::Public
    );
    assert_eq!(
        document.components()[0].visibility(),
        VisibilitySyntax::Private
    );
    assert_eq!(
        document.components()[1].visibility(),
        VisibilitySyntax::Public
    );
    assert_eq!(
        &source[document.connectors()[0].range().start() as usize
            ..document.connectors()[0].range().end() as usize],
        "public connector Pin {\n  across voltage: 1;\n  through current: A;\n}"
    );
    assert_eq!(
        &source[document.components()[0].range().start() as usize
            ..document.components()[0].range().end() as usize],
        "private component Internal() {}"
    );
}

#[test]
fn parser_retains_public_and_private_model_visibility() {
    let public = parse("entry.eqi", "public model Main() {}")
        .into_document()
        .expect("public Model is accepted");
    assert_eq!(public.models().len(), 1);
    assert_eq!(public.models()[0].visibility(), VisibilitySyntax::Public);
    assert_eq!(
        public.models()[0].range(),
        TextRange::new(0, "public model Main() {}".len() as u32)
    );
    assert_eq!(crate::format(&public), "public model Main() {\n}\n");

    let private = parse("entry.eqi", "private model Main() {}")
        .into_document()
        .expect("an explicitly package-local Model is accepted");
    assert_eq!(private.models()[0].visibility(), VisibilitySyntax::Private);
    assert_eq!(private.models()[0].name(), "Main");
}

#[test]
fn parser_discards_illegal_public_members_and_recovers() {
    let source = r#"
component Invalid() {
  public relation exposed { 1 = 0; }
  public instance child: Other();
  instance malformed: ;
  parameter retained: 1 = 1;
}
model root() {}
"#;
    let result = parse("visibility.eqi", source);
    let document = result.document().expect("recovered compilation unit");

    assert_eq!(result.diagnostics().len(), 3);
    assert_eq!(document.components()[0].items().len(), 1);
    assert!(matches!(
        document.components()[0].items()[0],
        ComponentItem::Parameter(_)
    ));
}

#[test]
fn parser_requires_compilation_unit_definitions_before_models() {
    let result = parse(
        "order.eqi",
        "model first() {} component Late() {} model second() {}",
    );
    let document = result.document().expect("declarations are recovered");

    assert_eq!(result.diagnostics().len(), 1);
    assert_eq!(document.components()[0].name(), "Late");
    assert_eq!(document.models().len(), 2);
}

#[test]
fn parser_retains_field_physical_connector_shapes_ports_and_flux_access() {
    let source = r#"
public connector MechanicalBoundary {
  pairing euclidean_boundary_duality;
  flux traction: kg / (m * s ^ 2);
  frame spatial;
  trace velocity: m / s;
  shape spatial_vector;
  orientation parent_outward;
}

model coupled() {
  domain fluid = box(0, 1, 0, 1);
  domain wall = boundary(fluid, axis = 0, side = upper);
  variable velocity: array<m / s, 2> on fluid;
  port interface: MechanicalBoundary over wall;
  relation balance on wall { interface.traction = 0; }
}
"#;
    let document = parse("field-physical.eqi", source)
        .into_document()
        .expect("field-valued boundary syntax parses");

    let ConnectorSyntax::FieldPhysical {
        trace,
        flux,
        shape,
        frame,
        pairing,
    } = document.connectors()[0].syntax()
    else {
        panic!("field-physical Connector retained");
    };
    assert_eq!(trace.name(), "velocity");
    assert_eq!(flux.name(), "traction");
    assert_eq!(shape, &ValueShapeSyntax::SpatialVector);
    assert_eq!(*frame, FrameSyntax::Spatial);
    assert_eq!(*pairing, BoundaryPairingSyntax::EuclideanBoundaryDuality);

    let Item::Field(field) = &document.models()[0].items()[2] else {
        panic!("third item is the shaped Field");
    };
    assert!(matches!(
        field.value_type().kind(),
        crate::ValueTypeSyntaxKind::Array { extent, .. } if matches!(extent.kind(), ExprKind::Number(n) if n.to_i64().ok() == Some(2))
    ));
    let Item::Port(port) = &document.models()[0].items()[3] else {
        panic!("fourth item is the boundary Port");
    };
    let PortSyntax::FieldPhysical { connector, support } = port.syntax() else {
        panic!("field-physical Port retained");
    };
    assert_eq!(connector.as_str(), "MechanicalBoundary");
    assert_eq!(support, "wall");
    let Item::Relation(relation) = &document.models()[0].items()[4] else {
        panic!("fifth item is the Relation");
    };
    assert!(matches!(
        relation.conditions().unwrap()[0].left().kind(),
        ExprKind::Path(path) if path.as_str() == "interface.traction"
    ));
}

#[test]
fn shaped_initial_equations_remain_explicit_for_shared_type_validation() {
    let source = "model M() { state velocity: array<m / s, 2>; initial { velocity = 1[m / s]; } }";
    let document = parse("shaped-field.eqi", source)
        .into_document()
        .expect("syntax preserves the equation; compiler validates shape");
    let Item::Field(field) = &document.models()[0].items()[0] else {
        panic!("field");
    };
    assert_eq!(field.role(), crate::FieldRoleSyntax::State);
    let Item::Initial(initial) = &document.models()[0].items()[1] else {
        panic!("initial equations");
    };
    assert!(matches!(
        initial.equations()[0].right().kind(),
        ExprKind::Quantity { value, .. } if value.canonical_text() == "1"
    ));
}

#[test]
fn field_physical_connector_fields_are_closed_and_exactly_once() {
    let invalid_sources = [
        (
            "duplicate",
            "connector C {\n  trace u: 1;\n  flux f: 1;\n  shape [];\n  frame invariant;\n  trace v: 1;\n  pairing euclidean_boundary_duality;\n  orientation parent_outward;\n}",
        ),
        (
            "missing",
            "connector C {\n  trace u: 1;\n  flux f: 1;\n  shape [];\n  frame invariant;\n  orientation parent_outward;\n}",
        ),
        (
            "unknown",
            "connector C { trace u: 1; flux f: 1; shape []; frame invariant; pairing euclidean_boundary_duality; orientation parent_outward; channels 2; }",
        ),
    ];

    for (case, source) in invalid_sources {
        let result = parse(format!("{case}.eqi"), source);
        assert!(!result.diagnostics().is_empty(), "{case} must fail closed");
        assert!(result.into_document().is_err(), "{case} cannot compile");
    }
}

#[test]
fn parser_retains_closed_complete_exterior_family_syntax() {
    let source = r#"
component BoundaryLaw(
  support body: volume(ambient_dimension = 2),
  support exterior: complete_exterior(parent = body), port mechanical[boundary in exterior]: MechanicalBoundary over boundary) {

  relation natural[boundary in exterior] on boundary {
mechanical[boundary = boundary].traction = 0;
  }
  connect [boundary in exterior] child.mechanical[boundary = boundary], mechanical[boundary = boundary];
}

model coupled() {
  instance law: BoundaryLaw(
body = fluid,
exterior = boundaries(x_lower, x_upper, y_lower, y_upper)
  );
  connect law.mechanical[boundary = x_lower], environment;
}
"#;
    let document = parse("complete-exterior.eqi", source)
        .into_document()
        .expect("restricted boundary-family syntax parses");
    let component = &document.components()[0];

    let crate::SignatureItem::Support(exterior) = &component.signature()[1] else {
        panic!("second component member is the complete exterior");
    };
    assert!(matches!(
        exterior.syntax(),
        SupportSlotSyntax::CompleteExterior { parent } if parent == "body"
    ));

    let crate::SignatureItem::PortFamily(port) = &component.signature()[2] else {
        panic!("third component member is a Port family");
    };
    assert_eq!(port.binder().member(), "boundary");
    assert_eq!(port.binder().set().as_str(), "exterior");

    let ComponentItem::RelationFamily(relation) = &component.items()[0] else {
        panic!("fourth component member is a Relation family");
    };
    assert_eq!(relation.relation().domain(), Some("boundary"));
    let ExprKind::Member { value, member } =
        relation.relation().conditions().unwrap()[0].left().kind()
    else {
        panic!("family Relation equation reads a named Port quantity");
    };
    assert_eq!(member, "traction");
    assert!(matches!(
        value.kind(),
        ExprKind::BoundaryPortSelection { port, selector }
            if port.as_str() == "mechanical"
                && selector.member() == "boundary" && selector.target() == "boundary"
    ));

    let ComponentItem::BoundaryConnection(connection) = &component.items()[1] else {
        panic!("fifth component member is a pointwise conserving Connection");
    };
    assert_eq!(
        connection.binder().expect("binder").set().as_str(),
        "exterior"
    );

    let Item::Instance(instance) = &document.models()[0].items()[0] else {
        panic!("first model member is the component instance");
    };
    assert_eq!(instance.bindings()[0].name(), "body");
    assert!(matches!(instance.bindings()[0].value().kind(), ExprKind::Name(n) if n == "fluid"));
    let ExprKind::Call { arguments, .. } = instance.bindings()[1].value().kind() else {
        panic!("boundaries call")
    };
    assert_eq!(
        arguments
            .expressions()
            .map(|a| match a.kind() {
                ExprKind::Name(n) => n.as_str(),
                _ => panic!("boundary name"),
            })
            .collect::<Vec<_>>(),
        ["x_lower", "x_upper", "y_lower", "y_upper"]
    );

    let Item::BoundaryConnection(connection) = &document.models()[0].items()[1] else {
        panic!("second model member is a selected conserving Connection");
    };
    assert!(connection.binder().is_none());
    assert_eq!(
        connection.ports()[0].selector().expect("selector").target(),
        "x_lower"
    );
}

#[test]
fn parser_retains_exact_model_spatial_periodic_pairs() {
    let source = r#"
model periodic() {
  connect periodic upper.interface, lower.interface;
}
"#;
    let document = parse("periodic.eqi", source)
        .into_document()
        .expect("closed Model spatial-periodic pair parses");
    let Item::BoundaryConnection(connection) = &document.models()[0].items()[0] else {
        panic!("periodic syntax is retained as a boundary Connection");
    };
    assert_eq!(connection.syntax(), ConnectionSyntax::SpatialPeriodic);
    assert_eq!(connection.ports().len(), 2);
    assert!(connection.binder().is_none());

    let component = parse(
        "component-periodic.eqi",
        "component C() { connect periodic upper, lower; }",
    );
    assert!(component.diagnostics().iter().any(|diagnostic| {
        diagnostic
            .message()
            .contains("allowed only in closed Models")
    }));
}

#[test]
fn parser_leaves_relation_and_connection_family_set_kinds_to_the_compiler() {
    for source in [
        "component C() { clock c = periodic(1[s]); relation r[b in exterior] on b at c { 1 = 0; } }",
        "model M() { relation r[b in exterior] on b { 1 = 0; } }",
        "model M() { connect [b in exterior] left, right; }",
    ] {
        // An identifier alone cannot distinguish an IndexSet from a boundary
        // set. The compiler owns the distinct activation and selector rules.
        assert!(parse("family.eqi", source).into_document().is_ok());
    }
    assert!(
        parse(
            "signal.eqi",
            "component C(port p[b in exterior]: signal input 1) {}"
        )
        .into_document()
        .is_err()
    );
}

#[test]
fn parser_and_formatter_retain_ordered_dimension_prefix_with_exact_ranges() {
    let source = "dimension Speed = m / s;\ndimension Acceleration = Speed / s;\nmodel M() { variable velocity: Speed; }";
    let document = parse("dimensions.eqi", source)
        .into_document()
        .expect("dimension prefix parses");

    assert_eq!(document.dimensions().len(), 2);
    let declaration = document.dimensions().first().expect("first alias");
    let (name, range) = (declaration.name(), declaration.range());
    assert_eq!(name, "Speed");
    assert_eq!(
        &source[range.start() as usize..range.end() as usize],
        "dimension Speed = m / s;"
    );
    let formatted = crate::format(&document);
    let reparsed = parse("dimensions.eqi", &formatted)
        .into_document()
        .expect("formatted prefix reparses");
    assert_eq!(crate::format(&reparsed), formatted);

    let later = parse(
        "later.eqi",
        "model M() { variable x: Length; } public dimension Length = m;",
    )
    .into_document()
    .expect("complete scope permits later dimensions");
    assert!(crate::format(&later).contains("public dimension Length = m;"));
}
