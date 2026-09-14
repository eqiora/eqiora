use crate::{BinaryOp, Expr, ExprKind, TextRange, UnaryOp, parse};

use super::*;

#[test]
fn factory_negative_literal_power_bases_are_grouped() {
    for kind in [
        ExprKind::Number(crate::DecimalLiteral::parse("-2.0").expect("exact literal")),
        ExprKind::Quantity {
            value: crate::DecimalLiteral::from_f64(-2.0).unwrap(),
            unit: Box::new(Expr {
                resolved_enum: None,
                resolved_nominal: None,
                kind: ExprKind::Name("m".into()),
                range: TextRange::new(0, 0),
            }),
        },
    ] {
        let expression = Expr {
            resolved_enum: None,
            resolved_nominal: None,
            kind: ExprKind::Binary {
                op: BinaryOp::Pow,
                left: Box::new(Expr {
                    resolved_enum: None,
                    resolved_nominal: None,
                    kind,
                    range: TextRange::new(0, 0),
                }),
                right: Box::new(Expr {
                    resolved_enum: None,
                    resolved_nominal: None,
                    kind: ExprKind::Number(
                        crate::DecimalLiteral::parse("2.0").expect("exact literal"),
                    ),
                    range: TextRange::new(0, 0),
                }),
            },
            range: TextRange::new(0, 0),
        };
        let mut output = comments::Output::default();
        format_expression(&expression, 0, &mut output);
        let text = output.finish();
        assert!(text.starts_with("(-2"), "{text}");
        let source = format!("model M() {{ relation r {{ {text} = 0; }} }}");
        let document = parse("factory.eqi", &source).into_document().unwrap();
        let Item::Relation(relation) = &document.models()[0].items()[0] else {
            panic!("relation")
        };
        let ExprKind::Binary {
            op: BinaryOp::Pow,
            left,
            ..
        } = relation.conditions().unwrap()[0].left().kind()
        else {
            panic!("power")
        };
        assert!(matches!(
            left.kind(),
            ExprKind::Unary {
                op: UnaryOp::Neg,
                ..
            }
        ));
    }
}

#[test]
fn signed_parameter_quantities_round_trip_with_exact_unit_powers() {
    let source = "model M() { parameter value: m ^ (1 / 2) = -4[(mm ^ 2) ^ (1 / 4)]; }";
    let document = parse("quantity.eqi", source).into_document().unwrap();
    let formatted = format(&document);
    assert!(formatted.contains("-4 [(mm ^ 2) ^ (1 / 4)]"));
    let reparsed = parse("quantity.eqi", &formatted).into_document().unwrap();
    assert_eq!(format(&reparsed), formatted);
    for value in ["1 + 2", "-1[ms] + 2"] {
        let source = format!("model M() {{ parameter value: m = {value}; }}");
        let parsed = parse("expression.eqi", &source).into_document().unwrap();
        let formatted = format(&parsed);
        let reparsed = parse("expression.eqi", &formatted).into_document().unwrap();
        assert_eq!(format(&reparsed), formatted);
    }
    for value in ["1[]", "1[m", "1[m ^ (1 / 0)]"] {
        let source = format!("model M() {{ parameter value: m = {value}; }}");
        assert!(
            parse("invalid.eqi", &source).into_document().is_err(),
            "{source}"
        );
    }
}

#[test]
fn canonical_format_is_idempotent() {
    let source = "model m() {state x: 1;initial{x=0;}relation r{derivative(x)-(1+x*2)=0;}}";
    let first = parse("m.eqi", source).into_document().expect("parse");
    let formatted = format(&first);
    let second = parse("m.eqi", &formatted)
        .into_document()
        .expect("formatted parse");

    assert_eq!(format(&second), formatted);
    assert!(formatted.contains("derivative(x) - (1 + x * 2) = 0;"));
}

#[test]
fn pure_operators_format_before_consumers_with_canonical_exact_integers() {
    let source = r#"component Consumer() {}
public operator dyadic(input left: spatial[01], input right: scalar): spatial[2] = component(left, 00) * component(right) + rational(03, 04) * delta(0, 01);
model M() {
  variable u: 1;
  relation law { catalog.dyadic(left = u, right = u) = 0; }
}"#;
    let document = parse("pure-format.eqi", source)
        .into_document()
        .expect("pure operator source");
    let formatted = format(&document);
    let reparsed = parse("pure-format.eqi", &formatted)
        .into_document()
        .expect("canonical pure operator source");

    assert_eq!(format(&reparsed), formatted);
    assert!(
        formatted.find("operator").expect("operator")
            < formatted.find("component Consumer").expect("component")
    );
    assert!(formatted.contains(
        "dyadic(input left: spatial[1], input right: scalar): spatial[2] = component(left, 0) * component(right) + rational(3, 4) * delta(0, 1);"
    ));
    assert!(formatted.contains("catalog.dyadic(left = u, right = u)"));
}

#[test]
fn spatial_source_roundtrips_with_explicit_scope() {
    let source = r#"
model bar() {
  domain body = box(0, 2);
  domain fixed = boundary(body, axis = 0, side = lower);
  variable u: m on body;
  relation clamp on fixed { trace(u) = 0; }
}
"#;
    let first = parse("bar.eqi", source).into_document().expect("parse");
    let formatted = format(&first);
    let second = parse("bar.eqi", &formatted)
        .into_document()
        .expect("formatted parse");

    assert_eq!(format(&second), formatted);
    assert!(formatted.contains("relation clamp on fixed"));
}

#[test]
fn coordinate_math_roundtrips_without_losing_structure() {
    let source = r#"
model poisson() {
  domain interval = box(0, 1);
  variable u: 1 on interval;
  parameter wave_number: 1 / m = 3.141592653589793;
  relation balance on interval {
-div(grad(u)) - math.sin(wave_number * coordinate(0)) = 0;
  }
}
"#;
    let first = parse("poisson.eqi", source).into_document().expect("parse");
    let formatted = format(&first);
    let second = parse("poisson.eqi", &formatted)
        .into_document()
        .expect("formatted parse");

    assert_eq!(format(&second), formatted);
    assert!(formatted.contains("math.sin(wave_number * coordinate(0))"));
}

#[test]
fn scalar_physical_source_roundtrips_with_nominal_domain() {
    let source = r#"
model circuit() {
  domain electrical = scalar_physical(across voltage: kg * m ^ 2 / (s ^ 3 * A), through current: A);
  port positive: electrical;
  relation source { positive.voltage = 0; }
}
"#;
    let first = parse("circuit.eqi", source)
        .into_document()
        .expect("physical source parses");
    let formatted = format(&first);
    let second = parse("circuit.eqi", &formatted)
        .into_document()
        .expect("formatted physical source parses");

    assert_eq!(format(&second), formatted);
    assert!(formatted.contains(
        "domain electrical = scalar_physical(across voltage: kg * m ^ 2 / (s ^ 3 * A), through current: A);"
    ));
    assert!(formatted.contains("port positive: electrical;"));
}

#[test]
fn component_source_roundtrips_with_canonical_qualified_names() {
    let source = r#"
connector Pin {
  across voltage: kg*m^2/(s^3*A);
  through current: A;
}
component Resistor(parameter resistance:kg*m^2/(s^3*A^2), port positive:Pin, port negative:Pin) {



relation law{positive.voltage-negative.voltage-resistance*positive.current=0;}
}
component Pair(parameter resistance:kg*m^2/(s^3*A^2)=2, port positive:Pin) {


instance inner:Library.Resistor(resistance=resistance);
connect positive,inner.positive;
}
model circuit() {
instance r2:Pair(resistance=2);
instance r4:Pair(resistance=4);
connect r2.positive,r4.positive;
}
"#;
    let first = parse("component.eqi", source)
        .into_document()
        .expect("component source parses");
    let formatted = format(&first);
    let second = parse("component.eqi", &formatted)
        .into_document()
        .expect("formatted component source parses");
    assert_eq!(format(&second), formatted);
    assert!(formatted.contains("connector Pin {"));
    assert!(formatted.contains("instance inner: Library.Resistor(resistance = resistance);"));
    assert!(formatted.contains("connect r2.positive, r4.positive;"));
}

#[test]
fn component_support_source_roundtrips_with_discriminated_bindings() {
    let source = r#"component BoundaryState(
  support body:volume(ambient_dimension=2),
  support interface:boundary(parent=body)
) {
variable state: 1 on body;
}
model coupled() {
domain fluid=box(0,1,0,1);
domain wall=boundary(fluid,axis=0,side=lower);
instance probe:BoundaryState(interface =wall,gain=2,body =fluid);
}"#;
    let first = parse("supports.eqi", source)
        .into_document()
        .expect("support source parses");
    let formatted = format(&first);
    let second = parse("supports.eqi", &formatted)
        .into_document()
        .expect("formatted support source parses");

    assert_eq!(format(&second), formatted);
    assert!(formatted.contains("support body: volume(ambient_dimension = 2),"));
    assert!(formatted.contains("support interface: boundary(parent = body),"));
    assert!(
        formatted
            .contains("instance probe: BoundaryState(interface = wall, gain = 2, body = fluid);")
    );
}

#[test]
fn package_visibility_roundtrips_in_canonical_source() {
    let source = r#"private connector Hidden {
  across voltage: 1;
  through current: A;
}
public connector Pin {
  across voltage: 1;
  through current: A;
}
private component Internal() {}
public component Resistor() {}"#;
    let first = parse("library.eqi", source)
        .into_document()
        .expect("declaration-only source parses");
    let formatted = format(&first);
    let second = parse("relocated/library.eqi", &formatted)
        .into_document()
        .expect("canonical declaration-only source parses");

    assert_eq!(format(&second), formatted);
    assert!(formatted.contains("connector Hidden {"));
    assert!(!formatted.contains("private connector"));
    assert!(formatted.contains("public connector Pin {"));
    assert!(formatted.contains("component Internal() {\n}"));
    assert!(!formatted.contains("private component"));
    assert!(formatted.contains("public component Resistor() {\n}"));
    assert_eq!(
        first.connectors()[0].visibility(),
        second.connectors()[0].visibility()
    );
    assert_eq!(
        first.components()[1].visibility(),
        second.components()[1].visibility()
    );
}

#[test]
fn field_physical_source_has_one_fixed_readable_order() {
    let source = r#"public connector MechanicalBoundary {
  pairing euclidean_boundary_duality;
  frame spatial;
  shape [2];
  flux traction: kg/(m*s^2);
  trace velocity: m/s;
  orientation parent_outward;
}
component Wall(
  support wall:boundary(parent=body),
  support body:volume(ambient_dimension=2), port interface:MechanicalBoundary over wall) {

variable velocity: vector<m/s, 2>;
relation load { interface.traction=0; }
}"#;
    let first = parse("boundary.eqi", source)
        .into_document()
        .expect("field-physical source parses");
    let formatted = format(&first);
    let second = parse("boundary.eqi", &formatted)
        .into_document()
        .expect("canonical field-physical source parses");

    assert_eq!(format(&second), formatted);
    assert!(formatted.contains(
        "connector MechanicalBoundary {\n  trace velocity: m / s;\n  flux traction: kg / (m * s ^ 2);\n  shape [2];\n  frame spatial;\n  pairing euclidean_boundary_duality;\n  orientation parent_outward;\n}"
    ));
    assert!(formatted.contains("port interface: MechanicalBoundary over wall,"));
    assert!(formatted.contains("variable velocity: vector<m / s, 2>;"));
    assert!(formatted.contains("interface.traction = 0;"));
}

#[test]
fn complete_exterior_families_have_one_closed_canonical_spelling() {
    let source = r#"component BoundaryLaw(
  support body:volume(ambient_dimension=2),
  support exterior:complete_exterior(parent=body), port mechanical[boundary in exterior]:MechanicalBoundary over boundary) {

relation natural[boundary in exterior] on boundary{mechanical[boundary=boundary].traction=0;}
connect[boundary in exterior] child.mechanical[boundary=boundary],mechanical[boundary=boundary];
}
model coupled() {
instance law:BoundaryLaw(body =fluid,exterior =boundaries(x_lower,x_upper,y_lower,y_upper));
connect law.mechanical[boundary=x_lower],environment;
}"#;
    let first = parse("complete-exterior.eqi", source)
        .into_document()
        .expect("boundary-family source parses");
    let formatted = format(&first);
    let second = parse("complete-exterior.eqi", &formatted)
        .into_document()
        .expect("canonical boundary-family source parses");

    assert_eq!(format(&second), formatted);
    assert!(formatted.contains("support exterior: complete_exterior(parent = body),"));
    assert!(
        formatted
            .contains("port mechanical[boundary in exterior]: MechanicalBoundary over boundary,")
    );
    assert!(formatted.contains("relation natural[boundary in exterior] on boundary {"));
    assert!(formatted.contains(
        "connect [boundary in exterior] child.mechanical[boundary = boundary], mechanical[boundary = boundary];"
    ));
    assert!(formatted.contains("exterior = boundaries(x_lower, x_upper, y_lower, y_upper)"));
    assert!(formatted.contains("connect law.mechanical[boundary = x_lower], environment;"));
}

#[test]
fn spatial_periodic_pair_has_one_closed_canonical_spelling() {
    let source = "model M() {connect periodic upper.interface,lower.interface;}";
    let first = parse("periodic.eqi", source)
        .into_document()
        .expect("spatial-periodic source parses");
    let formatted = format(&first);
    let second = parse("periodic.eqi", &formatted)
        .into_document()
        .expect("canonical spatial-periodic source parses");

    assert_eq!(format(&second), formatted);
    assert!(formatted.contains("connect periodic upper.interface, lower.interface;"));
}

#[test]
fn legacy_support_relations_and_connections_keep_their_canonical_spelling() {
    let source = r#"component Legacy(
  support body:volume(ambient_dimension=2),
  support wall:boundary(parent=body), port interface:MechanicalBoundary over wall) {

relation law on wall{interface.traction=0;}
connect interface,child.interface;
}
model use_legacy() {
instance legacy:Legacy(body =fluid,wall =fixed);
connect legacy.interface,environment;
}"#;
    let first = parse("legacy.eqi", source)
        .into_document()
        .expect("legacy source parses");
    let formatted = format(&first);
    let second = parse("legacy.eqi", &formatted)
        .into_document()
        .expect("canonical legacy source parses");

    assert_eq!(format(&second), formatted);
    assert!(formatted.contains("support wall: boundary(parent = body),"));
    assert!(formatted.contains("port interface: MechanicalBoundary over wall,"));
    assert!(formatted.contains("relation law on wall {"));
    assert!(formatted.contains("connect interface, child.interface;"));
    assert!(!formatted.contains("boundaries("));
    assert!(!formatted.contains("[boundary"));
}
