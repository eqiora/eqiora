use super::*;
use crate::BoundarySideSyntax;
use crate::{
    ConnectionSyntax, DomainDecl, DomainSyntax, FieldRoleSyntax, Item, PortDecl, PortSyntax,
};

#[test]
fn shared_expression_depth_rejects_before_copying_and_never_becomes_a_literal() {
    let mut expression = DraftExpression::constant(crate::DecimalLiteral::parse("1").unwrap());
    for _ in 1..crate::SourceAstFactory::MAX_EXPRESSION_DEPTH {
        expression = -expression;
    }
    assert_eq!(
        expression.depth,
        crate::SourceAstFactory::MAX_EXPRESSION_DEPTH
    );
    assert!(expression.source_ast(|_| None, |_| None).is_ok());
    let rejected = -expression;
    assert!(rejected.source_ast(|_| None, |_| None).is_err());
    let errors = Module::new(
        "M",
        [DraftRelation::continuous("law", [(rejected.clone(), rejected)]).into()],
    )
    .unwrap_err();
    assert!(
        errors
            .iter()
            .any(|error| error.message().contains("depth or node limit"))
    );
}

#[test]
fn module_bounds_the_total_before_projecting_shared_equation_subtrees() {
    let mut expression = DraftExpression::constant(crate::DecimalLiteral::parse("1").unwrap());
    for _ in 0..9 {
        expression = expression.clone() + expression;
    }
    assert_eq!(expression.nodes, 1023);
    let equation_count = crate::SourceAstFactory::MAX_EXPRESSION_NODES / (2 * expression.nodes) + 1;
    let relation = DraftRelation::continuous(
        "law",
        std::iter::repeat_n((expression.clone(), expression), equation_count),
    );
    let errors = Module::new("M", [relation.into()]).unwrap_err();
    assert!(errors[0].message().contains("shared expression node limit"));
}

#[test]
fn literal_projection_counts_prefixes_before_nested_array_allocation() {
    let integer = eqiora_core::ValueType::scalar(
        eqiora_core::ScalarDomain::Integer,
        DimExponents::DIMENSIONLESS,
    )
    .unwrap();
    let mut inner = integer;
    for _ in 0..26 {
        inner = inner.array(1).unwrap();
    }
    // One outer array plus 27 nodes for each member (26 singleton arrays
    // and one integer leaf): 1 + 37_037 * 27 = 1_000_000 exactly.
    let value = ValueLiteral::integer(
        inner.clone().array(37_037).unwrap(),
        std::iter::repeat_n(1, 37_037),
    )
    .unwrap();
    assert_eq!(
        crate::SourceAstFactory::value_literal_nodes(&value, false).unwrap(),
        1_000_000
    );
    let excessive =
        ValueLiteral::integer(inner.array(37_038).unwrap(), std::iter::repeat_n(1, 37_038))
            .unwrap();
    assert!(
        crate::SourceAstFactory::value_literal_nodes(&excessive, false)
            .unwrap_err()
            .message()
            .contains("node limit")
    );
    assert!(
        crate::SourceAstFactory::value_literal(
            &excessive,
            None,
            TextRange::new(0, 1),
            |_| None,
            |_| None
        )
        .is_err()
    );
    // A second scalar leaf exceeds the module's aggregate bound without
    // constructing the million-node first initializer at all.
    let first = DraftParameter::new("values", value);
    let second = DraftParameter::new(
        "extra",
        ValueLiteral::from_integer(
            eqiora_core::ValueType::scalar(
                eqiora_core::ScalarDomain::Integer,
                DimExponents::DIMENSIONLESS,
            )
            .unwrap(),
            1,
        )
        .unwrap(),
    );
    assert!(
        Module::new("M", [first.into(), second.into()]).unwrap_err()[0]
            .message()
            .contains("node limit")
    );
}

fn voltage_dimension() -> DimExponents {
    DimExponents::from_integers([1, 2, -3, -1, 0, 0, 0]).expect("bounded dimension")
}

fn current_dimension() -> DimExponents {
    DimExponents::from_integers([0, 0, 0, 1, 0, 0, 0]).expect("bounded dimension")
}

#[test]
fn native_draft_rejects_foreign_symbol_even_when_name_matches() {
    let included = DraftField::new(
        "x",
        eqiora_core::ValueType::scalar(
            eqiora_core::ScalarDomain::Real,
            DimExponents::DIMENSIONLESS,
        )
        .unwrap(),
        FieldRoleSyntax::Variable,
    );
    let foreign = DraftField::new(
        "x",
        eqiora_core::ValueType::scalar(
            eqiora_core::ScalarDomain::Real,
            DimExponents::DIMENSIONLESS,
        )
        .unwrap(),
        FieldRoleSyntax::Variable,
    );
    let relation = DraftRelation::continuous(
        "flow",
        [(
            foreign.expression(),
            DraftExpression::constant(crate::DecimalLiteral::parse("0").unwrap()),
        )],
    );

    let diagnostic = Module::new("decay", [included.into(), relation.into()]).unwrap_err();
    assert_eq!(diagnostic[0].code(), codes::LANGUAGE_TYPE_ERROR);
    assert_eq!(
        diagnostic[0].graph_path().unwrap().to_string(),
        "decay.flow"
    );
}

#[test]
fn typed_dimensions_and_expression_references_become_source_ast() {
    let state = DraftField::new(
        "x",
        eqiora_core::ValueType::scalar(
            eqiora_core::ScalarDomain::Real,
            DimExponents::DIMENSIONLESS,
        )
        .unwrap(),
        FieldRoleSyntax::State,
    );
    let rate = DraftParameter::new(
        "rate",
        eqiora_core::ValueLiteral::from_real(
            eqiora_core::ValueType::scalar(
                eqiora_core::ScalarDomain::Real,
                DimExponents::from_integers([0, 0, -1, 0, 0, 0, 0]).expect("bounded dimension"),
            )
            .unwrap(),
            1.0,
        )
        .unwrap(),
    );
    let initial = DraftDeclaration::Initial(vec![(
        state.expression(),
        DraftExpression::constant(crate::DecimalLiteral::parse("1.0").unwrap()),
    )]);
    let residual = DraftExpression::derivative(&state) + rate.expression() * state.expression();
    let draft = Module::new(
        "decay",
        [
            state.into(),
            rate.into(),
            initial,
            DraftRelation::continuous(
                "flow",
                [(
                    residual,
                    DraftExpression::constant(crate::DecimalLiteral::parse("0").unwrap()),
                )],
            )
            .into(),
        ],
    )
    .unwrap();

    let native = draft;
    assert_eq!(native.model().name(), "decay");
    assert_eq!(native.model().items().len(), 4);
    assert!(native.graph_path(native.model().range()).is_some());
    let Item::Field(field) = &native.model().items()[0] else {
        panic!("state");
    };
    assert_eq!(field.role(), FieldRoleSyntax::State);
    let Item::Parameter(parameter) = &native.model().items()[1] else {
        panic!("parameter");
    };
    assert!(matches!(
        parameter.value().kind(),
        ExprKind::Quantity { value, .. } if value.canonical_text() == "1"
    ));
    assert!(matches!(native.model().items()[2], Item::Initial(_)));
}

#[test]
fn native_draft_rejects_names_and_numbers_source_could_not_express() {
    let field = DraftField::new(
        "not valid",
        eqiora_core::ValueType::scalar(
            eqiora_core::ScalarDomain::Real,
            DimExponents::DIMENSIONLESS,
        )
        .unwrap(),
        FieldRoleSyntax::Variable,
    );
    let relation = DraftRelation::continuous(
        "flow",
        [(
            field.expression() + DraftExpression::complex(f64::NAN, 0.0),
            DraftExpression::constant(crate::DecimalLiteral::parse("0").unwrap()),
        )],
    );

    let diagnostics = Module::new(
        "",
        [
            field.into(),
            relation.into(),
            DraftDeclaration::Initial(vec![(
                DraftExpression::complex(f64::INFINITY, 0.0),
                DraftExpression::constant(crate::DecimalLiteral::parse("0").unwrap()),
            )]),
        ],
    )
    .unwrap_err();
    assert!(
        diagnostics
            .iter()
            .any(|diagnostic| diagnostic.message().contains("model name"))
    );
    assert!(
        diagnostics
            .iter()
            .any(|diagnostic| diagnostic.message().contains("declaration name"))
    );
    assert!(
        diagnostics
            .iter()
            .any(|diagnostic| diagnostic.message().contains("non-finite"))
    );
    assert!(
        diagnostics
            .iter()
            .any(|diagnostic| diagnostic.message().contains("numeric literal"))
    );
}

#[test]
fn physical_vocabulary_projects_only_to_existing_source_ast_forms() {
    let electrical = DraftPhysicalDomain::new(
        "electrical",
        "voltage",
        eqiora_core::ValueType::scalar(eqiora_core::ScalarDomain::Real, voltage_dimension())
            .unwrap(),
        "current",
        eqiora_core::ValueType::scalar(eqiora_core::ScalarDomain::Real, current_dimension())
            .unwrap(),
    );
    let positive = DraftConservingPort::new("positive", &electrical);
    let negative = DraftConservingPort::new("negative", &electrical);
    let resistance = DraftParameter::new(
        "resistance",
        eqiora_core::ValueLiteral::from_real(
            eqiora_core::ValueType::scalar(
                eqiora_core::ScalarDomain::Real,
                DimExponents::from_integers([1, 2, -3, -2, 0, 0, 0]).expect("bounded dimension"),
            )
            .unwrap(),
            2.0,
        )
        .unwrap(),
    );
    let relation = DraftRelation::continuous(
        "resistor",
        [
            (
                DraftExpression::across(&positive)
                    - DraftExpression::across(&negative)
                    - resistance.expression() * DraftExpression::through(&positive),
                DraftExpression::constant(crate::DecimalLiteral::parse("0").unwrap()),
            ),
            (
                DraftExpression::through(&positive) + DraftExpression::through(&negative),
                DraftExpression::constant(crate::DecimalLiteral::parse("0").unwrap()),
            ),
        ],
    );
    let connection = DraftConservingConnection::new([&positive, &negative]);
    let draft = Module::new(
        "resistor",
        [
            electrical.into(),
            positive.into(),
            negative.into(),
            resistance.into(),
            relation.into(),
            connection.into(),
        ],
    )
    .unwrap();

    let native = draft;
    let items = native.model().items();
    assert!(matches!(
        items[0],
        Item::Domain(DomainDecl {
            syntax: DomainSyntax::ScalarPhysical { .. },
            ..
        })
    ));
    assert!(matches!(
        items[1],
        Item::Port(PortDecl {
            syntax: PortSyntax::ScalarPhysical { .. },
            ..
        })
    ));
    let Item::Relation(relation) = &items[4] else {
        panic!("fifth item must be a Relation");
    };
    assert_eq!(relation.conditions().unwrap().len(), 2);
    let mut quantities = std::collections::BTreeSet::new();
    for equation in relation.conditions().unwrap() {
        let _ = equation.left().rewrite_name_paths(|path| {
            quantities.insert(path.as_str().to_owned());
            None
        });
    }
    for quantity in [
        "positive.voltage",
        "negative.voltage",
        "positive.current",
        "negative.current",
    ] {
        assert!(quantities.contains(quantity));
    }
    let Item::Connection(connection) = &items[5] else {
        panic!("sixth item must be a Connection");
    };
    assert_eq!(connection.syntax(), ConnectionSyntax::Conserving);
    assert_eq!(connection.port_expressions().len(), 2);
    assert!(
        matches!(connection.port_expressions()[0].kind(), ExprKind::Name(name) if name == "positive")
    );
    assert!(
        matches!(connection.port_expressions()[1].kind(), ExprKind::Name(name) if name == "negative")
    );
    assert!(native.graph_path(connection.range()).is_some());
}

#[test]
fn draft_closure_rejects_foreign_domain_and_port_identity_before_rebinding_names() {
    let declared_domain = DraftPhysicalDomain::new(
        "electrical",
        "voltage",
        eqiora_core::ValueType::scalar(eqiora_core::ScalarDomain::Real, voltage_dimension())
            .unwrap(),
        "current",
        eqiora_core::ValueType::scalar(eqiora_core::ScalarDomain::Real, current_dimension())
            .unwrap(),
    );
    let foreign_domain = DraftPhysicalDomain::new(
        "electrical",
        "voltage",
        eqiora_core::ValueType::scalar(eqiora_core::ScalarDomain::Real, voltage_dimension())
            .unwrap(),
        "current",
        eqiora_core::ValueType::scalar(eqiora_core::ScalarDomain::Real, current_dimension())
            .unwrap(),
    );
    let declared_port = DraftConservingPort::new("terminal", &declared_domain);
    let foreign_domain_port = DraftConservingPort::new("foreign_domain", &foreign_domain);
    let foreign_port = DraftConservingPort::new("terminal", &declared_domain);
    let relation = DraftRelation::continuous(
        "owner",
        [(
            DraftExpression::across(&foreign_port),
            DraftExpression::constant(crate::DecimalLiteral::parse("0").unwrap()),
        )],
    );

    let diagnostics = Module::new(
        "identity",
        [
            declared_domain.into(),
            declared_port.into(),
            foreign_domain_port.into(),
            relation.into(),
        ],
    )
    .unwrap_err();
    assert!(diagnostics.iter().any(|diagnostic| {
        diagnostic
            .message()
            .contains("foreign or omitted scalar physical Domain")
    }));
    assert!(diagnostics.iter().any(|diagnostic| {
        diagnostic
            .message()
            .contains("foreign or omitted conserving Port `terminal`")
    }));
    assert!(
        diagnostics
            .iter()
            .all(|diagnostic| diagnostic.source_span().is_none())
    );
}

#[test]
fn draft_closure_rejects_invalid_connection_membership_atomically() {
    let electrical = DraftPhysicalDomain::new(
        "electrical",
        "voltage",
        eqiora_core::ValueType::scalar(eqiora_core::ScalarDomain::Real, voltage_dimension())
            .unwrap(),
        "current",
        eqiora_core::ValueType::scalar(eqiora_core::ScalarDomain::Real, current_dimension())
            .unwrap(),
    );
    let other = DraftPhysicalDomain::new(
        "other",
        "voltage",
        eqiora_core::ValueType::scalar(eqiora_core::ScalarDomain::Real, voltage_dimension())
            .unwrap(),
        "current",
        eqiora_core::ValueType::scalar(eqiora_core::ScalarDomain::Real, current_dimension())
            .unwrap(),
    );
    let a = DraftConservingPort::new("a", &electrical);
    let b = DraftConservingPort::new("b", &electrical);
    let incompatible = DraftConservingPort::new("incompatible", &other);
    let foreign = DraftConservingPort::new("a", &electrical);
    let diagnostics = Module::new(
        "invalid_connections",
        [
            electrical.into(),
            other.into(),
            a.clone().into(),
            b.clone().into(),
            incompatible.clone().into(),
            DraftConservingConnection::new([&a]).into(),
            DraftConservingConnection::new([&a, &b, &b]).into(),
            DraftConservingConnection::new([&incompatible, &b, &foreign]).into(),
        ],
    )
    .unwrap_err();

    assert!(
        diagnostics
            .iter()
            .any(|diagnostic| diagnostic.message().contains("at least two Ports"))
    );
    assert!(
        diagnostics
            .iter()
            .any(|diagnostic| diagnostic.message().contains("repeats Port `b`"))
    );
    assert!(diagnostics.iter().any(|diagnostic| {
        diagnostic
            .message()
            .contains("already belongs to another Connection")
    }));
    assert!(
        diagnostics
            .iter()
            .any(|diagnostic| diagnostic.message().contains("foreign or omitted Port `a`"))
    );
    assert!(diagnostics.iter().any(|diagnostic| {
        diagnostic
            .message()
            .contains("exact same draft-local scalar physical Domain")
    }));
}

#[test]
fn duplicate_names_are_rejected_across_physical_and_scalar_declarations() {
    let domain = DraftPhysicalDomain::new(
        "shared",
        "voltage",
        eqiora_core::ValueType::scalar(eqiora_core::ScalarDomain::Real, voltage_dimension())
            .unwrap(),
        "current",
        eqiora_core::ValueType::scalar(eqiora_core::ScalarDomain::Real, current_dimension())
            .unwrap(),
    );
    let field = DraftField::new(
        "shared",
        eqiora_core::ValueType::scalar(
            eqiora_core::ScalarDomain::Real,
            DimExponents::DIMENSIONLESS,
        )
        .unwrap(),
        FieldRoleSyntax::Variable,
    );
    let diagnostics = Module::new("duplicates", [domain.into(), field.into()]).unwrap_err();
    assert!(
        diagnostics[0]
            .message()
            .contains("duplicate declaration `shared`")
    );
}

#[test]
fn anonymous_connection_diagnostic_paths_follow_membership_not_declaration_position() {
    let domain = DraftPhysicalDomain::new(
        "electrical",
        "voltage",
        eqiora_core::ValueType::scalar(eqiora_core::ScalarDomain::Real, voltage_dimension())
            .unwrap(),
        "current",
        eqiora_core::ValueType::scalar(eqiora_core::ScalarDomain::Real, current_dimension())
            .unwrap(),
    );
    let terminal = DraftConservingPort::new("terminal", &domain);
    let unrelated = DraftField::new(
        "x",
        eqiora_core::ValueType::scalar(
            eqiora_core::ScalarDomain::Real,
            DimExponents::DIMENSIONLESS,
        )
        .unwrap(),
        FieldRoleSyntax::Variable,
    );
    let connection = DraftConservingConnection::new([&terminal]);
    let forward = Module::new(
        "stable_path",
        [
            domain.clone().into(),
            terminal.clone().into(),
            connection.clone().into(),
            unrelated.clone().into(),
        ],
    )
    .unwrap_err();
    let reordered = Module::new(
        "stable_path",
        [
            unrelated.into(),
            connection.into(),
            terminal.into(),
            domain.into(),
        ],
    )
    .unwrap_err();

    assert_eq!(
        forward[0].graph_path().unwrap(),
        reordered[0].graph_path().unwrap()
    );
    assert_eq!(
        forward[0].graph_path().unwrap().to_string(),
        "stable_path.connection[terminal]"
    );
}

#[test]
fn spatial_draft_retains_exact_scope_identity_before_ast_projection() {
    let included = DraftSpatialDomain::cartesian_box("interval", [(0.0, 1.0)]);
    let foreign = DraftSpatialDomain::cartesian_box("interval", [(0.0, 1.0)]);
    let field = DraftField::spatial(
        "u",
        &foreign,
        eqiora_core::ValueType::scalar(
            eqiora_core::ScalarDomain::Real,
            DimExponents::DIMENSIONLESS,
        )
        .unwrap(),
        FieldRoleSyntax::Variable,
    );
    let diagnostics = Module::new("foreign_scope", [included.into(), field.into()]).unwrap_err();
    assert!(diagnostics.iter().any(|diagnostic| {
        diagnostic
            .message()
            .contains("foreign or omitted Domain `interval`")
    }));
}

#[test]
fn spatial_draft_projects_only_to_existing_source_ast_forms() {
    let interval = DraftSpatialDomain::cartesian_box("interval", [(0.0, 1.0)]);
    let lower = DraftSpatialDomain::boundary("lower", &interval, 0, BoundarySideSyntax::Lower);
    let field = DraftField::spatial(
        "u",
        &interval,
        eqiora_core::ValueType::scalar(
            eqiora_core::ScalarDomain::Real,
            DimExponents::DIMENSIONLESS,
        )
        .unwrap(),
        FieldRoleSyntax::Variable,
    );
    let balance = DraftRelation::continuous_on(
        "balance",
        &interval,
        [(
            -DraftExpression::divergence(DraftExpression::gradient(field.expression())),
            DraftExpression::constant(crate::DecimalLiteral::parse("0").unwrap()),
        )],
    );
    let boundary = DraftRelation::continuous_on(
        "lower_value",
        &lower,
        [(
            DraftExpression::trace(field.expression()),
            DraftExpression::constant(crate::DecimalLiteral::parse("0").unwrap()),
        )],
    );
    let draft = Module::new(
        "poisson",
        [
            interval.into(),
            lower.into(),
            field.into(),
            balance.into(),
            boundary.into(),
        ],
    )
    .unwrap();

    let native = draft;
    assert!(matches!(
        native.model().items()[0],
        Item::Domain(DomainDecl {
            syntax: DomainSyntax::CartesianBox(_),
            ..
        })
    ));
    assert!(matches!(
        native.model().items()[1],
        Item::Domain(DomainDecl {
            syntax: DomainSyntax::Boundary { .. },
            ..
        })
    ));
    let Item::Field(field) = &native.model().items()[2] else {
        panic!("third item must be a Field");
    };
    assert_eq!(field.domain(), Some("interval"));
    assert_eq!(field.role(), FieldRoleSyntax::Variable);
    let Item::Relation(relation) = &native.model().items()[3] else {
        panic!("fourth item must be a Relation");
    };
    assert_eq!(relation.domain(), Some("interval"));
    assert!(expression_contains_call(
        relation.conditions().unwrap()[0].left(),
        "grad"
    ));
    assert!(expression_contains_call(
        relation.conditions().unwrap()[0].left(),
        "div"
    ));
}

fn expression_contains_call(expression: &Expr, expected: &str) -> bool {
    match expression.kind() {
        ExprKind::Partial { value, .. } => expression_contains_call(value, expected),
        ExprKind::Case { value, arms } => {
            expression_contains_call(value, expected)
                || arms
                    .iter()
                    .any(|arm| expression_contains_call(arm.value(), expected))
        }
        ExprKind::Select {
            condition,
            then_value,
            else_value,
        } => {
            expression_contains_call(condition, expected)
                || expression_contains_call(then_value, expected)
                || expression_contains_call(else_value, expected)
        }
        ExprKind::Call { callee, arguments } => {
            callee.as_str() == expected
                || arguments
                    .expressions()
                    .any(|argument| expression_contains_call(argument, expected))
        }
        ExprKind::Array(values) => values
            .iter()
            .any(|value| expression_contains_call(value, expected)),
        ExprKind::Index { value, index } => {
            expression_contains_call(value, expected) || expression_contains_call(index, expected)
        }
        ExprKind::Slice {
            value,
            lower,
            upper,
        } => [value.as_ref(), lower.as_ref(), upper.as_ref()]
            .into_iter()
            .any(|value| expression_contains_call(value, expected)),
        ExprKind::Unary { value, .. }
        | ExprKind::Member { value, .. }
        | ExprKind::Reduction { value, .. } => expression_contains_call(value, expected),
        ExprKind::Binary { left, right, .. } => {
            expression_contains_call(left, expected) || expression_contains_call(right, expected)
        }
        ExprKind::Boolean(_)
        | ExprKind::Number(_)
        | ExprKind::Quantity { .. }
        | ExprKind::Name(_)
        | ExprKind::Path(_)
        | ExprKind::BoundaryPortSelection { .. } => false,
    }
}

#[test]
fn draft_channel_literals_cannot_hide_empty_arrays_or_foreign_symbols() {
    let omitted = DraftField::new(
        "x",
        eqiora_core::ValueType::scalar(
            eqiora_core::ScalarDomain::Real,
            DimExponents::DIMENSIONLESS,
        )
        .unwrap(),
        FieldRoleSyntax::Variable,
    );
    let array = DraftExpression::array([omitted.expression()]).index(0);
    assert!(
        Module::new(
            "foreign",
            [DraftRelation::continuous(
                "law",
                [(
                    array,
                    DraftExpression::constant(crate::DecimalLiteral::parse("0").unwrap())
                )]
            )
            .into()]
        )
        .is_err()
    );
    assert!(
        Module::new(
            "empty",
            [DraftRelation::continuous(
                "law",
                [(
                    DraftExpression::array([]),
                    DraftExpression::constant(crate::DecimalLiteral::parse("0").unwrap())
                )]
            )
            .into()]
        )
        .is_err()
    );
}

#[test]
fn observables_require_exact_references_without_introducing_unknowns() {
    let kind =
        ValueType::scalar(eqiora_core::ScalarDomain::Real, DimExponents::DIMENSIONLESS).unwrap();
    let field = DraftField::new("x", kind.clone(), FieldRoleSyntax::Variable);
    let foreign = DraftField::new("x", kind.clone(), FieldRoleSyntax::Variable);
    let output = DraftObservable::new(
        "double",
        kind.clone(),
        field.expression() + field.expression(),
    );
    let module = Module::new("M", [field.clone().into(), output.into()]).unwrap();
    assert!(matches!(&module.model().items()[1], Item::Observable(_)));
    let rejected = DraftObservable::new("double", kind.clone(), foreign.expression());
    assert!(
        Module::new("M", [field.into(), rejected.into()])
            .unwrap_err()
            .iter()
            .any(|error| error.message().contains("foreign or omitted Field"))
    );
    let domain = DraftSpatialDomain::cartesian_box("body", [(0.0, 1.0)]);
    let other = DraftSpatialDomain::cartesian_box("body", [(0.0, 1.0)]);
    let output = DraftObservable::new(
        "total",
        kind,
        DraftExpression::integral(
            DraftExpression::constant(crate::DecimalLiteral::parse("1").unwrap()),
            DraftExpression::measure(&other),
        ),
    );
    assert!(
        Module::new("M", [domain.into(), output.into()])
            .unwrap_err()
            .iter()
            .any(|error| error.message().contains("foreign or omitted Domain"))
    );
}
