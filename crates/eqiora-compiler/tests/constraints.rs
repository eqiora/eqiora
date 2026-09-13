use eqiora_compiler::compile;
use eqiora_graph::Op;
use eqiora_schema::kernel::{ExprNode, KernelNode, RelationConditionKind};

#[test]
fn source_conditions_keep_distinct_meaning_and_independently_dimensioned_operands() {
    let source = "model M(){variable gap:m;variable force:N;let opening=gap;relation contact{2[N/m]*gap-force=6[N];complementarity(0[m]<=opening,force>=0[N]);inequality(gap<=4[m]);}}";
    let compiled = compile("contact.eqi", source).unwrap_or_else(|errors| panic!("{errors:?}"));
    let relation = compiled[0]
        .transaction()
        .ops()
        .iter()
        .find_map(|op| match op {
            Op::DefineKernelNode {
                node: KernelNode::Relation(value),
            } => Some(value),
            _ => None,
        })
        .unwrap();
    assert_eq!(
        relation.conditions().unwrap(),
        &[
            RelationConditionKind::Equality,
            RelationConditionKind::Complementarity,
            RelationConditionKind::Inequality
        ]
    );
    assert_eq!(relation.expression().roots().len(), 6);
    // Explicit nonnegativity syntax becomes retained mathematical meaning, not a Boolean root.
    assert!(
        !relation
            .expression()
            .nodes()
            .iter()
            .any(|node| matches!(node, ExprNode::Compare(..)))
    );
    let document = eqiora_lang::parse("contact.eqi", source)
        .into_document()
        .unwrap();
    let formatted = eqiora_lang::format(&document);
    let rebuilt = compile("contact.eqi", &formatted)
        .unwrap_or_else(|errors| panic!("{formatted}\n{errors:?}"));
    let rebuilt = rebuilt[0]
        .transaction()
        .ops()
        .iter()
        .find_map(|op| match op {
            Op::DefineKernelNode {
                node: KernelNode::Relation(value),
            } => Some(value),
            _ => None,
        })
        .unwrap();
    assert_eq!(relation, rebuilt);
}

#[test]
fn kind_is_source_identity_even_when_operand_expressions_match() {
    let identity = |statement: &str| {
        let source = format!("model M(){{variable x:1;relation r{{{statement};}}}}");
        let compiled = compile("kind.eqi", &source).unwrap();
        compiled[0]
            .transaction()
            .ops()
            .iter()
            .find_map(|op| match op {
                Op::DefineKernelNode {
                    node: KernelNode::Relation(value),
                } => Some(value.id()),
                _ => None,
            })
            .unwrap()
    };
    assert_ne!(identity("x=0"), identity("inequality(x>=0)"));
}

#[test]
fn missing_nonnegativity_reversed_sign_units_and_unordered_values_fail() {
    for statement in [
        "complementarity(gap,force)",
        "complementarity(gap<=0[m],0[N]<=force)",
        "complementarity(0[s]<=gap,0[N]<=force)",
        "complementarity(0<=true,0[N]<=force)",
        "complementarity(0<=math.complex(1,2),0[N]<=force)",
        "inequality(gap>=0[N])",
        "inequality(true>=false)",
        "inequality(gap>0[m])",
    ] {
        let source = format!(
            "model M(){{variable gap:m;variable force:N;relation contact{{{statement};}}}}"
        );
        let errors = compile("invalid-constraint.eqi", &source).unwrap_err();
        assert!(!errors.is_empty(), "{statement}");
        assert!(
            errors.iter().all(|error| error.source_span().is_some()),
            "{errors:?}"
        );
    }
}

#[test]
fn different_support_and_unsupported_inclusion_do_not_become_penalties() {
    let source = "model M(){domain a=box(0,1);domain b=box(0,1);variable gap:m on a;variable force:N on b;relation c on a{complementarity(0[m]<=gap,0[N]<=force);}}";
    assert!(compile("foreign-support.eqi", source).is_err());
    let errors = compile(
        "inclusion.eqi",
        "model M(){variable x:1;relation c{inclusion(x,0);}}",
    )
    .unwrap_err();
    assert!(
        errors
            .iter()
            .any(|error| error.message().contains("inclusions")),
        "{errors:?}"
    );
    assert!(
        compile(
            "initial.eqi",
            "model M(){variable x:1;initial{inequality(x>=0);}}"
        )
        .is_err()
    );
}
