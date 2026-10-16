use eqiora_compiler::compile;
use eqiora_core::{ScalarDomain, ValueLiteral};
use eqiora_graph::Op;
use eqiora_schema::kernel::{ComparisonOp, ExprNode, KernelNode};

#[test]
fn boolean_equations_keep_ordered_sides_and_exact_integer_operands() {
    let models = compile("bool.eqi", "model M(parameter n: integer=9007199254740993) { variable matched: bool; relation r { matched = n == 9007199254740993; } }").unwrap();
    let relation = models[0]
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
    let (left, right) = relation.equation_sides().unwrap().next().unwrap();
    assert!(matches!(
        relation.expression().node(left),
        Some(ExprNode::Symbol(_))
    ));
    assert!(matches!(
        relation.expression().node(right),
        Some(ExprNode::Compare(ComparisonOp::Equal, _, _))
    ));
    assert!(relation.expression().nodes().iter().any(|node| matches!(node, ExprNode::Constant(value) if value.integer_scalar_value() == Some(9007199254740993))));
    assert!(
        !relation
            .expression()
            .nodes()
            .iter()
            .any(|node| matches!(node, ExprNode::Sub(..)))
    );
}

#[test]
fn booleans_do_not_coerce_to_numbers_or_contextual_zero() {
    for body in [
        "variable x: bool; relation r { x=0; }",
        "variable x: 1; relation r { x=true; }",
        "variable x: bool; relation r { x=not 1; }",
        "variable x: bool; relation r { x=true and 1; }",
        "variable x: bool; relation r { x=-true; }",
        "variable x: bool; relation r { x=true < false; }",
        "variable x: bool; relation r { x=[1,2] == [1,2]; }",
        "variable x: bool; relation r { x=1[m] < 1[s]; }",
    ] {
        let errors = compile("bad.eqi", &format!("model M() {{ {body} }}")).unwrap_err();
        assert!(
            errors
                .iter()
                .any(|e| e.code() == eqiora_core::diagnostic::codes::LANGUAGE_TYPE_ERROR),
            "{body}: {errors:?}"
        );
    }
}

#[test]
fn boolean_literals_and_negation_survive_source_identity_and_alias_lowering() {
    let source = "model M(parameter flag: bool=false) { variable result: bool; let predicate = not flag or true; relation r { result=predicate; } }";
    let models = compile("bool.eqi", source).unwrap();
    assert!(models[0].transaction().ops().iter().any(|op| matches!(op, Op::DefineKernelNode { node: KernelNode::Parameter(value) } if value.value()==&ValueLiteral::boolean(false) && value.value_type().scalar_domain()==ScalarDomain::Boolean)));
    let graph = models[0]
        .transaction()
        .ops()
        .iter()
        .find_map(|op| match op {
            Op::DefineKernelNode {
                node: KernelNode::Relation(value),
            } => Some(value.expression()),
            _ => None,
        })
        .unwrap();
    assert!(
        graph
            .nodes()
            .iter()
            .any(|node| matches!(node, ExprNode::Not(_)))
    );
    assert!(
        graph
            .nodes()
            .iter()
            .any(|node| matches!(node, ExprNode::Or(..)))
    );
}
