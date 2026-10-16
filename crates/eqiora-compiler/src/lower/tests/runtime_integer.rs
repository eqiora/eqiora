use super::*;
#[test]
fn exact_runtime_integer_literals_and_operations_retain_order_and_identity() {
    let source = "model M(parameter increment:integer=1, output y:integer at tick) { clock tick=periodic(1[s]); state n:integer at tick; initial {n=-9223372036854775808;} relation update at tick { next(n)=pre(n)+increment+9007199254740993; y=quotient(pre(n),3)+remainder(pre(n),3); } }";
    let compiled = compile("exact-runtime.eqi", source).unwrap_or_else(|e| panic!("{e:?}"));
    let nodes: Vec<_> = compiled[0]
        .transaction()
        .ops()
        .iter()
        .filter_map(|op| match op {
            Op::DefineKernelNode {
                node: KernelNode::Relation(r),
            } => Some(r.expression().nodes()),
            _ => None,
        })
        .flatten()
        .collect();
    assert!(nodes.iter().any(|node|matches!(node,eqiora_schema::kernel::ExprNode::Constant(value) if value.integer_scalar_value()==Some(i64::MIN))));
    assert!(nodes.iter().any(|node|matches!(node,eqiora_schema::kernel::ExprNode::Constant(value) if value.integer_scalar_value()==Some(9_007_199_254_740_993))));
    assert!(nodes.iter().any(|node| matches!(
        node,
        eqiora_schema::kernel::ExprNode::Symbol(SymbolRef::Parameter(_))
    )));
    assert!(
        nodes
            .iter()
            .any(|node| matches!(node, eqiora_schema::kernel::ExprNode::Quotient(..)))
    );
    assert!(
        nodes
            .iter()
            .any(|node| matches!(node, eqiora_schema::kernel::ExprNode::Remainder(..)))
    );
}
#[test]
fn integer_calls_are_explicit_and_zero_assignment_retains_target() {
    for rhs in [
        "to_integer(2.0)",
        "to_integer(to_real(pre(n)))",
        "0",
        "quotient(-7,3)",
        "remainder(-7,3)",
    ] {
        let source = format!(
            "model M() {{ clock tick=periodic(1[s]); state n:integer at tick; initial {{n=0;}} relation update at tick {{next(n)={rhs};}} }}"
        );
        let compiled =
            compile("integer-calls.eqi", &source).unwrap_or_else(|e| panic!("{rhs}: {e:?}"));
        for op in compiled[0].transaction().ops() {
            if let Op::DefineKernelNode {
                node: KernelNode::Relation(r),
            } = op
            {
                assert_eq!(r.equation_sides().unwrap().len(), 1);
                let (left, right) = r.equation_sides().unwrap().next().unwrap();
                let state = compiled[0].symbols().get("n").unwrap();
                assert!(match r.expression().node(left) {
                    Some(eqiora_schema::kernel::ExprNode::Symbol(SymbolRef::Field(id)))
                        if r.is_initial() =>
                        id.erase() == state,
                    Some(eqiora_schema::kernel::ExprNode::Symbol(SymbolRef::Next(id)))
                        if !r.is_initial() =>
                        id.erase() == state,
                    _ => false,
                });
                assert_ne!(left, right);
            }
        }
    }
    for rhs in [
        "9223372036854775808",
        "-9223372036854775809",
        "1.5",
        "pre(n)/2",
        "pre(n)+1[s]",
        "quotient(pre(n))",
        "to_real(1.0)",
    ] {
        let source = format!(
            "model M() {{clock tick=periodic(1[s]); state n:integer at tick; initial {{n=0;}} relation update at tick {{next(n)={rhs};}} }}"
        );
        assert!(compile("invalid-integer.eqi", &source).is_err(), "{rhs}");
    }
}
