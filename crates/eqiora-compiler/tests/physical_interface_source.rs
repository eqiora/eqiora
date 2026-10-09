//! Source interface declarations retain ordered sides without an implicit Connection.
use eqiora_compiler::compile;
use eqiora_graph::Op;
use eqiora_schema::kernel::{DomainKind, ExprNode, KernelNode};

fn source(expression: &str) -> String {
    format!(
        r#"model M() {{
        domain left=box(0,1,0,1);
        domain right=box(1,3,0,1);
        domain left_face=boundary(left,axis=0,side=upper);
        domain right_face=boundary(right,axis=0,side=lower);
        domain contact=interface(left_face,right_face);
        variable u_left:1 on left;
        variable u_right:1 on right;
        relation law on contact {{ {expression}=0; }}
    }}"#
    )
}

#[test]
fn explicit_one_sided_expressions_retain_the_declared_interface() {
    for expression in [
        "trace(u_left,on=contact,from=left)-trace(u_right,on=contact,from=right)",
        "normal(grad(u_left),on=contact,from=left)-normal(grad(u_right),on=contact,from=right)",
        "tangential_trace(grad(u_left),on=contact,from=left)-tangential_trace(grad(u_right),on=contact,from=right)",
    ] {
        let models = compile("physical-interface.eqi", &source(expression)).unwrap();
        let model = &models[0];
        let contact = model.symbols().get("contact").unwrap();
        let sides = [
            model.symbols().get("left_face").unwrap(),
            model.symbols().get("right_face").unwrap(),
        ];
        let mut targets = Vec::new();
        let mut found = false;
        for op in model.transaction().ops() {
            match op {
                Op::DefineKernelNode {
                    node: KernelNode::Domain(domain),
                } if domain.id().erase() == contact => {
                    let DomainKind::PhysicalInterface { boundaries } = domain.kind() else {
                        panic!("interface Domain");
                    };
                    assert_eq!(boundaries.map(|id| id.erase()), sides);
                    found = true;
                }
                Op::DefineKernelNode {
                    node: KernelNode::Relation(relation),
                } => {
                    targets.extend(relation.expression().nodes().iter().filter_map(
                        |node| match node {
                            ExprNode::Trace { on, .. } | ExprNode::NormalComponent { on, .. } => {
                                Some(on.erase())
                            }
                            _ => None,
                        },
                    ));
                }
                Op::DefineKernelNode {
                    node: KernelNode::Connection(_),
                } => panic!("interface declaration created a Connection"),
                _ => {}
            }
        }
        assert!(found);
        assert_eq!(targets, vec![contact, contact]);
    }
}

#[test]
fn from_cannot_select_the_other_operands_side() {
    let bad = source("trace(u_left,on=contact,from=right)");
    assert!(compile("wrong-side.eqi", &bad).is_err());
    let duplicate = source("trace(u_left,on=contact)").replace(
        "interface(left_face,right_face)",
        "interface(left_face,left_face)",
    );
    assert!(compile("duplicate-side.eqi", &duplicate).is_err());
}

#[test]
fn formatting_preserves_interface_order_and_source_identity() {
    use eqiora_compiler::source_identity::LocalSourceIdentity;
    let text = source("trace(u_left,on=contact)-trace(u_right,on=contact)");
    let original = eqiora_lang::parse("original.eqi", &text)
        .into_document()
        .unwrap();
    let formatted = eqiora_lang::format(&original);
    let reparsed = eqiora_lang::parse("formatted.eqi", &formatted)
        .into_document()
        .unwrap();
    assert_eq!(
        LocalSourceIdentity::from_document(&original).unwrap(),
        LocalSourceIdentity::from_document(&reparsed).unwrap()
    );
    compile("formatted.eqi", &formatted).unwrap();
    let reversed = text.replace(
        "interface(left_face,right_face)",
        "interface(right_face,left_face)",
    );
    let reversed = eqiora_lang::parse("reversed.eqi", &reversed)
        .into_document()
        .unwrap();
    assert_ne!(
        LocalSourceIdentity::from_document(&original).unwrap(),
        LocalSourceIdentity::from_document(&reversed).unwrap()
    );
}
