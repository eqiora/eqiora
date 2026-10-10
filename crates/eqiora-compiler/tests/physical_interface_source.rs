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
        variable u_left:1 on left in smooth;
        variable u_right:1 on right in smooth;
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

fn weak_source() -> String {
    let mut text = source("trace(u_left,on=contact)-trace(u_right,on=contact)");
    let end = text.rfind('}').unwrap();
    text.insert_str(
        end,
        r#"form weak for law {
            test eta:1 for u_left in h1;
            integrate(contact,trace(eta,on=contact,from=left)*
                (trace(u_left,on=contact,from=left)-trace(u_right,on=contact,from=right)))=0;
        }
        "#,
    );
    text
}

#[test]
fn weak_interface_scalar_and_normal_traces_retain_both_parents_and_replay() {
    for normal in [false, true] {
        for reversed in [false, true] {
            let mut text = weak_source();
            if normal {
                text = text
                    .replace("trace(u_left,", "normal(grad(u_left),")
                    .replace("trace(u_right,", "normal(grad(u_right),");
            }
            if reversed {
                text = text.replace(
                    "interface(left_face,right_face)",
                    "interface(right_face,left_face)",
                );
            }
            let models = compile("weak-physical-interface.eqi", &text).unwrap();
            let model = &models[0];
            let form = model.authored_formulations().next().unwrap().projection();
            assert_eq!(
                eqiora_compiler::AuthoredFormulationProjection::decode(form.canonical_bytes())
                    .unwrap(),
                *form
            );
            let encoded = String::from_utf8(form.canonical_bytes().to_vec()).unwrap();
            let contact = model.symbols().get("contact").unwrap().ulid().to_string();
            assert_eq!(
                encoded
                    .matches(&format!("\"on_ulid\":\"{contact}\""))
                    .count(),
                3
            );
            for name in ["u_left", "u_right"] {
                assert!(encoded.contains(&model.symbols().get(name).unwrap().ulid().to_string()));
            }
        }
    }
}

#[test]
fn weak_interface_rejects_foreign_sides_untraced_fields_and_missing_regularity() {
    for (old, new) in [
        (
            "trace(eta,on=contact,from=left)",
            "trace(eta,on=contact,from=right)",
        ),
        ("trace(eta,on=contact,from=left)", "eta"),
        (
            "trace(u_right,on=contact,from=right)",
            "trace(u_right,on=left_face,from=right)",
        ),
        ("test eta:1 for u_left in h1", "test eta:1 for u_left in l2"),
        (
            "variable u_left:1 on left in smooth",
            "variable u_left:1 on left in l2",
        ),
    ] {
        let text = weak_source().replace(old, new);
        assert!(
            compile("invalid-weak-interface.eqi", &text).is_err(),
            "accepted {new}"
        );
    }
    let boundary_gradient = weak_source().replace(
        "trace(u_left,on=contact,from=left)",
        "normal(grad(trace(u_left,on=contact)),on=contact)",
    );
    let errors = compile("boundary-gradient.eqi", &boundary_gradient).unwrap_err();
    assert!(
        errors
            .iter()
            .any(|error| error.message().contains("grad requires a parent-volume"))
    );
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
