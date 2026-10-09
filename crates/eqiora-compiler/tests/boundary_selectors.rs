use eqiora_compiler::compile;
use eqiora_graph::Op;
use eqiora_schema::kernel::{ExprNode, KernelNode};

fn source(expression: &str) -> String {
    format!(
        r#"model M() {{
        domain body=box(0,1,0,2);
        domain other=box(0,1,0,2);
        domain wall=boundary(body,axis=0,side=lower);
        variable u:1 on body;
        relation balance on body {{ u=1; }}
        relation fixed on wall {{ {expression}=0; }}
    }}"#
    )
}

#[test]
fn explicit_boundary_selectors_retain_the_exact_target() {
    for expression in [
        "trace(u,on=wall,from=body)",
        "trace(u,from=body,on=wall)",
        "normal(grad(u),on=wall,from=body)",
        "tangential_trace(grad(u),on=wall,from=body)",
    ] {
        let models = compile("boundary.eqi", &source(expression)).unwrap();
        let wall = models[0].symbols().get("wall").unwrap();
        let targets = models[0]
            .transaction()
            .ops()
            .iter()
            .filter_map(|op| match op {
                Op::DefineKernelNode {
                    node: KernelNode::Relation(relation),
                } => Some(relation),
                _ => None,
            })
            .flat_map(|relation| relation.expression().nodes())
            .filter_map(|node| match node {
                ExprNode::Trace { on, .. } | ExprNode::NormalComponent { on, .. } => {
                    Some(on.erase())
                }
                _ => None,
            })
            .collect::<Vec<_>>();
        assert_eq!(targets, vec![wall], "{expression}");
    }
}

#[test]
fn explicitly_selected_trace_can_be_authored_outside_a_relation() {
    let source = source("selected").replace(
        "relation fixed",
        "let selected=trace(u,on=wall,from=body); relation fixed",
    );
    compile("boundary-alias.eqi", &source).unwrap();
}

#[test]
fn selectors_reject_foreign_parents_and_ambiguous_targets() {
    for expression in [
        "trace(u,on=body)",
        "trace(u,on=missing)",
        "trace(u,on=wall,from=other)",
        "trace(u,on=wall,on=wall)",
        "trace(u,on=wall,from=body,from=body)",
        "normal(grad(u),on=wall,from=other)",
        "tangential_trace(grad(u),on=wall,from=other)",
    ] {
        assert!(
            compile("invalid-boundary.eqi", &source(expression)).is_err(),
            "{expression}"
        );
    }
}

#[test]
fn an_explicit_boundary_cannot_be_replaced_by_the_relation_scope() {
    let source = source("trace(u,on=opposite,from=body)").replace(
        "variable u",
        "domain opposite=boundary(body,axis=0,side=upper); variable u",
    );
    assert!(compile("wrong-boundary.eqi", &source).is_err());
}
