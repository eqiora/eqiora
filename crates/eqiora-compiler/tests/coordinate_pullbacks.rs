//! Authored pullbacks retain their explicit bindings and reject ambiguous selectors.
use eqiora_compiler::compile;
use eqiora_graph::Op;
use eqiora_schema::kernel::{ExprNode, KernelNode};

fn source(expression: &str) -> String {
    format!(
        r#"model M() {{
        domain reference=box(0,1,0,1);
        domain body=box(0,3,0,3);
        coordinate xi:m on reference from reference[0];
        coordinate eta:m on reference from reference[1];
        coordinate alias:m on reference from reference[0];
        coordinate x:m on body from body[0];
        coordinate y:m on body from body[1];
        variable anchor:1;
        relation r {{anchor=0;}}
        observable sample:m^2={expression};
        }}"#
    )
}

#[test]
fn authored_affine_pullback_retains_complete_coordinate_bindings() {
    let source = source(
        "evaluate(pullback(x*x+x*y,from=(xi,eta),at=(x=2*xi+eta,y=3*eta)),at=(xi=0.25[m],eta=0.5[m]))",
    );
    let models = compile("pullback.eqi", &source).unwrap();
    let observable = models[0]
        .transaction()
        .ops()
        .iter()
        .find_map(|op| match op {
            Op::DefineKernelNode {
                node: KernelNode::Observable(value),
            } => Some(value),
            _ => None,
        })
        .unwrap();
    assert!(observable.expression().nodes().iter().any(|node| matches!(
        node, ExprNode::Pullback { source, at, .. } if source.len() == 2 && at.len() == 2
    )));
}

#[test]
fn authored_pullbacks_reject_alias_duplicates_and_incompatible_bindings() {
    for expression in [
        "pullback(x*x,from=(xi,alias),at=(x=xi,y=eta))",
        "pullback(x*x,from=(xi,eta),at=(x=xi))",
        "pullback(x*x,from=(xi,eta),at=(x=1[s],y=eta))",
        "pullback(x*x,from=(xi,eta),at=(x=x,y=eta))",
    ] {
        let sampled = format!("evaluate({expression},at=(xi=0.25[m],eta=0.5[m]))");
        let errors = compile("invalid-pullback.eqi", &source(&sampled)).unwrap_err();
        assert!(
            errors
                .iter()
                .any(|error| error.message().contains("pullback")),
            "{expression}: {errors:?}"
        );
    }
}
