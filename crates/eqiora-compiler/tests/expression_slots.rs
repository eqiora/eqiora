use eqiora_compiler::compile;
use eqiora_graph::Op;
use eqiora_schema::kernel::{ExprNode, KernelNode, SymbolRef};

const LAW: &str = "component Law(support body:volume(ambient_dimension=2),variable u:K on body,expression source:K/m^2 on body){relation balance on body{-div(grad(u))=source;}}";
fn model(binding: &str) -> String {
    format!(
        "{LAW} model M(){{domain body=box(0,1,0,1);variable u:K on body;parameter A:K=2;parameter L:m=1;instance law:Law(body=body,u=u,{binding});}}"
    )
}
fn rejected(source: &str) {
    eqiora_lang::parse("slots.eqi", source)
        .into_document()
        .unwrap();
    assert!(compile("slots.eqi", source).is_err(), "{source}");
}
#[test]
fn typed_expression_preserves_parameter_ids_without_carriers() {
    let compiled = compile("slots.eqi", &model("source=-6*A*coordinate(0)/L^3")).unwrap();
    let nodes = compiled[0]
        .transaction()
        .ops()
        .iter()
        .filter_map(|op| match op {
            Op::DefineKernelNode { node } => Some(node),
            _ => None,
        })
        .collect::<Vec<_>>();
    assert_eq!(
        nodes
            .iter()
            .filter(|node| matches!(node, KernelNode::Field(_)))
            .count(),
        1
    );
    assert_eq!(
        nodes
            .iter()
            .filter(|node| matches!(node, KernelNode::Parameter(_)))
            .count(),
        2
    );
    assert_eq!(
        nodes
            .iter()
            .filter(|node| matches!(node, KernelNode::Relation(_)))
            .count(),
        1
    );
    let a = compiled[0].symbols().get("A").unwrap();
    assert!(nodes.iter().any(|node| matches!(node,KernelNode::Relation(r) if r.expression().nodes().iter().any(|node| matches!(node,ExprNode::Symbol(SymbolRef::Parameter(p)) if p.erase()==a)))));
}
#[test]
fn nested_forwarding_retains_caller_lexical_capture() {
    let outer = "component Outer(support body:volume(ambient_dimension=2),variable u:K on body,expression f:K/m^2 on body){parameter A:K=99;coordinate x:m on body from body[1];let forwarded=f;instance inner:Law(body=body,u=u,source=forwarded);}";
    let source = model("source=-6*A*x/L^3")
        .replace("model M()", &format!("{outer} model M()"))
        .replace("instance law:Law", "instance law:Outer")
        .replace(
            "variable u:K on body;",
            "coordinate x:m on body from body[0];variable u:K on body;",
        )
        .replace("u=u,source=-6", "u=u,f=-6");
    let compiled = compile("nested.eqi", &source).unwrap();
    let caller = compiled[0].symbols().get("A").unwrap();
    let length = compiled[0].symbols().get("L").unwrap();
    for op in compiled[0].transaction().ops() {
        if let Op::DefineKernelNode {
            node: KernelNode::Relation(r),
        } = op
        {
            let refs = r
                .expression()
                .nodes()
                .iter()
                .filter_map(|node| match node {
                    ExprNode::Symbol(SymbolRef::Parameter(p)) => Some(p.erase()),
                    _ => None,
                })
                .collect::<Vec<_>>();
            assert!(refs.contains(&caller));
            assert!(refs.iter().all(|id| *id == caller || *id == length));
        }
    }
}
#[test]
fn expression_binding_failures_are_typed_and_nonvacuous() {
    for binding in [
        "",
        "source=1[m]",
        "source=1[K/m^2],source=2[K/m^2]",
        "source=[1[K/m^2],2[K/m^2]]",
    ] {
        rejected(&model(binding));
    }
    rejected(&model("source=x*1[K/m^3]").replace("variable u:K on body;", "domain foreign=box(0,1,0,1);coordinate x:m on foreign from foreign[0];variable u:K on body;"));
    rejected(&model("source=z").replace(
        "variable u:K on body;",
        "clock tick=periodic(1[s]);state z:K/m^2 on body at tick;variable u:K on body;",
    ));
    rejected(&model("source=1[K/m^2]").replace(
        "expression source:K/m^2 on body",
        "expression source:K/m^2 on body at tick",
    ));
}

#[test]
fn vector_boundary_expression_uses_the_same_typed_contract() {
    let source = "component Prescribed(support body:volume(ambient_dimension=2),support face:boundary(parent=body),variable u:vector<m/s,2> on body,expression value:vector<m/s,2> on face){relation datum on face{trace(u)=value;}} model M(){domain body=box(0,1,0,1);domain left=boundary(body,axis=0,side=lower);variable u:vector<m/s,2> on body in h1;parameter speed:m/s=2;instance boundary:Prescribed(body=body,face=left,u=u,value=tensor_value(frame=body,components=[speed,0[m/s]]));}";
    compile("vector.eqi", source).unwrap();
    rejected(&source.replace("value=tensor_value(frame=body,components=[speed,0[m/s]])", "value=speed"));
    rejected(&source.replace("value=tensor_value(frame=body,components=[speed,0[m/s]])", "value=tensor_value(frame=body,components=[speed,0[m/s],0[m/s]])"));
}
