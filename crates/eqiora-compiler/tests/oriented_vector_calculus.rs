use eqiora_compiler::compile;
use eqiora_graph::Op;
use eqiora_schema::kernel::{ExprNode, KernelNode};

#[test]
fn curl_and_oriented_trace_retain_spatial_nodes_and_exact_component_definitions() {
    let source = r#"model M() {
        domain body=box(0,1,0,1,0,1);
        domain face=boundary(body,axis=0,side=upper);
        variable u:vector<m,3> on body;
        relation interior on body { curl(curl(u))=-div(grad(u)); }
        relation boundary_value on face { tangential_trace(u)=trace(u); }
    }"#;
    let models = compile("oriented.eqi", source).unwrap();
    let mut gradient = false;
    let mut normal = false;
    let mut pure = false;
    for op in models[0].transaction().ops() {
        if let Op::DefineKernelNode {
            node: KernelNode::Relation(relation),
        } = op
        {
            for node in relation.expression().nodes() {
                gradient |= matches!(node, ExprNode::Gradient(_));
                normal |= matches!(node, ExprNode::NormalComponent(_));
                pure |= matches!(node, ExprNode::PureOperatorApplication(_));
            }
        }
    }
    assert!(gradient && normal && pure);
    // A boundary's parent is an exact identity, even for equal-size boxes.
    let foreign = source.replace(
        "domain face=boundary(body",
        "domain other=box(0,1,0,1,0,1); domain face=boundary(other",
    );
    let errors = compile("foreign.eqi", &foreign).unwrap_err();
    assert!(
        errors
            .iter()
            .any(|error| error.message().contains("support")),
        "{errors:?}"
    );
}

#[test]
fn planar_curl_reductions_are_selected_by_declared_shape_and_dimension() {
    let source = r#"model M() {
        domain body=box(0,1,0,1);
        domain face=boundary(body,axis=0,side=upper);
        variable u:vector<m,2> on body;
        variable f:m on body;
        relation scalar_curl on body { curl(u)=1; }
        relation vector_curl on body { curl(f)=grad(f); }
        relation curl_curl on body { curl(curl(u))=-div(grad(u)); }
        relation boundary_value on face { tangential_trace(u)=1[m]; }
    }"#;
    compile("planar.eqi", source).unwrap();
    for invalid in [
        source
            .replace("box(0,1,0,1)", "box(0,1,0,1,0,1)")
            .replace("vector<m,2>", "vector<m,3>"),
        source.replace("curl(u)=1", "curl(u)=1[m]"),
        source.replace("vector<m,2>", "array<m,2>"),
        source.replace(
            "relation boundary_value on face",
            "relation boundary_value on body",
        ),
    ] {
        assert!(compile("invalid-oriented.eqi", &invalid).is_err());
    }
}
