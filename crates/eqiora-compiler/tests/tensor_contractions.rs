use eqiora_compiler::compile;
use eqiora_graph::Op;
use eqiora_schema::kernel::{ExprNode, KernelNode};

#[test]
fn source_tensor_contractions_retain_shared_pure_definitions() {
    let source = r#"model M() {
        domain body=box(0,1,0,1);
        parameter conductivity:tensor<1,2,2>=tensor_value(frame=body,components=[[2,3],[5,7]]);
        parameter gradient:vector<1,2>=tensor_value(frame=body,components=[11,13]);
        observable first:1=component(contract(conductivity,gradient,axes=((1,0),)),indices=(0,));
        observable second:1=component(contract(conductivity,gradient,axes=((1,0),)),indices=(1,));
        relation independent_reference { component(contract(conductivity,gradient,axes=((1,0),)),indices=(0,))=61; component(contract(conductivity,gradient,axes=((1,0),)),indices=(1,))=146; }
    }"#;
    let models = compile("tensor.eqi", source).unwrap();
    assert!(models[0].transaction().ops().iter().any(|op| {
        match op {
            Op::DefineKernelNode {
                node: KernelNode::Observable(value),
            } => value
                .expression()
                .nodes()
                .iter()
                .any(|node| matches!(node, ExprNode::PureOperatorApplication(_))),
            _ => false,
        }
    }));
}

#[test]
fn invalid_axes_channels_and_nonscalar_indices_reject_in_source() {
    for expression in [
        "contract(a,b,axes=((0,0),(0,1)))",
        "contract(a,b,axes=((2,0),))",
        "contract(a,b,axes=[[1,0]])",
        "permute_axes(a,order=(0,0))",
        "component(a,indices=(0,2))",
        "component(a,indices=(0,0.5))",
        "contract(channels,b,axes=((0,0),))",
    ] {
        let source = format!(
            r#"model M() {{
            domain body=box(0,1,0,1);
            parameter a:tensor<1,2,2>=tensor_value(frame=body,components=[[2,3],[5,7]]);
            parameter b:vector<1,2>=tensor_value(frame=body,components=[11,13]);
            parameter channels:array<1,2>=[11,13];
            let bad={expression};
            relation witness {{ bad=bad; }}
        }}"#
        );
        assert!(
            compile("invalid-tensor.eqi", &source).is_err(),
            "{expression}"
        );
    }
}

#[test]
fn source_spatial_transpose_preserves_units_and_uses_pure_component_calculus() {
    let models = compile(
        "transpose.eqi",
        r#"model M() {
        domain body=box(0,1,0,1);
        parameter a:tensor<Pa,2,2>=tensor_value(frame=body,components=[[2,3],[5,7]]);
        observable transposed:Pa=component(transpose(a),indices=(0,1));
        relation witness { component(transpose(a),indices=(0,1))=5[Pa]; }
    }"#,
    )
    .unwrap();
    assert!(models[0].transaction().ops().iter().any(|op| match op {
        Op::DefineKernelNode {
            node: KernelNode::Observable(value),
        } => value.expression().definitions().len() == 2,
        _ => false,
    }));
}

#[test]
fn cross_product_source_requires_three_cartesian_components() {
    let source = r#"model M() {
        domain body=box(0,1,0,1,0,1);
        parameter a:vector<m,3>=tensor_value(frame=body,components=[1,2,3]);
        parameter b:vector<N,3>=tensor_value(frame=body,components=[5,7,11]);
        relation determinant_reference {
            component(cross(a,b),indices=(0,))=1[J];
            component(cross(a,b),indices=(1,))=4[J];
            component(cross(a,b),indices=(2,))=-3[J];
        }
    }"#;
    let models = compile("cross.eqi", source).unwrap();
    assert!(models[0].transaction().ops().iter().any(|op| matches!(op,
        Op::DefineKernelNode { node: KernelNode::Relation(relation) }
            if relation.expression().nodes().iter().any(|node| matches!(node, ExprNode::PureOperatorApplication(_))))));
    let planar = source
        .replace("box(0,1,0,1,0,1)", "box(0,1,0,1)")
        .replace("vector<m,3>", "vector<m,2>")
        .replace("vector<N,3>", "vector<N,2>")
        .replace("[1,2,3]", "[1,2]")
        .replace("[5,7,11]", "[5,7]");
    assert!(compile("planar-cross.eqi", &planar).is_err());
}
