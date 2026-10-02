use eqiora::api::ModelDocument;
use eqiora::kernel::{ExprNode, KernelNode};
use eqiora::sem::{Interpreter, ReferenceConfig};

fn compile(body: &str) -> ModelDocument {
    ModelDocument::compile(
        "simultaneous.eqi",
        &format!("model M() {{ variable x: 1; variable y: 1; relation r {{ {body} }} }}"),
    )
    .unwrap()
}

#[test]
fn equation_permutation_preserves_simultaneous_solution_not_ordered_identity() {
    let models = [
        compile("x + y = 3; x - y = 1;"),
        compile("x - y = 1; x + y = 3;"),
    ];
    assert_ne!(
        models[0].structural_fingerprint().unwrap(),
        models[1].structural_fingerprint().unwrap()
    );
    // Adding/subtracting the two equations independently gives x=2,y=1.
    // A^-1 has infinity norm one, so the configured absolute residual bound
    // also bounds the solution error; no measured solver output sets this bound.
    const RESIDUAL_BOUND: f64 = 1e-12;
    let config = ReferenceConfig::new(0.0, 0.1)
        .unwrap()
        .with_nonlinear_tolerances(RESIDUAL_BOUND, 0.0)
        .unwrap();
    for (index, model) in models.iter().enumerate() {
        let relation = model
            .program()
            .nodes()
            .find_map(|node| {
                if let KernelNode::Relation(value) = node {
                    Some(value)
                } else {
                    None
                }
            })
            .unwrap();
        let dag = &model
            .program()
            .numerical_residuals(relation.id().erase())
            .unwrap();
        // At the independently fixed input (4,-1), residuals are (0,4),
        // swapped to (4,0) by the root permutation.
        let mut values = Vec::new();
        for node in dag.nodes() {
            values.push(match node {
                ExprNode::Constant(value) => value
                    .real_scalar_value()
                    .expect("real scalar fixture")
                    .value(),
                ExprNode::Symbol(eqiora::kernel::SymbolRef::Field(id)) => {
                    if id.erase() == model.aliases()["x"] {
                        4.0
                    } else {
                        assert_eq!(id.erase(), model.aliases()["y"]);
                        -1.0
                    }
                }
                ExprNode::Add(a, b) => values[a.index() as usize] + values[b.index() as usize],
                ExprNode::Sub(a, b) => values[a.index() as usize] - values[b.index() as usize],
                other => panic!("outside fixed affine corpus: {other:?}"),
            });
        }
        let residuals = dag
            .roots()
            .iter()
            .map(|root| values[root.index() as usize])
            .collect::<Vec<_>>();
        assert_eq!(
            residuals,
            if index == 0 {
                vec![0.0, 4.0]
            } else {
                vec![4.0, 0.0]
            }
        );
        let trajectory = Interpreter::default().run(model.program(), config).unwrap();
        for (name, expected) in [("x", 2.0), ("y", 1.0)] {
            let sample = trajectory
                .samples()
                .iter()
                .find(|sample| sample.coordinate() == model.aliases()[name])
                .unwrap();
            assert!((sample.value().value() - expected).abs() <= RESIDUAL_BOUND);
        }
    }
}
