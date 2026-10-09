//! Recognize the explicit 2D scalar curl-curl reduction without rewriting its DAG.
use eqiora_schema::kernel::pure_operator::PureOperatorDefinition;
use eqiora_schema::kernel::{ExprDag, ExprId, ExprNode, SymbolRef};

/// For a live typed source, return grad(u) only for the exact shared composition
/// curl_vector(grad(curl_scalar(grad(u)))). Its scalar value is -div(grad(u)).
/// The two explicit factories fix dimension, operand rank, and orientation.
pub(super) fn gradient(dag: &ExprDag, root: ExprId) -> Option<ExprId> {
    let operand = |id, rank| {
        let ExprNode::PureOperatorApplication(application) = dag.node(id)? else {
            return None;
        };
        let expected = PureOperatorDefinition::curl_from_gradient(2, rank).ok()?;
        (dag.definition(application.definition())? == &expected).then_some(())?;
        let [argument] = application.arguments() else {
            return None;
        };
        Some(*argument)
    };
    let outer_gradient = operand(root, 1)?;
    let ExprNode::Gradient(inner_curl) = dag.node(outer_gradient)? else {
        return None;
    };
    let gradient = operand(*inner_curl, 0)?;
    let ExprNode::Gradient(field) = dag.node(gradient)? else {
        return None;
    };
    matches!(dag.node(*field)?, ExprNode::Symbol(SymbolRef::Field(_))).then_some(gradient)
}

#[cfg(test)]
mod tests {
    use super::*;
    use eqiora_graph::{GraphStore, InMemoryGraphStore};

    #[test]
    fn reduction_requires_the_exact_scalar_planar_composition() {
        for (dimensions, field_type, operator, admitted) in [
            (2, "1", "curl(curl(u))", true),
            (2, "1", "curl(-curl(u))", false),
            (2, "vector<1,2>", "curl(curl(u))", false),
            (3, "vector<1,3>", "curl(curl(u))", false),
        ] {
            let bounds = vec!["0,1"; dimensions].join(",");
            let source = format!(
                "model M() {{domain body=box({bounds}); variable u:{field_type} on body; relation law on body {{{operator}=u*1[1/m^2];}}}}"
            );
            let compiled = eqiora_compiler::CompiledModel::compile_selected(
                "reduction.eqi",
                &source,
                "M",
                &[],
            )
            .unwrap();
            let (transaction, model, symbols) = compiled.into_parts();
            let mut store = InMemoryGraphStore::new();
            store.commit(transaction).unwrap();
            let program =
                eqiora_sem::KernelProgram::from_snapshot(&store.snapshot(), model).unwrap();
            let typed =
                crate::form_compiler::scalar::typed_relation(&program, symbols.get("law").unwrap())
                    .unwrap();
            let dag = typed.expression();
            let ExprNode::Sub(left, _) = dag.node(dag.roots()[0]).unwrap() else {
                panic!("retained equation");
            };
            assert_eq!(
                gradient(dag, *left).is_some(),
                admitted,
                "{dimensions} {field_type} {operator}"
            );
        }
    }
}
