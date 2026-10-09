//! Emit only the generated amplitude roots after validating every original node.
use super::*;

pub(super) fn compact(expression: ExprDag) -> Result<ExprDag, Diagnostic> {
    let mut reached = vec![false; expression.nodes().len()];
    for root in expression.roots() {
        reached[root.index() as usize] = true;
    }
    for (index, node) in expression.nodes().iter().enumerate().rev() {
        if reached[index] {
            map(node, |operand| {
                reached[operand.index() as usize] = true;
                operand
            })?;
        }
    }
    let mut builder = ExprDagBuilder::new();
    let mut mapped = vec![None; reached.len()];
    for (index, node) in expression.nodes().iter().enumerate() {
        if reached[index] {
            let node = map(node, |operand| {
                mapped[operand.index() as usize].expect("reached prior operand")
            })?;
            mapped[index] = Some(builder.push(node)?);
        }
    }
    builder.finish(
        expression
            .roots()
            .iter()
            .map(|root| mapped[root.index() as usize].expect("reached amplitude root")),
    )
}

fn map(node: &ExprNode, mut operand: impl FnMut(ExprId) -> ExprId) -> Result<ExprNode, Diagnostic> {
    Ok(match node {
        ExprNode::Constant(_) | ExprNode::Symbol(_) => node.clone(),
        ExprNode::Complex { real, imag } => ExprNode::Complex {
            real: operand(*real),
            imag: operand(*imag),
        },
        ExprNode::Neg(a) => ExprNode::Neg(operand(*a)),
        ExprNode::Add(a, b) => ExprNode::Add(operand(*a), operand(*b)),
        ExprNode::Sub(a, b) => ExprNode::Sub(operand(*a), operand(*b)),
        ExprNode::Mul(a, b) => ExprNode::Mul(operand(*a), operand(*b)),
        ExprNode::Div(a, b) => ExprNode::Div(operand(*a), operand(*b)),
        ExprNode::PowI(a, n) => ExprNode::PowI(operand(*a), *n),
        ExprNode::UnaryMath(op, a) => ExprNode::UnaryMath(*op, operand(*a)),
        ExprNode::Gradient(a) => ExprNode::Gradient(operand(*a)),
        ExprNode::Divergence(a) => ExprNode::Divergence(operand(*a)),
        ExprNode::Trace(a) => ExprNode::Trace(operand(*a)),
        ExprNode::NormalComponent(a) => ExprNode::NormalComponent(operand(*a)),
        _ => {
            return Err(invalid(
                "unexpected operator in a generated harmonic expression",
            ));
        }
    })
}
