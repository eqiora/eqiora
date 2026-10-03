//! Ordered expression projection and canonical operand traversal.
use super::*;

pub(super) fn encode_expression(
    encoder: &mut Encoder,
    expression: &ExprDag,
    extra_roots: &[eqiora_schema::kernel::ExprId],
    scope: u8,
    ids: &BTreeMap<RawId, usize>,
    references: &mut Vec<Reference>,
    budget: &mut ConstructionBudget,
) -> Result<Vec<u32>, Diagnostic> {
    budget.account_expression_nodes(expression.nodes().len())?;
    let (order, canonical_index) = canonical_expression_order(expression, extra_roots)?;
    encoder.len(order.len())?;
    for original_index in order {
        let node = expression.nodes().get(original_index).ok_or_else(|| {
            fingerprint_error("canonical expression order references an absent node")
        })?;
        let index = canonical_index[original_index];
        match node {
            ExprNode::Constant(value) => {
                encoder.u8(1)?;
                encode_literal(encoder, value)?;
                let mut label = Encoder::new(32);
                label.u8(3)?;
                label.u8(scope)?;
                label.u32(index)?;
                label.u8(13)?;
                type_reference(value.value_type(), label.finish()?, ids, references, budget)?;
            }
            ExprNode::Array { elements } => {
                encoder.u8(18)?;
                encoder.u32(
                    u32::try_from(elements.len())
                        .map_err(|_| fingerprint_error("array operands exceed u32"))?,
                )?;
                for element in elements {
                    encoder.u32(canonical_index[element.index() as usize])?;
                }
            }
            ExprNode::Index { value, index } => {
                unary_expr(encoder, 19, *value, &canonical_index)?;
                encoder.u32(*index)?;
            }
            ExprNode::Complex { real, imag } => {
                binary_expr(encoder, 20, *real, *imag, &canonical_index)?
            }
            ExprNode::Sample { value, clock } => {
                unary_expr(encoder, 21, *value, &canonical_index)?;
                let mut label = Encoder::new(32);
                label.u8(3)?;
                label.u8(scope)?;
                label.u32(index)?;
                label.u8(12)?;
                push_reference(
                    references,
                    label.finish()?,
                    lookup(ids, clock.erase(), "sample clock")?,
                    budget,
                )?;
            }
            ExprNode::Hold(value) => unary_expr(encoder, 22, *value, &canonical_index)?,
            ExprNode::Symbol(symbol) => {
                encoder.u8(2)?;
                encode_symbol(encoder, *symbol, scope, index, ids, references, budget)?;
            }
            ExprNode::Compare(op, left, right) => {
                encoder.u8(28)?;
                encoder.u8(match op {
                    eqiora_schema::kernel::ComparisonOp::Equal => 0,
                    eqiora_schema::kernel::ComparisonOp::NotEqual => 1,
                    eqiora_schema::kernel::ComparisonOp::Less => 2,
                    eqiora_schema::kernel::ComparisonOp::LessEqual => 3,
                    eqiora_schema::kernel::ComparisonOp::Greater => 4,
                    eqiora_schema::kernel::ComparisonOp::GreaterEqual => 5,
                })?;
                encoder.u32(canonical_expr_id(*left, &canonical_index)?)?;
                encoder.u32(canonical_expr_id(*right, &canonical_index)?)?;
            }
            ExprNode::Not(value) => unary_expr(encoder, 29, *value, &canonical_index)?,
            ExprNode::And(left, right) => {
                binary_expr(encoder, 30, *left, *right, &canonical_index)?
            }
            ExprNode::Or(left, right) => binary_expr(encoder, 31, *left, *right, &canonical_index)?,
            ExprNode::Select {
                condition,
                then_value,
                else_value,
            } => {
                encoder.u8(34)?;
                for operand in [condition, then_value, else_value] {
                    encoder.u32(canonical_expr_id(*operand, &canonical_index)?)?;
                }
            }
            ExprNode::Require { condition, value } => {
                binary_expr(encoder, 35, *condition, *value, &canonical_index)?
            }
            ExprNode::Quotient(left, right) => {
                binary_expr(encoder, 23, *left, *right, &canonical_index)?
            }
            ExprNode::Remainder(left, right) => {
                binary_expr(encoder, 24, *left, *right, &canonical_index)?
            }
            ExprNode::Ordinal(value) => unary_expr(encoder, 27, *value, &canonical_index)?,
            ExprNode::ToReal(value) => unary_expr(encoder, 25, *value, &canonical_index)?,
            ExprNode::ToInteger(value) => unary_expr(encoder, 26, *value, &canonical_index)?,
            ExprNode::Neg(value) => unary_expr(encoder, 3, *value, &canonical_index)?,
            ExprNode::Add(left, right) => binary_expr(encoder, 4, *left, *right, &canonical_index)?,
            ExprNode::Sub(left, right) => binary_expr(encoder, 5, *left, *right, &canonical_index)?,
            ExprNode::Mul(left, right) => binary_expr(encoder, 6, *left, *right, &canonical_index)?,
            ExprNode::Div(left, right) => binary_expr(encoder, 7, *left, *right, &canonical_index)?,
            ExprNode::PowI(value, exponent) => {
                encoder.u8(8)?;
                encoder.u32(canonical_expr_id(*value, &canonical_index)?)?;
                encoder.i32(*exponent)?;
            }
            ExprNode::SpatialCoordinate(axis) => {
                encoder.u8(9)?;
                encoder.usize(*axis)?;
            }
            ExprNode::UnaryMath(function, value) => {
                encoder.u8(10)?;
                match function {
                    UnaryMathFunction::Sin => encoder.u8(1)?,
                    UnaryMathFunction::Sqrt => encoder.u8(2)?,
                    UnaryMathFunction::Cos => encoder.u8(3)?,
                    UnaryMathFunction::Exp => encoder.u8(4)?,
                    UnaryMathFunction::Log => encoder.u8(5)?,
                    UnaryMathFunction::Conj => encoder.u8(6)?,
                    UnaryMathFunction::Real => encoder.u8(7)?,
                    UnaryMathFunction::Imag => encoder.u8(8)?,
                    UnaryMathFunction::Abs => encoder.u8(9)?,
                    UnaryMathFunction::Abs2 => encoder.u8(10)?,
                    UnaryMathFunction::Arg => encoder.u8(11)?,
                    _ => return Err(newer_vocabulary("unary math function")),
                }
                encoder.u32(canonical_expr_id(*value, &canonical_index)?)?;
            }
            ExprNode::FiniteUnary(operation, value) => {
                use eqiora_schema::kernel::FiniteUnaryOperation;
                let tag = match operation {
                    FiniteUnaryOperation::Transpose => 36,
                    FiniteUnaryOperation::Adjoint => 37,
                };
                unary_expr(encoder, tag, *value, &canonical_index)?;
            }
            ExprNode::FiniteBinary(operation, left, right) => {
                use eqiora_schema::kernel::FiniteBinaryOperation;
                let tag = match operation {
                    FiniteBinaryOperation::Apply => 38,
                    FiniteBinaryOperation::Compose => 39,
                    FiniteBinaryOperation::Pair => 40,
                };
                binary_expr(encoder, tag, *left, *right, &canonical_index)?;
            }
            ExprNode::Gradient(value) => unary_expr(encoder, 11, *value, &canonical_index)?,
            ExprNode::Divergence(value) => unary_expr(encoder, 12, *value, &canonical_index)?,
            ExprNode::SymmetricPart(value) => unary_expr(encoder, 13, *value, &canonical_index)?,
            ExprNode::IsotropicLift(value) => unary_expr(encoder, 14, *value, &canonical_index)?,
            ExprNode::Trace(value) => unary_expr(encoder, 15, *value, &canonical_index)?,
            ExprNode::NormalComponent(value) => unary_expr(encoder, 16, *value, &canonical_index)?,
            ExprNode::PureOperatorApplication(application) => {
                encoder.u8(17)?;
                encoder.raw(&application.definition().bytes())?;
                encoder.len(application.arguments().len())?;
                for argument in application.arguments() {
                    encoder.u32(canonical_expr_id(*argument, &canonical_index)?)?;
                }
            }
            _ => return Err(newer_vocabulary("expression node")),
        }
    }
    encoder.len(expression.roots().len())?;
    for root in expression.roots() {
        encoder.u32(canonical_expr_id(*root, &canonical_index)?)?;
    }
    encoder.len(expression.definitions().len())?;
    for (digest, definition) in expression.definitions() {
        encoder.raw(&digest.bytes())?;
        let bytes = definition.canonical_bytes();
        budget.account_bytes(bytes.len())?;
        encoder.bytes(&bytes)?;
    }
    property::encode(encoder, expression, &canonical_index)?;
    Ok(canonical_index)
}

fn encode_symbol(
    encoder: &mut Encoder,
    symbol: SymbolRef,
    scope: u8,
    expression_index: u32,
    ids: &BTreeMap<RawId, usize>,
    references: &mut Vec<Reference>,
    budget: &mut ConstructionBudget,
) -> Result<(), Diagnostic> {
    let (tag, target) = match symbol {
        SymbolRef::Field(id) => (1, Some(id.erase())),
        SymbolRef::Derivative(id) => (2, Some(id.erase())),
        SymbolRef::Pre(id) => (3, Some(id.erase())),
        SymbolRef::Next(id) => (4, Some(id.erase())),
        SymbolRef::Parameter(id) => (5, Some(id.erase())),
        SymbolRef::Port(id) => (6, Some(id.erase())),
        SymbolRef::Across(id) => (7, Some(id.erase())),
        SymbolRef::Through(id) => (8, Some(id.erase())),
        SymbolRef::PortTrace(id) => (9, Some(id.erase())),
        SymbolRef::PortFlux(id) => (10, Some(id.erase())),
        SymbolRef::Time => (11, None),
        _ => return Err(newer_vocabulary("expression symbol")),
    };
    encoder.u8(tag)?;
    if let Some(target) = target {
        let mut label = Encoder::new(32);
        label.u8(3)?;
        label.u8(scope)?;
        label.u32(expression_index)?;
        label.u8(tag)?;
        push_reference(
            references,
            label.finish()?,
            lookup(ids, target, "expression symbol")?,
            budget,
        )?;
    }
    Ok(())
}

fn unary_expr(
    encoder: &mut Encoder,
    tag: u8,
    value: eqiora_schema::kernel::ExprId,
    canonical_index: &[u32],
) -> Result<(), Diagnostic> {
    encoder.u8(tag)?;
    encoder.u32(canonical_expr_id(value, canonical_index)?)
}

fn binary_expr(
    encoder: &mut Encoder,
    tag: u8,
    left: eqiora_schema::kernel::ExprId,
    right: eqiora_schema::kernel::ExprId,
    canonical_index: &[u32],
) -> Result<(), Diagnostic> {
    encoder.u8(tag)?;
    encoder.u32(canonical_expr_id(left, canonical_index)?)?;
    encoder.u32(canonical_expr_id(right, canonical_index)?)
}

fn canonical_expression_order(
    expression: &ExprDag,
    extra_roots: &[eqiora_schema::kernel::ExprId],
) -> Result<(Vec<usize>, Vec<u32>), Diagnostic> {
    let nodes = expression.nodes();
    let mut state = vec![0_u8; nodes.len()];
    let mut order = Vec::new();
    order
        .try_reserve_exact(nodes.len())
        .map_err(|_| fingerprint_error("cannot reserve canonical expression order"))?;
    for root in expression.roots().iter().chain(extra_roots) {
        let root = expression_index(*root, nodes.len())?;
        let mut stack = vec![(root, false)];
        while let Some((index, exiting)) = stack.pop() {
            if exiting {
                if state[index] != 2 {
                    state[index] = 2;
                    order.push(index);
                }
                continue;
            }
            match state[index] {
                2 => continue,
                1 => {
                    return Err(fingerprint_error(
                        "structural semantic projection found a cyclic expression DAG",
                    ));
                }
                _ => state[index] = 1,
            }
            stack.push((index, true));
            let operands = expression_operands(&nodes[index]);
            for operand in operands.into_iter().rev() {
                let operand = expression_index(operand, nodes.len())?;
                if state[operand] != 2 {
                    stack.push((operand, false));
                }
            }
        }
    }
    if order.len() != nodes.len() {
        return Err(fingerprint_error(
            "structural semantic projection rejects unreachable expression nodes",
        ));
    }
    let mut canonical_index = vec![0_u32; nodes.len()];
    for (canonical, &original) in order.iter().enumerate() {
        canonical_index[original] = u32::try_from(canonical)
            .map_err(|_| fingerprint_error("canonical expression index exceeds u32"))?;
    }
    Ok((order, canonical_index))
}

fn expression_operands(node: &ExprNode) -> Vec<eqiora_schema::kernel::ExprId> {
    match node {
        ExprNode::Array { elements } => elements.clone(),
        ExprNode::Complex { real, imag } => vec![*real, *imag],
        ExprNode::Sample { value, .. }
        | ExprNode::Not(value)
        | ExprNode::Ordinal(value)
        | ExprNode::ToReal(value)
        | ExprNode::ToInteger(value)
        | ExprNode::Hold(value)
        | ExprNode::Index { value, .. }
        | ExprNode::Neg(value)
        | ExprNode::PowI(value, _)
        | ExprNode::UnaryMath(_, value)
        | ExprNode::FiniteUnary(_, value)
        | ExprNode::Gradient(value)
        | ExprNode::Divergence(value)
        | ExprNode::SymmetricPart(value)
        | ExprNode::IsotropicLift(value)
        | ExprNode::Trace(value)
        | ExprNode::NormalComponent(value) => vec![*value],
        ExprNode::Add(left, right)
        | ExprNode::FiniteBinary(_, left, right)
        | ExprNode::Sub(left, right)
        | ExprNode::Mul(left, right)
        | ExprNode::Compare(_, left, right)
        | ExprNode::And(left, right)
        | ExprNode::Or(left, right)
        | ExprNode::Quotient(left, right)
        | ExprNode::Remainder(left, right)
        | ExprNode::Div(left, right) => vec![*left, *right],
        ExprNode::Select {
            condition,
            then_value,
            else_value,
        } => vec![*condition, *then_value, *else_value],
        ExprNode::Require { condition, value } => vec![*condition, *value],
        ExprNode::PureOperatorApplication(application) => application.arguments().to_vec(),
        ExprNode::Constant(_) | ExprNode::Symbol(_) | ExprNode::SpatialCoordinate(_) => Vec::new(),
        _ => Vec::new(),
    }
}

pub(super) fn canonical_expr_id(
    id: eqiora_schema::kernel::ExprId,
    canonical_index: &[u32],
) -> Result<u32, Diagnostic> {
    let index = expression_index(id, canonical_index.len())?;
    canonical_index
        .get(index)
        .copied()
        .ok_or_else(|| fingerprint_error("canonical expression index is absent"))
}

fn expression_index(id: eqiora_schema::kernel::ExprId, upper: usize) -> Result<usize, Diagnostic> {
    usize::try_from(id.index())
        .ok()
        .filter(|index| *index < upper)
        .ok_or_else(|| fingerprint_error("expression operand is outside its DAG"))
}
