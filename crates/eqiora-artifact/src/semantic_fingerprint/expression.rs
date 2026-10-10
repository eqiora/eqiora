//! Ordered expression projection and canonical operand traversal.
use super::*;

// Rust rejects duplicate discriminants at this canonical expression owner.
#[repr(u8)]
enum ExpressionTag {
    Constant = 1,
    Symbol = 2,
    Neg = 3,
    Add = 4,
    Sub = 5,
    Mul = 6,
    Div = 7,
    PowI = 8,
    UnaryMath = 10,
    Gradient = 11,
    Divergence = 12,
    SymmetricPart = 13,
    IsotropicLift = 14,
    Trace = 15,
    NormalComponent = 16,
    PureOperator = 17,
    Array = 18,
    Index = 19,
    Complex = 20,
    Sample = 21,
    Hold = 22,
    Quotient = 23,
    Remainder = 24,
    ToReal = 25,
    ToInteger = 26,
    Ordinal = 27,
    Compare = 28,
    Not = 29,
    And = 30,
    Or = 31,
    Select = 34,
    Require = 35,
    Transpose = 36,
    Adjoint = 37,
    Apply = 38,
    Compose = 39,
    Pair = 40,
    TensorProduct = 41,
    PermuteFactors = 42,
    CoordinatePartial = 43,
    Evaluate = 44,
    MatrixTrace = 45,
    Determinant = 46,
    Inverse = 47,
    Pullback = 48,
    CoordinateMapFactor = 49,
    CoordinateMapFactorAction = 50,
}

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
                encoder.u8(ExpressionTag::Constant as u8)?;
                encode_literal(encoder, value)?;
                let mut label = Encoder::new(32);
                label.u8(3)?;
                label.u8(scope)?;
                label.u32(index)?;
                label.u8(13)?;
                type_reference(value.value_type(), label.finish()?, ids, references, budget)?;
            }
            ExprNode::Array { elements } => {
                encoder.u8(ExpressionTag::Array as u8)?;
                encoder.u32(
                    u32::try_from(elements.len())
                        .map_err(|_| fingerprint_error("array operands exceed u32"))?,
                )?;
                for element in elements {
                    encoder.u32(canonical_index[element.index() as usize])?;
                }
            }
            ExprNode::Index { value, index } => {
                unary_expr(encoder, ExpressionTag::Index, *value, &canonical_index)?;
                encoder.u32(*index)?;
            }
            ExprNode::Complex { real, imag } => binary_expr(
                encoder,
                ExpressionTag::Complex,
                *real,
                *imag,
                &canonical_index,
            )?,
            ExprNode::Sample { value, clock } => {
                unary_expr(encoder, ExpressionTag::Sample, *value, &canonical_index)?;
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
            ExprNode::Hold(value) => {
                unary_expr(encoder, ExpressionTag::Hold, *value, &canonical_index)?
            }
            ExprNode::Symbol(symbol) => {
                encoder.u8(ExpressionTag::Symbol as u8)?;
                encode_symbol(encoder, *symbol, scope, index, ids, references, budget)?;
            }
            ExprNode::Compare(op, left, right) => {
                encoder.u8(ExpressionTag::Compare as u8)?;
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
            ExprNode::Not(value) => {
                unary_expr(encoder, ExpressionTag::Not, *value, &canonical_index)?
            }
            ExprNode::And(left, right) => {
                binary_expr(encoder, ExpressionTag::And, *left, *right, &canonical_index)?
            }
            ExprNode::Or(left, right) => {
                binary_expr(encoder, ExpressionTag::Or, *left, *right, &canonical_index)?
            }
            ExprNode::Select {
                condition,
                then_value,
                else_value,
            } => {
                encoder.u8(ExpressionTag::Select as u8)?;
                for operand in [condition, then_value, else_value] {
                    encoder.u32(canonical_expr_id(*operand, &canonical_index)?)?;
                }
            }
            ExprNode::Require { condition, value } => binary_expr(
                encoder,
                ExpressionTag::Require,
                *condition,
                *value,
                &canonical_index,
            )?,
            ExprNode::Quotient(left, right) => binary_expr(
                encoder,
                ExpressionTag::Quotient,
                *left,
                *right,
                &canonical_index,
            )?,
            ExprNode::Remainder(left, right) => binary_expr(
                encoder,
                ExpressionTag::Remainder,
                *left,
                *right,
                &canonical_index,
            )?,
            ExprNode::Ordinal(value) => {
                unary_expr(encoder, ExpressionTag::Ordinal, *value, &canonical_index)?
            }
            ExprNode::ToReal(value) => {
                unary_expr(encoder, ExpressionTag::ToReal, *value, &canonical_index)?
            }
            ExprNode::ToInteger(value) => {
                unary_expr(encoder, ExpressionTag::ToInteger, *value, &canonical_index)?
            }
            ExprNode::Neg(value) => {
                unary_expr(encoder, ExpressionTag::Neg, *value, &canonical_index)?
            }
            ExprNode::Add(left, right) => {
                binary_expr(encoder, ExpressionTag::Add, *left, *right, &canonical_index)?
            }
            ExprNode::Sub(left, right) => {
                binary_expr(encoder, ExpressionTag::Sub, *left, *right, &canonical_index)?
            }
            ExprNode::Mul(left, right) => {
                binary_expr(encoder, ExpressionTag::Mul, *left, *right, &canonical_index)?
            }
            ExprNode::Div(left, right) => {
                binary_expr(encoder, ExpressionTag::Div, *left, *right, &canonical_index)?
            }
            ExprNode::PowI(value, exponent) => {
                encoder.u8(ExpressionTag::PowI as u8)?;
                encoder.u32(canonical_expr_id(*value, &canonical_index)?)?;
                encoder.i32(*exponent)?;
            }
            ExprNode::UnaryMath(function, value) => {
                encoder.u8(ExpressionTag::UnaryMath as u8)?;
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
                    FiniteUnaryOperation::MatrixTrace => ExpressionTag::MatrixTrace,
                    FiniteUnaryOperation::Determinant => ExpressionTag::Determinant,
                    FiniteUnaryOperation::Inverse => ExpressionTag::Inverse,
                    FiniteUnaryOperation::Transpose => ExpressionTag::Transpose,
                    FiniteUnaryOperation::Adjoint => ExpressionTag::Adjoint,
                    FiniteUnaryOperation::PermuteFactors(_) => ExpressionTag::PermuteFactors,
                };
                unary_expr(encoder, tag, *value, &canonical_index)?;
                if let FiniteUnaryOperation::PermuteFactors(order) = operation {
                    encoder.u8(order[0])?;
                    encoder.u8(order[1])?;
                }
            }
            ExprNode::FiniteBinary(operation, left, right) => {
                use eqiora_schema::kernel::FiniteBinaryOperation;
                let tag = match operation {
                    FiniteBinaryOperation::Apply => ExpressionTag::Apply,
                    FiniteBinaryOperation::Compose => ExpressionTag::Compose,
                    FiniteBinaryOperation::Pair => ExpressionTag::Pair,
                    FiniteBinaryOperation::TensorProduct => ExpressionTag::TensorProduct,
                };
                binary_expr(encoder, tag, *left, *right, &canonical_index)?;
            }
            ExprNode::CoordinatePartial { value, wrt } => binary_expr(
                encoder,
                ExpressionTag::CoordinatePartial,
                *value,
                *wrt,
                &canonical_index,
            )?,
            ExprNode::CoordinateMapFactorAction {
                value,
                parameter,
                directions,
            } => {
                encoder.u8(ExpressionTag::CoordinateMapFactorAction as u8)?;
                encoder.u32(canonical_expr_id(*value, &canonical_index)?)?;
                encoder.u32(canonical_expr_id(*parameter, &canonical_index)?)?;
                encoder.len(directions.len())?;
                for direction in directions {
                    encoder.u32(canonical_expr_id(*direction, &canonical_index)?)?;
                }
            }
            ExprNode::CoordinateMapFactor { factor, source, at } => {
                use eqiora_schema::kernel::CoordinateMapFactor;
                encoder.u8(ExpressionTag::CoordinateMapFactor as u8)?;
                encoder.u8(match factor {
                    CoordinateMapFactor::SignedJacobian => 0,
                    CoordinateMapFactor::VolumeScale => 1,
                    CoordinateMapFactor::Orientation => 2,
                })?;
                encoder.len(source.len())?;
                for coordinate in source {
                    encoder.u32(canonical_expr_id(*coordinate, &canonical_index)?)?;
                }
                encoder.len(at.len())?;
                for (coordinate, mapped) in at {
                    encoder.u32(canonical_expr_id(*coordinate, &canonical_index)?)?;
                    encoder.u32(canonical_expr_id(*mapped, &canonical_index)?)?;
                }
            }
            ExprNode::Pullback { value, source, at } => {
                encoder.u8(ExpressionTag::Pullback as u8)?;
                encoder.u32(canonical_expr_id(*value, &canonical_index)?)?;
                encoder.len(source.len())?;
                for coordinate in source {
                    encoder.u32(canonical_expr_id(*coordinate, &canonical_index)?)?;
                }
                encoder.len(at.len())?;
                for (coordinate, mapped) in at {
                    encoder.u32(canonical_expr_id(*coordinate, &canonical_index)?)?;
                    encoder.u32(canonical_expr_id(*mapped, &canonical_index)?)?;
                }
            }
            ExprNode::Evaluate { value, at, side } => {
                encoder.u8(ExpressionTag::Evaluate as u8)?;
                encoder.u32(canonical_expr_id(*value, &canonical_index)?)?;
                encoder.len(at.len())?;
                for (coordinate, point) in at {
                    encoder.u32(canonical_expr_id(*coordinate, &canonical_index)?)?;
                    encoder.u32(canonical_expr_id(*point, &canonical_index)?)?;
                }
                encoder.u8(match side {
                    None => 0,
                    Some(eqiora_schema::kernel::BoundarySide::Lower) => 1,
                    Some(eqiora_schema::kernel::BoundarySide::Upper) => 2,
                })?;
            }
            ExprNode::Gradient(value) => {
                unary_expr(encoder, ExpressionTag::Gradient, *value, &canonical_index)?
            }
            ExprNode::Divergence(value) => {
                unary_expr(encoder, ExpressionTag::Divergence, *value, &canonical_index)?
            }
            ExprNode::SymmetricPart(value) => unary_expr(
                encoder,
                ExpressionTag::SymmetricPart,
                *value,
                &canonical_index,
            )?,
            ExprNode::IsotropicLift(value) => unary_expr(
                encoder,
                ExpressionTag::IsotropicLift,
                *value,
                &canonical_index,
            )?,
            ExprNode::Trace { value, on } | ExprNode::NormalComponent { value, on } => {
                let tag = if matches!(node, ExprNode::Trace { .. }) {
                    ExpressionTag::Trace
                } else {
                    ExpressionTag::NormalComponent
                };
                unary_expr(encoder, tag, *value, &canonical_index)?;
                let mut label = Encoder::new(32);
                label.u8(3)?;
                label.u8(scope)?;
                label.u32(index)?;
                label.u8(14)?;
                push_reference(
                    references,
                    label.finish()?,
                    lookup(ids, on.erase(), "boundary operator support")?,
                    budget,
                )?;
            }
            ExprNode::PureOperatorApplication(application) => {
                encoder.u8(ExpressionTag::PureOperator as u8)?;
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
    if let SymbolRef::Coordinate {
        support,
        factor,
        axis,
    } = symbol
    {
        encoder.u8(13)?;
        encoder.usize(axis)?;
        for (role, target) in [(0, support.erase()), (1, factor.erase())] {
            let mut label = Encoder::new(32);
            label.u8(3)?;
            label.u8(scope)?;
            label.u32(expression_index)?;
            label.u8(13)?;
            label.u8(role)?;
            push_reference(
                references,
                label.finish()?,
                lookup(ids, target, "coordinate projection")?,
                budget,
            )?;
        }
        return Ok(());
    }
    let (tag, target) = match symbol {
        SymbolRef::Field(id) => (1, Some(id.erase())),
        SymbolRef::Derivative(id, _) => (2, Some(id.erase())),
        SymbolRef::Pre(id) => (3, Some(id.erase())),
        SymbolRef::Next(id) => (4, Some(id.erase())),
        SymbolRef::Parameter(id) => (5, Some(id.erase())),
        SymbolRef::Port(id) => (6, Some(id.erase())),
        SymbolRef::Across(id) => (7, Some(id.erase())),
        SymbolRef::Through(id) => (8, Some(id.erase())),
        SymbolRef::PortTrace(id) => (9, Some(id.erase())),
        SymbolRef::PortFlux(id) => (10, Some(id.erase())),
        SymbolRef::Time => (11, None),
        SymbolRef::Observable(id) => (12, Some(id.erase())),
        _ => return Err(newer_vocabulary("expression symbol")),
    };
    encoder.u8(tag)?;
    if let SymbolRef::Derivative(_, order) = symbol {
        encoder.u32(order.get())?;
    }
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
    tag: ExpressionTag,
    value: eqiora_schema::kernel::ExprId,
    canonical_index: &[u32],
) -> Result<(), Diagnostic> {
    encoder.u8(tag as u8)?;
    encoder.u32(canonical_expr_id(value, canonical_index)?)
}

fn binary_expr(
    encoder: &mut Encoder,
    tag: ExpressionTag,
    left: eqiora_schema::kernel::ExprId,
    right: eqiora_schema::kernel::ExprId,
    canonical_index: &[u32],
) -> Result<(), Diagnostic> {
    encoder.u8(tag as u8)?;
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
        ExprNode::CoordinatePartial { value, wrt } => vec![*value, *wrt],
        ExprNode::CoordinateMapFactorAction {
            value,
            parameter,
            directions,
        } => [*value, *parameter]
            .into_iter()
            .chain(directions.iter().copied())
            .collect(),
        ExprNode::CoordinateMapFactor { source, at, .. } => source
            .iter()
            .copied()
            .chain(
                at.iter()
                    .flat_map(|(coordinate, mapped)| [*coordinate, *mapped]),
            )
            .collect(),
        ExprNode::Pullback { value, source, at } => std::iter::once(*value)
            .chain(source.iter().copied())
            .chain(
                at.iter()
                    .flat_map(|(coordinate, mapped)| [*coordinate, *mapped]),
            )
            .collect(),
        ExprNode::Evaluate { value, at, .. } => std::iter::once(*value)
            .chain(
                at.iter()
                    .flat_map(|(coordinate, point)| [*coordinate, *point]),
            )
            .collect(),
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
        | ExprNode::Trace { value, .. }
        | ExprNode::NormalComponent { value, .. } => vec![*value],
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
        ExprNode::Constant(_) | ExprNode::Symbol(_) => Vec::new(),
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
