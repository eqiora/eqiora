//! One current canonical pure-definition representation.
use super::*;

pub(super) fn canonical_definition_bytes(definition: &PureOperatorDefinition) -> Vec<u8> {
    let mut bytes = DEFINITION_DOMAIN.to_vec();
    push_u32(&mut bytes, definition.formals.len());
    for formal in &definition.formals {
        push_value_class(&mut bytes, *formal);
    }
    push_value_class(&mut bytes, definition.result);
    push_u32(&mut bytes, definition.nodes.len());
    for node in &definition.nodes {
        match node {
            CalculusNode::Require { condition, value } => {
                bytes.push(13);
                bytes.extend_from_slice(&condition.index().to_be_bytes());
                bytes.extend_from_slice(&value.index().to_be_bytes());
            }
            CalculusNode::Rational { value, dimension } => {
                bytes.push(0);
                push_rational(&mut bytes, *value);
                push_dimension(&mut bytes, *dimension);
            }
            CalculusNode::Boolean(value) => {
                bytes.push(6);
                bytes.push(u8::from(*value));
            }
            CalculusNode::Compare(op, left, right) => {
                bytes.push(7);
                bytes.push(match op {
                    super::super::ComparisonOp::Equal => 0,
                    super::super::ComparisonOp::NotEqual => 1,
                    super::super::ComparisonOp::Less => 2,
                    super::super::ComparisonOp::LessEqual => 3,
                    super::super::ComparisonOp::Greater => 4,
                    super::super::ComparisonOp::GreaterEqual => 5,
                });
                bytes.extend_from_slice(&left.index().to_be_bytes());
                bytes.extend_from_slice(&right.index().to_be_bytes());
            }
            CalculusNode::Not(value) => {
                bytes.push(8);
                bytes.extend_from_slice(&value.index().to_be_bytes());
            }
            CalculusNode::And(left, right) | CalculusNode::Or(left, right) => {
                bytes.push(if matches!(node, CalculusNode::And(_, _)) {
                    9
                } else {
                    10
                });
                bytes.extend_from_slice(&left.index().to_be_bytes());
                bytes.extend_from_slice(&right.index().to_be_bytes());
            }
            CalculusNode::Select {
                condition,
                then_value,
                else_value,
            } => {
                bytes.push(11);
                for value in [condition, then_value, else_value] {
                    bytes.extend_from_slice(&value.index().to_be_bytes());
                }
            }
            CalculusNode::UnaryMath(function, value) => {
                bytes.push(12);
                bytes.push(match function {
                    super::super::UnaryMathFunction::Sqrt => 0,
                    super::super::UnaryMathFunction::Sin => 1,
                    super::super::UnaryMathFunction::Cos => 2,
                    super::super::UnaryMathFunction::Exp => 3,
                    super::super::UnaryMathFunction::Log => 4,
                    super::super::UnaryMathFunction::Conj => 5,
                    super::super::UnaryMathFunction::Real => 6,
                    super::super::UnaryMathFunction::Imag => 7,
                    super::super::UnaryMathFunction::Abs => 8,
                    super::super::UnaryMathFunction::Abs2 => 9,
                    super::super::UnaryMathFunction::Arg => 10,
                });
                bytes.extend_from_slice(&value.index().to_be_bytes());
            }
            CalculusNode::FormalComponent { formal, axes } => {
                bytes.push(1);
                push_u16(&mut bytes, *formal);
                push_u32(&mut bytes, axes.len());
                for axis in axes {
                    push_u16(&mut bytes, axis.index());
                }
            }
            CalculusNode::KroneckerDelta(left, right) => {
                bytes.push(2);
                push_u16(&mut bytes, left.index());
                push_u16(&mut bytes, right.index());
            }
            CalculusNode::Differentiated { value, source, wrt } => {
                bytes.push(14);
                bytes.extend_from_slice(&value.index().to_be_bytes());
                bytes.extend_from_slice(&source.index().to_be_bytes());
                bytes.extend_from_slice(&wrt.index().to_be_bytes());
            }
            CalculusNode::BoundInput(value) => {
                bytes.push(15);
                bytes.extend_from_slice(&value.index().to_be_bytes());
            }
            CalculusNode::Neg(value) => {
                bytes.push(3);
                bytes.extend_from_slice(&value.index().to_be_bytes());
            }
            CalculusNode::Add(left, right) => {
                bytes.push(4);
                bytes.extend_from_slice(&left.index().to_be_bytes());
                bytes.extend_from_slice(&right.index().to_be_bytes());
            }
            CalculusNode::Mul(left, right) => {
                bytes.push(5);
                bytes.extend_from_slice(&left.index().to_be_bytes());
                bytes.extend_from_slice(&right.index().to_be_bytes());
            }
        }
    }
    bytes.extend_from_slice(&definition.root.index().to_be_bytes());
    bytes
}

fn push_value_class(bytes: &mut Vec<u8>, class: PureValueClass) {
    match class.spatial_rank() {
        None => bytes.push(0),
        Some(rank) => {
            bytes.push(1);
            push_u16(bytes, rank);
        }
    }
    match class.scalar_domain() {
        None => bytes.push(0),
        Some(eqiora_core::ScalarDomain::Real) => bytes.push(1),
        Some(eqiora_core::ScalarDomain::Complex) => bytes.push(2),
        Some(_) => unreachable!("checked pure scalar domain"),
    }
    match class.dimension() {
        None => bytes.push(0),
        Some(dimension) => {
            bytes.push(1);
            for (numerator, denominator) in dimension.exponents() {
                bytes.extend_from_slice(&numerator.to_be_bytes());
                bytes.extend_from_slice(&denominator.to_be_bytes());
            }
        }
    }
}

fn push_rational(bytes: &mut Vec<u8>, value: ExactRational) {
    bytes.extend_from_slice(&value.numerator().to_be_bytes());
    bytes.extend_from_slice(&value.denominator().to_be_bytes());
}

fn push_u16(bytes: &mut Vec<u8>, value: u16) {
    bytes.extend_from_slice(&value.to_be_bytes());
}

fn push_u32(bytes: &mut Vec<u8>, value: usize) {
    let value = u32::try_from(value).expect("bounded pure-operator count fits u32");
    bytes.extend_from_slice(&value.to_be_bytes());
}

fn push_dimension(bytes: &mut Vec<u8>, dimension: DimExponents) {
    for (numerator, denominator) in dimension.exponents() {
        bytes.extend_from_slice(&numerator.to_be_bytes());
        bytes.extend_from_slice(&denominator.to_be_bytes());
    }
}
