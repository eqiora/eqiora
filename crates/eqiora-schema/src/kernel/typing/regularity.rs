//! Bounded trace-theorem admission, separate from numerical basis selection.
use super::*;
use crate::kernel::{
    FiniteUnaryOperation, SpatialRegularity,
    pure_operator::{CalculusNode, PureOperatorDefinition},
};
use std::convert::Infallible;

const FULL: u8 = 1;
const NORMAL: u8 = 2;
const TANGENTIAL: u8 = 4;
const SMOOTH: u8 = 8;
const CONSTANT: u8 = 16;
const WEAK_BOUNDARY: u8 = 32;
const H1: u8 = FULL | NORMAL | TANGENTIAL;
const CLASSICAL: u8 = H1 | SMOOTH;
const SPATIAL_CONSTANT: u8 = CLASSICAL | CONSTANT;

impl<I: Clone + Eq> TypedResidual<I> {
    /// Check boundary trace availability from authored symbol regularity.
    ///
    /// This bounded profile preserves sums and smooth scalar multipliers,
    /// admits full H1, normal H(div), and tangential H(curl) traces, and requires
    /// smooth operands for traces of derivatives. No numerical basis, shared
    /// interface value, or general distribution product is inferred.
    ///
    /// Tangential traces retain the existing exact skew lift followed by normal
    /// contraction. Admission checks that closed calculus definition, never its
    /// source name. Other component maps cannot acquire this trace theorem.
    pub fn validate_trace_regularity(
        &self,
        mut symbol_regularity: impl FnMut(SymbolRef) -> SpatialRegularity,
    ) -> Result<(), Vec<TypedResidualError<I, Infallible>>> {
        if !self.expression.nodes().iter().any(|node| {
            matches!(
                node,
                ExprNode::Trace { .. } | ExprNode::NormalComponent { .. }
            )
        }) {
            return Ok(());
        }
        let mut degrees = std::collections::BTreeMap::new();
        let mut capabilities = Vec::<u8>::with_capacity(self.expression.nodes().len());
        let mut errors = Vec::new();
        for (index, node) in self.expression.nodes().iter().enumerate() {
            let get = |id: ExprId| capabilities[id.index() as usize];
            let scalar = |id: ExprId| self.node_types[id.index() as usize].shape().is_scalar();
            let smooth = |flags: u8| if flags & SMOOTH != 0 { CLASSICAL } else { 0 };
            let mut weak_input = false;
            node.try_for_each_operand(|id| {
                weak_input |= get(id) & WEAK_BOUNDARY != 0;
                Ok::<_, Infallible>(())
            })
            .unwrap();
            let mut flags = match node {
                ExprNode::Constant(_) => SPATIAL_CONSTANT,
                ExprNode::Symbol(_) if self.node_types[index].support.is_none() => SPATIAL_CONSTANT,
                ExprNode::Symbol(symbol) => match symbol {
                    SymbolRef::Parameter(_) | SymbolRef::Time => SPATIAL_CONSTANT,
                    SymbolRef::Coordinate { .. } => CLASSICAL,
                    _ => match symbol_regularity(*symbol) {
                        SpatialRegularity::Unspecified | SpatialRegularity::L2 => 0,
                        SpatialRegularity::H1 => H1,
                        SpatialRegularity::HCurl => TANGENTIAL,
                        SpatialRegularity::HDiv => NORMAL,
                        SpatialRegularity::Smooth => CLASSICAL,
                    },
                },
                ExprNode::Neg(value) => get(*value),
                ExprNode::Add(left, right) | ExprNode::Sub(left, right) => get(*left) & get(*right),
                ExprNode::Mul(left, right) => {
                    let (a, b) = (get(*left), get(*right));
                    let result = if a & SMOOTH != 0 && scalar(*left) {
                        b
                    } else if b & SMOOTH != 0 && scalar(*right) {
                        a
                    } else {
                        smooth(a & b)
                    };
                    (result & !CONSTANT) | (a & b & CONSTANT)
                }
                ExprNode::Div(left, right) if get(*right) & CONSTANT != 0 => get(*left),
                ExprNode::PowI(_, 0) => SPATIAL_CONSTANT,
                ExprNode::PowI(value, 1) => get(*value),
                ExprNode::PowI(value, power) if *power > 0 => smooth(get(*value)),
                ExprNode::UnaryMath(operation, value) => match operation {
                    UnaryMathFunction::Conj | UnaryMathFunction::Real | UnaryMathFunction::Imag => {
                        get(*value)
                    }
                    UnaryMathFunction::Sin
                    | UnaryMathFunction::Cos
                    | UnaryMathFunction::Exp
                    | UnaryMathFunction::Abs2 => smooth(get(*value)),
                    UnaryMathFunction::Abs if get(*value) & FULL != 0 => H1,
                    _ => 0,
                },
                ExprNode::Gradient(value)
                | ExprNode::Divergence(value)
                | ExprNode::CoordinatePartial { value, .. } => smooth(get(*value)),
                ExprNode::SymmetricPart(value)
                | ExprNode::IsotropicLift(value)
                | ExprNode::FiniteUnary(
                    FiniteUnaryOperation::Transpose
                    | FiniteUnaryOperation::Adjoint
                    | FiniteUnaryOperation::MatrixTrace
                    | FiniteUnaryOperation::PermuteFactors(_),
                    value,
                )
                | ExprNode::Index { value, .. } => {
                    let input = get(*value);
                    if input & SMOOTH != 0 {
                        CLASSICAL
                    } else if input & FULL != 0 {
                        H1
                    } else {
                        0
                    }
                }
                ExprNode::FiniteUnary(FiniteUnaryOperation::Determinant, value) => {
                    smooth(get(*value))
                }
                ExprNode::FiniteUnary(FiniteUnaryOperation::Inverse, _) => 0,
                ExprNode::Complex { real, imag } => {
                    let common = get(*real) & get(*imag);
                    if common & SMOOTH != 0 {
                        CLASSICAL
                    } else if common & FULL != 0 {
                        H1
                    } else {
                        0
                    }
                }
                ExprNode::Array { elements } => {
                    let common = elements
                        .iter()
                        .fold(SPATIAL_CONSTANT, |flags, id| flags & get(*id));
                    if common & SMOOTH != 0 {
                        CLASSICAL
                    } else if common & FULL != 0 {
                        H1
                    } else {
                        0
                    }
                }
                ExprNode::Select {
                    condition,
                    then_value,
                    else_value,
                } if self.node_types[condition.index() as usize]
                    .support
                    .is_none() =>
                {
                    get(*then_value) & get(*else_value)
                }
                ExprNode::PureOperatorApplication(application) => {
                    let definition = self
                        .expression
                        .definition(application.definition())
                        .expect("closed definition table");
                    let common = application
                        .arguments()
                        .iter()
                        .fold(SPATIAL_CONSTANT, |flags, id| flags & get(*id));
                    let degree = *degrees
                        .entry(application.definition())
                        .or_insert_with(|| polynomial_degree(definition));
                    let mut result = if degree.is_some() && common & SMOOTH != 0 {
                        CLASSICAL
                    } else if degree.is_some_and(|degree| degree <= 1) && common & FULL != 0 {
                        H1
                    } else {
                        0
                    };
                    if let [argument] = application.arguments()
                        && let Some(dimensions) = self.node_types[argument.index() as usize]
                            .support
                            .as_ref()
                            .and_then(SpatialSupport::ambient_dimensions)
                        && let Ok(dimensions) = u32::try_from(dimensions)
                        && PureOperatorDefinition::tangential_lift(dimensions)
                            .is_ok_and(|lift| lift == *definition)
                        && get(*argument) & TANGENTIAL != 0
                    {
                        result |= NORMAL;
                    }
                    result
                }
                ExprNode::Trace { value, .. } | ExprNode::NormalComponent { value, .. } => {
                    let flags = get(*value);
                    let normal = matches!(node, ExprNode::NormalComponent { .. });
                    let already_on_boundary = normal
                        && self.node_types[value.index() as usize].support
                            == self.node_types[index].support;
                    let required = if normal { NORMAL } else { FULL };
                    if !already_on_boundary && flags & required == 0 {
                        errors.push(TypedResidualError::Type {
                            node_index: index as u32,
                            error: TypeViolation::TraceRegularityInsufficient,
                        });
                    }
                    smooth(flags)
                        | if normal && !already_on_boundary && flags & FULL == 0 {
                            WEAK_BOUNDARY
                        } else {
                            0
                        }
                }
                _ => 0,
            };
            // Material coefficients may be compound global expressions (for
            // example 1 - poisson_ratio^2). Their typed absence of spatial
            // support, rather than their syntax, establishes spatial constancy.
            if self.node_types[index].support.is_none() && !weak_input {
                flags = SPATIAL_CONSTANT;
            }
            if weak_input {
                let admitted = match node {
                    ExprNode::Neg(_)
                    | ExprNode::Add(_, _)
                    | ExprNode::Sub(_, _)
                    | ExprNode::Index { .. }
                    | ExprNode::Array { .. }
                    | ExprNode::Complex { .. }
                    | ExprNode::SymmetricPart(_)
                    | ExprNode::IsotropicLift(_)
                    | ExprNode::UnaryMath(
                        UnaryMathFunction::Conj | UnaryMathFunction::Real | UnaryMathFunction::Imag,
                        _,
                    ) => true,
                    ExprNode::Mul(left, right) => {
                        (get(*left) & SMOOTH != 0 && scalar(*left))
                            || (get(*right) & SMOOTH != 0 && scalar(*right))
                    }
                    ExprNode::Div(_, right) => get(*right) & CONSTANT != 0,
                    ExprNode::NormalComponent { value, .. } => {
                        self.node_types[value.index() as usize].support
                            == self.node_types[index].support
                    }
                    ExprNode::PureOperatorApplication(application) => degrees
                        .get(&application.definition())
                        .copied()
                        .flatten()
                        .is_some_and(|degree| degree <= 1),
                    ExprNode::FiniteUnary(
                        FiniteUnaryOperation::Transpose
                        | FiniteUnaryOperation::Adjoint
                        | FiniteUnaryOperation::MatrixTrace
                        | FiniteUnaryOperation::PermuteFactors(_),
                        _,
                    ) => true,
                    _ => false,
                };
                if !admitted {
                    errors.push(TypedResidualError::Type {
                        node_index: index as u32,
                        error: TypeViolation::WeakTraceOperationUnsupported,
                    });
                }
                flags |= WEAK_BOUNDARY;
            }
            capabilities.push(flags);
        }
        if errors.is_empty() {
            Ok(())
        } else {
            Err(errors)
        }
    }
}

/// A conservative degree bound on a closed polynomial component map. Conditions,
/// nonlinear scalar functions and unproved branches do not enter this profile.
fn polynomial_degree(definition: &PureOperatorDefinition) -> Option<u8> {
    let mut degrees = Vec::<Option<u8>>::with_capacity(definition.nodes().len());
    for node in definition.nodes() {
        let get = |id: crate::kernel::pure_operator::CalculusNodeId| degrees[id.index() as usize];
        let degree = match node {
            CalculusNode::Rational { .. } | CalculusNode::KroneckerDelta(_, _) => Some(0),
            CalculusNode::FormalComponent { .. } => Some(1),
            CalculusNode::Neg(value)
            | CalculusNode::BoundInput(value)
            | CalculusNode::Differentiated { value, .. } => get(*value),
            CalculusNode::Add(left, right) => get(*left).zip(get(*right)).map(|(a, b)| a.max(b)),
            CalculusNode::Mul(left, right) => get(*left)
                .zip(get(*right))
                .map(|(a, b)| a.saturating_add(b).min(2)),
            _ => None,
        };
        degrees.push(degree);
    }
    degrees[definition.root().index() as usize]
}

#[cfg(test)]
mod tests;
