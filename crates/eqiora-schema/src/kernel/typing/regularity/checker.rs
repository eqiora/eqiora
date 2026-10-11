//! One incremental trace proof shared by typed expression consumers.
use super::*;
use crate::kernel::pure_operator::OperatorDefinitionDigest;
use std::collections::BTreeMap;

const FULL: u8 = 1;
const NORMAL: u8 = 2;
const TANGENTIAL: u8 = 4;
const SMOOTH: u8 = 8;
const CONSTANT: u8 = 16;
const WEAK_BOUNDARY: u8 = 32;
const H1: u8 = FULL | NORMAL | TANGENTIAL;
const CLASSICAL: u8 = H1 | SMOOTH;
const SPATIAL_CONSTANT: u8 = CLASSICAL | CONSTANT;

/// Incremental boundary-regularity admission for already typed native expression nodes.
///
/// Feed nodes in their expression's topological order. Operand types and closed
/// operator definitions must be the same ones used to type those nodes. Symbol
/// assertions belong to the actual symbols, independently of numerical bases.
#[derive(Default)]
pub struct TraceRegularityChecker {
    capabilities: Vec<u8>,
    degrees: BTreeMap<OperatorDefinitionDigest, Option<u8>>,
}

impl TraceRegularityChecker {
    /// Retain one node's derived trace profile and report an inadmissible operation.
    ///
    /// # Errors
    /// Rejects a trace beyond the authored hypothesis or an unsupported operation
    /// on a weak boundary distribution. The node remains retained after rejection
    /// so callers can collect diagnostics for the rest of the typed expression.
    pub fn check_node<'a, I: Clone + Eq + 'a>(
        &mut self,
        node: &ExprNode,
        result_type: &ExpressionType<I>,
        node_type: impl Fn(ExprId) -> &'a ExpressionType<I>,
        mut symbol_regularity: impl FnMut(SymbolRef) -> SpatialRegularity,
        definition: impl Fn(OperatorDefinitionDigest) -> &'a PureOperatorDefinition,
    ) -> Result<(), TypeViolation<I>> {
        let mut error = None;
        let get = |id: ExprId| self.capabilities[id.index() as usize];
        let scalar = |id: ExprId| node_type(id).shape().is_scalar();
        let smooth = |flags: u8| if flags & SMOOTH != 0 { CLASSICAL } else { 0 };
        let mut weak_input = false;
        node.try_for_each_operand(|id| {
            weak_input |= get(id) & WEAK_BOUNDARY != 0;
            Ok::<_, Infallible>(())
        })
        .unwrap();
        let mut flags = match node {
            ExprNode::Constant(_) => SPATIAL_CONSTANT,
            ExprNode::Symbol(_) if result_type.support.is_none() => SPATIAL_CONSTANT,
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
            ExprNode::FiniteUnary(FiniteUnaryOperation::Determinant, value) => smooth(get(*value)),
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
            } if node_type(*condition).support.is_none() => get(*then_value) & get(*else_value),
            ExprNode::PureOperatorApplication(application) => {
                let definition = definition(application.definition());
                let common = application
                    .arguments()
                    .iter()
                    .fold(SPATIAL_CONSTANT, |flags, id| flags & get(*id));
                let degree = *self
                    .degrees
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
                    && let Some(dimensions) = node_type(*argument)
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
                let already_on_boundary =
                    normal && node_type(*value).support == result_type.support;
                let required = if normal { NORMAL } else { FULL };
                if !already_on_boundary && flags & required == 0 {
                    error.get_or_insert(TypeViolation::TraceRegularityInsufficient);
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
        if result_type.support.is_none() && !weak_input {
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
                    node_type(*value).support == result_type.support
                }
                ExprNode::PureOperatorApplication(application) => self
                    .degrees
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
                error.get_or_insert(TypeViolation::WeakTraceOperationUnsupported);
            }
            flags |= WEAK_BOUNDARY;
        }
        self.capabilities.push(flags);
        error.map_or(Ok(()), Err)
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
