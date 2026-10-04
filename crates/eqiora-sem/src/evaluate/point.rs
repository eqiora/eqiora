//! A point is admitted against exact Model factors before a representation can sample it.
use super::*;
use eqiora_core::{Id, entity::kinds};
use eqiora_schema::kernel::{BoundarySide, DomainKind, KernelNode, typing::SpatialSupport};

/// A value or first coordinate derivative requested from an admitted reconstruction.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EvaluationInput {
    /// Ordinary symbol value, optionally at an exact point.
    Value(SymbolRef),
    /// First coordinate derivative of a Field at the accompanying exact point.
    CoordinatePartial {
        /// Retained Field identity.
        field: Id<kinds::Field>,
        /// Exact coordinate factor.
        factor: Id<kinds::Domain>,
        /// Axis local to that factor.
        axis: usize,
    },
}

/// Exact, dimensioned evaluation context admitted by the retained Model.
/// The requested side selects an approach from lower or higher coordinate values.
/// It carries no mesh search, interpolation, extrapolation or time-advancement policy.
#[derive(Debug, Clone, PartialEq)]
pub struct EvaluationPoint {
    domain: Id<kinds::Domain>,
    coordinates: Vec<((RawId, usize), DynQuantity)>,
    side: Option<BoundarySide>,
}

impl EvaluationPoint {
    /// Exact source support whose complete coordinates are bound.
    #[must_use]
    pub const fn domain(&self) -> Id<kinds::Domain> {
        self.domain
    }

    /// Factor identities, local axes, and dimensioned values in the authored binding order.
    pub fn coordinates(&self) -> impl ExactSizeIterator<Item = ((RawId, usize), DynQuantity)> + '_ {
        self.coordinates.iter().copied()
    }

    /// Explicit one-sided approach, if requested.
    #[must_use]
    pub const fn side(&self) -> Option<BoundarySide> {
        self.side
    }

    pub(super) fn bind(
        program: &KernelProgram,
        expression: &ExprDag,
        at: &[(ExprId, ExprId)],
        values: &[ValueLiteral],
        side: Option<BoundarySide>,
    ) -> Result<Self, Diagnostic> {
        if at.is_empty()
            || at.len() > 64
            || at.len() != values.len()
            || (side.is_some() && at.len() != 1)
        {
            return Err(invalid(
                "point evaluation has an invalid coordinate inventory or side",
            ));
        }
        let mut domain = None;
        let mut coordinates = Vec::with_capacity(at.len());
        for ((selector, _), value) in at.iter().zip(values) {
            let Some(ExprNode::Symbol(SymbolRef::Coordinate {
                support,
                factor,
                axis,
            })) = expression.node(*selector)
            else {
                return Err(invalid("evaluation selector is not an exact coordinate"));
            };
            if domain.is_some_and(|domain| domain != *support) {
                return Err(invalid(
                    "evaluation coordinates have different source supports",
                ));
            }
            domain = Some(*support);
            let value = value
                .real_scalar_value()
                .ok_or_else(|| invalid("evaluation point requires a real scalar"))?;
            checked_coordinate(program, *support, *factor, *axis, value, side, at.len())?;
            let key = (factor.erase(), *axis);
            if coordinates.iter().any(|(existing, _)| *existing == key) {
                return Err(invalid("evaluation repeats an exact coordinate"));
            }
            coordinates.push((key, value));
        }
        Ok(Self {
            domain: domain.expect("nonempty point"),
            coordinates,
            side,
        })
    }

    pub(crate) fn validate(&self, program: &KernelProgram) -> Result<(), Diagnostic> {
        for ((factor, axis), value) in &self.coordinates {
            let factor = factor
                .downcast()
                .ok_or_else(|| invalid("evaluation factor is not a Domain"))?;
            checked_coordinate(
                program,
                self.domain,
                factor,
                *axis,
                *value,
                self.side,
                self.coordinates.len(),
            )?;
        }
        Ok(())
    }

    pub(super) fn coordinate(
        &self,
        support: Id<kinds::Domain>,
        factor: Id<kinds::Domain>,
        axis: usize,
    ) -> Result<ValueLiteral, Diagnostic> {
        if support != self.domain {
            return Err(invalid(
                "coordinate read belongs to a foreign point support",
            ));
        }
        self.coordinates
            .iter()
            .find(|(key, _)| *key == (factor.erase(), axis))
            .ok_or_else(|| invalid("coordinate read is absent from the exact evaluation point"))
            .and_then(|(_, value)| ValueLiteral::try_from(*value).map_err(discrete_error))
    }
}

fn checked_coordinate(
    program: &KernelProgram,
    support: Id<kinds::Domain>,
    factor: Id<kinds::Domain>,
    axis: usize,
    value: DynQuantity,
    side: Option<BoundarySide>,
    count: usize,
) -> Result<(), Diagnostic> {
    let support_type = program
        .spatial_support(support)
        .ok_or_else(|| invalid("evaluation support is outside the Model"))?;
    let admitted_factor = match support_type {
        SpatialSupport::Coordinates { factors, .. } => factors
            .iter()
            .any(|(id, _, axes)| *id == factor.erase() && axis < *axes),
        SpatialSupport::Volume { domain, dimensions } => {
            *domain == factor.erase() && axis < *dimensions
        }
        _ => false,
    };
    if !admitted_factor || support_type.intrinsic_dimensions() != count {
        return Err(invalid(
            "evaluation omits or substitutes an exact support coordinate",
        ));
    }
    let Some(KernelNode::Domain(definition)) = program.node(factor.erase()) else {
        return Err(invalid("evaluation factor is outside the Model"));
    };
    let bounds = match definition.kind() {
        DomainKind::CoordinateInterval { bounds } if axis == 0 => *bounds,
        _ => *program
            .resolved_cartesian_bounds(factor)?
            .get(axis)
            .ok_or_else(|| invalid("evaluation axis is outside its factor"))?,
    };
    if value.dim() != bounds.lower().dim()
        || !value.value().is_finite()
        || value.value() < bounds.lower().value()
        || value.value() > bounds.upper().value()
    {
        return Err(invalid(
            "evaluation coordinate has wrong units or lies outside its exact support",
        ));
    }
    if (side == Some(BoundarySide::Lower) && value.value() == bounds.lower().value())
        || (side == Some(BoundarySide::Upper) && value.value() == bounds.upper().value())
    {
        return Err(invalid(
            "evaluation side approaches from outside the exact support",
        ));
    }
    Ok(())
}

fn invalid(message: &str) -> Diagnostic {
    Diagnostic::error(codes::INVALID_EXPRESSION_DAG, message)
}

// A one-sided value is a limit, not a substitution at a branch boundary.
// This bounded profile admits continuous arithmetic plus representation-owned
// Field limits; it does not invent limits for authored piecewise definitions.
pub(super) fn require_side_regularity(
    expression: &ExprDag,
    node: &ExprNode,
) -> Result<(), Diagnostic> {
    use eqiora_schema::kernel::{UnaryMathFunction, pure_operator::CalculusNode};
    let unsupported = match node {
        ExprNode::Select { .. }
        | ExprNode::Require { .. }
        | ExprNode::Compare(..)
        | ExprNode::ToInteger(_)
        | ExprNode::Quotient(..)
        | ExprNode::Remainder(..)
        | ExprNode::UnaryMath(
            UnaryMathFunction::Arg | UnaryMathFunction::Log | UnaryMathFunction::Sqrt,
            _,
        ) => true,
        ExprNode::PureOperatorApplication(application) => expression
            .definition(application.definition())
            .expect("admitted operator")
            .nodes()
            .iter()
            .any(|node| {
                matches!(
                    node,
                    CalculusNode::Select { .. }
                        | CalculusNode::Require { .. }
                        | CalculusNode::Compare(..)
                        | CalculusNode::UnaryMath(
                            UnaryMathFunction::Arg
                                | UnaryMathFunction::Log
                                | UnaryMathFunction::Sqrt,
                            _
                        )
                )
            }),
        _ => false,
    };
    if unsupported {
        return Err(invalid(
            "one-sided analytic evaluation requires admitted continuous arithmetic; piecewise or branch limits are unavailable",
        ));
    }
    Ok(())
}
