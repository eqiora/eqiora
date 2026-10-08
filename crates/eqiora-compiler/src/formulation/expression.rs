//! One bounded source-DAG projection shared by derived scalar forms.
use super::wire::rejection;
use super::*;
use eqiora_schema::kernel::{ExprDag, ExprId, ExprNode, SymbolRef, UnaryMathFunction};

impl AuthoredFormExpressionV1 {
    /// Project a retained scalar expression without changing its mathematical terms.
    /// # Errors
    /// Rejects malformed source expressions. Returns `None` outside the closed inventory.
    pub fn from_expression(dag: &ExprDag, id: ExprId) -> Result<Option<Self>, Diagnostic> {
        match from_dag(dag, id, &mut 65536, 0) {
            Ok(value) => Ok(Some(value)),
            Err(ProjectionFailure::Unsupported) => Ok(None),
            Err(ProjectionFailure::Invalid(error)) => Err(error),
        }
    }
}

enum ProjectionFailure {
    Unsupported,
    Invalid(Diagnostic),
}

impl From<Diagnostic> for ProjectionFailure {
    fn from(error: Diagnostic) -> Self {
        Self::Invalid(error)
    }
}

fn from_dag(
    dag: &ExprDag,
    id: ExprId,
    remaining: &mut usize,
    depth: usize,
) -> Result<AuthoredFormExpressionV1, ProjectionFailure> {
    if *remaining == 0 || depth > 128 {
        return Err(ProjectionFailure::Unsupported);
    }
    *remaining -= 1;
    let mut convert = |id| from_dag(dag, id, remaining, depth + 1).map(Box::new);
    Ok(
        match dag
            .node(id)
            .ok_or_else(|| rejection("missing source expression"))?
        {
            ExprNode::Constant(value) if !value.value_type().shape().is_scalar() => {
                if value.value_type().array_rank() != 0 || value.value_type().map_bases().is_none()
                {
                    return Err(ProjectionFailure::Unsupported);
                }
                let shape = value
                    .value_type()
                    .shape()
                    .extents()
                    .iter()
                    .map(|extent| extent.get())
                    .collect::<Vec<_>>();
                let count = shape
                    .iter()
                    .try_fold(1usize, |n, extent| n.checked_mul(*extent as usize))
                    .ok_or(ProjectionFailure::Unsupported)?;
                *remaining = remaining
                    .checked_sub(count)
                    .ok_or(ProjectionFailure::Unsupported)?;
                AuthoredFormExpressionV1::Components {
                    shape,
                    values: (0..count)
                        .map(|i| value.component(i))
                        .collect::<Option<Vec<_>>>()
                        .ok_or(ProjectionFailure::Unsupported)?,
                }
            }
            ExprNode::Constant(value)
                if value.value_type().shape().is_scalar()
                    && value.value_type().scalar_domain() == ScalarDomain::Complex =>
            {
                let (real, imag) = value.component(0).ok_or(ProjectionFailure::Unsupported)?;
                AuthoredFormExpressionV1::Complex {
                    real: Box::new(AuthoredFormExpressionV1::Number { value: real }),
                    imag: Box::new(AuthoredFormExpressionV1::Number { value: imag }),
                }
            }
            ExprNode::Constant(value) => AuthoredFormExpressionV1::Number {
                value: value
                    .real_scalar_value()
                    .ok_or(ProjectionFailure::Unsupported)?
                    .value(),
            },
            ExprNode::Complex { real, imag } => AuthoredFormExpressionV1::Complex {
                real: convert(*real)?,
                imag: convert(*imag)?,
            },
            ExprNode::Symbol(SymbolRef::Field(id)) => AuthoredFormExpressionV1::Field {
                ulid: id.ulid().to_string(),
            },
            ExprNode::Symbol(SymbolRef::Parameter(id)) => AuthoredFormExpressionV1::Parameter {
                ulid: id.ulid().to_string(),
            },
            ExprNode::Symbol(SymbolRef::Coordinate {
                support,
                factor,
                axis,
            }) => AuthoredFormExpressionV1::Coordinate {
                support_ulid: support.ulid().to_string(),
                factor_ulid: factor.ulid().to_string(),
                axis: *axis,
            },
            ExprNode::Neg(value) => AuthoredFormExpressionV1::Neg {
                value: convert(*value)?,
            },
            ExprNode::Add(left, right) => AuthoredFormExpressionV1::Add {
                left: convert(*left)?,
                right: convert(*right)?,
            },
            ExprNode::Sub(left, right) => AuthoredFormExpressionV1::Sub {
                left: convert(*left)?,
                right: convert(*right)?,
            },
            ExprNode::FiniteBinary(
                eqiora_schema::kernel::FiniteBinaryOperation::Apply,
                left,
                right,
            ) => AuthoredFormExpressionV1::Apply {
                left: convert(*left)?,
                right: convert(*right)?,
            },
            ExprNode::Mul(left, right) => AuthoredFormExpressionV1::Mul {
                left: convert(*left)?,
                right: convert(*right)?,
            },
            ExprNode::Div(left, right) => AuthoredFormExpressionV1::Div {
                left: convert(*left)?,
                right: convert(*right)?,
            },
            ExprNode::PowI(base, exponent) => AuthoredFormExpressionV1::Pow {
                base: convert(*base)?,
                exponent: *exponent,
            },
            ExprNode::CoordinatePartial { value, wrt } => {
                AuthoredFormExpressionV1::CoordinatePartial {
                    value: convert(*value)?,
                    wrt: convert(*wrt)?,
                }
            }
            ExprNode::Gradient(value) => AuthoredFormExpressionV1::Gradient {
                value: convert(*value)?,
            },
            ExprNode::UnaryMath(UnaryMathFunction::Sin, value) => AuthoredFormExpressionV1::Sin {
                value: convert(*value)?,
            },
            ExprNode::UnaryMath(UnaryMathFunction::Conj, value) => {
                AuthoredFormExpressionV1::Conjugate {
                    value: convert(*value)?,
                }
            }
            _ => {
                return Err(ProjectionFailure::Unsupported);
            }
        },
    )
}
