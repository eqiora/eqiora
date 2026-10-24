//! Structural activity never depends on a sampled value or a numerical tangent.
use super::*;

impl<S: Coefficient> Data<S> {
    /// Availability is distinct from derivative activity. Even a zero tangent
    /// keeps the primal input domains demanded by ordinary evaluation.
    pub(in crate::form_compiler) fn inputs_bound(&self) -> bool {
        match self.0.as_ref() {
            Node::Constant { .. } => true,
            Node::Input { input, .. } => input.inputs_bound(),
            Node::Pointwise(data) => data.inputs_bound(),
            Node::Affine(data) => data.inputs_bound(),
            Node::Chart(data, _) | Node::Part { data, .. } => data.inputs_bound(),
            Node::Derivative(a, b) | Node::Add(a, b) | Node::Mul(a, b) | Node::Div(a, b) => {
                a.inputs_bound() && b.inputs_bound()
            }
        }
    }

    pub(in crate::form_compiler) fn constant_in_direction(
        &self,
        direction: Direction<'_>,
    ) -> Result<bool, Diagnostic> {
        Ok(match self.0.as_ref() {
            Node::Constant { .. } => true,
            Node::Input {
                coordinate, input, ..
            } => input.constant_in_direction(coordinate, direction)?,
            Node::Pointwise(data) => data.constant_in_direction(direction)?,
            Node::Affine(data) => data.constant_in_direction(direction)?,
            Node::Chart(data, _) | Node::Part { data, .. } => {
                data.constant_in_direction(direction)?
            }
            // The original operand remains demanded by evaluate/bind; the value
            // being differentiated here is the retained tangent expression.
            Node::Derivative(_, tangent) => tangent.constant_in_direction(direction)?,
            Node::Add(a, b) | Node::Mul(a, b) | Node::Div(a, b) => {
                a.constant_in_direction(direction)? && b.constant_in_direction(direction)?
            }
        })
    }
}

impl<S: Coefficient> Input<S> {
    pub(super) fn inputs_bound(&self) -> bool {
        match self {
            Self::Scalar { data, .. } => data.inputs_bound(),
            Self::BoundScalar { value } => value.is_some(),
            Self::Coordinate { .. } => true,
        }
    }

    pub(super) fn constant_in_direction(
        &self,
        coordinate: &ScalarSymbolCoordinate,
        direction: Direction<'_>,
    ) -> Result<bool, Diagnostic> {
        match self {
            Self::Scalar { data, .. } => data.constant_in_direction(direction),
            Self::BoundScalar { .. } | Self::Coordinate { .. } => Ok(direction(coordinate) == 0.),
        }
    }
}
