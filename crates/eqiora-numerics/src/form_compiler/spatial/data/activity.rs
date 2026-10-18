//! Structural activity never depends on a sampled value or a numerical tangent.
use super::*;

impl<S: Coefficient> Data<S> {
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
    pub(super) fn constant_in_direction(
        &self,
        coordinate: &ScalarSymbolCoordinate,
        direction: Direction<'_>,
    ) -> Result<bool, Diagnostic> {
        match self {
            Self::Scalar { data, .. } => data.constant_in_direction(direction),
            Self::ParameterComponent { .. } | Self::Coordinate { .. } => {
                Ok(direction(coordinate) == 0.)
            }
        }
    }
}
