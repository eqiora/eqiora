//! Polynomial degree follows the same coefficient inputs and operations as evaluation.
use super::*;

impl<S: Coefficient> Data<S> {
    pub(in crate::form_compiler) fn polynomial_degree(&self) -> Option<usize> {
        if !self.spatial() {
            return Some(0);
        }
        match self.0.as_ref() {
            Node::Constant { .. } => Some(0),
            Node::Derivative(_, tangent) => tangent.polynomial_degree(),
            Node::Input { input, .. } => input.polynomial_degree(),
            Node::Pointwise(data) => data.polynomial_degree(),
            Node::Chart(data, _) | Node::Part { data, .. } => data.polynomial_degree(),
            Node::Add(a, b) => a
                .polynomial_degree()
                .zip(b.polynomial_degree())
                .map(|(a, b)| a.max(b)),
            Node::Mul(a, b) => a
                .polynomial_degree()
                .zip(b.polynomial_degree())
                .map(|(a, b)| a.saturating_add(b)),
            Node::Div(a, b) => (b.polynomial_degree() == Some(0))
                .then(|| a.polynomial_degree())
                .flatten(),
            Node::Affine(_) => None,
        }
    }
}

impl<S: Coefficient> Input<S> {
    pub(super) fn polynomial_degree(&self) -> Option<usize> {
        match self {
            Self::Coordinate { .. } => Some(1),
            Self::BoundScalar { .. } => Some(0),
            Self::Scalar { data, .. } => data.polynomial_degree(),
        }
    }
}
