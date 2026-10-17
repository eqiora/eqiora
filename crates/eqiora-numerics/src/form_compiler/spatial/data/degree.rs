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
            Node::Input { input, .. } => match input {
                Input::Coordinate { .. } => Some(1),
                Input::ParameterComponent { .. } => Some(0),
                Input::Scalar { data, .. } => data.polynomial_degree(),
            },
            Node::Chart(data, _)
            | Node::Math(
                UnaryMathFunction::Conj | UnaryMathFunction::Real | UnaryMathFunction::Imag,
                data,
            ) => data.polynomial_degree(),
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
            Node::Pow(a, n) => match (a.polynomial_degree(), *n) {
                (_, 0) | (Some(0), _) => Some(0),
                (Some(degree), n) if n > 0 => Some(degree.saturating_mul(n as usize)),
                _ => None,
            },
            Node::MapFactor(_) => Some(0),
            Node::Pointwise(_)
            | Node::Affine(_)
            | Node::Math(_, _)
            | Node::Cos(_)
            | Node::SqrtDerivativeRoot(_) => None,
        }
    }
}
