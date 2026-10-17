//! Degree projection of composed prescribed data onto its spatial tape owner.
use super::*;

impl<S: Coefficient> Data<S> {
    pub(in crate::form_compiler) fn polynomial_degree(&self) -> Option<usize> {
        if !self.spatial() {
            return Some(0);
        }
        match self.0.as_ref() {
            Node::Tape(tape) => tape.polynomial_degree(),
            Node::CoordinateDerivative(tape, _) => tape
                .polynomial_degree()
                .map(|degree| degree.saturating_sub(1)),
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
