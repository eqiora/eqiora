//! One algebraic proof reads assembly nodes and retained scalar SSA by reference.
use super::super::pointwise::PointwiseData;
use super::*;
use eqiora_ir::{ScalarAlgebraicNode, ScalarExpressionView};

#[derive(Clone, Copy)]
pub(super) enum Expression<'a, S: Coefficient> {
    Data(&'a Data<S>),
    Pointwise(&'a PointwiseData<S>, ScalarExpressionView<'a>),
}

pub(super) enum Algebra<'a, S: Coefficient> {
    Constant(S),
    Input(usize, &'a Input<S>, &'a ScalarSymbolCoordinate),
    Alias(Expression<'a, S>),
    Neg(Expression<'a, S>),
    Add(Expression<'a, S>, Expression<'a, S>),
    Sub(Expression<'a, S>, Expression<'a, S>),
    Mul(Expression<'a, S>, Expression<'a, S>),
    Div(Expression<'a, S>, Expression<'a, S>),
    Opaque,
}

impl<'a, S: Coefficient> Expression<'a, S> {
    pub(super) fn key(self) -> (usize, Option<ScalarExpressionView<'a>>) {
        match self {
            Self::Data(data) => (Arc::as_ptr(&data.0) as usize, None),
            Self::Pointwise(data, view) => (std::ptr::from_ref(data) as usize, Some(view)),
        }
    }

    pub(super) fn algebra(self) -> Algebra<'a, S> {
        match self {
            Self::Data(data) => match data.0.as_ref() {
                Node::Constant { value, .. } => Algebra::Constant(*value),
                Node::Input {
                    dimension,
                    input,
                    coordinate,
                } => Algebra::Input(*dimension, input, coordinate),
                Node::Pointwise(data) => data
                    .expression_view()
                    .map(|view| Algebra::Alias(Self::Pointwise(data, view)))
                    .unwrap_or(Algebra::Opaque),
                Node::Add(a, b) => Algebra::Add(Self::Data(a), Self::Data(b)),
                Node::Mul(a, b) => Algebra::Mul(Self::Data(a), Self::Data(b)),
                Node::Div(a, b) => Algebra::Div(Self::Data(a), Self::Data(b)),
                _ => Algebra::Opaque,
            },
            Self::Pointwise(data, view) => {
                let at = |view| Self::Pointwise(data, view);
                match view.node() {
                    ScalarAlgebraicNode::Constant(value) => {
                        Algebra::Constant(<S as From<f64>>::from(value))
                    }
                    ScalarAlgebraicNode::Input(slot) => {
                        let input = data.input(slot.ordinal() as usize);
                        if let Input::Scalar {
                            data,
                            imaginary: false,
                        } = input
                        {
                            Algebra::Alias(Self::Data(data))
                        } else {
                            Algebra::Input(data.dimension(), input, slot.source())
                        }
                    }
                    ScalarAlgebraicNode::Neg(a) => Algebra::Neg(at(a)),
                    ScalarAlgebraicNode::Add(a, b) => Algebra::Add(at(a), at(b)),
                    ScalarAlgebraicNode::Sub(a, b) => Algebra::Sub(at(a), at(b)),
                    ScalarAlgebraicNode::Mul(a, b) => Algebra::Mul(at(a), at(b)),
                    ScalarAlgebraicNode::Div(a, b) => Algebra::Div(at(a), at(b)),
                    ScalarAlgebraicNode::Opaque => Algebra::Opaque,
                }
            }
        }
    }

    pub(super) fn same_atom(self, other: Self) -> bool {
        match (self.algebra(), other.algebra()) {
            (Algebra::Constant(a), Algebra::Constant(b)) => a == b,
            (Algebra::Input(a, b, c), Algebra::Input(d, e, f)) => a == d && b.same_binding(c, e, f),
            (Algebra::Div(a, b), Algebra::Div(c, d)) => {
                same_expression(a, c) && same_expression(b, d)
            }
            _ => match (self, other) {
                (Self::Pointwise(a, av), Self::Pointwise(b, bv)) => {
                    a.dimension() == b.dimension()
                        && av.same_expression(bv, |left, right| {
                            a.input(left.ordinal() as usize).same_binding(
                                left.source(),
                                b.input(right.ordinal() as usize),
                                right.source(),
                            )
                        })
                }
                (Self::Data(a), Self::Data(b)) => same_data_atom(a, b),
                _ => false,
            },
        }
    }
}
