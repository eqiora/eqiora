//! Volume terms of the Green identity; boundary discharge remains a separate proof.
use eqiora_compiler::AuthoredFormExpressionV1 as E;
use eqiora_core::RawId;
use eqiora_schema::kernel::{ExprId, ExprNode, typing::TypedResidual};

pub(in crate::form_compiler) fn volume(
    typed: &TypedResidual<RawId>,
    test: E,
    conjugated: bool,
) -> Option<E> {
    let [root] = typed.expression().roots() else {
        return None;
    };
    Context {
        typed,
        test,
        conjugated,
        remaining: 65536,
    }
    .expression(*root, 0)
}

struct Context<'a> {
    typed: &'a TypedResidual<RawId>,
    test: E,
    conjugated: bool,
    remaining: usize,
}

impl Context<'_> {
    fn expression(&mut self, id: ExprId, depth: usize) -> Option<E> {
        if depth > 96 {
            return None;
        }
        self.remaining = self.remaining.checked_sub(1)?;
        let scalar_coefficient = |id| {
            self.typed
                .node_type(id)
                .is_some_and(|ty| ty.shape().is_scalar() && ty.support.is_none())
        };
        let expression = match self.typed.expression().node(id)? {
            ExprNode::Add(a, b) => E::Add {
                left: Box::new(self.expression(*a, depth + 1)?),
                right: Box::new(self.expression(*b, depth + 1)?),
            },
            ExprNode::Sub(a, b) => E::Sub {
                left: Box::new(self.expression(*a, depth + 1)?),
                right: Box::new(self.expression(*b, depth + 1)?),
            },
            ExprNode::Neg(a) => E::Neg {
                value: Box::new(self.expression(*a, depth + 1)?),
            },
            ExprNode::Mul(a, b) if scalar_coefficient(*a) => E::Mul {
                left: Box::new(E::from_expression(self.typed.expression(), *a).ok()??),
                right: Box::new(self.expression(*b, depth + 1)?),
            },
            ExprNode::Mul(a, b) if scalar_coefficient(*b) => E::Mul {
                left: Box::new(self.expression(*a, depth + 1)?),
                right: Box::new(E::from_expression(self.typed.expression(), *b).ok()??),
            },
            ExprNode::Divergence(flux) => E::Neg {
                value: Box::new(pair(
                    E::Gradient {
                        value: Box::new(self.test.clone()),
                    },
                    E::from_expression(self.typed.expression(), *flux).ok()??,
                    self.typed.node_type(*flux)?.shape().rank(),
                    self.conjugated,
                )?),
            },
            _ => pair(
                self.test.clone(),
                E::from_expression(self.typed.expression(), id).ok()??,
                self.typed.node_type(id)?.shape().rank(),
                self.conjugated,
            )?,
        };
        Some(expression)
    }
}

pub(in crate::form_compiler) fn pair(
    test: E,
    value: E,
    rank: usize,
    conjugated: bool,
) -> Option<E> {
    let test = if conjugated {
        E::Conjugate {
            value: Box::new(test),
        }
    } else {
        test
    };
    let (left, right) = (Box::new(test), Box::new(value));
    Some(match rank {
        0 => E::Mul { left, right },
        1 => E::Dot { left, right },
        2 => E::Frobenius { left, right },
        _ => return None,
    })
}
