//! Derived scalar execution projection of a retained canonical pure application.
use super::*;
use crate::kernel::pure_operator::{CalculusNode, PureOperatorInstantiation};
use eqiora_core::DynQuantity;

impl ExprDagBuilder {
    /// Append an ordered scalar execution body from a checked pure instantiation.
    /// The canonical Model retains its original application and definition identity.
    /// `max_nodes` bounds the complete destination arena before expansion allocation.
    pub fn project_scalar_operator<I>(
        &mut self,
        instance: &PureOperatorInstantiation<'_, I>,
        arguments: &[ExprId],
        max_nodes: usize,
    ) -> Result<ExprId, Diagnostic> {
        let definition = instance.definition();
        if arguments.len() != definition.formals().len()
            || !definition.result_rule().is_invariant_scalar()
            || definition
                .formals()
                .iter()
                .any(|formal| !formal.is_invariant_scalar())
        {
            return Err(invalid_pure_operator(
                "scalar projection requires exact scalar formal arity and result",
            ));
        }
        for argument in arguments {
            self.validate_prior_operand(*argument)?;
        }
        let additional = definition
            .nodes()
            .iter()
            .filter(|node| !matches!(node, CalculusNode::FormalComponent { .. }))
            .count();
        self.nodes
            .len()
            .checked_add(additional)
            .filter(|count| *count <= max_nodes && u32::try_from(*count).is_ok())
            .ok_or_else(|| {
                invalid_pure_operator(
                    "scalar pure-operator projection exceeds the expression node budget",
                )
            })?;
        let mut ids = Vec::with_capacity(definition.nodes().len());
        for node in definition.nodes() {
            let mapped = |id: crate::kernel::pure_operator::CalculusNodeId| {
                ids.get(id.index() as usize).copied().ok_or_else(|| {
                    invalid_pure_operator("scalar pure-operator projection has a forward operand")
                })
            };
            let id = match node {
                CalculusNode::FormalComponent { formal, axes } if axes.is_empty() => {
                    arguments[usize::from(*formal)]
                }
                CalculusNode::Rational { value, dimension } => {
                    self.constant(DynQuantity::new(value.as_f64(), *dimension))?
                }
                CalculusNode::Boolean(value) => self.constant(ValueLiteral::boolean(*value))?,
                CalculusNode::Compare(op, left, right) => {
                    self.compare(*op, mapped(*left)?, mapped(*right)?)?
                }
                CalculusNode::Not(value) => self.not(mapped(*value)?)?,
                CalculusNode::And(left, right) => self.and(mapped(*left)?, mapped(*right)?)?,
                CalculusNode::Or(left, right) => self.or(mapped(*left)?, mapped(*right)?)?,
                CalculusNode::UnaryMath(function, value) => {
                    self.unary_math(*function, mapped(*value)?)?
                }
                CalculusNode::Select {
                    condition,
                    then_value,
                    else_value,
                } => self.select(
                    mapped(*condition)?,
                    mapped(*then_value)?,
                    mapped(*else_value)?,
                )?,
                CalculusNode::Require { condition, value } => {
                    self.require(mapped(*condition)?, mapped(*value)?)?
                }
                CalculusNode::BoundInput(value) | CalculusNode::Differentiated { value, .. } => {
                    mapped(*value)?
                }
                CalculusNode::Neg(value) => self.neg(mapped(*value)?)?,
                CalculusNode::Add(left, right) => self.add(mapped(*left)?, mapped(*right)?)?,
                CalculusNode::Mul(left, right) => self.mul(mapped(*left)?, mapped(*right)?)?,
                _ => {
                    return Err(invalid_pure_operator(
                        "scalar projection cannot expand spatial component calculus",
                    ));
                }
            };
            ids.push(id);
        }
        mapped_root(&ids, definition.root().index())
    }
}

fn mapped_root(ids: &[ExprId], root: u32) -> Result<ExprId, Diagnostic> {
    ids.get(root as usize)
        .copied()
        .ok_or_else(|| invalid_pure_operator("scalar pure-operator projection root is unavailable"))
}
