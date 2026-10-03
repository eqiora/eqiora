//! Derived scalar execution projection of a retained canonical pure application.
use super::*;
use crate::kernel::pure_operator::{CalculusNode, PureOperatorInstantiation};
use eqiora_core::DynQuantity;

impl ExprDagBuilder {
    /// Append one ordered scalar component of a checked pure instantiation.
    /// Each argument supplies its complete row-major scalar coordinates.
    /// The canonical Model retains its original application and definition identity.
    /// `max_nodes` bounds the complete destination arena before expansion allocation.
    pub fn project_operator_component<I>(
        &mut self,
        instance: &PureOperatorInstantiation<'_, I>,
        arguments: &[impl AsRef<[ExprId]>],
        component: &[u32],
        max_nodes: usize,
    ) -> Result<ExprId, Diagnostic> {
        let definition = instance.definition();
        component_offset(instance.result_type().shape(), component)?;
        if arguments.len() != instance.arguments().len() {
            return Err(invalid_pure_operator(
                "component projection has incorrect formal arity",
            ));
        }
        for (argument, ty) in arguments.iter().zip(instance.arguments()) {
            let argument = argument.as_ref();
            if ty.shape().component_count() != Some(argument.len()) {
                return Err(invalid_pure_operator(
                    "component projection has incorrect argument shape",
                ));
            }
            for value in argument {
                self.validate_prior_operand(*value)?;
            }
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
                CalculusNode::FormalComponent { formal, axes } => {
                    let formal = usize::from(*formal);
                    let coordinates = axes
                        .iter()
                        .map(|axis| axis.resolve(component))
                        .collect::<Result<Vec<_>, _>>()
                        .map_err(|error| invalid_pure_operator(error.to_string()))?;
                    let offset =
                        component_offset(instance.arguments()[formal].shape(), &coordinates)?;
                    arguments[formal].as_ref()[offset]
                }
                CalculusNode::KroneckerDelta(left, right) => {
                    let left = left
                        .resolve(component)
                        .map_err(|error| invalid_pure_operator(error.to_string()))?;
                    let right = right
                        .resolve(component)
                        .map_err(|error| invalid_pure_operator(error.to_string()))?;
                    self.constant(DynQuantity::new(
                        if left == right { 1.0 } else { 0.0 },
                        eqiora_core::DimExponents::DIMENSIONLESS,
                    ))?
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

fn component_offset(
    shape: &eqiora_core::ValueShape,
    coordinates: &[u32],
) -> Result<usize, Diagnostic> {
    if shape.rank() != coordinates.len() {
        return Err(invalid_pure_operator(
            "component projection coordinate rank differs",
        ));
    }
    shape
        .extents()
        .iter()
        .zip(coordinates)
        .try_fold(0usize, |offset, (extent, coordinate)| {
            if *coordinate >= extent.get() {
                return Err(invalid_pure_operator(
                    "component projection coordinate is out of range",
                ));
            }
            offset
                .checked_mul(extent.get() as usize)
                .and_then(|offset| offset.checked_add(*coordinate as usize))
                .ok_or_else(|| invalid_pure_operator("component projection offset overflows"))
        })
}
