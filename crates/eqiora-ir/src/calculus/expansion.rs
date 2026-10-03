//! Exact scalar expansion of schema-owned pure-operator definitions.

use eqiora_core::ValueShape;
use eqiora_schema::kernel::pure_operator::{
    CalculusNode, CalculusNodeId, ExactRational, PureOperatorInstantiation,
};
use eqiora_schema::kernel::typing::ExpressionType;

use super::CalculusError;

/// Lowered component expansion for a schema-owned typed instantiation.
///
/// Keeping this operation in L2 prevents the canonical definition vocabulary
/// from acquiring scalar SSA, normalization, or execution policy.
pub trait OperatorExpansionExt<I> {
    /// Expand one exact row-major result component without changing
    /// expression order.
    ///
    /// # Errors
    /// Rejects a component outside the derived result shape.
    fn component(&self, component: &[u32]) -> Result<ScalarCalculus<I>, CalculusError>;
}

impl<I: Clone> OperatorExpansionExt<I> for PureOperatorInstantiation<'_, I> {
    fn component(&self, component: &[u32]) -> Result<ScalarCalculus<I>, CalculusError> {
        validate_component(self.result_type().shape(), component)?;
        let definition = self.definition();
        let mut nodes = Vec::with_capacity(definition.nodes().len());
        for node in definition.nodes() {
            let node = match node {
                CalculusNode::Rational { value, dimension } => ScalarCalculusNode::Rational {
                    value: *value,
                    dimension: *dimension,
                },
                CalculusNode::FormalComponent { formal, axes } => {
                    let argument = self
                        .arguments()
                        .get(usize::from(*formal))
                        .ok_or(CalculusError::InvalidFormal(*formal))?;
                    let coordinates = axes
                        .iter()
                        .map(|axis| {
                            axis.resolve(component)
                                .map_err(|_| CalculusError::ResultAxisOutOfRange)
                        })
                        .collect::<Result<Box<[_]>, _>>()?;
                    validate_component(argument.shape(), &coordinates)?;
                    ScalarCalculusNode::FormalComponent(ScalarCalculusAtom {
                        formal: *formal,
                        component: coordinates,
                    })
                }
                CalculusNode::KroneckerDelta(left, right) => {
                    let left = left
                        .resolve(component)
                        .map_err(|_| CalculusError::ResultAxisOutOfRange)?;
                    let right = right
                        .resolve(component)
                        .map_err(|_| CalculusError::ResultAxisOutOfRange)?;
                    ScalarCalculusNode::Rational {
                        value: ExactRational::integer(i64::from(left == right)),
                        dimension: eqiora_core::DimExponents::DIMENSIONLESS,
                    }
                }
                CalculusNode::Differentiated { value, source, wrt } => {
                    ScalarCalculusNode::Differentiated {
                        value: *value,
                        source: *source,
                        wrt: *wrt,
                    }
                }
                CalculusNode::BoundInput(value) => ScalarCalculusNode::BoundInput(*value),
                CalculusNode::Neg(value) => ScalarCalculusNode::Neg(*value),
                CalculusNode::Add(left, right) => ScalarCalculusNode::Add(*left, *right),
                CalculusNode::Mul(left, right) => ScalarCalculusNode::Mul(*left, *right),
                _ => return Err(CalculusError::NonPolynomialCalculus),
            };
            nodes.push(node);
        }
        Ok(ScalarCalculus {
            argument_types: self.arguments().to_vec(),
            result_type: self.result_type().clone(),
            result_component: component.into(),
            nodes,
            root: definition.root(),
        })
    }
}

fn validate_component(shape: &ValueShape, component: &[u32]) -> Result<(), CalculusError> {
    if shape.rank() != component.len()
        || shape
            .extents()
            .iter()
            .zip(component)
            .any(|(extent, coordinate)| *coordinate >= extent.get())
    {
        Err(CalculusError::ComponentOutOfRange)
    } else {
        Ok(())
    }
}

/// Exact formal coordinate in one expanded scalar component.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct ScalarCalculusAtom {
    formal: u16,
    component: Box<[u32]>,
}

impl ScalarCalculusAtom {
    /// Zero-based formal slot.
    #[must_use]
    pub const fn formal(&self) -> u16 {
        self.formal
    }

    /// Exact component coordinate; empty for a scalar formal.
    #[must_use]
    pub const fn component(&self) -> &[u32] {
        &self.component
    }
}

/// Scalar calculus after exact result-component substitution.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ScalarCalculusNode {
    /// Exact mathematical literal.
    Rational {
        value: ExactRational,
        dimension: eqiora_core::DimExponents,
    },
    /// One exact formal component.
    FormalComponent(ScalarCalculusAtom),
    /// One retained local formal binding.
    BoundInput(CalculusNodeId),
    /// Retained differentiation history, with an already expanded value.
    Differentiated {
        value: CalculusNodeId,
        source: CalculusNodeId,
        wrt: CalculusNodeId,
    },
    /// Ordered negation.
    Neg(CalculusNodeId),
    /// Ordered addition.
    Add(CalculusNodeId, CalculusNodeId),
    /// Ordered multiplication.
    Mul(CalculusNodeId, CalculusNodeId),
}

/// Ordered executable component expression plus optional exact proof view.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ScalarCalculus<I> {
    pub(super) argument_types: Vec<ExpressionType<I>>,
    pub(super) result_type: ExpressionType<I>,
    pub(super) result_component: Box<[u32]>,
    pub(super) nodes: Vec<ScalarCalculusNode>,
    pub(super) root: CalculusNodeId,
}

impl<I> ScalarCalculus<I> {
    /// Complete types of the formal arguments in slot order.
    #[must_use]
    pub fn argument_types(&self) -> &[ExpressionType<I>] {
        &self.argument_types
    }

    /// Complete shaped result type from which this component was selected.
    #[must_use]
    pub const fn result_type(&self) -> &ExpressionType<I> {
        &self.result_type
    }

    /// Exact row-major result coordinate represented by this scalar calculus.
    #[must_use]
    pub const fn result_component(&self) -> &[u32] {
        &self.result_component
    }

    /// Topologically ordered scalar nodes. Execution must retain this order.
    #[must_use]
    pub fn nodes(&self) -> &[ScalarCalculusNode] {
        &self.nodes
    }

    /// Scalar root.
    #[must_use]
    pub const fn root(&self) -> CalculusNodeId {
        self.root
    }
}

#[cfg(test)]
mod tests {
    use eqiora_core::ValueFrame;
    use eqiora_core::{DimExponents, ValueShape};
    use eqiora_schema::kernel::pure_operator::PureOperatorDefinition;
    use eqiora_schema::kernel::typing::{ExpressionType, SpatialSupport};

    use super::*;

    fn volume_tensor(domain: &str) -> ExpressionType<&str> {
        ExpressionType::shaped(
            DimExponents::DIMENSIONLESS,
            ValueShape::new([2, 2]).unwrap(),
            ValueFrame::SpatialCartesian,
            Some(SpatialSupport::Volume {
                domain,
                dimensions: 2,
            }),
        )
        .unwrap()
    }

    fn volume_scalar(domain: &str) -> ExpressionType<&str> {
        ExpressionType::scalar(
            DimExponents::DIMENSIONLESS,
            Some(SpatialSupport::Volume {
                domain,
                dimensions: 2,
            }),
        )
    }

    fn volume_vector(domain: &str) -> ExpressionType<&str> {
        ExpressionType::shaped(
            DimExponents::DIMENSIONLESS,
            ValueShape::new([2]).unwrap(),
            ValueFrame::SpatialCartesian,
            Some(SpatialSupport::Volume {
                domain,
                dimensions: 2,
            }),
        )
        .unwrap()
    }

    #[test]
    fn standard_operators_share_one_ordered_component_expansion() {
        let symmetric = PureOperatorDefinition::symmetric_part().unwrap();
        let expansion = symmetric.instantiate(&[volume_tensor("body")]).unwrap();
        let off_diagonal = expansion.component(&[0, 1]).unwrap();
        assert_eq!(off_diagonal.nodes().len(), 5);
        off_diagonal
            .normalize()
            .unwrap()
            .verify(&off_diagonal)
            .unwrap();

        let isotropic = PureOperatorDefinition::isotropic_lift().unwrap();
        let expansion = isotropic.instantiate(&[volume_scalar("body")]).unwrap();
        let diagonal = expansion.component(&[1, 1]).unwrap();
        let off_diagonal = expansion.component(&[0, 1]).unwrap();
        assert_eq!(off_diagonal.nodes().len(), 3);
        assert!(matches!(
            off_diagonal.nodes()[usize::try_from(off_diagonal.root().index()).unwrap()],
            ScalarCalculusNode::Mul(_, _)
        ));
        assert_ne!(
            diagonal.normalize().unwrap().after_digest(),
            off_diagonal.normalize().unwrap().after_digest()
        );
    }

    #[test]
    fn dyadic_product_expands_through_the_generic_formal_calculus() {
        let definition = PureOperatorDefinition::dyadic_product().unwrap();
        let expansion = definition
            .instantiate(&[volume_vector("body"), volume_vector("body")])
            .unwrap();
        let component = expansion.component(&[0, 1]).unwrap();
        assert_eq!(component.nodes().len(), 3);
        assert!(matches!(
            &component.nodes()[0],
            ScalarCalculusNode::FormalComponent(atom)
                if atom.formal() == 0 && atom.component() == [0]
        ));
        assert!(matches!(
            &component.nodes()[1],
            ScalarCalculusNode::FormalComponent(atom)
                if atom.formal() == 1 && atom.component() == [1]
        ));
        assert!(matches!(
            component.nodes()[2],
            ScalarCalculusNode::Mul(_, _)
        ));
        component.normalize().unwrap().verify(&component).unwrap();
    }
}
