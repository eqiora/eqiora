use eqiora_schema::kernel::typing::{ExpressionType, TypedResidual};
use eqiora_schema::kernel::{ExprId, ExprNode};

use super::{CalculusError, OperatorDefinitionDigest, PureOperatorDefinition, expr_index};

/// Standard calculus definitions already represented by immutable V4 Kernel
/// expression nodes.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum StandardPureOperator {
    /// Symmetric part of a square spatial tensor.
    SymmetricPart,
    /// Isotropic lift of an invariant scalar.
    IsotropicLift,
}

impl StandardPureOperator {
    fn definition(self) -> Result<PureOperatorDefinition, CalculusError> {
        Ok(match self {
            Self::SymmetricPart => PureOperatorDefinition::symmetric_part(),
            Self::IsotropicLift => PureOperatorDefinition::isotropic_lift(),
        }?)
    }
}

/// Typed proof that one immutable Kernel node is the selected standard pure
/// operator application.
#[derive(Debug, Clone)]
pub struct OperatorApplicationProof<I> {
    operator: StandardPureOperator,
    definition_digest: OperatorDefinitionDigest,
    operand: ExprId,
    result_type: ExpressionType<I>,
}

/// Typed proof that one immutable Kernel node applies one exact
/// content-addressed pure-operator definition.
///
/// Unlike [`OperatorApplicationProof`], this proof is not keyed by a closed
/// Rust enum. The expected definition supplies mathematical identity, while
/// the Kernel application supplies ordered argument nodes. This is the
/// extension seam for package-defined pure operators.
#[derive(Debug, Clone)]
pub struct PureOperatorApplicationProof<I> {
    definition_digest: OperatorDefinitionDigest,
    arguments: Box<[ExprId]>,
    result_type: ExpressionType<I>,
}

impl<I: Clone + Eq> OperatorApplicationProof<I> {
    /// Classify one exact node from a fully typed residual.
    ///
    /// A different node returns `Ok(None)`. A matching node whose type cannot
    /// replay through the content-addressed definition fails explicitly.
    pub fn classify(
        residual: &TypedResidual<I>,
        value: ExprId,
        operator: StandardPureOperator,
    ) -> Result<Option<Self>, CalculusError> {
        let Some(node) = residual.expression().node(value) else {
            return Err(CalculusError::InvalidExpressionNode);
        };
        let operand = match (operator, node) {
            (StandardPureOperator::SymmetricPart, ExprNode::SymmetricPart(operand))
            | (StandardPureOperator::IsotropicLift, ExprNode::IsotropicLift(operand)) => *operand,
            _ => return Ok(None),
        };
        let operand_type = residual
            .node_types()
            .get(expr_index(operand, residual.node_types().len())?)
            .ok_or(CalculusError::InvalidExpressionNode)?
            .clone();
        let expected_result = residual
            .node_types()
            .get(expr_index(value, residual.node_types().len())?)
            .ok_or(CalculusError::InvalidExpressionNode)?;
        let definition = operator.definition()?;
        let expansion = definition.instantiate(&[operand_type])?;
        if expansion.result_type() != expected_result {
            return Err(CalculusError::ApplicationResultMismatch);
        }
        Ok(Some(Self {
            operator,
            definition_digest: definition.digest(),
            operand,
            result_type: expansion.result_type().clone(),
        }))
    }

    /// Standard operator family.
    #[must_use]
    pub const fn operator(&self) -> StandardPureOperator {
        self.operator
    }

    /// Exact content identity of the replayed definition.
    #[must_use]
    pub const fn definition_digest(&self) -> OperatorDefinitionDigest {
        self.definition_digest
    }

    /// Canonical operand node.
    #[must_use]
    pub const fn operand(&self) -> ExprId {
        self.operand
    }

    /// Exact derived result type.
    #[must_use]
    pub const fn result_type(&self) -> &ExpressionType<I> {
        &self.result_type
    }
}

impl<I: Clone + Eq> PureOperatorApplicationProof<I> {
    /// Classify one exact application from a fully typed residual.
    ///
    /// A non-application or an application of another definition returns
    /// `Ok(None)`. A matching digest whose retained definition or typed
    /// instantiation cannot be replayed fails explicitly.
    pub fn classify(
        residual: &TypedResidual<I>,
        value: ExprId,
        expected: &PureOperatorDefinition,
    ) -> Result<Option<Self>, CalculusError> {
        let Some(node) = residual.expression().node(value) else {
            return Err(CalculusError::InvalidExpressionNode);
        };
        let ExprNode::PureOperatorApplication(application) = node else {
            return Ok(None);
        };
        let definition_digest = expected.digest();
        if application.definition() != definition_digest {
            return Ok(None);
        }
        let retained = residual
            .expression()
            .definition(definition_digest)
            .ok_or(CalculusError::ApplicationDefinitionMissing)?;
        if retained != expected {
            return Err(CalculusError::ApplicationDefinitionMismatch);
        }
        let argument_types = application
            .arguments()
            .iter()
            .map(|argument| {
                residual
                    .node_types()
                    .get(expr_index(*argument, residual.node_types().len())?)
                    .cloned()
                    .ok_or(CalculusError::InvalidExpressionNode)
            })
            .collect::<Result<Vec<_>, _>>()?;
        let instantiation = retained.instantiate(&argument_types)?;
        let expected_result = residual
            .node_types()
            .get(expr_index(value, residual.node_types().len())?)
            .ok_or(CalculusError::InvalidExpressionNode)?;
        if instantiation.result_type() != expected_result {
            return Err(CalculusError::ApplicationResultMismatch);
        }
        Ok(Some(Self {
            definition_digest,
            arguments: application.arguments().into(),
            result_type: instantiation.result_type().clone(),
        }))
    }

    /// Exact content identity of the replayed definition.
    #[must_use]
    pub const fn definition_digest(&self) -> OperatorDefinitionDigest {
        self.definition_digest
    }

    /// Canonical argument nodes in formal-slot order.
    #[must_use]
    pub const fn arguments(&self) -> &[ExprId] {
        &self.arguments
    }

    /// Exact derived result type.
    #[must_use]
    pub const fn result_type(&self) -> &ExpressionType<I> {
        &self.result_type
    }
}

#[cfg(test)]
mod tests {
    use eqiora_core::ValueFrame;
    use eqiora_core::entity::kinds;
    use eqiora_core::{DimExponents, Id, ValueShape};
    use eqiora_schema::kernel::typing::{
        ExpressionType, RootContract, SpatialSupport, TypedResidual,
    };
    use eqiora_schema::kernel::{ExprDagBuilder, SymbolRef};

    use super::*;

    #[test]
    fn typed_kernel_applications_replay_definition_identity() {
        let field = Id::<kinds::Field>::new();
        let mut expression = ExprDagBuilder::new();
        let field_value = expression.symbol(SymbolRef::Field(field)).unwrap();
        let symmetric = expression.symmetric_part(field_value).unwrap();
        let dag = expression.finish([symmetric]).unwrap();
        let tensor_type = ExpressionType::shaped(
            DimExponents::DIMENSIONLESS,
            ValueShape::new([2, 2]).unwrap(),
            ValueFrame::SpatialCartesian,
            Some(SpatialSupport::Volume {
                domain: "body",
                dimensions: 2,
            }),
        )
        .unwrap();
        let typed = TypedResidual::infer(
            dag,
            Some(SpatialSupport::Volume {
                domain: "body",
                dimensions: 2,
            }),
            |_| None,
            RootContract::ComponentwiseResidual,
            |_| Ok::<_, ()>(tensor_type.clone()),
        )
        .unwrap();
        let proof = OperatorApplicationProof::classify(
            &typed,
            symmetric,
            StandardPureOperator::SymmetricPart,
        )
        .unwrap()
        .unwrap();
        assert_eq!(proof.operand(), field_value);
        assert_eq!(
            proof.definition_digest(),
            PureOperatorDefinition::symmetric_part().unwrap().digest()
        );
        assert_eq!(proof.result_type(), &tensor_type);
    }

    #[test]
    fn content_addressed_application_proof_preserves_definition_and_argument_order() {
        let left = Id::<kinds::Field>::new();
        let right = Id::<kinds::Field>::new();
        let definition = PureOperatorDefinition::dyadic_product().unwrap();
        let mut expression = ExprDagBuilder::new();
        let left_value = expression.symbol(SymbolRef::Field(left)).unwrap();
        let right_value = expression.symbol(SymbolRef::Field(right)).unwrap();
        let dyadic = expression
            .pure_operator(&definition, [left_value, right_value])
            .unwrap();
        let dag = expression.finish([dyadic]).unwrap();
        let support = SpatialSupport::Volume {
            domain: "body",
            dimensions: 2,
        };
        let vector_type = ExpressionType::shaped(
            DimExponents::DIMENSIONLESS,
            ValueShape::new([2]).unwrap(),
            ValueFrame::SpatialCartesian,
            Some(support.clone()),
        )
        .unwrap();
        let typed = TypedResidual::infer(
            dag,
            Some(support),
            |_| None,
            RootContract::ComponentwiseResidual,
            |_| Ok::<_, ()>(vector_type.clone()),
        )
        .unwrap();

        let proof = PureOperatorApplicationProof::classify(&typed, dyadic, &definition)
            .unwrap()
            .unwrap();
        assert_eq!(proof.definition_digest(), definition.digest());
        assert_eq!(proof.arguments(), [left_value, right_value]);
        assert_eq!(
            proof.result_type().shape().extents(),
            [
                std::num::NonZeroU32::new(2).unwrap(),
                std::num::NonZeroU32::new(2).unwrap(),
            ]
        );
        assert!(
            PureOperatorApplicationProof::classify(
                &typed,
                dyadic,
                &PureOperatorDefinition::symmetric_part().unwrap(),
            )
            .unwrap()
            .is_none()
        );
    }
}
