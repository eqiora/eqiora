//! Explicit scalar-domain constraints, separate from dimensions and component rank.
use super::*;
use eqiora_core::ScalarDomain;

/// The body owns its result domain; unused inputs cannot change it.
pub(super) fn validate_result(
    formals: &[PureValueClass],
    result: &PureValueClass,
    nodes: &[CalculusNode],
    root: CalculusNodeId,
) -> Result<(), PureOperatorError> {
    result_domain(
        result.scalar_domain(),
        expression_domain(formals, nodes, root)?,
    )
    .map(|_| ())
}

pub(super) fn result_domain(
    expected: Option<ScalarDomain>,
    actual: Option<ScalarDomain>,
) -> Result<Option<ScalarDomain>, PureOperatorError> {
    if let Some(expected) = expected {
        if actual != Some(expected)
            && !(expected == ScalarDomain::Complex && actual == Some(ScalarDomain::Real))
        {
            return Err(PureOperatorError::FormalTypeMismatch);
        }
        Ok(Some(expected))
    } else {
        Ok(actual)
    }
}

pub(super) use super::dimensions::expression_domain;

#[cfg(test)]
mod tests {
    use super::*;
    use eqiora_core::ValueType;

    fn identity(class: PureValueClass) -> PureOperatorDefinition {
        let mut builder = CalculusBuilder::new([class.clone()], class.clone()).unwrap();
        let root = builder
            .push(CalculusNode::FormalComponent {
                formal: 0,
                axes: Box::new([]),
            })
            .unwrap();
        builder.finish(root).unwrap()
    }
    fn argument(domain: ScalarDomain) -> ExpressionType<u32> {
        ExpressionType::new(
            ValueType::scalar(domain, DimExponents::DIMENSIONLESS).expect("checked scalar type"),
            None,
        )
    }

    #[test]
    fn native_concrete_real_constraint_rejects_complex_without_changing_generic_classes() {
        let generic =
            PureValueClass::invariant_scalar().with_dimension(DimExponents::DIMENSIONLESS);
        let real = generic
            .clone()
            .with_scalar_domain(ScalarDomain::Real)
            .unwrap();
        assert!(
            identity(real.clone())
                .instantiate(&[argument(ScalarDomain::Real)])
                .is_ok()
        );
        assert_eq!(
            identity(real.clone())
                .instantiate(&[argument(ScalarDomain::Complex)])
                .unwrap_err(),
            PureOperatorError::FormalTypeMismatch
        );
        assert!(
            identity(generic.clone())
                .instantiate(&[argument(ScalarDomain::Complex)])
                .is_ok()
        );
        assert_ne!(
            identity(real.clone()).digest(),
            identity(generic.clone()).digest()
        );
        for domain in [ScalarDomain::Boolean, ScalarDomain::Integer] {
            assert!(generic.clone().with_scalar_domain(domain).is_err());
        }
        let tensor = PureOperatorDefinition::symmetric_part().unwrap();
        let shape = eqiora_core::ValueShape::new([2, 2]).unwrap();
        let value = ValueType::shaped(
            ScalarDomain::Complex,
            DimExponents::DIMENSIONLESS,
            shape,
            ValueFrame::SpatialCartesian,
        )
        .unwrap();
        let input = ExpressionType::new(
            value,
            Some(SpatialSupport::Volume {
                domain: 1,
                dimensions: 2,
            }),
        );
        assert_eq!(
            tensor
                .instantiate(&[input])
                .unwrap()
                .result_type()
                .value_type
                .scalar_domain(),
            ScalarDomain::Complex
        );
    }

    #[test]
    fn composition_cannot_erase_a_concrete_real_formal_constraint() {
        let generic = PureValueClass::invariant_scalar();
        let real = generic
            .clone()
            .with_scalar_domain(ScalarDomain::Real)
            .unwrap();
        let callee = identity(real.clone());
        for class in [
            generic.clone(),
            generic
                .clone()
                .with_scalar_domain(ScalarDomain::Complex)
                .unwrap(),
        ] {
            let mut wrapper = CalculusBuilder::new([class.clone()], class.clone()).unwrap();
            let input = wrapper
                .push(CalculusNode::FormalComponent {
                    formal: 0,
                    axes: Box::new([]),
                })
                .unwrap();
            assert_eq!(
                wrapper.apply_scalar_inputs(&callee, &[input]).unwrap_err(),
                PureOperatorError::FormalTypeMismatch
            );
            assert_eq!(wrapper.finish(input).unwrap().nodes().len(), 1);
        }
        let mut wrapper = CalculusBuilder::new([real.clone()], real.clone()).unwrap();
        let input = wrapper
            .push(CalculusNode::FormalComponent {
                formal: 0,
                axes: Box::new([]),
            })
            .unwrap();
        let output = wrapper.apply_scalar_inputs(&callee, &[input]).unwrap();
        assert!(wrapper.finish(output).is_ok());
        let mut invalid_result = CalculusBuilder::new([generic.clone()], real.clone()).unwrap();
        let root = invalid_result
            .push(CalculusNode::FormalComponent {
                formal: 0,
                axes: Box::new([]),
            })
            .unwrap();
        assert_eq!(
            invalid_result.finish(root).unwrap_err(),
            PureOperatorError::FormalTypeMismatch
        );
    }
}
