//! Explicit scalar-domain constraints, separate from dimensions and component rank.
use super::*;
use eqiora_core::ScalarDomain;

impl PureValueClass {
    /// Pin an existing real/complex polynomial class to one scalar domain.
    /// Discrete domains have no pure-polynomial embedding.
    pub fn with_scalar_domain(mut self, domain: ScalarDomain) -> Result<Self, PureOperatorError> {
        if !matches!(domain, ScalarDomain::Real | ScalarDomain::Complex) {
            return Err(PureOperatorError::FormalTypeMismatch);
        }
        self.scalar_domain = Some(domain);
        Ok(self)
    }

    /// Explicit scalar-domain constraint, or none for the existing generic class.
    #[must_use]
    pub const fn scalar_domain(self) -> Option<ScalarDomain> {
        self.scalar_domain
    }
}

fn common(left: Option<ScalarDomain>, right: Option<ScalarDomain>) -> Option<ScalarDomain> {
    match (left, right) {
        (Some(ScalarDomain::Complex), _) | (_, Some(ScalarDomain::Complex)) => {
            Some(ScalarDomain::Complex)
        }
        (Some(ScalarDomain::Real), Some(ScalarDomain::Real)) => Some(ScalarDomain::Real),
        _ => None,
    }
}

pub(super) fn validate_result(
    formals: &[PureValueClass],
    result: PureValueClass,
) -> Result<(), PureOperatorError> {
    if let Some(expected) = result.scalar_domain() {
        let mut actual = Some(ScalarDomain::Real);
        // Unknown generic domains can still become certainly Complex when a later
        // constrained formal supplies Complex, so this must not short-circuit.
        for formal in formals {
            actual = common(actual, formal.scalar_domain());
        }
        if actual != Some(expected) {
            return Err(PureOperatorError::FormalTypeMismatch);
        }
    }
    Ok(())
}

pub(super) use super::dimensions::expression_domain;

#[cfg(test)]
mod tests {
    use super::*;
    use eqiora_core::ValueType;

    fn identity(class: PureValueClass) -> PureOperatorDefinition {
        let mut builder = CalculusBuilder::new([class], class).unwrap();
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
        let real = generic.with_scalar_domain(ScalarDomain::Real).unwrap();
        assert!(
            identity(real)
                .instantiate(&[argument(ScalarDomain::Real)])
                .is_ok()
        );
        assert_eq!(
            identity(real)
                .instantiate(&[argument(ScalarDomain::Complex)])
                .unwrap_err(),
            PureOperatorError::FormalTypeMismatch
        );
        assert!(
            identity(generic)
                .instantiate(&[argument(ScalarDomain::Complex)])
                .is_ok()
        );
        assert_ne!(identity(real).digest(), identity(generic).digest());
        for domain in [ScalarDomain::Boolean, ScalarDomain::Integer] {
            assert!(generic.with_scalar_domain(domain).is_err());
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
        let real = generic.with_scalar_domain(ScalarDomain::Real).unwrap();
        let callee = identity(real);
        for class in [
            generic,
            generic.with_scalar_domain(ScalarDomain::Complex).unwrap(),
        ] {
            let mut wrapper = CalculusBuilder::new([class], class).unwrap();
            let input = wrapper
                .push(CalculusNode::FormalComponent {
                    formal: 0,
                    axes: Box::new([]),
                })
                .unwrap();
            assert_eq!(
                wrapper.apply_scalar(&callee, &[input]).unwrap_err(),
                PureOperatorError::FormalTypeMismatch
            );
            assert_eq!(wrapper.finish(input).unwrap().nodes().len(), 1);
        }
        let mut wrapper = CalculusBuilder::new([real], real).unwrap();
        let input = wrapper
            .push(CalculusNode::FormalComponent {
                formal: 0,
                axes: Box::new([]),
            })
            .unwrap();
        let output = wrapper.apply_scalar(&callee, &[input]).unwrap();
        assert!(wrapper.finish(output).is_ok());
        let mut invalid_result = CalculusBuilder::new([generic], real).unwrap();
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
