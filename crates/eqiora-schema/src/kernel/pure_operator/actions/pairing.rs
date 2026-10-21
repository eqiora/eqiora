//! Complete real pairings share coordinate enumeration and scalar contraction.
use super::*;

impl PureOperatorDefinition {
    /// Contract matching numeric values with `Re(sum(conj(left) * right))`.
    /// Complete component meaning must agree; units multiply and support is
    /// combined by ordinary typed application. No derivative rule is introduced.
    ///
    /// # Errors
    /// Rejects nonnumeric or mismatched layouts, dimension overflow and calculus limits.
    pub fn real_pairing(left: ValueType, right: ValueType) -> Result<Self, PureOperatorError> {
        let normalized =
            |value: &ValueType| value.clone().with_dimension(DimExponents::DIMENSIONLESS);
        if !matches!(
            left.scalar_domain(),
            ScalarDomain::Real | ScalarDomain::Complex
        ) || normalized(&left) != normalized(&right)
        {
            return Err(PureOperatorError::FormalTypeMismatch);
        }
        let dimension = left
            .dimension()
            .mul(right.dimension())
            .ok_or(PureOperatorError::ResultDimensionOverflow)?;
        let result = ValueType::scalar(ScalarDomain::Real, dimension)
            .map_err(|_| PureOperatorError::FormalTypeMismatch)?;
        let mut builder = CalculusBuilder::new(
            [
                PureValueClass::from_value_type(left.clone())?,
                PureValueClass::from_value_type(right)?,
            ],
            PureValueClass::from_value_type(result)?,
        )?;
        let mut sum = None;
        let domain = left.scalar_domain();
        for axes in coordinates(&left)? {
            let left = builder.push(CalculusNode::FormalComponent {
                formal: 0,
                axes: axes.clone(),
            })?;
            let right = builder.push(CalculusNode::FormalComponent { formal: 1, axes })?;
            let term = component(&mut builder, left, right, domain)?;
            sum = Some(match sum {
                Some(prior) => builder.push(CalculusNode::Add(prior, term))?,
                None => term,
            });
        }
        builder.finish(sum.ok_or(PureOperatorError::FormalTypeMismatch)?)
    }
}

pub(super) fn component(
    builder: &mut CalculusBuilder,
    left: CalculusNodeId,
    right: CalculusNodeId,
    domain: ScalarDomain,
) -> Result<CalculusNodeId, PureOperatorError> {
    if domain == ScalarDomain::Complex {
        let conjugate = builder.push(CalculusNode::UnaryMath(UnaryMathFunction::Conj, left))?;
        let product = builder.push(CalculusNode::Mul(conjugate, right))?;
        builder.push(CalculusNode::UnaryMath(UnaryMathFunction::Real, product))
    } else {
        builder.push(CalculusNode::Mul(left, right))
    }
}
