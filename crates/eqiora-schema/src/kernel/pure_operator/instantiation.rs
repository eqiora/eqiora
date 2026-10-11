//! Exact spatial extent, domain, frame and support admission of pure applications.
use super::dimensions::instantiate_dimension;
use super::*;

impl PureOperatorDefinition {
    /// Derive and validate one typed application.
    ///
    /// Rational polynomial definitions embed real inputs into the common
    /// real/complex scalar domain without changing dimension or component roles.
    ///
    /// # Errors
    /// Rejects arity, shape, frame, support, and result-rule mismatches before
    /// any lowered component expansion.
    pub fn instantiate<'a, I: Clone + Eq>(
        &'a self,
        arguments: &[ExpressionType<I>],
    ) -> Result<PureOperatorInstantiation<'a, I>, PureOperatorError> {
        if arguments.len() != self.formals.len() {
            return Err(PureOperatorError::ArityMismatch);
        }
        let mut common_support = None;
        let mut spatial_extent = None;
        let mut scalar_domain = eqiora_core::ScalarDomain::Real;
        for (rule, argument) in self.formals.iter().zip(arguments) {
            validate_argument_class(*rule, argument)?;
            if rule.spatial_rank().is_some() {
                let extent = argument.shape().extents()[0].get();
                if spatial_extent.is_some_and(|expected| expected != extent) {
                    return Err(PureOperatorError::FormalTypeMismatch);
                }
                spatial_extent = Some(extent);
            }
            scalar_domain = scalar_domain
                .common(argument.value_type.scalar_domain())
                .ok_or(PureOperatorError::FormalTypeMismatch)?;
            let Some(support) = argument.support.as_ref() else {
                continue;
            };
            if !matches!(
                support,
                SpatialSupport::Volume { .. }
                    | SpatialSupport::Boundary { .. }
                    | SpatialSupport::Coordinates { .. }
            ) {
                return Err(PureOperatorError::FormalTypeMismatch);
            }
            match &common_support {
                Some(expected) if expected != support => {
                    return Err(PureOperatorError::CommonSupportMismatch);
                }
                Some(_) => {}
                None => common_support = Some(support.clone()),
            }
        }
        if self
            .result
            .scalar_domain()
            .is_some_and(|expected| expected != scalar_domain)
        {
            return Err(PureOperatorError::FormalTypeMismatch);
        }
        let result_dimension = instantiate_dimension(&self.dimension, arguments)?;
        if self
            .result
            .dimension()
            .is_some_and(|expected| expected != result_dimension)
        {
            return Err(PureOperatorError::FormalTypeMismatch);
        }
        if let Some(support) = &common_support
            && let Some(dimensions) = support.ambient_dimensions()
        {
            let extent =
                u32::try_from(dimensions).map_err(|_| PureOperatorError::FormalTypeMismatch)?;
            if spatial_extent.is_some_and(|expected| expected != extent) {
                return Err(PureOperatorError::FormalTypeMismatch);
            }
            spatial_extent = Some(extent);
        }
        if self
            .result
            .spatial_extent()
            .is_some_and(|expected| spatial_extent != Some(expected))
        {
            return Err(PureOperatorError::FormalTypeMismatch);
        }
        for node in &self.nodes {
            let coordinates: &[ComponentIndex] = match node {
                CalculusNode::FormalComponent { axes, .. } => axes,
                CalculusNode::KroneckerDelta(left, right) => &[*left, *right],
                _ => continue,
            };
            if coordinates.iter().any(|index| {
                matches!(index,
                ComponentIndex::Fixed(coordinate)
                    if spatial_extent.is_none_or(|extent| *coordinate >= extent))
            }) {
                return Err(PureOperatorError::FormalTypeMismatch);
            }
        }
        let result_type = expression_type_for_class(
            self.result,
            scalar_domain,
            result_dimension,
            common_support,
            spatial_extent,
        )?;
        Ok(PureOperatorInstantiation {
            definition: self,
            arguments: arguments.to_vec(),
            result_type,
        })
    }
}

fn validate_argument_class<I>(
    class: PureValueClass,
    argument: &ExpressionType<I>,
) -> Result<(), PureOperatorError> {
    if class
        .scalar_domain()
        .is_some_and(|expected| expected != argument.value_type.scalar_domain())
    {
        return Err(PureOperatorError::FormalTypeMismatch);
    }
    if class
        .dimension()
        .is_some_and(|expected| expected != argument.dimension())
    {
        return Err(PureOperatorError::FormalTypeMismatch);
    }
    if class.spatial_extent().is_some_and(|expected| {
        argument
            .shape()
            .extents()
            .iter()
            .any(|extent| extent.get() != expected)
    }) {
        return Err(PureOperatorError::FormalTypeMismatch);
    }
    let dimensions = match argument.support.as_ref() {
        Some(
            SpatialSupport::Volume { dimensions, .. } | SpatialSupport::Boundary { dimensions, .. },
        ) => Some(*dimensions),
        Some(SpatialSupport::Coordinates { .. }) => None,
        None => argument
            .shape()
            .extents()
            .first()
            .map(|extent| extent.get() as usize),
        _ => return Err(PureOperatorError::FormalTypeMismatch),
    };
    match class.spatial_rank() {
        None if argument.shape().is_scalar() && argument.frame() == ValueFrame::Invariant => Ok(()),
        Some(rank)
            if argument.frame() == ValueFrame::SpatialCartesian
                && argument.value_type.array_rank() == 0
                && argument.shape().rank() == usize::from(rank)
                && dimensions
                    .and_then(|dimensions| u32::try_from(dimensions).ok())
                    .is_some_and(|dimension| {
                        dimension != 0
                            && argument
                                .shape()
                                .extents()
                                .iter()
                                .all(|extent| extent.get() == dimension)
                    }) =>
        {
            Ok(())
        }
        _ => Err(PureOperatorError::FormalTypeMismatch),
    }
}

fn expression_type_for_class<I>(
    class: PureValueClass,
    scalar_domain: eqiora_core::ScalarDomain,
    dimension: eqiora_core::DimExponents,
    support: Option<SpatialSupport<I>>,
    spatial_extent: Option<u32>,
) -> Result<ExpressionType<I>, PureOperatorError> {
    match class.spatial_rank() {
        None => Ok(ExpressionType::new(
            eqiora_core::ValueType::scalar(scalar_domain, dimension)
                .map_err(|_| PureOperatorError::FormalTypeMismatch)?,
            support,
        )),
        Some(rank) => {
            let extent = spatial_extent
                .filter(|extent| *extent != 0)
                .ok_or(PureOperatorError::FormalTypeMismatch)?;
            let shape =
                eqiora_core::ValueShape::new(std::iter::repeat_n(extent, usize::from(rank)))
                    .map_err(|_| PureOperatorError::FormalTypeMismatch)?;
            eqiora_core::ValueType::shaped(
                scalar_domain,
                dimension,
                shape,
                ValueFrame::SpatialCartesian,
            )
            .map(|value_type| ExpressionType::new(value_type, support))
            .map_err(|_| PureOperatorError::FormalTypeMismatch)
        }
    }
}
