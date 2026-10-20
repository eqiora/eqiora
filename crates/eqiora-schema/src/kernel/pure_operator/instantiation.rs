//! Spatial extent, domain, frame, and support checks for pure applications.
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
        for (rule, argument) in self.formals.iter().zip(arguments) {
            validate_argument_class(rule, argument)?;
            if argument.frame() == ValueFrame::SpatialCartesian {
                let extent = argument.shape().extents()[argument.value_type.array_rank()].get();
                if spatial_extent.is_some_and(|expected| expected != extent) {
                    return Err(PureOperatorError::FormalTypeMismatch);
                }
                spatial_extent = Some(extent);
            }
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
        let formals = arguments
            .iter()
            .map(|argument| PureValueClass::from_value_type(argument.value_type.clone()))
            .collect::<Result<Vec<_>, _>>()?;
        let component_types = dimensions::component_types(&formals, &self.nodes)?;
        let scalar_domain = domains::result_domain(
            self.result.scalar_domain(),
            Some(component_types[self.root.index() as usize].scalar_domain()),
        )?
        .ok_or(PureOperatorError::FormalTypeMismatch)?;
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
            .is_some_and(|expected| spatial_extent.is_some_and(|actual| actual != expected))
        {
            return Err(PureOperatorError::FormalTypeMismatch);
        }
        let result_type = expression_type_for_class(
            &self.result,
            scalar_domain,
            result_dimension,
            common_support,
            spatial_extent.or(self.result.spatial_extent()),
        )?;
        let result_class = PureValueClass::from_value_type(result_type.value_type.clone())?;
        for node in &self.nodes {
            match node {
                CalculusNode::FormalComponent { formal, axes } => {
                    formals[usize::from(*formal)].validate_component(axes, &result_class)?;
                }
                CalculusNode::KroneckerDelta(left, right) => {
                    if let (ComponentIndex::Fixed(left), ComponentIndex::Fixed(right)) =
                        (left, right)
                        && spatial_extent
                            .or(self.result.spatial_extent())
                            .is_none_or(|extent| *left >= extent || *right >= extent)
                    {
                        return Err(PureOperatorError::FormalTypeMismatch);
                    }
                    for (axis, other) in [(*left, *right), (*right, *left)] {
                        if let ComponentIndex::Result(index) = axis {
                            let extent = result_type.shape().extents()[usize::from(index)].get();
                            validate_component_axis(other, extent, &result_type)?;
                        }
                    }
                }
                _ => {}
            }
        }
        Ok(PureOperatorInstantiation {
            definition: self,
            arguments: arguments.to_vec(),
            result_type,
            component_types,
        })
    }
}

pub(super) fn validate_argument_class<I>(
    class: &PureValueClass,
    argument: &ExpressionType<I>,
) -> Result<(), PureOperatorError> {
    if let Some(exact) = class.exact_type() {
        if exact != &argument.value_type {
            return Err(PureOperatorError::FormalTypeMismatch);
        }
        return validate_spatial_shape(argument);
    }
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
    class: &PureValueClass,
    scalar_domain: eqiora_core::ScalarDomain,
    dimension: eqiora_core::DimExponents,
    support: Option<SpatialSupport<I>>,
    spatial_extent: Option<u32>,
) -> Result<ExpressionType<I>, PureOperatorError> {
    if let Some(exact) = class.exact_type() {
        let result = ExpressionType::new(exact.clone(), support);
        validate_spatial_shape(&result)?;
        if result.frame() == ValueFrame::SpatialCartesian
            && spatial_extent.is_some_and(|extent| {
                result.shape().extents()[exact.array_rank()..]
                    .iter()
                    .any(|actual| actual.get() != extent)
            })
        {
            return Err(PureOperatorError::FormalTypeMismatch);
        }
        return Ok(result);
    }
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

fn validate_spatial_shape<I>(value: &ExpressionType<I>) -> Result<(), PureOperatorError> {
    if value.frame() != ValueFrame::SpatialCartesian {
        return Ok(());
    }
    let axes = &value.shape().extents()[value.value_type.array_rank()..];
    let extent = value
        .support
        .as_ref()
        .and_then(SpatialSupport::ambient_dimensions)
        .and_then(|extent| u32::try_from(extent).ok())
        .or_else(|| axes.first().map(|axis| axis.get()));
    if extent.is_none() || axes.iter().any(|axis| Some(axis.get()) != extent) {
        return Err(PureOperatorError::FormalTypeMismatch);
    }
    Ok(())
}

fn validate_component_axis<I>(
    axis: ComponentIndex,
    extent: u32,
    result: &ExpressionType<I>,
) -> Result<(), PureOperatorError> {
    let valid = match axis {
        ComponentIndex::Fixed(coordinate) => coordinate < extent,
        ComponentIndex::Result(index) => result
            .shape()
            .extents()
            .get(usize::from(index))
            .is_some_and(|actual| actual.get() == extent),
    };
    if valid {
        Ok(())
    } else {
        Err(PureOperatorError::FormalTypeMismatch)
    }
}
