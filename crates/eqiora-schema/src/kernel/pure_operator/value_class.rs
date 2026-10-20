//! Closed value contracts and the existing dimension-polymorphic templates.
use super::*;
use eqiora_core::{ScalarDomain, ValueType};
use std::num::{NonZeroU16, NonZeroU32};

/// Pointwise value contract of a formal argument or result.
///
/// Concrete contracts retain the complete shared `ValueType`. Generic scalar
/// and spatial templates constrain only the properties they declare; application
/// typing supplies the remaining dimensions, domain and spatial extent.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct PureValueClass {
    rule: Rule,
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
enum Rule {
    Generic(Generic),
    Exact(ValueType),
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
struct Generic {
    spatial_rank: Option<NonZeroU16>,
    spatial_extent: Option<NonZeroU32>,
    dimension: Option<DimExponents>,
    scalar_domain: Option<ScalarDomain>,
}

impl PureValueClass {
    /// Retain an exact numeric value type, including its shape, frame and bases.
    ///
    /// # Errors
    /// Rejects discrete domains and ranks outside the component-calculus budget.
    pub fn from_value_type(value: ValueType) -> Result<Self, PureOperatorError> {
        if !matches!(
            value.scalar_domain(),
            ScalarDomain::Real | ScalarDomain::Complex
        ) || value.shape().rank() > usize::from(MAX_TENSOR_RANK)
        {
            return Err(PureOperatorError::FormalTypeMismatch);
        }
        Ok(Self {
            rule: Rule::Exact(value),
        })
    }

    /// Complete value type, when this contract has no unresolved constraints.
    #[must_use]
    pub const fn exact_type(&self) -> Option<&ValueType> {
        match &self.rule {
            Rule::Exact(value) => Some(value),
            Rule::Generic(_) => None,
        }
    }

    /// Invariant scalar on the common support.
    #[must_use]
    pub const fn invariant_scalar() -> Self {
        Self {
            rule: Rule::Generic(Generic {
                spatial_rank: None,
                spatial_extent: None,
                dimension: None,
                scalar_domain: None,
            }),
        }
    }

    /// Spatial Cartesian tensor with a positive rank and common axis extent.
    pub fn spatial_tensor(rank: u16) -> Result<Self, PureOperatorError> {
        let rank = NonZeroU16::new(rank).ok_or(PureOperatorError::InvalidResultRule)?;
        if rank.get() > MAX_TENSOR_RANK {
            return Err(PureOperatorError::InvalidResultRule);
        }
        Ok(Self {
            rule: Rule::Generic(Generic {
                spatial_rank: Some(rank),
                spatial_extent: None,
                dimension: None,
                scalar_domain: None,
            }),
        })
    }

    /// Constrain the common extent of an ordinary spatial template.
    pub fn with_spatial_extent(mut self, extent: u32) -> Result<Self, PureOperatorError> {
        let rank = self
            .spatial_rank()
            .ok_or(PureOperatorError::InvalidResultRule)?;
        let extent = NonZeroU32::new(extent).ok_or(PureOperatorError::InvalidResultRule)?;
        let shape =
            eqiora_core::ValueShape::new(std::iter::repeat_n(extent.get(), usize::from(rank)))
                .map_err(|_| PureOperatorError::InvalidResultRule)?;
        if shape.component_count().is_none() {
            return Err(PureOperatorError::InvalidResultRule);
        }
        self.rule = Rule::Generic(Generic {
            spatial_rank: NonZeroU16::new(rank),
            spatial_extent: Some(extent),
            dimension: self.dimension(),
            scalar_domain: self.scalar_domain(),
        });
        Ok(self.normalized())
    }

    /// Common spatial extent; channel extents remain part of the exact type.
    #[must_use]
    pub fn spatial_extent(&self) -> Option<u32> {
        match &self.rule {
            Rule::Generic(value) => value.spatial_extent.map(NonZeroU32::get),
            Rule::Exact(value) if self.spatial_rank().is_some() => {
                value.shape().extents().first().map(|extent| extent.get())
            }
            Rule::Exact(_) => None,
        }
    }

    /// Preserve the value contract while assigning its derived physical dimension.
    #[must_use]
    pub fn with_dimension(mut self, dimension: DimExponents) -> Self {
        match &mut self.rule {
            Rule::Generic(value) => value.dimension = Some(dimension),
            Rule::Exact(value) => {
                *value = value
                    .clone()
                    .with_dimension(dimension)
                    .expect("numeric value contract");
            }
        }
        self.normalized()
    }

    /// Physical dimension constraint, absent for a generic class.
    #[must_use]
    pub const fn dimension(&self) -> Option<DimExponents> {
        match &self.rule {
            Rule::Generic(value) => value.dimension,
            Rule::Exact(value) => Some(value.dimension()),
        }
    }

    /// Constrain the real or complex scalar domain without changing component meaning.
    pub fn with_scalar_domain(mut self, domain: ScalarDomain) -> Result<Self, PureOperatorError> {
        if !matches!(domain, ScalarDomain::Real | ScalarDomain::Complex) {
            return Err(PureOperatorError::FormalTypeMismatch);
        }
        match &mut self.rule {
            Rule::Generic(value) => value.scalar_domain = Some(domain),
            Rule::Exact(value) => {
                *value = value
                    .clone()
                    .with_scalar_domain(domain)
                    .ok_or(PureOperatorError::FormalTypeMismatch)?;
            }
        }
        Ok(self.normalized())
    }

    /// Explicit scalar-domain constraint, absent for a generic class.
    #[must_use]
    pub const fn scalar_domain(&self) -> Option<ScalarDomain> {
        match &self.rule {
            Rule::Generic(value) => value.scalar_domain,
            Rule::Exact(value) => Some(value.scalar_domain()),
        }
    }

    /// Rank of an ordinary spatial template; other shapes use their exact type.
    #[must_use]
    pub fn spatial_rank(&self) -> Option<u16> {
        match &self.rule {
            Rule::Generic(value) => value.spatial_rank.map(NonZeroU16::get),
            Rule::Exact(value)
                if value.frame() == ValueFrame::SpatialCartesian && value.array_rank() == 0 =>
            {
                Some(value.shape().rank() as u16)
            }
            Rule::Exact(_) => None,
        }
    }

    /// Number of ordered mathematical component axes.
    #[must_use]
    pub fn rank(&self) -> usize {
        match &self.rule {
            Rule::Generic(value) => value.spatial_rank.map_or(0, |rank| usize::from(rank.get())),
            Rule::Exact(value) => value.shape().rank(),
        }
    }

    /// Whether this contract denotes an invariant scalar.
    #[must_use]
    pub fn is_invariant_scalar(&self) -> bool {
        match &self.rule {
            Rule::Generic(value) => value.spatial_rank.is_none(),
            Rule::Exact(value) => {
                value.shape().is_scalar() && value.frame() == ValueFrame::Invariant
            }
        }
    }

    pub(super) fn validate_component(
        &self,
        axes: &[ComponentIndex],
        result: &Self,
    ) -> Result<(), PureOperatorError> {
        if axes.len() != self.rank() {
            return Err(PureOperatorError::FormalComponentRank);
        }
        for (index, axis) in axes.iter().enumerate() {
            if !axis.valid_result_axis(result.rank()) {
                return Err(PureOperatorError::ResultAxisOutOfRange);
            }
            let extent = self.axis_extent(index);
            let invalid = match axis {
                ComponentIndex::Fixed(coordinate) => extent.is_some_and(|n| *coordinate >= n),
                ComponentIndex::Result(index) => extent
                    .zip(result.axis_extent(usize::from(*index)))
                    .is_some_and(|(formal, result)| formal != result),
            };
            if invalid {
                return Err(PureOperatorError::FormalTypeMismatch);
            }
        }
        Ok(())
    }

    fn axis_extent(&self, axis: usize) -> Option<u32> {
        self.exact_type()
            .map(|value| value.shape().extents()[axis].get())
            .or_else(|| self.spatial_extent())
    }

    // Fully constrained templates and declared exact types have one identity.
    fn normalized(mut self) -> Self {
        let Rule::Generic(generic) = &self.rule else {
            return self;
        };
        let (Some(dimension), Some(domain)) = (generic.dimension, generic.scalar_domain) else {
            return self;
        };
        let value = match (generic.spatial_rank, generic.spatial_extent) {
            (None, _) => ValueType::scalar(domain, dimension),
            (Some(rank), Some(extent)) => ValueType::shaped(
                domain,
                dimension,
                eqiora_core::ValueShape::new(std::iter::repeat_n(
                    extent.get(),
                    usize::from(rank.get()),
                ))
                .expect("positive constrained extent"),
                ValueFrame::SpatialCartesian,
            ),
            (Some(_), None) => return self,
        };
        self.rule = Rule::Exact(value.expect("numeric constrained value type"));
        self
    }
}
